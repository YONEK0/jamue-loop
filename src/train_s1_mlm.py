"""S1: MLM 循环适应训练 (E/M/D 循环 MoE, fp16 amp, 存档协议, 硬时限).

核心机制:
- 稀疏深度监督: 每 step 从 DEPTHS 采样 K 个深度各做一次前向, loss 取均值;
  R=1 也被监督 (深度级联检索需要浅深度可用)。
- aux: load-balance + router z-loss (fp32)。
- 监控: expert load/entropy/cross-loop overlap 每 monitor_every 步落盘 history.jsonl;
  转折点 (首次路由分化 / 首次深度价值) 触发永久档。
- 存档: atomic tmp->replace, rolling latest/prev, RNG+optimizer+data 偏移全保存。

用法 (350M 平台):
  python3 src/train_s1_mlm.py --config configs/350m_warmstart.json \
      --data data/train.bin --val data/val.bin --warmstart assets/jaume/350m-mlm \
      --run runs/s1_350m --sup-tokens 30000000 --max-minutes 140
本地小验证:
  python3 src/train_s1_mlm.py --config configs/local_smoke.json --run runs/smoke \
      --sup-tokens 50000 --max-minutes 3
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from arch import LoopMoEModel, mlm_mask
from data import BinCorpus
from monitor import m_stats_from_blocks, overlap_from_per_loop

DEPTHS = [1, 2, 4, 8]


def save_ckpt(model, opt, scaler, args, st, kind, turning=None):
    import shutil
    run = args.run
    os.makedirs(run, exist_ok=True)
    payload = {
        "model": model.state_dict(), "opt": opt.state_dict(),
        "scaler": scaler.state_dict(), "step": st["step"], "sup_tokens": st["sup"],
        "data_off": st["off"], "rng_py": random.getstate(),
        "rng_np": np.random.get_state(), "rng_torch": torch.get_rng_state(),
        "cfg": model.cfg, "args": vars(args),
        "cuda_rng": [t.cpu() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }
    if kind == "latest":
        cur = os.path.join(run, "ckpt_latest.pt")
        prev = os.path.join(run, "ckpt_prev.pt")
        tmp = cur + ".tmp"
        torch.save(payload, tmp)
        if os.path.exists(cur):
            os.replace(cur, prev)
        os.replace(tmp, cur)
        if turning:  # 永久档副本, 不被 rolling 覆盖
            tp = os.path.join(run, f"ckpt_turning_{turning}_{st['step']}.pt")
            shutil.copyfile(cur, tp)
            return os.path.basename(tp)
        return "ckpt_latest.pt"
    name = f"ckpt_{kind}.pt"
    tmp = os.path.join(run, name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(run, name))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--val", default="data/val.bin")
    ap.add_argument("--run", required=True)
    ap.add_argument("--warmstart", default=None)
    ap.add_argument("--sup-tokens", type=int, default=30_000_000)
    ap.add_argument("--max-minutes", type=float, default=140.0)
    ap.add_argument("--bs", type=int, default=None, help="micro batch (默认读 config)")
    ap.add_argument("--seq", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=2, help="每 step 采样深度数")
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--z-coef", type=float, default=1e-3)
    ap.add_argument("--monitor-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 42))
    model = LoopMoEModel(cfg).to(dev)
    n_par = model.param_count()
    print(f"[s1] params={n_par/1e6:.1f}M device={dev} cfg={os.path.basename(args.config)}")

    if args.warmstart:
        from warmstart import load_jaume
        rep = load_jaume(model, args.warmstart)
        print(f"[s1] warmstart: matched={rep.get('matched')} rate={rep.get('match_rate')} "
              f"missed={len(rep.get('missed', []))}")
        if rep.get("vocab_note"):
            print(f"[s1] !! {rep['vocab_note']}")
        os.makedirs(args.run, exist_ok=True)
        _ = json.dump(rep, open(os.path.join(args.run, "warmstart_report.json"), "w"), indent=1)

    data = BinCorpus(args.data)
    bs = args.bs or cfg.get("bs", 16)
    seq = args.seq or cfg.get("seq", 512)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))

    st = {"step": 0, "sup": 0, "off": 0, "t0": time.time()}
    os.makedirs(args.run, exist_ok=True)
    hist_path = os.path.join(args.run, "history.jsonl")
    flags = {"routing_div": False, "depth_value": False}
    if args.resume:
        # 分段会话模式: 在 rolling/final/prev 中选 step 最新的档恢复
        best = None
        for name in ["ckpt_latest.pt", "ckpt_final.pt", "ckpt_prev.pt"]:
            p = os.path.join(args.run, name)
            if not os.path.exists(p):
                continue
            try:
                pay = torch.load(p, map_location=dev, weights_only=False)
            except Exception as e:
                print(f"[s1] !! {name} 损坏跳过: {type(e).__name__}")
                continue
            if best is None or pay["step"] > best[1]["step"]:
                best = (name, pay)
        if best:
            name, pay = best
            model.load_state_dict(pay["model"])
            opt.load_state_dict(pay["opt"])
            scaler.load_state_dict(pay["scaler"])
            st.update(step=pay["step"], sup=pay["sup_tokens"], off=pay["data_off"])
            random.setstate(pay["rng_py"])
            np.random.set_state(pay["rng_np"])
            torch.set_rng_state(pay["rng_torch"].cpu())  # RNG 必须是 CPU ByteTensor
            if pay.get("cuda_rng") and dev == "cuda":
                torch.cuda.set_rng_state_all([t.cpu() for t in pay["cuda_rng"]])
            print(f"[s1] resumed from {name} step={st['step']} sup={st['sup']/1e6:.2f}M")

    tokens_per_step = bs * seq
    logf = open(hist_path, "a", encoding="utf-8")

    while st["sup"] < args.sup_tokens and \
            (time.time() - st["t0"]) / 60 < args.max_minutes:
        st["step"] += 1
        x = data.batch(st["off"], bs, seq).to(dev)
        st["off"] = (st["off"] + tokens_per_step) % data.n
        depths = random.sample(DEPTHS, k=min(args.K, len(DEPTHS)))
        losses, ces = [], []
        for d in depths:
            mi, labels = mlm_mask(x, cfg["vocab"])
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                logits = model(mi, R=d)
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, cfg["vocab"]).float(), labels.view(-1),
                    ignore_index=-100)
            losses.append(ce)
        loss = torch.stack(losses).mean()
        aux = model.aux_losses()
        if aux is not None:
            bal, zl = aux
            loss = loss + args.aux_coef * bal + args.z_coef * zl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        st["sup"] += tokens_per_step
        ces = [float(l.detach()) for l in losses]

        if st["step"] % args.monitor_every == 0:
            mst = m_stats_from_blocks(model)
            # cross-loop overlap: 额外一次 no-grad R=8 前向, 收集每层每圈 top-k
            with torch.no_grad():
                collect = []
                _ = model.encode(x, R=model.max_R, collect=collect)
            per_layer = [[collect[d * len(model.M) + l] for d in range(model.max_R)]
                         for l in range(len(model.M))]
            ov = overlap_from_per_loop(per_layer)
            rec = {"step": st["step"], "sup_m": round(st["sup"] / 1e6, 3),
                   "ce_by_depth": dict(zip(map(str, depths), ces)),
                   "ce_mean": sum(ces) / len(ces),
                   "cross_loop_overlap": [round(v, 3) for v in ov],
                   "min": round((time.time()-st["t0"])/60, 1)}
            if mst:
                rec.update({k: mst[k] for k in
                            ["expert_load", "load_maxmin", "dead_expert", "router_entropy"]})
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            el = rec.get("expert_load")
            print(f"[s1] {rec['step']} sup={rec['sup_m']}M ce={rec['ce_mean']:.3f} "
                  f"ov={rec['cross_loop_overlap'][:3]} "
                  f"load={[round(v,2) for v in el] if el else None} "
                  f"ent={rec.get('router_entropy', -1):.3f}")
            # 转折点: 首次路由分化 (overlap 有意义且不锁死) / 首次深度价值 (CE(d8)<CE(d1))
            if not flags["routing_div"] and ov and 0.15 < min(ov[:1] or [0]) < 0.95 \
                    and mst and mst["router_entropy"] > 0.3 and mst["load_maxmin"] < 10:
                flags["routing_div"] = True
                _ = save_ckpt(model, opt, scaler, args, st, "latest", "div")
                print("[s1] 转折点: 路由分化首现 -> 永久档")
            if not flags["depth_value"] and len(depths) == 2 and \
                    min(depths) == 1 and max(depths) == 8 and ces[1] < ces[0]:
                flags["depth_value"] = True
                _ = save_ckpt(model, opt, scaler, args, st, "latest", "depth")
                print("[s1] 转折点: 深度价值首现 (CE8<CE1) -> 永久档")
        if st["step"] % args.save_every == 0:
            _ = save_ckpt(model, opt, scaler, args, st, "latest")

    _ = save_ckpt(model, opt, scaler, args, st, "latest")  # rolling 链同步到最新, 供下会话 resume
    _ = save_ckpt(model, opt, scaler, args, st, "final")
    print(f"[s1] done step={st['step']} sup={st['sup']/1e6:.2f}M "
          f"({(time.time()-st['t0'])/60:.1f} min)")


if __name__ == "__main__":
    main()
