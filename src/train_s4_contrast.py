"""S4: 对比训练 (InfoNCE in-batch + 温度 + Matryoshka 截断维度)。

在 S1 产物之上加 attention pooling + projection head; 支持冻结主干 (省时) 或全参。
Matryoshka dims 从 config 读 (Jaume: [128,256,384,512])。
产出: runs/<run>/ckpt_s4_final.pt + history.jsonl (各维度 in-batch acc)。

用法: python3 src/train_s4_contrast.py --ckpt runs/s1_350m/ckpt_final.pt \
        --pairs data/pairs.jsonl --run runs/s4_350m --steps 2000 --max-minutes 60
"""
import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from arch import LoopMoEModel
from data import PairData
from train_s1_mlm import save_ckpt  # 存档协议复用


class PoolHead(nn.Module):
    """attention pooling (单可学习 query) + projection, 输出 L2 归一化."""

    def __init__(self, d, dims):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / d ** 0.5)
        self.proj = nn.Linear(d, max(dims), bias=False)
        self.dims = dims

    def forward(self, h, pad_mask=None):
        # h [B,T,D]
        a = F.scaled_dot_product_attention(
            self.q.view(1, 1, -1).expand(h.shape[0], -1, -1),
            h, h)                                   # [B,1,D]
        pooled = a.squeeze(1)
        e = F.normalize(self.proj(pooled), dim=-1)
        return {dm: e[..., :dm] for dm in self.dims}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="S1 产物 ckpt")
    ap.add_argument("--pairs", default="data/pairs.jsonl")
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--freeze-backbone", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pay = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = LoopMoEModel(pay["cfg"]).to(dev)
    model.load_state_dict(pay["model"])
    head = PoolHead(model.cfg["hidden"], model.cfg.get("mrl_dims", [128, 256, 384, 512])).to(dev)
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)
        params = head.parameters()
        print("[s4] backbone 冻结, 只训 pooling/projection")
    else:
        params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))

    data = PairData(args.pairs)
    os.makedirs(args.run, exist_ok=True)
    logf = open(os.path.join(args.run, "history.jsonl"), "a", encoding="utf-8")
    t0 = time.time()

    class _A:  # save_ckpt 需要的 args 形状
        run = args.run

    for step in range(1, args.steps + 1):
        q, p = data.batch(step, args.bs, args.seq)
        q, p = q.to(dev), p.to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            hq, _ = model.encode(q, R=args.R)
            hp, _ = model.encode(p, R=args.R)
            eq, ep = head(hq), head(hp)
            losses, accs = [], {}
            for dm in head.dims:
                logits = eq[dm] @ ep[dm].T / args.temp
                tgt = torch.arange(len(q), device=dev)
                losses.append(F.cross_entropy(logits.float(), tgt))
                with torch.no_grad():
                    accs[dm] = round((logits.argmax(-1) == tgt).float().mean().item(), 3)
            loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if step % 20 == 0:
            rec = {"step": step, "loss": round(loss.item(), 4), "acc": accs,
                   "min": round((time.time() - t0) / 60, 1)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[s4] {step} loss={rec['loss']} acc@{max(head.dims)}={accs[max(head.dims)]}")
        if (time.time() - t0) / 60 > args.max_minutes:
            print("[s4] 硬时限到, 提前结束")
            break

    torch.save({"model": model.state_dict(), "head": head.state_dict(),
                "cfg": model.cfg, "args": vars(args), "step": step},
               os.path.join(args.run, "ckpt_s4_final.pt.tmp"))
    os.replace(os.path.join(args.run, "ckpt_s4_final.pt.tmp"),
               os.path.join(args.run, "ckpt_s4_final.pt"))
    print(f"[s4] done {step} steps ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
