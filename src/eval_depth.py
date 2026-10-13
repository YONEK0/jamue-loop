"""评测: CE(d) 深度曲线 + 样本级退出扫描 (相邻圈 hidden 收敛阈值 -> 有效平均深度).

用法: python3 src/eval_depth.py --ckpt runs/s1/ckpt_final.pt --data data/val.bin \
        --out results/depth_s1.json [--batches 20] [--depths 1,2,4,8]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from arch import LoopMoEModel, mlm_mask
from data import BinCorpus


def load_model(ckpt, dev):
    pay = torch.load(ckpt, map_location=dev, weights_only=False)
    model = LoopMoEModel(pay["cfg"]).to(dev)
    model.load_state_dict(pay["model"])
    model.eval()
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/val.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--depths", default="1,2,4,8")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, dev)
    data = BinCorpus(args.data)
    V = model.cfg["vocab"]
    depths = [int(d) for d in args.depths.split(",")]

    ces = {}
    for d in depths:
        tot, nb = 0.0, 0
        for i in range(args.batches):
            x = data.batch(i * 4096 * 7 + 13, args.bs, args.seq).to(dev)
            mi, labels = mlm_mask(x, V)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                logits = model(mi, R=d)
            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, V).float(), labels.view(-1), ignore_index=-100)
            tot += ce.item()
            nb += 1
        ces[d] = round(tot / max(nb, 1), 4)
        print(f"CE(d={d}) = {ces[d]}")

    # 样本级退出扫描: 相邻圈 M 段末端 hidden 相对变化率, 阈值 tau -> 有效平均深度
    x = data.batch(31, args.bs, args.seq).to(dev)
    _, hs = model.encode(x, R=model.max_R, collect=[])
    deltas = []
    for a, b in zip(hs[:-1], hs[1:]):
        deltas.append(((b.float() - a.float()).norm(dim=-1) /
                       a.float().norm(dim=-1).clamp_min(1e-6)).mean(-1))  # [B] 每样本
    D = torch.stack(deltas)  # [R, B] 圈间变化率
    exits = {}
    for tau in [0.05, 0.10, 0.20, 0.30, 0.50]:
        # 首个 delta<tau 的圈 +1 即该样本退出深度; 全不满足则 max_R
        ok = D < tau
        first = torch.where(ok.any(0), ok.float().argmax(0),
                            torch.full((D.shape[1],), model.max_R - 1, device=dev))
        exits[tau] = round(float(first.float().mean() + 1), 3)

    out = {"ckpt": args.ckpt, "ce_by_depth": ces,
           "best_depth": min(ces, key=ces.get),
           "depth_gain": round(ces[depths[0]] - ces[depths[-1]], 4),
           "loop_delta_mean": [round(float(d.mean()), 4) for d in deltas],
           "sample_exit_mean_depth_by_tau": exits}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _ = json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(list(ces.keys()), list(ces.values()), "o-")
        ax[0].set_xlabel("depth R"); ax[0].set_ylabel("MLM CE"); ax[0].set_title("CE(d)")
        ax[1].plot(range(1, len(deltas) + 1), out["loop_delta_mean"], "s-")
        ax[1].set_xlabel("loop"); ax[1].set_ylabel("rel. delta"); ax[1].set_title("hidden drift")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot -> {args.plot}")


if __name__ == "__main__":
    main()
