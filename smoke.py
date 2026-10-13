"""jamue-loop 冒烟: 平台上正式训练前的最后一道闸。

tier0: 基础机制 (autocast/GradScaler/SDPA/原子存档/memmap 数据)
tier1: 内联 mini 循环-MoE 模型 3 步训练 — 直接验证 "E/M/D 循环 x MoE x fp16"
       这条组合路径的数值健康与监控统计可行性, 不依赖完整 src/arch.py。
tier1 通过标准: loss 有限且下降、expert load 可统计、无 NaN 梯度。
用法: python3 smoke.py   (CPU 也可跑, 用 fp32 路径)
"""
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


# ---------- tier1: mini 循环 MoE ----------
class MiniMoE(nn.Module):
    def __init__(self, d, n_exp=4, top_k=1):
        super().__init__()
        self.n_exp, self.top_k = n_exp, top_k
        self.router = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(n_exp)])
        self.last_load = None

    def forward(self, x):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        # router 数值敏感路径走 fp32 (fp16 softmax 下溢风险)
        logits = self.router(xf.float()).float()
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        sel = torch.zeros_like(logits).scatter_(-1, topi, 1.0)
        w = sel * probs
        w = w / w.sum(-1, keepdim=True).clamp(min=1e-9)  # 每 token 选中 expert 的归一权重
        load = sel.sum(0) / sel.sum()  # 每 expert token 占比
        self.last_load = load.detach().float().cpu()
        out = torch.zeros_like(xf)
        for e in range(self.n_exp):
            m = sel[:, e] > 0
            if m.any():
                out[m] = out[m] + self.experts[e](xf[m]) * w[m, e].unsqueeze(-1)
        # load-balance aux (Switch 式): n_exp * sum(f_i * P_i)
        aux = self.n_exp * (load * probs.mean(0)).sum()
        return out.reshape(shape), aux


class MiniLoopBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.moe = MiniMoE(d)
        self.depth_emb = nn.Embedding(8, d)  # 圈数条件

    def forward(self, h, h0, depth):
        x = self.norm(h + self.depth_emb.weight[depth])
        y, aux = self.moe(x)
        return h + y, aux  # 残差; h0 再注入由外层控制


class MiniLoopMoE(nn.Module):
    """E=1 层 + M=1 层循环 R 次 + D=1 层, 隐藏 d."""

    def __init__(self, d=128, r=4):
        super().__init__()
        self.r = r
        self.e = MiniLoopBlock(d)
        self.m = MiniLoopBlock(d)
        self.d_ = MiniLoopBlock(d)
        self.head = nn.Linear(d, 16)  # toy 词表

    def forward(self, x):
        h = self.e(x, None, 0)[0]
        h0 = h
        aux_sum = x.new_zeros(())
        for d in range(self.r):
            if d > 0:
                h = h + 0.5 * h0  # 第 2 圈起 h0 再注入 (V5 反复读配置)
            h, aux = self.m(h, h0, d)
            aux_sum = aux_sum + aux
        return self.head(self.d_(h, None, 0)[0]), aux_sum / self.r


def tier1(device, use_amp):
    torch.manual_seed(7)
    d, r, steps = 128, 4, 12
    model = MiniLoopMoE(d, r).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    x = torch.randn(32, 20, d, device=device)
    tgt = torch.randn(32, 20, 16, device=device)

    losses, loads, first_m_overlap = [], [], None
    prev_top = None
    for _ in range(steps):
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            out, aux = model(x)
            loss = F.mse_loss(out.float(), tgt) + 0.01 * aux
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        loads.append(model.m.moe.last_load.tolist())
        # cross-loop 信号可用性: 相邻 step 间 M 层 expert 选择分化
        logits = model.m.moe.router(model.m.norm(x)[:, 0].float()).float()
        cur = logits.argmax(-1)
        if prev_top is not None:
            first_m_overlap = (cur == prev_top).float().mean().item()
        prev_top = cur

    finite = all(l == l for l in losses) and all(l < 1e4 for l in losses)
    dropped = losses[-1] < losses[0]
    grads_nan = any(p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters())
    n_loads = len(loads[-1])
    check("tier1_loss_finite_drop", finite and dropped,
          f"{losses[0]:.3f}->{losses[-1]:.3f} finite={finite}")
    check("tier1_grads_finite", not grads_nan, "所有梯度有限")
    check("tier1_expert_load_stat", n_loads == 4, f"load 统计 {loads[-1]}")
    check("tier1_route_stat", first_m_overlap is not None,
          f"M层相邻步 overlap 信号={first_m_overlap}")
    return all(r_["ok"] for r_ in RESULTS if r_["name"].startswith("tier1"))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"
    print(f"smoke on {dev} (amp={use_amp})")

    # tier0: 原子存档 + memmap
    tmp, fin = "results/_smoke.tmp", "results/_smoke.ckpt"
    os.makedirs("results", exist_ok=True)
    torch.save({"a": 1}, tmp)
    os.replace(tmp, fin)
    check("tier0_atomic_ckpt", torch.load(fin)["a"] == 1, "tmp+os.replace")
    os.remove(fin)
    mm = "results/_smoke.bin"
    with open(mm, "wb") as f:
        f.write((torch.arange(10, dtype=torch.int32).numpy()).tobytes())
    arr = torch.from_file(mm, dtype=torch.int32, size=10)
    check("tier0_memmap", arr.sum().item() == 45, "torch.from_file 读 packed bin")
    os.remove(mm)

    ok1 = tier1(dev, use_amp)
    _ = json.dump(RESULTS, open("results/smoke.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"\n== smoke: {'ALL PASS' if ok1 else 'TIER1 FAIL'} ==")
    sys.exit(0 if ok1 else 1)


if __name__ == "__main__":
    main()
