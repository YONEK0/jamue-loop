"""E/M/D 原生循环 MoE 架构 (V6 试验版)。

结构: prelude E (dense) -> recurrent core M (shared+4 routed top-2, 循环 R 次)
      -> coda D (dense)。attention/norm 逐物理层独立; M 段每层带 depth embedding
      参与路由; 第 2 圈起 h0 再注入 (V5 反复读配置)。
数值: V100 无 bf16, 训练走 fp16 amp; router/softmax 等 fp32 (见 forward 内 .float())。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 1  # <mask>; 0=pad, 2..vocab 起为正常 token


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.float()
        n = x * torch.rsqrt(dt.pow(2).mean(-1, keepdim=True) + self.eps)
        return n.type_as(x) * self.w


def apply_rope(x, cos, sin):
    # x [B,H,T,hd]; cos/sin [T,hd/2]
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).type_as(x)


class GQA(nn.Module):
    def __init__(self, d, n_head, n_kv):
        super().__init__()
        assert d % n_head == 0
        self.h, self.kv, self.hd = n_head, n_kv, d // n_head
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, n_kv * self.hd, bias=False)
        self.v = nn.Linear(d, n_kv * self.hd, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v)  # 双向, 无 causal mask
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Expert(nn.Module):
    """SwiGLU FFN, 单 expert (shared 或 routed 之一)."""

    def __init__(self, d, inter):
        super().__init__()
        self.gate = nn.Linear(d, inter, bias=False)
        self.up = nn.Linear(d, inter, bias=False)
        self.down = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, d, inter, n_routed=4, top_k=2):
        super().__init__()
        self.n_routed, self.top_k = n_routed, top_k
        self.shared = Expert(d, inter)
        self.routed = nn.ModuleList(Expert(d, inter) for _ in range(n_routed))
        self.router = nn.Linear(d, n_routed, bias=False)
        # 圈数条件: 路由输入 = h + depth_emb[d], 跨圈分化的结构性来源 (V6 S4)
        self.depth_emb = nn.Embedding(16, d)
        nn.init.zeros_(self.depth_emb.weight)  # 初始等价于无深度条件, 让训练自己学分化
        self.stats = None  # 最近一次 forward 的路由统计 (monitor 用)

    def forward(self, x, depth):
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        r_in = xf + self.depth_emb.weight[depth]
        logits = self.router(r_in.float()).float()           # fp32 数值路径
        z_loss = logits.pow(2).mean()
        probs = logits.softmax(-1)
        tv, ti = probs.topk(self.top_k, dim=-1)              # softmax-then-topK
        w = tv / tv.sum(-1, keepdim=True)
        sel = torch.zeros_like(probs).scatter_(-1, ti, 1.0)
        w_full = torch.zeros_like(probs).scatter_(-1, ti, w)  # [N, n_routed] 按 expert id 的权重
        load = sel.sum(0) / sel.sum(0).sum()
        balance = self.n_routed * (load * probs.mean(0)).sum()
        out = self.shared(xf)
        for e in range(self.n_routed):
            m = sel[:, e] > 0
            if m.any():
                out[m] = out[m] + self.routed[e](xf[m]) * w_full[m, e].unsqueeze(-1).to(out.dtype)
        # 统计 (no_grad, 不进计算图)
        with torch.no_grad():
            ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
            self.stats = {
                "load": load.cpu(), "entropy": ent.item(),
                "topi": ti.detach().reshape(*shape[:-1], self.top_k),
                "z_loss": float(z_loss.detach()),
            }
        return out.reshape(shape), balance, z_loss


class Block(nn.Module):
    def __init__(self, d, n_head, n_kv, inter, moe=False, n_routed=4, top_k=2):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = GQA(d, n_head, n_kv)
        self.moe = MoE(d, inter, n_routed, top_k) if moe else Expert(d, inter)

    def forward(self, x, cos, sin, depth=0, collect=None):
        x = x + self.attn(self.n1(x), cos, sin)
        if isinstance(self.moe, MoE):
            y, bal, zl = self.moe(self.n2(x), depth)
            x = x + y
            self.aux = (bal, zl)
            if collect is not None:
                collect.append(self.moe.stats["topi"])
        else:
            x = x + self.moe(self.n2(x))
            self.aux = None
        return x


class LoopMoEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["hidden"]
        self.embed = nn.Embedding(cfg["vocab"], d)
        self.E = nn.ModuleList(Block(d, cfg["heads"], cfg["kv"], cfg["inter"])
                               for _ in range(cfg["n_e"]))
        self.M = nn.ModuleList(Block(d, cfg["heads"], cfg["kv"], cfg["inter"],
                                     moe=True, n_routed=cfg["n_routed"],
                                     top_k=cfg["top_k"]) for _ in range(cfg["n_m"]))
        self.D = nn.ModuleList(Block(d, cfg["heads"], cfg["kv"], cfg["inter"])
                               for _ in range(cfg["n_d"]))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, cfg["vocab"], bias=False)
        self.h0_coef = cfg.get("h0_coef", 0.5)
        self.max_R = cfg.get("max_R", 8)
        self._cos = self._sin = None

    def _rope(self, T, device):
        if self._cos is None or self._cos.shape[0] < T:
            hd = self.cfg["hidden"] // self.cfg["heads"]
            base = self.cfg.get("rope_theta", 1000000.0)  # Jaume 实键 rope_theta=1e6
            inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
            t = torch.arange(8192, device=device).float()
            fr = torch.outer(t, inv)
            self._cos, self._sin = fr.cos(), fr.sin()
        return self._cos[:T], self._sin[:T]

    def encode(self, ids, R, collect=None):
        """返回 M 段末端的逐圈 hidden 列表 (collect=True 时) 与最终 hidden."""
        x = self.embed(ids)
        cos, sin = self._rope(ids.shape[1], ids.device)
        for b in self.E:
            x = b(x, cos, sin)
        h0 = x
        per_loop = collect is not None
        hs = [x] if per_loop else None
        for d in range(R):
            if d > 0:
                x = x + self.h0_coef * h0            # 第 2 圈起 h0 再注入
            for b in self.M:
                x = b(x, cos, sin, depth=d, collect=collect)
            if per_loop:
                hs.append(x)
        for b in self.D:
            x = b(x, cos, sin)
        return (x, hs) if per_loop else (x, None)

    def forward(self, ids, R):
        x, _ = self.encode(ids, R)
        return self.head(self.norm(x))

    def aux_losses(self):
        bal = x = 0.0
        items = [b.aux for b in self.M if b.aux is not None]
        if not items:
            return None
        bal = torch.stack([a[0] for a in items]).mean()
        zl = torch.stack([a[1] for a in items]).mean()
        return bal, zl

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def mlm_mask(ids, vocab, mask_prob=0.15):
    """BERT 式 80/10/10 动态 mask, 返回 (model_input, labels)."""
    labels = ids.clone()
    p = torch.rand_like(ids.float())
    m = p < mask_prob
    labels[~m] = -100
    r = torch.rand_like(ids.float())
    ids = ids.clone()
    ids[m & (r < 0.8)] = MASK_ID
    rnd = m & (r >= 0.8) & (r < 0.9)
    ids[rnd] = torch.randint(2, vocab, (int(rnd.sum()),), device=ids.device)
    return ids, labels
