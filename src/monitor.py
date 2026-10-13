"""训练健康度监控: expert load / router entropy / cross-loop overlap / hidden norm."""
import torch


def overlap_from_per_loop(per_layer_per_loop):
    """per_layer_per_loop: [n_layers][n_loops] tensor[N..., k] -> 每 adjacent 圈对的 overlap 均值."""
    n_layers, n_loops = len(per_layer_per_loop), len(per_layer_per_loop[0])
    if n_loops < 2:
        return []
    out = []
    for li in range(n_loops - 1):
        vals = []
        for layer in per_layer_per_loop:
            a = _multi_hot(layer[li], n_e_max=32)
            b = _multi_hot(layer[li + 1], n_e_max=32)
            inter = (a & b).sum(-1).float()
            union = (a | b).sum(-1).float().clamp_min(1)
            vals.append((inter / union).mean().item())
        out.append(sum(vals) / len(vals))
    return out


def _multi_hot(ti, n_e_max):
    shape = ti.shape[:-1]
    oh = torch.zeros(*shape, n_e_max, device=ti.device, dtype=torch.bool)
    oh.scatter_(-1, ti.long(), True)
    return oh


def m_stats_from_blocks(model):
    """从最近一次 forward 收集 M 段 MoE 统计 (Block.moe.stats)."""
    loads, ents, zs = [], [], []
    for b in model.M:
        s = b.moe.stats
        if s is None:
            continue
        loads.append(s["load"])
        ents.append(s["entropy"])
        zs.append(s["z_loss"])
    if not loads:
        return None
    load = torch.stack(loads).mean(0)  # [n_routed] M 段平均 expert load
    return {
        "expert_load": load.tolist(),
        "load_maxmin": float(load.max() / load.min().clamp_min(1e-9)),
        "dead_expert": bool((load < 0.01).any()),
        "router_entropy": sum(ents) / len(ents),
        "router_z": sum(zs) / len(zs),
    }
