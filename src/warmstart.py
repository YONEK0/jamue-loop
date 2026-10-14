"""Jaume 350m 权重 -> LoopMoEModel 的 warm-start 映射 (继承式部分 upcycling)。

映射规则 (E4/M12/D4, Jaume 16 层 hidden 512):
  E 层 i        <- Jaume L0..3   (dense 化: 4 个 expert 权重均值)
  M 物理层 j    <- Jaume L4..15  (routed experts 1:1; shared 用 4 expert 均值)
  D 层 k        <- Jaume L12..15 (尾部层复制)
  embed/norm/head 照搬。
注意: Jaume 实际键名以其发布的 checkpoint 为准, 本模块做"尽力匹配 + 完整报告",
首次运行必须看 match report; 匹配率低不会崩, 会 fallback 到部分随机初始化。
"""
import glob
import json
import os

import torch


def _load_sd(path):
    files = glob.glob(os.path.join(path, "*.safetensors")) or glob.glob(os.path.join(path, "*.bin"))
    sd = {}
    for f in files:
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd.update(load_file(f))
        else:
            sd.update(torch.load(f, map_location="cpu"))
    return sd


def _avg_expert(sd, ja_layer, comp):
    """Jaume 某 MoE 层 4 个 expert 的某分量均值 -> dense/shared 初始化。

    实键: backbone.layers.N.mlp.experts.{0..3}.{gate,up,down}_proj.weight
    """
    ws = [sd.get(f"{ja_layer}.mlp.experts.{e}.{comp}.weight") for e in range(4)]
    ws = [w for w in ws if w is not None]
    if not ws:
        return None
    return torch.stack([w.float() for w in ws]).mean(0)


def load_jaume(model, path, vocab_override=None):
    """返回 (report dict)。直接 in-place 修改 model 参数。"""
    sd = _load_sd(path)
    if not sd:
        return {"error": f"no weight files under {path}", "matched": 0}
    cfgf = [f for f in glob.glob(os.path.join(path, "*.json"))
            if "config" in os.path.basename(f).lower()]
    jacfg = json.load(open(cfgf[0])) if cfgf else {}
    if vocab_override is None and "vocab_size" in jacfg:
        vocab_override = jacfg["vocab_size"]

    # 键名前缀自适应 (Jaume 可能有 backbone. 前缀)
    pre = ""
    if any(k.startswith("backbone.") for k in sd):
        pre = "backbone."
    lay = [k for k in sd if ".layers.0." in k or k.startswith(f"{pre}layers.0.")]
    prefix = (lay[0].split("layers.0.")[0] + "layers.") if lay else f"{pre}layers."
    n_ja = 16

    report = {"jaume_keys": len(sd), "prefix": prefix, "matched": 0, "missed": []}

    def take(k, dst_p):
        v = sd.get(k)
        if v is not None and tuple(v.shape) == tuple(dst_p.shape):
            dst_p.data.copy_(v.float())
            report["matched"] += 1
            return True
        report["missed"].append(k)
        return False

    with torch.no_grad():
        # backbone 顶层: embed/norm (实键 backbone.embed_tokens.weight / backbone.norm.weight)
        bb = prefix.rsplit("layers.", 1)[0]  # "backbone."
        take(f"{bb}embed_tokens.weight", model.embed.weight)
        take(f"{bb}norm.weight", model.norm.w)
        # E 段 (dense 化)
        for i, blk in enumerate(model.E):
            _take_block(blk, sd, prefix, i, report, dense_from_moe=True)
        # M 段 (routed 1:1 + shared 均值)
        for j, blk in enumerate(model.M):
            _take_block(blk, sd, prefix, 4 + j, report, moe=True)
        # D 段 (尾部复制)
        for k, blk in enumerate(model.D):
            _take_block(blk, sd, prefix, 12 + k, report, dense_from_moe=True)
        # head: Jaume 350m-mlm 无独立 lm_head (实键只有 backbone./pooling./projection.),
        # 用 embed 词表权重平铺初始化 (训练会调整)
        if hasattr(model, "head"):
            hk = None
            for cand in [f"{bb}lm_head.weight", "lm_head.weight", f"{prefix}lm_head.weight"]:
                if cand in sd:
                    hk = cand
                    break
            if hk:
                take(hk, model.head.weight)
            else:
                ev = sd.get(f"{bb}embed_tokens.weight")
                if ev is not None and tuple(ev.shape) == tuple(model.head.weight.shape):
                    model.head.weight.data.copy_(ev.float())
                    report["matched"] += 1
                    report["head_init"] = "copied_from_embed"
                else:
                    report["head_init"] = "random"
    report["has_pooling"] = "pooling.query" in sd
    report["has_projection"] = any(k.startswith("projection.") for k in sd)
    if vocab_override and vocab_override != model.cfg["vocab"]:
        report["vocab_note"] = f"jaume vocab={vocab_override} != cfg {model.cfg['vocab']}: 重训 head 或改 config"
    rate = report["matched"] / max(1, report["matched"] + len(report["missed"]))
    report["match_rate"] = round(rate, 3)
    return report


def _take_block(blk, sd, prefix, li, report, dense_from_moe=False, moe=False):
    p = f"{prefix}{li}."

    def take(k, dst):
        v = sd.get(k)
        if v is not None and tuple(v.shape) == tuple(dst.shape):
            dst.data.copy_(v.float())
            report["matched"] += 1
            return
        report["missed"].append(k)

    # 先探测存在性, 避免备用键名进入 missed 统计
    take(p + ("norm1.weight" if (p + "norm1.weight") in sd else "input_layernorm.weight"), blk.n1.w)
    take(p + ("norm2.weight" if (p + "norm2.weight") in sd else "post_attention_layernorm.weight"), blk.n2.w)
    for a, b in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")]:
        take(p + f"self_attn.{a}.weight", getattr(blk.attn, b).weight)

    if moe:
        for comp in ["gate_proj", "up_proj", "down_proj"]:
            for e in range(len(blk.moe.routed)):
                take(p + f"mlp.experts.{e}.{comp}.weight",
                     getattr(blk.moe.routed[e], {"gate_proj": "gate", "up_proj": "up",
                                                 "down_proj": "down"}[comp]).weight)
        # shared expert 用 4 个 routed expert 均值初始化 (无现成 dense 权重)
        for comp, attr in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
            avg = _avg_expert(sd, p[:-1], comp)
            dst = getattr(blk.moe.shared, attr).weight
            if avg is not None and tuple(avg.shape) == tuple(dst.shape):
                dst.data.copy_(avg)
                report["matched"] += 1
        # router: 实键 mlp.gate.weight (与 expert 的 gate_proj 不同)
        take(p + "mlp.gate.weight", blk.moe.router.weight)
    elif dense_from_moe:
        # Jaume 层是 MoE: 用 4 expert 均值给 dense FFN
        for comp, attr in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
            avg = _avg_expert(sd, p[:-1], comp)
            dst = getattr(blk.moe, attr).weight
            if avg is not None and tuple(avg.shape) == tuple(dst.shape):
                dst.data.copy_(avg)
                report["matched"] += 1
