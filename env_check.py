"""jamue-loop 环境体检：GPU/torch/sm70/SDPA/amp fp16 全探测，输出 go/no-go。

用法: python3 env_check.py [--json results/env_check.json]
退出码 0 = 全部通过; 1 = 有致命项失败。
CPU 上运行时只做 tier-C 检查(逻辑冒烟用), 不判失败。
"""
import argparse
import json
import os
import sys
import time

RESULTS = []


def check(name, ok, detail="", fatal=True):
    RESULTS.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400], "fatal": fatal})
    tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"[{tag}] {name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/env_check.json")
    args = ap.parse_args()

    import torch

    print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("!! 无 CUDA（CPU 模式）：只做逻辑层检查，用于本地冒烟，不判失败")
        check("torch_import", True, torch.__version__)
        check("sdpa_math_backend", _sdpa_probe(torch, "cpu"), "CPU math 后端")
        _ = json.dump(RESULTS, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        sys.exit(0)

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    check("gpu_model", True, f"{name} sm_{cap[0]}{cap[1]} mem={props.total_memory/2**30:.1f}GB")

    is_volta = cap == (7, 0)
    if is_volta:
        # Volta sm70: torch 2.11+ 官方 wheel 已移除, 必须确认 arch list 里有 sm_70
        archs = torch.cuda.get_arch_list()
        check("sm70_in_wheel", any("70" in a for a in archs), f"archs={archs}")
        check("no_bf16_on_volta", True,
              "Volta 无 bf16: 训练代码必须走 fp16+GradScaler 路径", fatal=False)

    # matmul 实测: fp16 应显著快于 fp32 (Tensor Core 生效的粗证据)
    def bench(dtype, n=2048, iters=30):
        a = torch.randn(n, n, dtype=dtype, device=dev)
        b = torch.randn(n, n, dtype=dtype, device=dev)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize()
        return (time.time() - t0) / iters

    t32 = bench(torch.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        t16 = bench(torch.float16)
    check("fp16_tensor_core", t16 < t32,
          f"fp32={t32*1e3:.2f}ms fp16={t16*1e3:.2f}ms speedup={t32/max(t16,1e-9):.1f}x",
          fatal=False)

    # SDPA 后端探测: Volta 无 flash, 期望 efficient 或 math 可用
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for bk_name in ["FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]:
        try:
            bk = getattr(SDPBackend, bk_name)
            with sdpa_kernel([bk]):
                _sdpa_probe(torch, "cuda")
            check(f"sdpa_{bk_name.lower()}", bk_name != "FLASH_ATTENTION" or not is_volta,
                  "可用" + ("(Volta 上意外可用)" if bk_name == "FLASH_ATTENTION" else ""))
        except Exception as e:
            # flash 失败永远不致命(我们不依赖它, efficient/math 兜底);
            # efficient/math 失败才致命
            fatal = bk_name != "FLASH_ATTENTION"
            check(f"sdpa_{bk_name.lower()}", not fatal, f"不可用: {type(e).__name__}", fatal=fatal)

    # amp fp16 + GradScaler 数值冒烟: 20 步线性回归, loss 应下降且无 NaN
    ok, detail = _amp_train_probe(torch, dev)
    check("amp_fp16_gradscaler", ok, detail)

    # 原子存档机制冒烟 (V5 协议核心: tmp + os.replace)
    try:
        tmp, fin = "/tmp/jl_probe.tmp", "/tmp/jl_probe.pt"
        torch.save({"x": torch.arange(4)}, tmp)
        os.replace(tmp, fin)
        assert torch.equal(torch.load(fin)["x"], torch.arange(4))
        os.remove(fin)
        check("atomic_ckpt", True, "tmp+os.replace OK")
    except Exception as e:
        check("atomic_ckpt", False, repr(e))

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    _ = json.dump(RESULTS, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    fatal_fails = [r for r in RESULTS if not r["ok"] and r["fatal"]]
    print(f"\n== env_check: {len(RESULTS)-len(fatal_fails)}/{len(RESULTS)} 通过, "
          f"致命失败 {len(fatal_fails)} ==")
    sys.exit(1 if fatal_fails else 0)


def _sdpa_probe(torch, device):
    import torch.nn.functional as F
    q = torch.randn(2, 4, 64, 32, device=device)
    k = torch.randn(2, 4, 64, 32, device=device)
    v = torch.randn(2, 4, 64, 32, device=device)
    o = F.scaled_dot_product_attention(q, k, v)
    return bool(torch.isfinite(o).all().item())


def _amp_train_probe(torch, dev, steps=20):
    torch.manual_seed(0)
    w = torch.randn(64, 64, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([w], lr=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(256, 64, device=dev)
    tgt = x @ torch.randn(64, 64, device=dev) + 0.1
    losses = []
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ((x @ w - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    finite = all(l == l for l in losses)
    dropped = losses[-1] < losses[0]
    return finite and dropped, f"loss {losses[0]:.3f}->{losses[-1]:.3f} finite={finite}"


if __name__ == "__main__":
    main()
