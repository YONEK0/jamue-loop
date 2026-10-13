"""拉取 Jaume 350m 权重与配置 (ModelScope 国内 CDN)。

只拉试验需要的子目录, 不拉整个仓库(仓库还含 data/logs/多份 checkpoint)。
用法: python3 fetch_assets.py [--stage 350m-mlm]
"""
import argparse
import os
import sys

MODEL_ID = "LightNeko/Jaume-Embedding"
STAGES = ["350m-mlm", "350m-mlm-v2", "350m-stage2", "350m-stage2-ml",
          "350m-stage3", "350m-stage3-ml", "350m-decision-v2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="350m-mlm",
                    help=f" checkpoint 子目录, 之一: {STAGES}")
    ap.add_argument("--out", default="assets/jaume")
    args = ap.parse_args()
    assert args.stage in STAGES, f"未知 stage: {args.stage}"

    os.makedirs(args.out, exist_ok=True)
    marker = os.path.join(args.out, f".done_{args.stage}")
    if os.path.exists(marker):
        print(f"[fetch] {args.stage} 已存在, 跳过 (删 {marker} 可强制重拉)")
        return

    from modelscope import snapshot_download
    print(f"[fetch] {MODEL_ID}/{args.stage} -> {args.out}")
    # allow_patterns 精确到子目录+顶层配置, 避免拉全仓
    pats = [f"{args.stage}/**", "*.json", "*.md", "tokenizer*", "*.model"]
    snapshot_download(MODEL_ID, local_dir=args.out, allow_patterns=pats)
    _ = open(marker, "w").write("ok")
    print("[fetch] 完成")

    # 校验权重文件存在且非空
    import glob
    files = glob.glob(os.path.join(args.out, args.stage, "*"))
    big = [f for f in files if os.path.getsize(f) > 10 * 2**20]
    if not big:
        print(f"!! {args.stage} 下没有 >10MB 的权重文件, 下载可能不完整")
        sys.exit(1)
    print(f"[fetch] 权重文件 {len(big)} 个, 共 {sum(os.path.getsize(f) for f in big)/2**30:.2f} GB")


if __name__ == "__main__":
    main()
