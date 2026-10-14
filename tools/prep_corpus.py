"""真实语料预处理: JSON/JSONL 文本 -> tokenize -> packed uint32 bin.

vocab 151647 > uint16 上限 65535, 真实数据必须 uint32。
用法: python3 tools/prep_corpus.py --raw data/raw/wiki_cn.json --out data --val-tokens 100000
"""
import argparse
import json
import os
import sys

import numpy as np


def load_texts(raw):
    texts = []
    with open(raw, encoding="utf-8") as f:
        head = f.read(2)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            for d in data:
                if isinstance(d, dict):
                    t = d.get("completion") or d.get("text") or d.get("content") or ""
                    if t and len(t) > 50:
                        texts.append(t)
                elif isinstance(d, str) and len(d) > 50:
                    texts.append(d)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    t = d.get("completion") or d.get("text") or d.get("content") or "" \
                        if isinstance(d, dict) else (d if isinstance(d, str) else "")
                    if t and len(t) > 50:
                        texts.append(t)
                except Exception:
                    pass
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-tokens", type=int, default=100_000)
    ap.add_argument("--tok", default="assets/jaume/350m-mlm/tokenizer/tokenizer.json")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tok)
    texts = load_texts(args.raw)
    print(f"[prep] docs={len(texts)}")
    if not texts:
        sys.exit("no texts")

    from array import array
    ids = array("I")  # uint32 紧凑累积, 避免 Python list 的 ~36B/元素开销
    B = 2000
    for i in range(0, len(texts), B):
        for e in tok.encode_batch(texts[i:i + B]):
            ids.extend(e.ids)
        if i % 40000 == 0:
            print(f"[prep] encoded {i}/{len(texts)} docs, {len(ids)/1e6:.1f}M tok", flush=True)
    a = np.frombuffer(ids, dtype=np.uint32)
    print(f"[prep] total tokens: {len(a)/1e6:.2f}M", flush=True)

    os.makedirs(args.out, exist_ok=True)
    nv = min(args.val_tokens, len(a) // 20)
    a[:-nv].tofile(os.path.join(args.out, "real_train.bin"))
    a[-nv:].tofile(os.path.join(args.out, "real_val.bin"))
    print(f"[prep] train={(len(a)-nv)/1e6:.2f}M val={nv/1e3:.0f}k -> {args.out}/real_*.bin (uint32)")


if __name__ == "__main__":
    main()
