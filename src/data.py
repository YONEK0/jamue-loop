"""memmap token 数据 (uint16 .bin) 与对比 pairs 读取 + 合成数据生成.

bin 格式: 连续 token id 流 (GPT/nanogpt 风格), 读取窗口确定性顺序, 无 shuffle。
pairs 格式: jsonl, 每行 {"q": [ids...], "pos": [ids...]} (ids 级, tokenize 在本地做)。
"""
import argparse
import json
import os

import numpy as np
import torch


class BinCorpus:
    def __init__(self, path, dtype="uint16"):
        # vocab 151647 > uint16 上限: 真实语料用 uint32
        self.dtype = np.uint32 if dtype == "uint32" else np.uint16
        self.a = np.memmap(path, dtype=self.dtype, mode="r")
        self.n = len(self.a)

    def batch(self, offset_tokens, bs, seq):
        """确定性顺序读取: 指定起始 token 偏移, 循环回绕."""
        idx = (offset_tokens + np.arange(bs * seq)) % self.n
        x = torch.from_numpy(self.a[idx].astype(np.int64)).view(bs, seq)
        return x


class PairData:
    def __init__(self, path):
        self.pairs = [json.loads(l) for l in open(path, encoding="utf-8")]

    def batch(self, step, bs, seq):
        q, p = [], []
        for i in range(bs):
            e = self.pairs[(step * bs + i) % len(self.pairs)]
            q.append(_fit(e["q"], seq))
            p.append(_fit(e["pos"], seq))
        return torch.tensor(q), torch.tensor(p)


def _fit(ids, seq):
    ids = [i if i >= 2 else 2 for i in ids][:seq]
    return ids + [0] * (seq - len(ids))


def make_dummy(outdir="data", vocab=32000, n_tokens=2_000_000, n_pairs=4096, seq_hint=256):
    """合成 Zipf 语料 + 弱结构 pairs, 用于本地/平台管线验证 (无语义, 只测机制)."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    zipf_p = 1.0 / np.arange(2, vocab + 2) ** 1.1
    zipf_p /= zipf_p.sum()
    # token id 空间: 0=pad 1=<mask> 2..vocab-1 正常 token
    ids = 2 + rng.choice(vocab - 2, size=n_tokens, p=zipf_p[2:] / zipf_p[2:].sum()).astype(np.uint16)
    # 注入重复 n-gram 制造可学习的局部结构
    grams = ids[: n_tokens // 4].reshape(-1, 8)
    ids[n_tokens // 2 :] = grams.repeat(2, axis=0).reshape(-1)[: n_tokens // 2]
    ids.astype(np.uint16).tofile(os.path.join(outdir, "train.bin"))
    ids[-65536:].astype(np.uint16).tofile(os.path.join(outdir, "val.bin"))

    with open(os.path.join(outdir, "pairs.jsonl"), "w") as f:
        for i in range(n_pairs):
            q = 2 + rng.choice(vocab - 2, size=int(seq_hint * 0.6), p=zipf_p[2:] / zipf_p[2:].sum())
            pos = np.concatenate([q[: len(q) // 3], 2 + rng.choice(vocab - 2, size=len(q) // 3, p=zipf_p[2:] / zipf_p[2:].sum())])
            f.write(json.dumps({"q": q.tolist(), "pos": pos.tolist()}) + "\n")
    print(f"[dummy] {outdir}/train.bin {n_tokens} tok, val.bin 64k, pairs {n_pairs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--pairs", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=32000)
    args = ap.parse_args()
    make_dummy(args.out, args.vocab, args.tokens, args.pairs)
