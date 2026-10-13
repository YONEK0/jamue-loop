#!/usr/bin/env bash
# 总入口: 环境 -> 权重 -> 冒烟 -> 本地小验证 (可再接 350M 主线)
set -uo pipefail
cd "$(dirname "$0")"

echo "########## [1/4] setup ##########"
bash setup.sh || { echo "!! setup 失败"; exit 1; }

echo "########## [2/4] fetch assets ##########"
python3 fetch_assets.py || { echo "!! 权重拉取失败 (检查 ModelScope 网络)"; exit 1; }

echo "########## [3/4] smoke ##########"
python3 smoke.py || { echo "!! 冒烟失败, 禁止开训"; exit 1; }

echo "########## [4/4] 管线小验证 (合成数据全链路) ##########"
python3 src/data.py --out data --tokens 500000 --pairs 256 --vocab 1000 || exit 1
python3 src/train_s1_mlm.py --config configs/local_smoke.json --run runs/pipeline_smoke \
    --sup-tokens 30000 --monitor-every 5 || exit 1
python3 src/eval_depth.py --ckpt runs/pipeline_smoke/ckpt_final.pt --data data/val.bin \
    --out results/depth_pipeline_smoke.json --batches 5 --bs 4 --seq 128 || exit 1
python3 src/train_s4_contrast.py --ckpt runs/pipeline_smoke/ckpt_final.pt \
    --pairs data/pairs.jsonl --run runs/s4_pipeline_smoke --steps 20 --bs 8 --seq 128 --R 4 || exit 1

echo ""
echo "== 全链路管线验证通过 =="
echo "350M 主线 (真实数据就位后, 参考 RUN.md 时间表):"
echo "  python3 src/train_s1_mlm.py --config configs/350m_warmstart.json \\"
echo "      --warmstart assets/jaume/350m-mlm --run runs/s1_350m \\"
echo "      --sup-tokens 30000000 --max-minutes 140"
