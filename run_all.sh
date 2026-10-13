#!/usr/bin/env bash
# 总入口: 环境 -> 权重 -> 冒烟 -> (训练接入点)
# 训练脚本就绪后, 在 TRAIN 区取消注释接入。
set -uo pipefail
cd "$(dirname "$0")"

echo "########## [1/3] setup ##########"
bash setup.sh || { echo "!! setup 失败"; exit 1; }

echo "########## [2/3] fetch assets ##########"
python3 fetch_assets.py || { echo "!! 权重拉取失败 (检查 ModelScope 网络)"; exit 1; }

echo "########## [3/3] smoke ##########"
python3 smoke.py || { echo "!! 冒烟失败, 禁止开训"; exit 1; }

echo "########## TRAIN (接入点) ##########"
# python3 src/train_s1_mlm.py --config configs/350m_warmstart.json
echo "(训练脚本待接入 — 冒烟已通过, 管线就绪)"
