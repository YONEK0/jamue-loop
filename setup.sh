#!/usr/bin/env bash
# jamue-loop 一键环境配置 (Tesla V100 32G / 国内 GPU 平台)
# 策略: 1) 探测平台预置 torch, 若含 sm70 直接复用  2) 否则装 torch==2.10.0+cu126
#       3) 装锁定小依赖(国内源优先)  4) 跑 env_check.py 出 go/no-go
set -uo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
echo "== [0/4] 系统 =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "!! nvidia-smi 不可用"
$PY --version || { echo "!! 无 python3"; exit 1; }

echo "== [1/4] 探测预置 torch =="
PRESET_OK=0
if $PY -c "import torch" 2>/dev/null; then
  TV=$($PY -c "import torch;print(torch.__version__)")
  if $PY -c "import torch;import sys;sys.exit(0 if any('70' in a for a in torch.cuda.get_arch_list()) and torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "预置 torch=$TV 含 sm70 且 CUDA 可用 -> 复用, 跳过安装"
    PRESET_OK=1
  else
    echo "预置 torch=$TV 不可用(无 sm70 或无 CUDA), 需要另装"
  fi
else
  echo "无预置 torch"
fi

if [ "$PRESET_OK" -eq 0 ]; then
  echo "== [2/4] 安装 torch==2.10.0 (Volta 最后支持版本) =="
  # 通道1: pytorch 官方; 通道2: 上交镜像; 通道3: 无 CUDA 约束的兜底 CPU 版(仅本地冒烟用)
  $PY -m pip install --no-cache-dir "torch==2.10.0+cu126" \
      --index-url https://download.pytorch.org/whl/cu126 \
    || $PY -m pip install --no-cache-dir "torch==2.10.0+cu126" \
      -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/ \
    || { echo "!! torch 安装失败, 检查网络/驱动后重试"; exit 1; }
else
  echo "== [2/4] 跳过 torch 安装 =="
fi

echo "== [3/4] 安装锁定依赖 =="
$PY -m pip install --no-cache-dir -r requirements-lock.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || $PY -m pip install --no-cache-dir -r requirements-lock.txt \
  || { echo "!! 依赖安装失败"; exit 1; }

echo "== [4/4] 环境体检 =="
$PY env_check.py
EC=$?
if [ $EC -ne 0 ]; then
  echo "!! env_check 有致命失败, 先解决再继续 (详见 results/env_check.json)"
  exit $EC
fi
echo "== setup 完成 =="
