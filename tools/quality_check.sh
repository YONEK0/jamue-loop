#!/usr/bin/env bash
# 平台质检序列: env_check -> 权重 -> 冒烟 -> warmstart 匹配率 -> 吞吐探针
set -uo pipefail
cd ~/jamue-loop
PY=~/miniconda3/bin/python
export PATH=~/miniconda3/bin:$PATH

echo "=== [1/5] env_check ==="
$PY env_check.py 2>&1 | grep -v Warning | tail -14

echo "=== [2/5] fetch jaume 350m-mlm ==="
$PY fetch_assets.py 2>&1 | tail -4

echo "=== [3/5] smoke ==="
$PY smoke.py 2>&1 | tail -9

echo "=== [4/5] warmstart match probe ==="
$PY - <<'EOF'
import sys, json
sys.path.insert(0, 'src')
from arch import LoopMoEModel
from warmstart import load_jaume
cfg = json.load(open('configs/350m_warmstart.json'))
m = LoopMoEModel(cfg)
rep = load_jaume(m, 'assets/jaume/350m-mlm')
print('matched:', rep.get('matched'), 'rate:', rep.get('match_rate'),
      'missed:', len(rep.get('missed', [])))
print('sample missed:', rep.get('missed', [])[:10])
json.dump(rep, open('results/warmstart_report.json', 'w'), indent=1)
EOF

echo "=== [5/5] throughput probe (warm-start S1, 3M tokens) ==="
$PY src/data.py --out data --tokens 500000 --pairs 256 --vocab 32000 | tail -1
$PY src/train_s1_mlm.py --config configs/350m_warmstart.json \
    --warmstart assets/jaume/350m-mlm --run runs/s1_probe \
    --sup-tokens 3000000 --monitor-every 20 --save-every 200 2>&1 | tail -14

echo "=== quality check done ==="
