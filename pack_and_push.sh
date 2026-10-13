#!/usr/bin/env bash
# 结果回传: 小产物(指标/曲线/日志)自动 commit+push; 大 ckpt 提示走网盘/ModelScope。
# 幂等, 可被定时任务每 30 分钟调用一次。
set -uo pipefail
cd "$(dirname "$0")"

TS=$(date +%H%M%S)
git add results/ *.md 2>/dev/null || true
if git diff --cached --quiet; then
  echo "[push] 无新产物, 跳过"
  exit 0
fi
git commit -m "results auto-push ${TS}" || exit 0
if git push origin HEAD 2>/dev/null; then
  echo "[push] OK ${TS}"
else
  echo "[push] !! git push 失败(网络/token), 产物仍在本地 results/"
  echo "[push] 手动兜底: 打包 results/ 走网盘/ModelScope"
  tar czf "results_backup_${TS}.tar.gz" results/ 2>/dev/null || true
fi

# 大文件提醒 (不进 git)
BIG=$(find runs assets -name "*.pt" -size +100M 2>/dev/null | head -3)
if [ -n "$BIG" ]; then
  echo "[push] 大 ckpt 不走 git, 请手动上传: ${BIG}"
fi
