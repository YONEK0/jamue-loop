# RUN.md — V100 实例开机操作清单（按顺序执行，每步有判定）

> **2 小时会话模式**（平台单次上限 2h，磁盘持久）：每次会话 = 开机清单 + 训练段 +
> 收尾清单。推荐训练参数 `--max-minutes 105 --save-every 250`——留 15 分钟收尾缓冲，
> 滚动档约 100 秒一存，硬杀最多丢 2 分钟。
> 存档四层：rolling（每 save-every 步，latest/prev）→ 转折点永久档（事件触发）
> → 会话 final 档 → resume 自动选三者中 step 最新的恢复。
> 持久盘布局约定（**单块 200G 系统盘，无数据盘**）：一切放 `~/jamue-loop`，
> 无需挂载。容量预算表见下方"磁盘账本"。
> 每次会话只推小产物（jsonl/json/png）。
>
> **平台镜像选型（已定）：Ubuntu22.04 `ubuntu22_cuda12.4`**（预置 torch 2.5.0+cu124
> 含 sm_70 → setup.sh 自动复用跳过安装；驱动 550+ 给 torch 2.10+cu126 留余量；
> headless 无桌面件）。不选 cu118 两档（torch 2.0 老、驱动旧）。
> Anaconda 环境装依赖一律 `python3 -m pip install ...`。

## 磁盘账本（200G 系统盘）

| 项 | 占用 | 说明 |
|---|---|---|
| 镜像系统 + conda + 预置 torch | ~25G | 固定 |
| 仓库 + 小依赖 | ~2G | torch 复用预置，只装 numpy/mpl 等 |
| Jaume 权重 | ~1.5G | 350m-mlm 一个 stage |
| 训练数据 .bin | ~1–2G | 30M token=60MB；500M token 也才 1G |
| S1 ckpt 池 | ~23G | **每档 ~4.5G**（fp32 权重 1.5G + AdamW 动量 3G）：latest+prev+final+2 转折档 |
| C2 对照 ckpt 池 | ~23G | 如果跑对照线 |
| **合计** | **~75G / 200G** | 余量充足，但受 ckpt 支配 |

ckpt 纪律（防膨胀）：
- 转折点档是事件触发不会无限涨（已知 div/depth 两类），但**每次会话手动 `du -sh runs/`**
  超过 60G 时清理 `ckpt_prev` 与已消费的中间 final；
- S1 完成进入 S4/eval 前，把 final 档瘦身为"仅 model 权重"档（~1.5G，丢 opt 状态，
  不再续训才允许）：
  `python3 -c "import torch;p=torch.load('runs/s1_350m/ckpt_final.pt',map_location='cpu',weights_only=False);torch.save({'model':p['model'],'cfg':p['cfg'],'step':p['step']},'runs/s1_350m/model_only.pt')"`
- 质检会话第一项：在 `/` 与 `~/` 各写 marker 文件，下次会话检查**哪片区域真正持久**
  （有的平台只持久 home），确认后所有路径以实测为准。

## T+0:00 第 0 分钟（网络与回传通道验证，最优先）

```bash
git clone <repo-url> jamue-loop && cd jamue-loop   # 若超时/断连 -> 立刻改用 gitee 镜像
sed -i 's/\r$//' *.sh 2>/dev/null                   # 防 CRLF（.gitattributes 已配，双保险）
git config user.name "runner" && git config user.email "runner@local"
# 配回传 token（最小权限 fine-grained PAT）:
git remote set-url origin https://<TOKEN>@github.com/<user>/<repo>.git
echo test > results/net_probe.txt && bash pack_and_push.sh   # 验证回传通道!!
```

**判定：`git push` 成功才继续。失败 → 换 gitee remote / 检查 token，这一步不通，
后面所有产物只能靠网盘，风险大增。**

## T+0:05 一键环境

```bash
bash run_all.sh        # setup -> fetch 权重 -> smoke, 全自动
```

**判定：**
- setup 卡在 torch 安装 >15 分钟 → Ctrl+C，手动只跑 `pip install torch==2.10.0+cu126 -i 清华镜像`（搜阿里 pytorch-wheels 镜像），再重跑 `python3 env_check.py`；
- env_check 里 `sm70_in_wheel` FAIL → 装的 torch 版本 ≥2.11，必须降级；
- smoke tier1 FAIL → 不要开训，把 `results/*.json` 推上来排查。

## T+1:00（预期）训练

`run_all.sh` 已含合成数据全链路验证。真实数据就位后跑 350M 主线。

**实测锚定（2026-10-14 质检会话，必须遵守）**：
- 环境非预置：用 `/root/Miniconda3-*.sh` 离线装 + pip torch 2.10.0+cu126（29 分钟）；
- **bs16 必 OOM，固定 `--bs 8` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**；
- 实测吞吐 1422 tok/s（2.88 s/step @ bs8）→ 30M sup-tokens ≈ 5.9h ≈ 3 个会话；
- 启动命令（唯一发布一次，超时先 pgrep 再决定）：
```bash
cd ~/jamue-loop
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/miniconda3/bin/python -u \
  src/train_s1_mlm.py --config configs/350m_warmstart.json \
  --warmstart assets/jaume/350m-mlm --run runs/s1_350m \
  --sup-tokens 30000000 --monitor-every 50 --save-every 150 \
  --max-minutes 105 --bs 8 > ~/s1_$(date +%H%M).log 2>&1 &
sleep 20; pgrep -fc '[t]rain_s1_mlm'   # 必须 ≥1, 否则看日志
```

续训（ckpt_latest 已在 step 200 / 0.82M，直接 --resume 接力）：
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup ~/miniconda3/bin/python -u \
  src/train_s1_mlm.py --config configs/350m_warmstart.json \
  --warmstart assets/jaume/350m-mlm --run runs/s1_350m \
  --sup-tokens 30000000 --monitor-every 50 --save-every 150 \
  --max-minutes 105 --bs 8 --resume > ~/s1_$(date +%H%M).log 2>&1 &
```

**开训前 3 分钟做实测吞吐**：前 200 step 的 tokens/s × 剩余时间 = 本次 token 预算上限，
宁少勿超。warmstart 首次运行必看 `runs/s1_350m/warmstart_report.json` 的 match_rate
（质检会话实测 **1.0 / 零缺失**，此风险已消除）。

## 会话内降级链（到点自动降，不犹豫；T = 会话开始）

| 时点 | 状态 | 动作 |
|---|---|---|
| T+0:30 | 环境/下载还没就绪 | 跳过 350M 真实训练，只完成管线验证 + warmstart 匹配率检测 |
| T+1:00 | 还没就绪 | 只跑 env_check + smoke，把 platform_report 推上去，本会话定位为纯质检 |
| T+1:45 | 无论训练在哪 | **停训（--max-minutes 105 会自动触发），评测+打包+回传** |

## 收尾（每次会话必做，约 5 分钟）

```bash
bash pack_and_push.sh          # history.jsonl / results/* 自动 commit+push
# 大 ckpt 只在转折档/final 时手动外传一次: 网盘或 modelscope upload
# 平时依赖持久盘 + 下次会话 --resume 续训, 不丢进度
```

下次会话续训（命令见上方 T+1:00 节，固定 `--bs 8 --resume`）：
**会话开始第 2 件事**（第 1 件是验证回传通道）：检查持久化 marker —
`ls /marker_jl.txt ~/marker_jl.txt` 与 `runs/s1_350m/ckpt_latest.pt` 是否存在，
以实测结果决定本会话是"直接续训"还是"重装环境"。

## 环境事实卡（Tesla V100 32G，质检会话实测锚定）

- sm_70 架构：**torch ≤ 2.10.0（cu126/cu128 是最后支持线）**，2.11+ 官方 wheel 无 Volta；
- 镜像（ubuntu22_cuda12.4）实际无预置 torch → 用 `/root` 的 Miniconda 安装器离线装：
  `bash ~/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh -b -p ~/miniconda3`；
- torch 下载慢/停滞时：`/tmp/pip-unpack-*/*.whl` 里完整 wheel 直接用
  `pip install --no-deps <whls>` + triton 走清华源；apt 源已过期（404）不可用；
- **无 bf16**：fp16 + GradScaler，router/softmax/norm 全 fp32（代码已内置）；
- flash-attention 不可装；SDPA efficient/math 后端可用；
- **显存红线**：bs8 + gradient checkpoint = 24.4GB 稳定；bs16 OOM。
  151647 词表的 CE fp32 计算是大头之一（bs8 约 2.5GB）；
- 磁盘：196G 系统盘，一次 S1 档约 4.6GB；`/root` 与 `/` 为同一分区且**持久**（待下会话
  以 marker 文件最终确认）。
