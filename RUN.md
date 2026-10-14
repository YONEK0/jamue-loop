# RUN.md — V100 实例开机操作清单（按顺序执行，每步有判定）

> **2 小时会话模式**（平台单次上限 2h，磁盘持久）：每次会话 = 开机清单 + 训练段 +
> 收尾清单。推荐训练参数 `--max-minutes 105 --save-every 250`——留 15 分钟收尾缓冲，
> 滚动档约 100 秒一存，硬杀最多丢 2 分钟。
> 存档四层：rolling（每 save-every 步，latest/prev）→ 转折点永久档（事件触发）
> → 会话 final 档 → resume 自动选三者中 step 最新的恢复。
> 持久盘布局约定：仓库/venv/权重/ckpt 全放持久盘；每次会话只推小产物（jsonl/json/png）。

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

`run_all.sh` 已含合成数据全链路验证。真实数据就位后跑 350M 主线：

```bash
# S1: MLM 循环适应 (warm-start Jaume 350m-mlm), 硬时限 140 分钟
python3 src/train_s1_mlm.py --config configs/350m_warmstart.json \
    --warmstart assets/jaume/350m-mlm --run runs/s1_350m \
    --sup-tokens 30000000 --max-minutes 140

# 评测: CE(d) 深度曲线 + 样本级退出扫描
python3 src/eval_depth.py --ckpt runs/s1_350m/ckpt_final.pt --data data/val.bin \
    --out results/depth_s1_350m.json --plot results/depth_s1_350m.png

# S4: 对比训练 (pairs 数据换成真实 token ids 后)
python3 src/train_s4_contrast.py --ckpt runs/s1_350m/ckpt_final.pt \
    --pairs data/pairs.jsonl --run runs/s4_350m --steps 2000 --max-minutes 60
```

**开训前 3 分钟做实测吞吐**：前 200 step 的 tokens/s × 剩余时间 = 本次 token 预算上限，
宁少勿超。warmstart 首次运行必看 `runs/s1_350m/warmstart_report.json` 的 match_rate，
低匹配说明 Jaume 键名与预设映射不符，先看 missed 列表再决定是否继续。

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

下次会话续训：

```bash
python3 src/train_s1_mlm.py --config configs/350m_warmstart.json \
    --warmstart assets/jaume/350m-mlm --run runs/s1_350m \
    --sup-tokens 30000000 --max-minutes 105 --save-every 250 --resume
```

## 环境事实卡（Tesla V100 32G）

- sm_70 架构：**torch ≤ 2.10.0（cu126/cu128 是最后支持线）**，2.11+ 官方 wheel 无 Volta；
- **无 bf16**：所有训练代码走 fp16 + `torch.amp.GradScaler`，router/softmax/norm 转 fp32；
- flash-attention 不可装（需 sm80+）：SDPA 用 memory-efficient 或 math 后端，
  env_check 会自动探测报告；
- 显存 32G：350M 全参 + AdamW ≈ 10GB，充裕，优先加大 micro-batch。
