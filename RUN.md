# RUN.md — V100 实例开机操作清单（按顺序执行，每步有判定）

> 硬预算：6 小时窗口。环境+冒烟目标 1 小时内完成；T+4:30 是评测回传硬截止。
> 原则：每一步失败先走该步的 fallback，不要跳步。

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

## 降级链（到点自动降，不犹豫）

| 时点 | 状态 | 动作 |
|---|---|---|
| T+1:00 | 环境还没就绪 | 砍 C2 对照线，只跑 C3 主线 |
| T+2:00 | 还没就绪 | 砍到单线 50M token 最低目标（跑通+路由不塌） |
| T+4:30 | 无论训练在哪 | **停训，评测+打包+回传** |

## 结束前

```bash
bash pack_and_push.sh          # 指标/曲线全量回传
# 大 ckpt (runs/*.pt) 手动: 网盘 或 modelscope upload
```

## 环境事实卡（Tesla V100 32G）

- sm_70 架构：**torch ≤ 2.10.0（cu126/cu128 是最后支持线）**，2.11+ 官方 wheel 无 Volta；
- **无 bf16**：所有训练代码走 fp16 + `torch.amp.GradScaler`，router/softmax/norm 转 fp32；
- flash-attention 不可装（需 sm80+）：SDPA 用 memory-efficient 或 math 后端，
  env_check 会自动探测报告；
- 显存 32G：350M 全参 + AdamW ≈ 10GB，充裕，优先加大 micro-batch。
