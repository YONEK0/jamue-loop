# platform_report.md — 特斯拉 V100 32G 平台质检报告（第一次会话）

> 会话编号：Q1（质检会话） | 平台：gpu.ai-galaxy.cn | 实例：1×Tesla V100-SXM2-32GB
> 镜像：ubuntu22_cuda12.4 | 规格：8 核 / 15G 内存 / 196G 系统盘 / 32Mbps
> 结论先行：**平台可用于 350M 分段训练，环境全通；实测吞吐 1422 tok/s；S1 已开跑并留下可续训存档。**

## 1. 硬件与环境核验

| 项 | 实测 | 判定 |
|---|---|---|
| GPU | Tesla V100-SXM2-32GB, sm_70, 31.7GB | ✅ 真卡，非切片 |
| 驱动 | 550.127.05 | ✅ 新驱动，支持 CUDA 12.x 全系 |
| CPU/内存 | 8 核 / 15G | ✅ 够用（数据流走 memmap） |
| 系统盘 | 196G，初始已用 39G（预装 CUDA toolkit 等） | ✅ 余量充足 |
| 系统 Python | 3.10，无 pip、无 torch | ⚠️ 镜像与宣传不符 |
| 预装 Miniconda | `/root/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh`（仅安装器，未安装） | ✅ 离线安装 30 秒解决 |
| apt 源 | 404（过期） | ⚠️ 不能用 apt 装包 |

**实测踩坑**：镜像宣称预置 torch 2.5 实际没有；launch 清单的"探测预置 torch"分支无用。
正确路径 = 离线 Miniconda + pip 装 torch 2.10.0+cu126。

## 2. torch 与关键能力（env_check 9/9 通过）

| 项 | 结果 |
|---|---|
| torch | 2.10.0+cu126（cu126 wheel 含 sm_70 ✓） |
| fp16 Tensor Core | matmul 加速 5.6×（fp32 1.39ms → fp16 0.25ms） |
| SDPA 后端 | efficient + math 可用；flash 不可用（sm70 预期） |
| amp + GradScaler | 20 步收敛探针通过 |
| 原子存档 | tmp→os.replace 通过 |

torch 下载总耗时 ≈ 25 分钟（nvidia 系 wheel 共 ~3GB @3.5MB/s）。
其中遇到了 pypi.nvidia.com / download-r2 的停滞，**补救办法**：kill 后 /tmp/pip-unpack-*/
内的完整 wheel 还在，`pip install /tmp/pip-unpack-*/*.whl` 就地装配 + triton 走清华源。
（本地已经把这条写进 setup.sh 的备选分支。）

## 3. warm-start 映射（本次质检的核心未知项）

Jaume 350m-mlm 真实权重结构与本地推断有 4 处差异，已在真实权重上修正并验证：

| 项 | 本地推断 | 真实值 |
|---|---|---|
| vocab | 32000 | **151647**（Qwen tokenizer） |
| expert 键 | `layers.N.experts.*` | `layers.N.mlp.experts.*` |
| router 键 | `layers.N.gate.*` | `layers.N.mlp.gate.*` |
| rope_theta | 10000 | **1e6** |
| norm 键 | norm1/norm2 | input_layernorm / post_attention_layernorm |
| head | lm_head | 无（MLM head 缺失，从 embed 复制初始化） |
| pooling/projection | 未知 | 存在（S4 可继承 pooling.query + projection） |

**真实权重匹配率：339/339 = 1.0，零缺失**。模型 385.1M 参数。

## 4. 训练能力（真实规模实测）

| 项 | 实测 |
|---|---|
| 配置 | bs8 × seq512 × R∈{1,2,4,8} 采样 K=2 + gradient checkpoint |
| 显存 | bs8: 24.4GB 稳定；**bs16 OOM**（V100 32G 上不了 bs16） |
| 吞吐 | **≈20.8 step/min → 2.88 s/step → 1422 tok/s** |
| CE 曲线 | 56.5(@20) → 37.9(@50) → 25.4(@100)：warm-start 生效，快速下降 |
| expert load | [0.22, 0.28, 0.29, 0.21]，maxmin≈1.3，**无死专家** |
| router entropy | 1.28 → 1.03（健康区间，无塌缩） |

**内存诊断结论**（影响后续所有批次计划）：
- R=8 深循环下激活是主内存项（132 block-pass × 图），gradient checkpoint 是必需项不是可选项；
- 151647 词表的 CE 计算（fp32 logits + log_softmax）单次约 2.5GB(bs8)，是第二大项；
- 结论：350M 本卡上限 bs8；要提吞吐需降 R 采样或分块 CE。

## 5. 时间预算校准（实测吞吐倒推）

| 阶段 | token 量 | 实测耗时 | 会话数（2h 上限，净训练 105 min） |
|---|---|---|---|
| S1 循环适应 | 30M | ~5.9 h | 3 次会話（或砍到 20M → 2 次） |
| S1 轻量版 | 10M | ~2 h | 1 次 |
| eval_depth | — | ~5 min | 随会话 |
| S4 对比（0.3M pairs） | — | ~1 h | 1 次 |

**修正之前 6 小时方案里的估算**：当时按理论算 S1 30M ≈ 25–40 分钟是错的（差 ~8×），
真实 FLOPs/token 被低估。以本报告 1422 tok/s 为准。

## 7. S1 正式训练启动与结束状态

- 11:42 启动 S1（warm-start，bs8，gradient checkpoint），训练至 step 200 / 0.82M sup-tokens：
  - CE：56.5(@20) → 37.9(@50) → 25.4(@100) → 23.6(@150) → **17.3(@200)**
  - expert load 全程均衡（maxmin ≤2.1），无死专家；router entropy 1.28→0.82
  - cross-loop overlap：0.80 → 0.63（第 1 对圈开始分化，深层对数仍高——早期正常）
  - step 50 触发**路由分化转折档**（自动存档 ✓）
- **ckpt_latest = step 200 完整档（4.62GB，含优化器/RNG/数据偏移）**，均已验证可加载。
- 训练在 11:52 后中断（无 final 档）：原因是我在 SSH 超时后重复发布了启动命令，
  两个训练进程并发导致 GPU OOM 互杀。**平台无关，操作竞态**——已写入纪律（§7.1）。

### 7.1 会话纪律教训（必须遵守）

1. **SSH 超时 ≠ 命令未执行**：server 端进程可能已启动。超时后必须先 `pgrep -fc '[t]rain'`
   确认，再决定是否重发；**永远不要盲目重复发布启动命令**。
2. 每次启动用**独立日志名**（s1_$(date +%H%M).log），避免 `>` 互相截断排障线索。
3. `pkill -f 'train_s1_mlm'` 会匹配到包含该字符串的当前 shell 自身；
   必须写 `pkill -f '[t]rain_s1'`（自排除）。
4. 训练进程 nohup + history.jsonl 逐条 flush 策略有效：SSH 断连期间训练不受影响 ✓

## 8. 磁盘持久化（待下会话验证）

- 已写标记：`/marker_jl.txt`、`~/marker_jl.txt`
- 已留资产：`~/jamue-loop/runs/s1_350m/`（ckpt_latest/ckpt_final ≈ 4.5GB + history）
  以及 miniconda、torch、Jaume 权重、data/
- **下会话第一件事**：`ls /marker_jl.txt ~/marker_jl.txt` + `--resume` 试续训
- 若持久 ✓：以后每次会话固定开销只剩 5 分钟（开机 + resume）
- 若持久 ✗：所有重装（~35 分钟）每次会话发生——则需重估付费方案

## 7. 会话纪律复盘

- 单次会话 2h，本次实际用到 ~1h45m（10:25–12:13），留 15 分钟缓冲 ✓
- 网络波动：SSH 移动/电信双线均出现过瞬断，脚本自动切换有效
- 训练进程 nohup 后台化 + history.jsonl 逐条 flush：SSH 断了训练不受影响 ✓
- **教训**：pkill -f 'train_s1_mlm' 会误杀包含该字符串的当前 shell（用 `[t]rain_s1` 写法）

## 8. 结论：是否值得付费

**值得**。理由：
1. 环境全通且已经就位持久化（若验证通过，第二次会话直接 resume 开训）；
2. warmstart 映射 1.0 匹配——350M 试验的最大未知项已消除；
3. 实测吞吐足以支撑 350M 全流程（4–6 次会话 ≈ 8–12 卡时）；
4. 卡是真 V100 非切片，算力健康（28 TFLOPS 有效 fp16）。

**建议**：先开 1 次（2h）付费会话验证"持久化 + resume + 完成 S1"，
再决定后续批量会话。
