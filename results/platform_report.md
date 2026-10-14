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

## 7. S1 正式训练（完成，非中断）

**更正备注**：会话中段我曾误判训练被中断——实际是把 `grep -c done` 的计数输出
错读成了进程数。真实情况：训练全程健康运行并干净收尾。

- 11:41 启动 S1（warm-start，bs8，gradient checkpoint），**22.3 分钟干净完成**：
  - `[s1] done step=437 sup=1.79M`，ckpt_final + ckpt_latest 正常落盘（12:04）
- CE 曲线：56.5(@20) → 25.4(@100) → 17.3(@200) → 12.8(@250) → 11.6(@300) → **9.2(@350)**
- expert load 全程均衡（maxmin ≤2.1），无死专家；router entropy 1.28→0.76
- cross-loop overlap 第 1 对圈 0.80→0.74 缓慢分化（深层对数偏高——早期正常）
- step 50 触发路由分化转折档；**step ~450 深度价值真实出现（CE8<CE1）**——
  修复采样顺序 bug 后第一次正确捕获该信号
- 随后按会话纪律单次发布续跑段（--resume，12 分钟预算）

### 7.1 会话纪律教训（必须遵守）

1. **SSH 超时 ≠ 命令未执行**：server 端进程可能已启动。超时后必须先 `pgrep -fc '[t]rain'`
   确认，再决定是否重发；**永远不要盲目重复发布启动命令**。
2. 每次启动用**独立日志名**（如 `s1_r2_$(date +%H%M).log`），避免 `>` 互相截断排障线索。
3. `pkill -f 'train_s1_mlm'` 会匹配到包含该字符串的当前 shell 自身；
   必须写 `pkill -f '[t]rain_s1'`（自排除）。
4. 训练进程 nohup + history.jsonl 逐条 flush：SSH 断连（本会话出现 4 次）训练不受影响 ✓
5. **resume 时 flags 复位**导致转折信号会在续跑段重新触发一次存盘（已知行为，
   时间成本约 1–2 分钟 I/O，下一步可改为随 ckpt 持久化 flags）。

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


---

# 付费会话 #1（4h）：S1 完成与首份 CE(d) 判定

> 2026-10-14 12:30-16:10 | 持久化 ✓（marker/ckpt/env 全存活，开机 5 分钟即续训）

## 结果

- **S1 循环适应完成：17.71M sup-tokens / step 4324**（三段接力 22+85+20 min，resume 无损）
- CE 轨迹：56.5 → 17.3(0.8M) → 9.2(1.4M) → 5.0(5.3M) → 3.4(8.4M) → **1.04(17.0M)**
- expert load 全程均衡（maxmin ≤2.7），无死专家；router entropy 1.28 → ~0.76 稳定
- flags 持久化生效：续跑不再重复触发转折存盘

## 首份 CE(d) 判定（17.7M checkpoint, dummy 验证集）

| d=1 | d=2 | d=4 | d=8 |
|---|---|---|---|
| 4.648 | **4.632** | 4.677 | 4.692 |

- **best_depth=2, depth_gain=-0.044**：在本验证集上 d>2 不再带来 CE 收益
- loop_delta_mean 单调衰减 [3.88, 0.88, 0.47, 0.32, 0.24, 0.19, 0.16, 0.14]：
  循环收敛行为健康，样本级退出有清晰信号
- 样本级平均退出深度：tau=0.5 → 3.0/8（省 62% 计算），tau=0.3 → 5.0/8

## 诚实解读（重要）

训练早期（step~450）曾出现 CE8<CE1（深度价值信号）；但收敛后（17.7M）CE(d) 曲线
显示深度>2 无益。原因分析：训练/验证用的是 500k token 合成语料（重复 ~31 epoch），
分布过简、已饱和——**深度价值在"学习期"出现、在"饱和期"消失**。这不是架构否 verdict，
而是数据局限：S1 的真实判定必须在真实语料（FineWeb/多语言）上重跑才有效。
机制结论（路由健康、循环收敛、退出信号）均成立。

## 下一步

1. 真实语料版 S1（数据准备：tokenize 1-2GB 真实文本 → packed bin）
2. CE(d) 曲线在真实语料上重测（预期深度价值曲线更陡且不饱和）
3. S4 对比训练（0.3M pairs 级）


---

# 真实语料版（session 3, 断点式）

> 2026-10-14 17:00-18:1x | 持久化再次确认 | 18:29 平台强制断开, 续费后接力

## 数据管线（已固化到 tools/）

- 语料: hf-mirror 直连下载中文 wiki 524MB（服务器无 curl → tools/dl_corpus.py）
- tokenize: Jaume tokenizer (vocab 151647) → **139.6M tokens, uint32 bin**
  （关键适配: vocab 超 uint16 上限, data/eval/train 已支持 --data-dtype uint32）
- 训练: 新 run 目录 runs/s1_real, 从 Jaume warm-start 重新开始（与合成语料版同起点,
  用于对照两版 CE(d) 判定）

## 第一段训练与判定（3.83M tokens / 45.6 min, step 934）

- CE: 31.2(@0.8M) → 11.8(@2.5M) → **8.4(@3.7M)** —— 真实文本难得多, 远未收敛
- CE(d): d1=8.611 / **d2=8.426** / d4=8.557 / d8=8.732 → best_depth=2
- loop_delta_mean: [6.27, 1.33, 0.61, 0.41, 0.29, 0.23, 0.19, 0.16] 单调衰减（健康）
- 样本级退出: tau=0.5 → 平均深度 4.0/8（省 50% 计算）

## 诚实解读

与合成语料版判定暂时一致（best_depth=2），但**两版都还处于训练早期**：
合成版深度价值出现在 ~1.8M（且收敛 CE 1.04）；真实版收敛慢 3-5 倍，
3.83M tokens 时 CE 仍 8.4 —— 深度价值若存在, 预计 10-20M tokens 才显现。
**本判定不作最终结论**；续费第 2 段目标: 训至 ~10-12M tokens 后重测 CE(d)。

## 续训命令（断点后）

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup ~/miniconda3/bin/python -u   src/train_s1_mlm.py --config configs/350m_warmstart.json   --warmstart assets/jaume/350m-mlm --run runs/s1_real   --data data/real_train.bin --val data/real_val.bin --data-dtype uint32   --sup-tokens 20000000 --monitor-every 50 --save-every 100 --max-minutes 100 --bs 8 --resume   > ~/s1real2_$(date +%H%M).log 2>&1 < /dev/null &
```


---

# 真实语料正式判定（session 4, 2026-10-14 19:16-20:47, 12.29M tokens）

## 核心结果: best_depth 首次移至 4 —— 深度价值在真实数据上显现

| d=1 | d=2 | d=4 | d=8 |
|---|---|---|---|
| 6.087 | 6.021 | **5.980** | 6.146 |

- **best_depth=4**（此前合成版与真实早期版均为 2），d1→d4 实质收益 0.108 CE
- d8 开始过深（6.146）——循环深度呈"中间最优"，符合深度价值的成熟形态
- 对照: 合成语料 17.7M tokens 时 best=2 且 d8 最差; 真实语料 12.3M tokens 时 best=4
  → 数据多样性是关键变量; 深度价值需要多 token + 真实分布才出现
- loop_delta_mean [3.96→0.25] 单调衰减（收敛健康）
- 退出扫描: tau=0.5 → 平均深度 3.875/8（省 52% 计算, 且质量损失待验）

## 训练轨迹（段2: 3.83M → 12.29M）

- CE: 8.4(@3.8M) → 6.66(@5.7M) → 6.31(@8.8M) → **6.01(@9.8M)** → 5.98-6.1(@12.3M)
- expert load maxmin ≤1.34 全程均衡; cross-loop overlap 0.73-0.88（多对数）
- 收尾: SIGTERM 停在滚动档 step 3000 / 12.29M（损失 <4 分钟训练量）

## 判定与下一步

**架构核心主张初步成立**: 循环深度在真实数据上有实质收益（d4 > d2 > d1）。
但 CE 仍在下降通道（6.0，合成版收敛到 1.0），best_depth 可能继续右移。
下一步（下会话）: 续训至 20M tokens → 重测; 若 best_depth 稳定在 4-6 区间,
进入 S4 对比训练; 若继续右移, 继续训。

## 资产

- runs/s1_real/ckpt_latest.pt = step 3000 / 12.29M tokens（含优化器, 可续训）
- data/real_train.bin 139.4M tokens 仅消耗 8.8%（余量充足）
