# jamue-loop

Jaume 350M 原生循环 × MoE（V6）试验包：在 Tesla V100 32G 单卡 6 小时窗口内
完成 350M warm-start 全流程微量试验（S1 循环适应 → S4 对比 → S6 样本级退出），
产出架构可行性判定报告。主计划见
`llm-test/docs/plans/TRAINING_PLAN_V6_JAUME_700M_LOOP_MOE_FINAL.md`。

## 开箱即用

```bash
bash run_all.sh          # 环境 -> Jaume 权重 -> 冒烟, 一条命令
```

开机完整操作顺序（含网络验证/降级链/硬截止）见 **RUN.md**。

## 结构

| 文件 | 作用 |
|---|---|
| `run_all.sh` | 总入口 |
| `setup.sh` | 一键环境：探测预置 torch → 锁定 torch 2.10.0+cu126 → 锁定依赖 → 体检 |
| `env_check.py` | GPU/sm70/SDPA/amp fp16/原子存档 全探测，输出 go/no-go JSON |
| `fetch_assets.py` | ModelScope 拉 Jaume 350m 子目录权重（默认 350m-mlm） |
| `smoke.py` | tier0 机制冒烟 + tier1 内联 mini 循环-MoE fp16 数值冒烟 |
| `pack_and_push.sh` | 产物自动回传（幂等，可 30 分钟一次定时调用） |
| `src/arch.py` | E/M/D 循环 MoE 模型（GQA+RoPE+SwiGLU+shared/4 routed top-2+圈数条件路由） |
| `src/train_s1_mlm.py` | S1 MLM 循环适应训练：稀疏深度监督、fp16+GradScaler、atomic rolling 存档、RNG/数据顺序恢复、硬时限、路由分化/深度价值转折点自动永久档 |
| `src/train_s4_contrast.py` | S4 InfoNCE in-batch 对比 + Matryoshka 截断维度 |
| `src/eval_depth.py` | CE(d) 深度曲线 + 样本级退出阈值扫描（平均有效深度） |
| `src/warmstart.py` | Jaume 350m → 本架构权重映射（E←L0-3 / M←L4-15 / D←尾部，含匹配率报告） |
| `src/data.py` | memmap .bin 语料 + pairs 读取 + 合成数据生成（本地管线验证用） |
| `configs/` | 350m_warmstart.json（主线）/ local_smoke.json（秒级验证） |

## V100 事实卡

torch ≤ 2.10.0（2.11 移除 Volta）；无 bf16，走 fp16+GradScaler；
无 flash-attention，SDPA 走 memory-efficient/math。详见 RUN.md 末尾。
