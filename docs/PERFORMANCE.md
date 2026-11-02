# 性能：JAX 与 Torch/VMAS

## 1. 结论先行

在相同的 `B=800`、`T=200`、每 iter `160,000` 个环境步下：

- Torch/VMAS 稳态约 **159 s/iter**
- JAX 基础训练稳态约 **53.6 s/iter**
- JAX 联赛 + Aux Loss 训练约 **65.0 s/iter**

端到端约：

```text
Torch / JAX basic       ≈ 159.1 / 53.6 = 2.97×
Torch / JAX league+aux  ≈ 159.1 / 65.0 = 2.45×
```

**JAX 的优势主要来自 rollout。** 纯 PPO update 与 Torch 同量级，当前实现甚至可能更慢。

## 2. 测量环境与口径

硬件与栈：

- NVIDIA RTX 3080 Ti 12 GB
- JAX 0.10.2 CUDA
- Torch/VMAS 与 JAX 使用同一台机器
- Torch sampling/storing：CPU
- JAX：GPU `jit + vmap + scan`

共同训练形状：

```text
B = 800
T = 200
frames / iter = 800 × 200 = 160,000
```

JAX 稳态统计排除前 100 iter 以及编译阶段；Torch 统计排除前 20 个样本和明显外部停顿。

## 3. 稳态对比

| 阶段 | Torch v64 | JAX basic | JAX league + aux |
|---|---:|---:|---:|
| rollout / collection | **94.4 s** | **1.27 s** | **2.25 s** |
| GAE | Torch 内部 | **0.05 s** | **0.04 s** |
| 辅助标签生成 | 无 | 无 | **0.36 s** |
| PPO update / launch | **62.3 s** | **52.3 s** | **60.4 s** |
| 完整 iteration | **159.1 s** | **53.6 s** | **65.0 s** |
| 端到端 env steps/s | **1,005** | **2,986** | **2,460** |

说明：

- Torch 的 PPO 数字来自 `opt_loops`，不含 collection。
- JAX 的 update 使用 `timers_training_time`，包含 jitted PPO/launch 路径。
- JAX league 的 `total` 还包含少量 league evaluation 时间。

## 4. Rollout 为什么快

Torch 每个 iteration 的 collection 约 94.4 s，JAX 基础 rollout 约 1.27 s：

```text
rollout speedup ≈ 74×
```

主要来源：

- 环境状态和规则在 device 上更新
- 800 个环境用 `vmap`
- 200 步用 `scan`
- actor encoder、GRU 和 rollout 融合到编译图
- 避免逐步 Python/TorchRL transform 调度
- CPU→GPU 数据搬运次数显著减少

这个数字只描述 **rollout/collection**，不能当成完整训练 speedup。

## 5. PPO update 不是 JAX 的强项

| 训练 | PPO/update 时间 |
|---|---:|
| Torch v64 | 约 62.3 s |
| JAX basic | 约 52.3 s |
| JAX league + aux | 约 60.4 s |

Aux Loss 与 league 会增加反向图和优化成本，使 JAX update 接近 Torch。

所以正确结论是：

- JAX 把环境和 rollout 变成近乎免费
- update 仍是主要瓶颈
- JAX 训练端到端更快，但不是“所有环节都快几十倍”

## 6. 编译与总时长

JAX 首次 iteration 含编译：

```text
JAX basic 首次 iteration：约 191.6 s
JAX warm/persistent cache 后：约 53–55 s
```

Torch 不需要 XLA 编译。

实际总时长：

| 训练 | 实际/外推总时长 |
|---|---:|
| JAX cold500，500 iter | **7 h 32 min** |
| JAX 900→1200，300 iter | **5 h 20 min** |
| Torch at 159 s/iter，300 iter | 约 13 h 15 min（按稳态外推） |
| Torch at 159 s/iter，500 iter | 约 22 h 06 min（按稳态外推） |

Torch 总时长是外推值，不是某次完整 500 iter 的实测 wall time。

## 7. JAX 优化历程

同一 JAX 训练从初版演进到约 54 s/iter：

| 阶段 | 约 iteration 时间 |
|---|---:|
| 初始 full PPO | 273 s |
| 分离 actor/critic 参数与梯度 | 212 s |
| actor encoder 预计算 | 115 s |
| `scan-all` + head 移出 scan | 38–55 s |
| bf16 + remat + minibatch 16 + cache | 52–55 s |

关键优化：

- 参数和梯度树分离，避免重复计算
- encoder 结果复用
- minibatch 更新 `lax.scan`
- bf16 compute，参数/优化器保持 fp32
- activation remat
- persistent XLA cache
- `minibatch_envs=16`

## 8. 不能忽略的配置差异

Torch/JAX 不是完全同模型横比：

| 项 | Torch v64 | JAX current |
|---|---|---|
| actor | 11 tokens，d128 Transformer ×2，GRU256 | 相同主干 |
| critic | d128，2 层 | d192，3 层，FFN384 |
| compute | Torch AMP bf16 | JAX bf16 |
| update | Torch minibatch | `scan-all + remat + lag1` |
| league/aux | 无 | 最新 run 有 |

JAX critic 更重，但 rollout 仍然快得多。反过来，update 时间不能脱离这些结构差异解释。

## 9. 原始证据

汇总数据同时保存在
[`benchmarks/rm_perf_summary.csv`](benchmarks/rm_perf_summary.csv)，可直接用于表格或作图。

Torch：

```text
BenchMARL/outputs/train_rm_v64_resume2.log
BenchMARL/outputs/train_bench_torch_512_200.log
```

JAX 基础：

```text
JaxMARL/outputs/rm_jax_cold500_lag1/train.log
JaxMARL/outputs/rm_jax_cold500_lag1/logs/progress.csv
JaxMARL/outputs/rm_jax_timing_b800/train.log
```

JAX 联赛 + Aux：

```text
JaxMARL/outputs/rm_jax_league900_1200/train_resident.log
JaxMARL/outputs/rm_jax_league900_1200/logs/progress.csv
```

复现统计：

```bash
python - <<'PY'
import csv, statistics

paths = [
    ("basic", "JaxMARL/outputs/rm_jax_cold500_lag1/logs/progress.csv", 100, 500),
    ("league", "JaxMARL/outputs/rm_jax_league900_1200/logs/progress.csv", 1000, 1200),
]
for name, path, lo, hi in paths:
    rows = [
        r for r in csv.DictReader(open(path))
        if lo <= int(float(r["iteration"])) <= hi
    ]
    total = statistics.median(float(r["timers_total_time"]) for r in rows)
    rollout = statistics.median(float(r["timers_collection_time"]) for r in rows)
    train = statistics.median(float(r["timers_training_time"]) for r in rows)
    print(name, total, rollout, train)
PY
```

## 10. 重要结论

- JAX 的最大收益是把向量化环境和 recurrent rollout 编译到 device。
- 完整训练约快 2.4–3 倍。
- rollout 可快约 70 倍。
- PPO update 仍占 JAX iteration 的 90% 左右，是最重要的后续优化对象。
- 任何新性能结论都应同时报告 B、T、epochs、minibatch、bf16、scan/remat、aux 和 league 配置。
