# 训练、恢复与评估

## 1. 最新完整配方

最新 run 目录：

```text
outputs/rm_jax_league900_1200/
```

最关键的保存参数来自 `logs/hparams.json`：

| 参数 | 值 |
|---|---:|
| 环境数 `B` | 800 |
| rollout 长度 `T` | 200 |
| 结束 iteration | 1200 |
| 起始 iteration | 900 |
| PPO epochs | 10 |
| minibatch envs | 16 |
| learning rate | `5e-5` |
| gamma / lambda | `0.999 / 0.95` |
| clip | `0.2` |
| entropy coefficient | `0.01` |
| value coefficient | `1.0` |
| rollout lag | 1 |
| bf16 compute | 是 |
| `scan-all` | 是 |
| activation remat | 是 |
| sync interval | 8 minibatches |
| fixed / random | 90% / 10% |
| fixed spawn rotation | 是，±180° |
| body-push terminal | 关闭；dense 惩罚保留 |
| 自博弈 / 冻结最优代 | 75% / 25% |

辅助头：

```text
--aux-heads
--aux-coef 0.02
--aux-warmup-iters 10
--aux-warmup-start-iter 500
--aux-lr 1e-4
```

## 2. 从 900 恢复到 1200 的命令模板

下面的 checkpoint 名按下述历史 run 约定填写；路径可由环境变量覆盖。

```bash
export ALPHARM_ROOT="$PWD"
export CKPT900="$ALPHARM_ROOT/outputs/rm_jax_league550_900/jax_ckpt_000900.pkl"
export LEAGUE550_900="$ALPHARM_ROOT/outputs/rm_jax_league550_900/league_best.pkl"
export BASE500="$ALPHARM_ROOT/outputs/rm_jax_cold500_lag1/jax_ckpt_000500.pkl"
export RUN="$ALPHARM_ROOT/outputs/rm_jax_league900_1200"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
PYTHONPATH="$ALPHARM_ROOT" \
python baselines/MAPPO/train_rm_jax.py \
  --B 800 --T 200 --updates 1200 --start-iter 900 \
  --resume-jax "$CKPT900" \
  --epochs 10 --minibatch-envs 16 \
  --lr 5e-5 --gamma 0.999 --lam 0.95 --clip 0.2 \
  --ent-coef 0.01 --vf-coef 1.0 --seed 0 \
  --bf16 --scan-all --remat --rollout-lag 1 \
  --reset-mode curriculum \
  --fixed-prob-start 0.9 --fixed-prob-end 0.9 \
  --fixed-spawn-rotate \
  --no-body-push-terminal \
  --league \
  --league-opponent "$LEAGUE550_900" \
  --league-base "$BASE500" \
  --league-self-play-frac 0.75 \
  --league-promote-win-rate 0.55 \
  --league-base-min-win-rate 0.50 \
  --aux-heads --aux-coef 0.02 \
  --aux-warmup-iters 10 --aux-warmup-start-iter 500 \
  --live-view --live-dir "$ALPHARM_ROOT/outputs/live" --live-every 1 \
  --save-dir "$RUN" \
  --save-every 10 \
  --timing --log-every 1 --fail-on-nan
```

要点：

- `--updates 1200 --start-iter 900` 表示继续训练到绝对 iteration 1200。
- `--no-body-push-terminal` 只关闭推人终局，dense 推人惩罚仍存在。
- `--fixed-spawn-rotate` 是训练增强；strict fixed 部署评估不加该参数。
- `--league-self-play-frac 0.75` 让 75% 环境做当前自博弈，25% 对冻结最优代。

## 3. 观察窗

训练命令写入：

```text
outputs/live/rm_live.json
```

启动：

```bash
bash liveview/start_rm_live.sh
```

页面：

- `http://localhost:8768/`
- `http://localhost:8768/stats.html`
- `http://localhost:8768/files`

## 4. 快速 smoke

不加载 checkpoint、不写日志：

```bash
PYTHONPATH=. python baselines/MAPPO/train_rm_jax.py \
  --B 16 --T 16 --updates 1 --epochs 1 \
  --disable-jit --no-log
```

## 5. 评测

```bash
PYTHONPATH=. python baselines/MAPPO/eval_rm_jax.py \
  --ckpt outputs/rm_jax_league900_1200/jax_ckpt_001200.pkl \
  --episodes 8 \
  --seed 0 \
  --mode fixed \
  --out outputs/eval_1200
```

默认为 deterministic action。添加 `--stochastic` 进行随机采样。环境观测噪声仍按 `RmConfig` 默认开启。

输出：

```text
outputs/eval_1200/eval.json
outputs/eval_1200/eval.csv
```

## 6. 1200 轮评估结果

最后联赛评估文件：

```text
outputs/rm_jax_league900_1200/league_eval.json
```

当前 1200 对上一冻结最优：

| 指标 | 数值 |
|---|---:|
| 总胜率 | 0.5625 |
| 平局率 | 0 |
| 红方胜率 | 0.3125 |
| 蓝方胜率 | 0.8125 |
| 红方平均分差 | -18.76 |
| 蓝方平均分差 | +19.12 |
| 平均步数 | 802.2 |

对 base500：

- 64/64 全胜，总胜率 1.0
- 平均分差约 79.05
- 红蓝两侧胜率均为 1.0

**不要只看总胜率。** 1200 对上一代仍有明显的红蓝不对称，说明颜色/队伍视角或训练匹配仍可能有偏置。

## 7. 辅助预测评估

评估样本：`B=64`、`T=100`、seed `100003`。

1200 checkpoint 的代表指标：

| 指标 | 数值 |
|---|---:|
| 可见敌人位置 MAE | 0.0269 m |
| 隐藏敌人位置 MAE | 0.3344 m |
| 敌人可见性准确率 | 0.9706 |
| 开火事件准确率 | 0.9985 |
| 条件命中概率 MAE | 0.0295 |
| 过热总体准确率 | 0.9580 |
| 过热 balanced accuracy | 0.7384 |
| 过热正例召回率 | 0.4933 |
| 自身资源 MAE | 0.0335 |
| 自身存活准确率 | 0.9994 |
| 自身过热准确率 | 0.9990 |

结论：

- 辅助头确实学到了下一帧位置、可见性、伤害/开火和自身资源信息。
- 隐藏敌人位置误差会随遮挡年龄增长，不应被解释为持续全知。
- 过热正例召回仍偏低；总体 accuracy 高主要受负例占比影响。
- 辅助头是训练模块，部署 actor 不依赖它们。

## 8. 跨代视频

历史视频位于运行机器的 `outputs/eval_550_900_final/`，不提交到 Git。包括：

- 550 vs 880
- 880 vs 900
- 900 vs 900
- 1200 vs 550 / 900 / 1070 / base500

生成脚本和派生资产需要在对应 run 环境中复现，仓库只保存源码与评估口径。
