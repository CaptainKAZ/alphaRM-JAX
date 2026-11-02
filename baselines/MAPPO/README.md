# MAPPO Baseline

Pure JAX MAPPO implementation, based on the PureJaxRL PPO implementation.

## 🔎 Implementation Details
General features:
* Agents are controlled by a single network architecture (either FF or RNN).
* Parameters are shared between agents.
* Each script has a `WorldStateWrapper` which provides a global `"world_state"` observation.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the algorithms as scripts, e.g.
```
python baselines/MAPPO/mappo_rnn_smax.py
```
Each file has a distinct config file which resides within [`config`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO/config).
The config file contains the MAPPO hyperparameters, the environment's parameters and the `wandb` details (`wandb` is disabled by default).

## alphaRM 3v3

alphaRM uses a standalone JAX entry point instead of a Hydra config:

```bash
PYTHONPATH=. python baselines/MAPPO/train_rm_jax.py --help
PYTHONPATH=. python baselines/MAPPO/eval_rm_jax.py --help
```

- `train_rm_jax.py`：MAPPO、自博弈、冻结最优代、辅助头、resume 和 liveview
- `league.py`：checkpoint 加载、best promotion、跨代评估
- `logger.py`：CSV/HJSON 日志和 `rm_live.json`
- `eval_rm_jax.py`：deterministic/stochastic 回合评测

完整训练配方见仓库根目录
[`docs/TRAINING_AND_EVALUATION.md`](../../docs/TRAINING_AND_EVALUATION.md)。
