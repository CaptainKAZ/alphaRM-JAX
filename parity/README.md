# Torch/JAX 对拍与转换工具

## 自包含测试

以下测试不依赖 PyTorch/VMAS：

```bash
python parity/test_ppo_math.py
python -m pytest parity/test_ppo_math.py -q
```

actor/critic parity 测试用于比较从 Torch checkpoint 转换出的 Flax 参数：

```bash
python -m pytest \
  parity/test_rm_actor_parity.py \
  parity/test_rm_critic_parity.py \
  parity/test_rm_parity.py -q
```

这些测试需要预先准备对应的 NPZ/参考数据；生成文件不提交到 Git。

## 生成 VMAS reference

`ref_*.py` 需要外部 PyTorch/VMAS 仓库。显式设置：

```bash
export VMAS_ROOT=/path/to/VectorizedMultiAgentSimulator
python parity/ref_snapshot.py
python parity/ref_step.py
python parity/ref_collision.py
```

输出写到 `parity/` 当前目录，文件名与脚本中的 `output_path(...)` 一致。

当前工具覆盖：

- fixed reset
- 单步移动
- collision / contact resolve
- push 与 foul
- Torch actor 到 Flax 参数转换
- Torch critic 到 Flax 参数转换
- attention、环境、rollout、PPO update 性能 profile

## 约束

- reference generator 使用 PyTorch，JAX 仓库本身不依赖 VMAS。
- 生成 `.npz` 只用于本地对拍，已被 `.gitignore` 排除。
- 单步通过不代表完整随机过程一致；报告时必须区分 deterministic 与 noisy 路径。
- 转换脚本变更后要同时检查 actor raw、hidden_next 和 critic value，而不只检查参数量。
