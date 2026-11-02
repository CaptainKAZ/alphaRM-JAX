# RM 3v3 JAX Environment

Registration:

```python
from jaxmarl import make
env = make("rm")
```

Minimal reset/step:

```python
import jax
import jax.numpy as jnp

env = make("rm")
obs, state = env.reset(jax.random.PRNGKey(0), mode="fixed")
actions = {agent: jnp.zeros((6,), jnp.float32) for agent in env.agents}
nobs, nstate, rewards, dones, infos = env.step_env(
    jax.random.PRNGKey(1), state, actions
)
```

关键文件：

- `config.py`：规则和奖励常量
- `env.py`：环境、几何、规则和观测
- `actor.py`：Transformer + GRU policy
- `critic.py`：team-canonical centralized critic
- `auxiliary.py`：训练期辅助预测头

当前维度：

- actor obs：140
- critic state：191
- action：6
- agents：6

完整说明：

- [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md)
- [`docs/IMPLEMENTATION_NOTES.md`](../../../docs/IMPLEMENTATION_NOTES.md)
