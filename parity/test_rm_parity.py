"""Parity smoke test: fixed reset + one movement step vs VMAS reference."""
import numpy as np
import jax
import jax.numpy as jnp
from jaxmarl import make

ROOT = __file__.rsplit("/", 1)[0]
base_cfg = make("rm").cfg
# The reference snapshot has no observation noise; production training has
# statistically matching random noise enabled.
env = type(make("rm"))(type(base_cfg)(obs_noise=False))
reset_ref = np.load(f"{ROOT}/ref_fixed_reset.npz")
step_ref = np.load(f"{ROOT}/ref_fixed_step.npz")

obs, st = env.reset(jax.random.PRNGKey(0))
checks = ["pos", "yaw", "hp", "heat", "target", "score", "time", "episode_elapsed"]
for k in checks:
    a = np.asarray(getattr(st, k))[None]
    np.testing.assert_allclose(a, reset_ref[k], rtol=1e-6, atol=1e-5, err_msg=k)
np.testing.assert_allclose(np.asarray(obs["agent_0"]), reset_ref["obs"][0, 0], rtol=1e-6, atol=1e-5)
np.testing.assert_allclose(np.asarray(env.get_global_state(st))[None], reset_ref["global_state"], rtol=1e-6, atol=1e-5)

raw = jnp.array([0.5, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
o, s, r, d, info = env.step_env(
    jax.random.PRNGKey(0), st, {a: raw for a in env.agents}
)
for k in checks:
    a = np.asarray(getattr(s, k))[None]
    np.testing.assert_allclose(a, step_ref[k], rtol=1e-6, atol=1e-5, err_msg=k)
np.testing.assert_allclose(np.asarray(o["agent_0"]), step_ref["obs"][0, 0], rtol=1e-6, atol=1e-5)
np.testing.assert_allclose(np.asarray(env.get_global_state(s))[None], step_ref["global_state"], rtol=1e-6, atol=1e-5)
np.testing.assert_allclose(np.asarray([r[a] for a in env.agents]), step_ref["reward_dense"][0], rtol=1e-5, atol=1e-6)

# JIT and vmap smoke
keys = jax.random.split(jax.random.PRNGKey(3), 8)
bo, bs = jax.vmap(env.reset)(keys)
ba = {a: jnp.zeros((8, 6), dtype=jnp.float32) for a in env.agents}
bo, bs, br, bd, bi = jax.jit(jax.vmap(env.step_env))(keys, bs, ba)
assert bo["agent_0"].shape == (8, 140)
assert bd["__all__"].shape == (8,)

# random / late-game reset smoke
cfg = type(env.cfg)(rsi_late_game=True, rsi_remaining_min=80.0, rsi_remaining_max=150.0)
env2 = type(env)(cfg)
ro, rs = env2.reset(jax.random.PRNGKey(4), mode="random")
assert 79.0 <= float(rs.time) <= 151.0
assert abs(float(rs.time) + float(rs.episode_elapsed) - 180.0) < 1e-4
print("PARITY OK: fixed reset/step, reward, JIT/vmap, random late-game reset")
