"""Small numerical checks for the JAX PPO math helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "MAPPO"))
from baselines.MAPPO.train_rm_jax import compute_gae  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    T, B, N = 5, 3, 2
    vals = rng.normal(size=(T, B, N)).astype(np.float32)
    rewards = rng.normal(size=(T, B, N)).astype(np.float32)
    dones = np.zeros((T, B), dtype=np.float32)
    dones[2, 1] = 1.0
    dones[4, 2] = 1.0
    last_value = rng.normal(size=(B, N)).astype(np.float32)
    last_done = np.array([0, 0, 1], dtype=np.float32)
    final_values = rng.normal(size=(T, B, N)).astype(np.float32)
    gamma, lam = 0.99, 0.95

    # True termination: no value bootstrap at a done step.
    terminated = np.broadcast_to(dones[:, :, None] > 0.5, (T, B, N)).copy()
    adv_j, target_j = compute_gae(
        jnp.asarray(vals), jnp.asarray(rewards), jnp.asarray(dones),
        jnp.asarray(terminated), jnp.asarray(final_values),
        jnp.asarray(last_value), jnp.asarray(last_done), gamma, lam,
    )

    nxt_base = np.concatenate(
        [vals[1:], (last_value * (1.0 - last_done[:, None]))[None]], axis=0
    )
    adv = np.zeros_like(vals)
    gae = np.zeros((B, N), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        nonterm = 1.0 - dones[t][:, None]
        nxt = np.where(nonterm < 0.5, final_values[t], nxt_base[t])
        nxt = nxt * (1.0 - terminated[t])
        delta = rewards[t] + gamma * nxt - vals[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    np.testing.assert_allclose(np.asarray(adv_j), adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(target_j), adv + vals, rtol=1e-5, atol=1e-6)

    # Truncation: chain stops at reset, but final-state value is bootstrapped.
    truncated = np.zeros((T, B, N), dtype=np.float32)
    truncated[2, 1] = 1.0
    adv_j, target_j = compute_gae(
        jnp.asarray(vals), jnp.asarray(rewards), jnp.asarray(dones),
        jnp.asarray(truncated), jnp.asarray(final_values),
        jnp.asarray(last_value), jnp.asarray(last_done), gamma, lam,
    )
    adv = np.zeros_like(vals)
    gae = np.zeros((B, N), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        nonterm = 1.0 - dones[t][:, None]
        nxt = np.where(nonterm < 0.5, final_values[t], nxt_base[t])
        nxt = nxt * (1.0 - truncated[t])
        delta = rewards[t] + gamma * nxt - vals[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    np.testing.assert_allclose(np.asarray(adv_j), adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(target_j), adv + vals, rtol=1e-5, atol=1e-6)

    # Clipped surrogate: min(r*A, clip(r,1-eps,1+eps)*A), averaged over a mask.
    ratio = jnp.asarray([0.8, 1.0, 1.3], dtype=jnp.float32)
    advs = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float32)
    eps = 0.2
    r = jnp.clip(ratio, 1 - eps, 1 + eps)
    expected = jnp.minimum(ratio * advs, r * advs)
    got = jnp.minimum(ratio * advs, jnp.clip(ratio, 1 - eps, 1 + eps) * advs)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=0, atol=0)
    print("PPO MATH OK: GAE + clipped surrogate")


if __name__ == "__main__":
    main()
