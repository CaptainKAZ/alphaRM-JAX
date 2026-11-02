"""Benchmark the JAX RM environment on CPU or GPU.

This is environment-only throughput: no policy network, no optimizer.
The environment is vmapped over a batch of independent initial states.
"""
from __future__ import annotations

import argparse
import os
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.rm.env import RmEnv


def make_batch(env: RmEnv, batch: int, steps: int, key):
    k_reset, k_act, k_step = jax.random.split(key, 3)
    reset_keys = jax.random.split(k_reset, batch)
    _, states = jax.vmap(lambda kk: env.reset(kk, mode="fixed"))(reset_keys)
    acts = jax.random.uniform(k_act, (steps, batch, 6, 6), minval=-1.0, maxval=1.0)
    actions = {a: acts[:, :, i] for i, a in enumerate(env.agents)}
    step_keys = jax.random.split(k_step, (steps, batch))
    return states, step_keys, actions


def run(env, states, step_keys, actions, steps: int):
    # Keep the whole scan inside one jit.  Jitting step_env inside the scan
    # body creates a nested call that is not fused and is especially slow on
    # GPU / small batches.
    def body(state, xs):
        key, act = xs
        _, state, _, _, _ = jax.vmap(env.step_env)(key, state, act)
        return state, None

    return jax.lax.scan(
        body, states,
        (step_keys[:steps], {a: actions[a][:steps] for a in env.agents}),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = RmEnv()
    states, keys, actions = make_batch(env, args.batch, args.steps, jax.random.PRNGKey(args.seed))

    # Compile and allocate outside the timed region.
    run_jit = jax.jit(lambda s, k, a: run(env, s, k, a, args.steps))
    t0 = time.perf_counter()
    out = run_jit(states, keys, actions)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = run_jit(states, keys, actions)
    jax.block_until_ready(out)
    total = time.perf_counter() - t0
    dt = total / args.steps
    rss = int(open(f"/proc/{os.getpid()}/statm").read().split()[1]) * 4096 / 1e9
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    dev = jax.devices()[0]
    gpu = ""
    if hasattr(dev, "memory_stats"):
        stats = dev.memory_stats() or {}
        gpu_peak = stats.get("peak_bytes_in_use")
        if gpu_peak:
            gpu = f" gpu_peak={gpu_peak/1e9:.2f}GB"
    print(
        f"device={dev} batch={args.batch} steps={args.steps} "
        f"compile={compile_s:.3f}s total={total:.3f}s "
        f"ms/step={dt*1000:.2f} env_steps/s={args.batch/dt:.0f} "
        f"rss={rss:.2f}GB host_peak={host_peak:.2f}GB{gpu}"
    )


if __name__ == "__main__":
    main()
