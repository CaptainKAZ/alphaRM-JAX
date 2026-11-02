#!/usr/bin/env python3
"""End-to-end JAX RM rollout benchmark: env + converted PyTorch actor.

This measures the same per-step work as a JAX training collector:
  current obs -> actor (stochastic, masks) -> env.step -> next obs
The GRU carry is kept across steps; done envs are not reset (speed only).
"""
from __future__ import annotations

import argparse
import os
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from convert_rm_actor import convert
from jaxmarl.environments.rm.actor import RmActor, act
from jaxmarl.environments.rm.env import RmEnv


def load_params(checkpoint: str):
    ck = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    # BenchMARL stores the trainable actor under loss_agents.
    return convert(ck["loss_agents"])


def make_data(env, batch, steps, key):
    k_reset, k_actor, k_step = jax.random.split(key, 3)
    reset_keys = jax.random.split(k_reset, batch)
    obs, states = jax.vmap(lambda kk: env.reset(kk, mode="fixed"))(reset_keys)
    obs = jnp.stack([obs[a] for a in env.agents], axis=1)
    actor_keys = jax.random.split(k_actor, (steps, batch * len(env.agents)))
    step_keys = jax.random.split(k_step, (steps, batch))
    return obs, states, actor_keys, step_keys


def run(env, model, params, obs, states, actor_keys, step_keys, steps: int):
    batch = states.pos.shape[0]
    n = len(env.agents)
    h = jnp.zeros((batch * n, 256), dtype=jnp.float32)

    def body(carry, xs):
        state, h, obs = carry
        k_actor, k_step = xs
        am = jax.vmap(env.get_action_mask)(state).reshape(batch * n, 4)
        fm = jax.vmap(env.get_fire_mask)(state).reshape(batch * n, 2)
        raw, h = act(
            model, params, h, obs.reshape(batch * n, 140), k_actor,
            action_mask=am, fire_mask=fm, deterministic=False,
        )
        acts = {a: raw.reshape(batch, n, 6)[:, i] for i, a in enumerate(env.agents)}
        obs_d, state, _, _, _ = jax.vmap(env.step_env)(k_step, state, acts)
        obs = jnp.stack([obs_d[a] for a in env.agents], axis=1)
        return (state, h, obs), None

    out, _ = jax.lax.scan(
        body, (states, h, obs),
        (actor_keys[:steps], step_keys[:steps]),
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = RmEnv()
    model = RmActor()
    params = load_params(args.checkpoint)
    obs, states, actor_keys, step_keys = make_data(
        env, args.batch, args.steps, jax.random.PRNGKey(args.seed)
    )
    run_jit = jax.jit(
        lambda o, s, ak, sk: run(env, model, params, o, s, ak, sk, args.steps)
    )

    t0 = time.perf_counter()
    out = run_jit(obs, states, actor_keys, step_keys)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = run_jit(obs, states, actor_keys, step_keys)
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
