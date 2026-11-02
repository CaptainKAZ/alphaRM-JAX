#!/usr/bin/env python3
"""Generate a short no-noise JAX trace for validating the standalone JS engine."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.actor import RmActor, split_params
from jaxmarl.environments.rm.config import RmConfig
from jaxmarl.environments.rm.env import RmEnv


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--human", type=int, default=0)
    args = parser.parse_args()

    with open(args.ckpt, "rb") as handle:
        checkpoint = pickle.load(handle)
    params = jax.device_put(checkpoint["actor_params"])
    actor = RmActor()
    cfg = RmConfig(
        robot_foul_speed=1.0e9,
        wall_foul_speed=1.0e9,
        wall_push_time=1.0e9,
        body_push_time=1.0e9,
        obs_noise=False,
        hit_cue_noise_deg=0.0,
    )
    env = RmEnv(cfg)
    reset_key = jax.random.PRNGKey(0)
    key = jax.random.PRNGKey(17)
    obs_dict, state = env.reset(reset_key, mode="fixed")
    obs = jnp.stack([obs_dict[agent] for agent in env.agents], axis=0)
    hidden = jnp.zeros((6, 256), dtype=jnp.float32)

    trace = []
    for _ in range(args.steps):
        raw, next_hidden = actor.apply(
            {"params": params}, obs, hidden
        )
        raw = raw.astype(jnp.float32)
        masks = env.get_action_mask(state)
        fire_masks = env.get_fire_mask(state)
        cont_raw, target_logits, fire_logits = split_params(raw)
        loc = 3.0 * jnp.tanh(cont_raw[..., :4] / 3.0)
        cont = jnp.tanh(loc)
        target_logits = jnp.where(masks, target_logits, -1e9)
        fire_logits = jnp.where(fire_masks, fire_logits, -1e9)
        target = jnp.argmax(target_logits, axis=-1)
        fire = jnp.argmax(fire_logits, axis=-1)
        actions = jnp.concatenate(
            [
                cont,
                target.astype(jnp.float32)[:, None],
                fire.astype(jnp.float32)[:, None],
            ],
            axis=-1,
        )

        human_action = np.zeros(6, dtype=np.float32)
        i = args.human
        sign = 1.0 if cfg.team(i) == 0 else -1.0
        human_action[1] = sign
        yaw_error = wrap_angle(0.0 - float(state.yaw[i]))
        human_action[2] = np.clip(
            yaw_error / (cfg.yaw_rate_max * cfg.dt), -1.0, 1.0
        )
        actions = actions.at[i].set(jnp.asarray(human_action))
        next_obs_dict, next_state, _, dones, _ = env.step_env(
            key, state, {
                agent: actions[index]
                for index, agent in enumerate(env.agents)
            },
            deterministic=True,
        )
        trace.append(
            {
                "raw": np.asarray(raw, dtype=np.float32).tolist(),
                "actions": np.asarray(actions, dtype=np.float32).tolist(),
                "time": float(np.asarray(next_state.time)),
                "pos": np.asarray(next_state.pos, dtype=np.float32).tolist(),
                "hp": np.asarray(next_state.hp, dtype=np.float32).tolist(),
                "done": bool(np.asarray(dones["__all__"])),
            }
        )
        state = next_state
        obs = jnp.stack([next_obs_dict[a] for a in env.agents], axis=0)
        hidden = next_hidden

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"steps": trace}), encoding="utf-8")
    print(f"wrote {out} steps={len(trace)}")


if __name__ == "__main__":
    main()
