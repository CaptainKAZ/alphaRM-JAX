#!/usr/bin/env python3
"""Export deterministic RM fixed-reset state and actor/env metadata for the web game."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.config import RmConfig
from jaxmarl.environments.rm.env import RmEnv


def _encode(value) -> dict:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "data": base64.b64encode(
            np.ascontiguousarray(array).tobytes(order="C")
        ).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = RmConfig(
        robot_foul_speed=1.0e9,
        wall_foul_speed=1.0e9,
        wall_push_time=1.0e9,
        body_push_time=1.0e9,
        obs_noise=False,
        hit_cue_noise_deg=0.0,
    )
    env = RmEnv(cfg)
    obs_dict, state = env.reset(jax.random.PRNGKey(args.seed), mode="fixed")
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    state_values = []
    for index, (path, value) in enumerate(leaves):
        name = ".".join(
            str(
                getattr(part, "key", None)
                or getattr(part, "name", None)
                or part
            )
            for part in path
        )
        state_values.append({"index": index, "name": name, **_encode(value)})

    obs = jnp.stack([obs_dict[agent] for agent in env.agents], axis=0)
    payload = {
        "format": "rm_web_fixture_v1",
        "seed": int(args.seed),
        "agents": list(env.agents),
        "state": state_values,
        "obs": _encode(obs),
        "hidden": {
            "dtype": "float32",
            "shape": [6, 256],
            "data": base64.b64encode(bytes(6 * 256 * 4)).decode("ascii"),
        },
        "config": {
            "dt": cfg.dt,
            "episode_time": cfg.episode_time,
            "field_hx": cfg.field_hx,
            "field_hy": cfg.field_hy,
            "bound_x": cfg.bound_x,
            "bound_y": cfg.bound_y,
            "robot_radius": cfg.robot_radius,
            "v_max": cfg.v_max,
            "hp_max": cfg.hp_max,
            "heat_max": cfg.heat_max,
            "target_score": cfg.target_score,
            "respawn_time": cfg.respawn_time,
            "invuln_time": cfg.invuln_time,
            "yaw_rate_max": cfg.yaw_rate_max,
            "lock_range": cfg.lock_range,
            "center_radius": cfg.center_radius,
            "hp_noise_base": cfg.hp_noise_base,
            "hp_noise_spin": cfg.hp_noise_spin,
            "yaw_noise_deg": cfg.yaw_noise_deg,
            "obs_dim": cfg.obs_dim,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(
        f"wrote {out} state_leaves={len(state_values)} "
        f"obs={obs.shape} bytes={out.stat().st_size}"
    )


if __name__ == "__main__":
    main()
