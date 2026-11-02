"""Statistics of RM slot-3 spawn distribution and initial target-3 availability.

Runs in alphaRM-jax-test.  No training and no environment stepping: only reset
and action-mask computation.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.rm.env import RmEnv
from jaxmarl.environments.rm.config import RmConfig


def main():
    env = RmEnv(RmConfig())
    cfg = env.cfg
    n = 1024
    key = jax.random.PRNGKey(20261028)

    # Fixed reference positions.
    _, fs = env.reset(key, "fixed", 1.0)
    fixed_pos = np.asarray(fs.pos)

    @jax.jit
    def reset_random_batch(keys):
        return jax.vmap(lambda k: env.reset(k, "random", 1.0))(keys)

    @jax.jit
    def mask_batch(states):
        return jax.vmap(env.get_action_mask)(states)

    out = {
        "n_random": n,
        "fixed_pos": fixed_pos.tolist(),
        "slots": {},
        "target3": {},
        "settings": {
            "rsi_late_game": bool(cfg.rsi_late_game),
            "rsi_remaining": [float(cfg.rsi_remaining_min), float(cfg.rsi_remaining_max)],
            "rsi_symmetric_prob": float(cfg.rsi_symmetric_prob),
            "spawn_anchor_init": bool(cfg.spawn_anchor_init),
            "spawn_anchor_jitter": float(cfg.spawn_anchor_jitter),
        },
    }

    for mode in ("fixed", "random"):
        if mode == "fixed":
            states = jax.tree_util.tree_map(lambda x: x[None], fs)
            masks = mask_batch(states)
        else:
            keys = jax.random.split(key, n)
            _, states = reset_random_batch(keys)
            masks = mask_batch(states)

        pos = np.asarray(states.pos)  # [S,6,2]
        mask = np.asarray(masks)     # [S,6,4]
        s = pos.shape[0]
        rec = {}
        for team, base in ((0, 0), (1, 3)):
            for slot in range(3):
                p = pos[:, base + slot, :]
                home = fixed_pos[base + slot]
                d_home = np.linalg.norm(p - home, axis=-1)
                d_center = np.linalg.norm(p, axis=-1)
                rec[f"r{slot+1}" if team == 0 else f"b{slot+1}"] = {
                    "home_dist_mean": float(d_home.mean()),
                    "home_dist_median": float(np.median(d_home)),
                    "home_dist_p90": float(np.percentile(d_home, 90)),
                    "near_home_1m": float((d_home <= 1.0).mean()),
                    "near_home_2m": float((d_home <= 2.0).mean()),
                    "center_dist_mean": float(d_center.mean()),
                    "own_half": float((p[:, 0] < 0).mean() if team == 0 else (p[:, 0] > 0).mean()),
                }

            # target class 3 is the enemy's slot 3
            m3 = mask[:, base:base + 3, 3]  # observers on this team, target3
            any_valid = mask[:, base:base + 3, 1:4].any(axis=-1)
            denom = mask[:, base:base + 3, 1:4].sum(axis=(-2, -1))
            share = np.divide(
                m3.sum(axis=-1),
                denom,
                out=np.zeros((s,), dtype=np.float64),
                where=denom > 0,
            )
            rec[f"target3_team{team}"] = {
                "mask_rate_per_observer": float(m3.mean()),
                "any_observer_can_target3": float(m3.any(axis=-1).mean()),
                "share3_among_available": float(share.mean()),
                "has_any_enemy_visible": float(any_valid.any(axis=-1).mean()),
            }

        out["slots"][mode] = rec

    # Explicit fixed-vs-random slot-3 home distance comparison.
    out["conclusion_inputs"] = {
        "fixed_slot3_home_dist": [
            float(np.linalg.norm(fixed_pos[2] - fixed_pos[2])),
            float(np.linalg.norm(fixed_pos[5] - fixed_pos[5])),
        ],
        "random_slot3_near_home_1m": [
            out["slots"]["random"]["r3"]["near_home_1m"],
            out["slots"]["random"]["b3"]["near_home_1m"],
        ],
        "random_slot3_own_half": [
            out["slots"]["random"]["r3"]["own_half"],
            out["slots"]["random"]["b3"]["own_half"],
        ],
    }

    path = Path("/tmp/slot3_reset_stats.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWROTE {path}")


if __name__ == "__main__":
    main()
