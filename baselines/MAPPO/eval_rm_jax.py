"""Deterministic/stochastic evaluation for the JAX RM policy.

This is the JAX-side counterpart of ``.opencode/render_rm_eval_manual.py``:
it loads a JAX checkpoint, runs complete episodes (no PyTorch env/loop), and
writes an episode CSV/JSON summary.  It is intentionally headless; rendering
can consume the policy later.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.actor import RmActor, act, sample_actions
from jaxmarl.environments.rm.env import RmEnv

OUTCOME_NAMES = {
    0: "进行中", 1: "红方先到目标分", 2: "蓝方先到目标分",
    3: "红方时间胜", 4: "蓝方时间胜", 5: "红方伤害tiebreak",
    6: "蓝方伤害tiebreak", 7: "红方血量tiebreak", 8: "蓝方血量tiebreak",
    9: "红方胜(蓝方碰撞犯规)", 10: "蓝方胜(红方碰撞犯规)",
    20: "平局(时间tiebreak)", 21: "平局(双方同帧200)",
    22: "平局(双方同帧犯规)", 23: "空局(10秒无事件)",
    24: "红方胜(蓝方推墙/掩体犯规)", 25: "蓝方胜(红方推墙/掩体犯规)",
    26: "平局(双方推墙/掩体犯规)", 27: "红方胜(蓝方推人犯规)",
    28: "蓝方胜(红方推人犯规)", 29: "平局(双方推人犯规)",
}


def _stack(obs, agents):
    return jnp.stack([jnp.reshape(obs[a], (-1,)) for a in agents], axis=0)


def _load_params(path: str):
    with open(path, "rb") as f:
        ck = pickle.load(f)
    if "actor_params" in ck:
        return ck["actor_params"], ck
    if "actor" in ck:
        raise ValueError(
            "legacy checkpoint has serialized actor bytes, not actor_params; "
            "rerun training with the current trainer to produce a full checkpoint"
        )
    raise ValueError(f"unknown JAX checkpoint format: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    env = RmEnv()
    actor = RmActor()
    params, ck = _load_params(args.ckpt)
    N = env.num_agents
    out_dir = Path(args.out or Path("outputs") / time.strftime("eval_jax_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for ep in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        key, rk = jax.random.split(key)
        obs_d, state0 = env.reset(rk, mode=args.mode)
        obs = _stack(obs_d, env.agents)                 # [N,140]
        state = jax.tree.map(lambda x: x[None], state0)  # batch of one
        obs_b = obs[None]                               # [1,N,140]
        h = jnp.zeros((1, N, 256), dtype=jnp.float32)
        ep_return = jnp.zeros((N,), dtype=jnp.float32)
        done = False
        steps = 0
        while not bool(done) and steps < args.max_steps:
            masks = jax.vmap(env.get_action_mask)(state)[0]
            fire_masks = jax.vmap(env.get_fire_mask)(state)[0]
            key, ak = jax.random.split(key)
            keys = jax.random.split(ak, N)[None]        # [1,N,2]
            obs_flat = obs_b.reshape(N, 140)
            h_flat = h.reshape(N, 256)
            keys_flat = keys.reshape(N, 2)
            masks_flat = masks
            fire_flat = fire_masks
            if args.stochastic:
                action, _, _, h_flat = sample_actions(
                    actor, params, obs_flat, h_flat, keys_flat,
                    masks_flat, fire_flat,
                )
            else:
                action, h_flat = act(
                    actor, params, h_flat, obs_flat, keys_flat,
                    masks_flat, fire_flat, deterministic=True,
                )
            h = h_flat.reshape(1, N, 256)
            action = action.reshape(1, N, 6)
            action = action[0]
            ad = {a: action[i][None, :] for i, a in enumerate(env.agents)}
            key, sk = jax.random.split(key)
            nobs, nstate, reward, dones, _ = jax.vmap(env.step_env)(
                sk[None], state, ad
            )
            ep_return = ep_return + jnp.stack([reward[a][0] for a in env.agents])
            obs = _stack(nobs, env.agents)
            obs_b = obs[None]
            state = nstate
            steps += 1
            done = bool(dones["__all__"][0])
        state = jax.tree.map(lambda x: x[0], state)
        episodes.append(
            {
                "episode": ep,
                "steps": steps,
                "time_left": float(state.time),
                "score_red": float(state.score[0]),
                "score_blue": float(state.score[1]),
                "outcome": int(state.outcome),
                "outcome_name": OUTCOME_NAMES.get(int(state.outcome), "未知"),
                "return_mean": float(jnp.mean(ep_return)),
            }
        )
        print(
            f"ep={ep} steps={steps} time_left={float(state.time):.1f} "
            f"score={float(state.score[0]):.1f}:{float(state.score[1]):.1f} "
            f"outcome={int(state.outcome)}({OUTCOME_NAMES.get(int(state.outcome), '未知')})",
            flush=True,
        )

    red = np.asarray([e["score_red"] for e in episodes], dtype=np.float64)
    blue = np.asarray([e["score_blue"] for e in episodes], dtype=np.float64)
    summary = {
        "ckpt": args.ckpt,
        "mode": args.mode,
        "stochastic": args.stochastic,
        "episodes": len(episodes),
        "score_red_mean": float(red.mean()),
        "score_blue_mean": float(blue.mean()),
        "score_red_max": float(red.max()),
        "score_blue_max": float(blue.max()),
        "outcomes": {str(e["outcome"]): sum(1 for x in episodes if x["outcome"] == e["outcome"]) for e in episodes},
    }
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, ensure_ascii=False, indent=2)
    with open(out_dir / "eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0].keys()))
        w.writeheader()
        w.writerows(episodes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
