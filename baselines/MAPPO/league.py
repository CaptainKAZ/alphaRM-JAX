"""Small batched checkpoint league for the JAX RM trainer."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.rm.actor import RmActor, act


def load_actor_params(path: str) -> Mapping[str, Any]:
    with open(path, "rb") as f:
        ck = pickle.load(f)
    if "actor_params" not in ck:
        raise ValueError(f"checkpoint has no actor_params: {path}")
    # Pickled checkpoints may restore to host memory. Place the immutable
    # frozen policy once so rollout/evaluation do not transfer it every call.
    return jax.device_put(ck["actor_params"])


def save_best_actor(
    path: str,
    actor_params: Mapping[str, Any],
    iteration: int,
    metrics: Mapping[str, float],
    args: Mapping[str, Any],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "format": "rm_jax_league_actor_v1",
                "iteration": int(iteration),
                "actor_params": jax.device_get(actor_params),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "args": dict(args),
            },
            f,
        )
    tmp.replace(out)


def make_crossplay_evaluator(
    actor: RmActor,
    env,
    batch_size: int,
    max_steps: int,
    stochastic: bool = False,
):
    """Return a jitted, batched current-vs-frozen evaluator.

    Half the environments put the current policy on red, half on blue, with
    paired red/blue reset and step randomness. Finished environments are frozen
    until the scan ends.
    """
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("league eval batch must be a positive even number")
    if max_steps <= 0:
        raise ValueError("league eval steps must be positive")
    n = env.num_agents
    m = batch_size * n
    team = env.team
    current_red = jnp.arange(batch_size) < batch_size // 2
    current_team = jnp.where(current_red, 0, 1)
    current_agent = team[None, :] == current_team[:, None]

    def run(current_params, opponent_params, key):
        key, rk = jax.random.split(key)
        # Pair current-red and current-blue envs with identical reset and
        # stochastic-step keys, making the two colour halves directly
        # comparable instead of two independently sampled mini-suites.
        pair_reset_keys = jax.random.split(rk, batch_size // 2)
        reset_keys = jnp.concatenate([pair_reset_keys, pair_reset_keys], axis=0)
        obs_d, state = jax.vmap(lambda k: env.reset(k, mode="fixed"))(reset_keys)
        obs = jnp.stack([obs_d[a] for a in env.agents], axis=1)
        h_cur = jnp.zeros((m, 256), dtype=jnp.float32)
        h_opp = jnp.zeros((m, 256), dtype=jnp.float32)
        finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
        final_score = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        final_outcome = jnp.zeros((batch_size,), dtype=jnp.int32)
        steps = jnp.zeros((batch_size,), dtype=jnp.int32)

        def body(carry, key):
            state, obs, h_cur, h_opp, finished, final_score, final_outcome, steps = carry
            masks = jax.vmap(env.get_action_mask)(state).reshape((m, 4))
            fire = jax.vmap(env.get_fire_mask)(state).reshape((m, 2))
            k_cur, k_opp, k_step = jax.random.split(key, 3)
            a_cur, h_cur_next = act(
                actor,
                current_params,
                h_cur,
                obs.reshape((m, -1)),
                jax.random.split(k_cur, m),
                masks,
                fire,
                deterministic=not stochastic,
            )
            a_opp, h_opp_next = act(
                actor,
                opponent_params,
                h_opp,
                obs.reshape((m, -1)),
                jax.random.split(k_opp, m),
                masks,
                fire,
                deterministic=not stochastic,
            )
            actions = jnp.where(
                current_agent[..., None],
                a_cur.reshape((batch_size, n, 6)),
                a_opp.reshape((batch_size, n, 6)),
            )
            action_dict = {a: actions[:, i] for i, a in enumerate(env.agents)}
            pair_step_keys = jax.random.split(k_step, batch_size // 2)
            step_keys = jnp.concatenate([pair_step_keys, pair_step_keys], axis=0)
            nobs_d, nstate, _, dones, _ = jax.vmap(env.step_env)(
                step_keys, state, action_dict
            )
            gd = dones["__all__"]
            first_done = gd & (~finished)
            final_score = jnp.where(
                first_done[:, None], nstate.score, final_score
            )
            final_outcome = jnp.where(
                first_done, nstate.outcome, final_outcome
            )
            steps = jnp.where(finished, steps, steps + 1)
            finished_next = finished | gd
            nobs = jnp.stack([nobs_d[a] for a in env.agents], axis=1)
            state = jax.tree.map(
                lambda new, old: jnp.where(
                    finished_next.reshape((batch_size,) + (1,) * (new.ndim - 1)),
                    old,
                    new,
                ),
                nstate,
                state,
            )
            obs = jnp.where(finished_next[:, None, None], obs, nobs)
            h_cur_next = jnp.where(
                finished_next.repeat(n)[:, None], h_cur, h_cur_next
            )
            h_opp_next = jnp.where(
                finished_next.repeat(n)[:, None], h_opp, h_opp_next
            )
            return (
                state,
                obs,
                h_cur_next,
                h_opp_next,
                finished_next,
                final_score,
                final_outcome,
                steps,
            ), None

        (_, _, _, _, finished, final_score, final_outcome, steps), _ = jax.lax.scan(
            body,
            (state, obs, h_cur, h_opp, finished, final_score, final_outcome, steps),
            jax.random.split(key, max_steps),
        )
        # For an unfinished max-step rollout, use the latest state score.
        last_score = state.score
        score = jnp.where(
            finished[:, None],
            final_score,
            last_score,
        )
        outcome = jnp.where(finished, final_outcome, state.outcome)
        current_score = jnp.where(current_red, score[:, 0], score[:, 1])
        opponent_score = jnp.where(current_red, score[:, 1], score[:, 0])
        win = current_score > opponent_score
        draw = current_score == opponent_score
        return {
            "score": score,
            "outcome": outcome,
            "steps": steps,
            "finished": finished,
            "current_red": current_red,
            "current_score": current_score,
            "opponent_score": opponent_score,
            "win": win,
            "draw": draw,
            "score_diff": current_score - opponent_score,
        }

    return jax.jit(run)


def summarize_eval(result: Mapping[str, jax.Array]) -> dict[str, Any]:
    win = np.asarray(result["win"], dtype=np.float32)
    draw = np.asarray(result["draw"], dtype=np.float32)
    diff = np.asarray(result["score_diff"], dtype=np.float32)
    steps = np.asarray(result["steps"], dtype=np.float32)
    finished = np.asarray(result["finished"], dtype=np.bool_)
    outcome = np.asarray(result["outcome"], dtype=np.int32)
    current_red = np.asarray(result["current_red"], dtype=np.bool_)
    current_blue = ~current_red
    outcome_codes, outcome_counts = np.unique(outcome, return_counts=True)
    return {
        "win_rate": float(win.mean()),
        "draw_rate": float(draw.mean()),
        "loss_rate": float((1.0 - win - draw).mean()),
        "win_or_draw_rate": float((win + draw).mean()),
        "mean_score_diff": float(diff.mean()),
        "median_score_diff": float(np.median(diff)),
        "mean_steps": float(steps.mean()),
        "finished_rate": float(finished.mean()),
        "as_red_win_rate": float(win[current_red].mean()),
        "as_red_mean_score_diff": float(diff[current_red].mean()),
        "as_blue_win_rate": float(win[current_blue].mean()),
        "as_blue_mean_score_diff": float(diff[current_blue].mean()),
        "outcome_counts": {
            str(int(code)): int(count)
            for code, count in zip(outcome_codes, outcome_counts)
        },
    }


def write_eval_json(path: str, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(out)
