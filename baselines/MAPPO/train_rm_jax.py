"""Minimal end-to-end JAX MAPPO prototype for the RM 3v3 environment.

This is intentionally self-contained: it uses the JAX RM environment, the
Flax actor (warm-started from a PyTorch checkpoint when available), and the
team-canonical Flax critic.  It is a full rollout + GAE + PPO update loop, but
uses a single full-batch update per iteration (no minibatching) so it is easy
to validate before scaling up.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training.train_state import TrainState
from tqdm import tqdm

# Make JaxMARL importable when launched from the repository root.
ROOT = Path(__file__).resolve().parents[2]  # JaxMARL/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.actor import RmActor, sample_actions, evaluate_actions, evaluate_raw
from jaxmarl.environments.rm.auxiliary import (
    AUX_METRIC_NAMES,
    RmAuxHeads,
    auxiliary_loss,
    build_aux_targets,
)
from jaxmarl.environments.rm.critic import RmCritic, STATE_DIM
from parity.convert_rm_critic import convert as convert_critic
from jaxmarl.environments.rm.env import RmEnv
from jaxmarl.environments.rm.config import RmConfig
if __package__:
    from .logger import JaxScalarLogger, OUTCOME_NAMES, write_rm_live
    from .league import (
        load_actor_params,
        make_crossplay_evaluator,
        save_best_actor,
        summarize_eval,
        write_eval_json,
    )
else:
    from logger import JaxScalarLogger, OUTCOME_NAMES, write_rm_live
    from league import (
        load_actor_params,
        make_crossplay_evaluator,
        save_best_actor,
        summarize_eval,
        write_eval_json,
    )


def _normalize_advantage(adv: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    """Per-agent unbiased advantage standardization.

    Matches the PyTorch `normalize_advantage_exclude_dims=[-2]` intent:
    each agent gets its own mean/std across time and batch.  The reference
    uses torch.std (unbiased, ddof=1).  No done mask is used for statistics.
    """
    if mask is None:
        m = jnp.ones_like(adv)
    else:
        m = mask.astype(adv.dtype)
    count = jnp.maximum(jnp.sum(m, axis=(0, 1), keepdims=True), 1.0)
    mean = jnp.sum(adv * m, axis=(0, 1), keepdims=True) / count
    var = jnp.sum((adv - mean) ** 2 * m, axis=(0, 1), keepdims=True) / jnp.maximum(
        count - 1.0, 1.0
    )
    return (adv - mean) / jnp.sqrt(var + 1e-8)

OUTCOME_CODES = list(range(0, 11)) + list(range(20, 30))


def _episode_stats(traj):
    """Return episode-level statistics from a rollout.

    The environment auto-resets on done; trans stores the pre-reset state, so
    the score/outcome at a done timestep belongs to the episode that just
    finished.  Rewards after an auto-reset belong to the next episode.
    """
    rewards = np.asarray(traj["reward"]).sum(axis=2)   # [T,B]
    dones = np.asarray(traj["done"])                   # [T,B]
    scores = np.asarray(traj["score"])                 # [T,B,2]
    outcomes = np.asarray(traj["outcome"])             # [T,B]
    times = np.asarray(traj["time_left"])              # [T,B]
    returns, lengths, score_red, score_blue, times_left = [], [], [], [], []
    counts = {int(c): 0 for c in OUTCOME_CODES}
    for b in range(rewards.shape[1]):
        ep_ret, ep_len = 0.0, 0
        for t in range(rewards.shape[0]):
            ep_ret += float(rewards[t, b])
            ep_len += 1
            if bool(dones[t, b] > 0.5):
                returns.append(ep_ret)
                lengths.append(ep_len)
                score_red.append(float(scores[t, b, 0]))
                score_blue.append(float(scores[t, b, 1]))
                times_left.append(float(times[t, b]))
                code = int(outcomes[t, b])
                if code in counts:
                    counts[code] += 1
                ep_ret, ep_len = 0.0, 0
    out = {
        "done_count": int(dones.sum()),
        "episode_return_mean": float(np.mean(returns)) if returns else float("nan"),
        "episode_return_min": float(np.min(returns)) if returns else float("nan"),
        "episode_return_max": float(np.max(returns)) if returns else float("nan"),
        "episode_len_mean": float(np.mean(lengths)) if lengths else float("nan"),
        "score_red": float(np.mean(score_red)) if score_red else float(scores[-1, :, 0].mean()),
        "score_blue": float(np.mean(score_blue)) if score_blue else float(scores[-1, :, 1].mean()),
        "time_left": float(np.mean(times_left)) if times_left else float(times[-1].mean()),
        "last_score_red": float(scores[-1, :, 0].mean()),
        "last_score_blue": float(scores[-1, :, 1].mean()),
        "outcome_counts": counts,
    }
    return out


def _outcome_text(counts):
    """Compact terminal-code summary for the tqdm/log line."""
    items = sorted(((int(c), int(n)) for c, n in counts.items() if int(n) > 0),
                   key=lambda kv: (-kv[1], kv[0]))
    return " ".join(f"{OUTCOME_NAMES.get(c, str(c))}={n}" for c, n in items) or "-"


def _stack_obs(obs, agents):
    return jnp.stack([obs[a] for a in agents], axis=1)  # [B,N,140]


def _freeze_params(params):
    """Copy a parameter tree so later donated updates cannot reuse its buffers."""
    return jax.tree.map(jnp.copy, params)


def _action_dict(actions, agents):
    return {a: actions[:, i] for i, a in enumerate(agents)}


def _selected_enemy_matrix(cfg, state, values):
    """Gather values[b, i, selected_enemy(i)] for a batched per-pair tensor."""
    selected = jnp.zeros((values.shape[0], cfg.n_agents), dtype=values.dtype)
    for i in range(cfg.n_agents):
        for k in range(3):
            j = cfg.enemy_indices(i)[k]
            selected = selected.at[:, i].set(
                jnp.where(state.target[:, i] == k + 1, values[:, i, j], selected[:, i])
            )
    return selected


def compute_gae(
    vals,
    rewards,
    dones,
    terminated,
    final_values,
    last_value,
    last_done,
    gamma,
    lam,
):
    """Compute GAE and value targets for [T,B,N] tensors.

    ``last_done`` is [B]. ``done`` breaks the GAE chain for every agent;
    ``terminated`` disables bootstrap, while truncated agents use the critic
    value of the pre-reset final state in ``final_values``.
    """
    last_value = last_value * (1.0 - last_done[..., None])
    next_vals = jnp.concatenate([vals[1:], last_value[None]], axis=0)
    gae0 = jnp.zeros_like(last_value)

    def rev_body(gae, x):
        d, terminal, r, v, nv, final_v = x
        # ``done`` breaks the episode/GAE chain. ``terminated`` additionally
        # removes the value bootstrap. Truncated agents use the value of the
        # pre-reset final state captured by the environment.
        next_value = jnp.where(d[..., None] > 0.5, final_v, nv)
        next_value = next_value * (1.0 - terminal)
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * (1.0 - d[..., None]) * gae
        return gae, gae

    xs = (
        dones[::-1],
        terminated[::-1],
        rewards[::-1],
        vals[::-1],
        next_vals[::-1],
        final_values[::-1],
    )
    _, adv_rev = jax.lax.scan(rev_body, gae0, xs)
    adv = adv_rev[::-1]
    return adv, adv + vals


def make_rollout(env, actor, critic, cfg, reset_mode="fixed", league=False):
    N = env.num_agents
    B = cfg["B"]
    T = cfg["T"]
    M = B * N

    def _step(carry, key):
        state, obs, h, h_opp, ap, bp, cp, last_done, fixed_prob, league_mask, league_teams = carry
        obs_flat = obs.reshape((M, -1))
        tm = jax.vmap(env.get_action_mask)(state).reshape((M, 4))
        fm = jax.vmap(env.get_fire_mask)(state).reshape((M, 2))
        keys = jax.random.split(key, M)
        act_cur, logp_cur, ent_cur, h2_cur = sample_actions(
            actor, ap, obs_flat, h, keys, tm, fm
        )
        if league:
            opp_keys = jax.random.split(jax.random.fold_in(key, 1), M)
            act_opp, logp_opp, ent_opp, h2_opp = sample_actions(
                actor, bp, obs_flat, h_opp, opp_keys, tm, fm
            )
            # Self-play envs train both sides. League envs train only the
            # current-policy team; assignments are fixed per environment so
            # the learner does not swap sides mid-episode.
            train_agents = (~league_mask[:, None]) | (
                env.team[None, :] == league_teams[:, None]
            )
            train_agents_flat = train_agents.reshape((M,))
            act_flat = jnp.where(train_agents_flat[:, None], act_cur, act_opp)
            logp = jnp.where(train_agents_flat, logp_cur, logp_opp)
            ent = jnp.where(train_agents_flat, ent_cur, ent_opp)
        else:
            act_flat, logp, ent = act_cur, logp_cur, ent_cur
            h2_opp = h_opp
            train_agents = jnp.ones((B, N), dtype=jnp.bool_)
        actions = act_flat.reshape((B, N, 6))
        ad = _action_dict(actions, env.agents)
        g = jax.vmap(env.get_global_state)(state)
        value = critic.apply({"params": cp}, g)
        key_step, key_reset = jax.random.split(key)
        step_keys = jax.random.split(key_step, B)
        nobs, nstate, reward, done, info = jax.vmap(env.step_env)(step_keys, state, ad)
        # Keep the raw post-step state for logging *before* auto-reset.  For
        # finished envs `_pick` below replaces nstate with the reset state, so
        # using nstate in logs would report score/time/outcome from t=0.
        step_state = nstate
        step_global_state = jax.vmap(env.get_global_state)(step_state)
        r = jnp.stack([reward[a] for a in env.agents], axis=1)
        gd = jnp.asarray(done["__all__"], dtype=jnp.float32)
        collision_foul = (
            (step_state.outcome == 9)
            | (step_state.outcome == 10)
            | (step_state.outcome == 22)
        )
        wall_push_foul = (
            (step_state.outcome == 24)
            | (step_state.outcome == 25)
            | (step_state.outcome == 26)
        )
        body_push_foul = (
            (step_state.outcome == 27)
            | (step_state.outcome == 28)
            | (step_state.outcome == 29)
        )
        foul_terminal = collision_foul | wall_push_foul | body_push_foul
        foul_culprit = jnp.where(
            collision_foul[:, None],
            step_state.foul_agent,
            jnp.where(
                wall_push_foul[:, None],
                step_state.wall_push_foul_intent > 0.0,
                jnp.where(
                    body_push_foul[:, None],
                    step_state.body_push_foul,
                    False,
                ),
            ),
        )
        done_agents = gd[:, None] > 0.5
        # Every responsibility-bearing foul uses the same semantics: the
        # culprit terminates, while unaffected/victim agents are truncated by
        # the match-ending event and bootstrap from the pre-reset final state.
        terminated = done_agents & ((~foul_terminal[:, None]) | foul_culprit)
        truncated = done_agents & (~terminated)
        final_value = jax.lax.cond(
            jnp.any(gd > 0.5),
            lambda _: critic.apply({"params": cp}, step_global_state),
            lambda _: jnp.zeros((B, N), dtype=jnp.float32),
            operand=None,
        )

        # Auxiliary-world-model labels. Damage matrices are indexed
        # [attacker, victim]; sum(dim=1) is therefore damage taken per robot.
        selected_damage = _selected_enemy_matrix(env.cfg, step_state, step_state.last_damage)
        selected_p_hit = jnp.zeros((B, N), dtype=jnp.float32)
        selected_kill = jnp.zeros((B, N), dtype=jnp.bool_)
        for i in range(N):
            for k in range(3):
                j = int(env.cfg.enemy_indices(i)[k])
                mask = step_state.target[:, i] == k + 1
                p_hit = step_state.p_t[:, i] * (
                    1.0 - env.cfg.spin_incoming_coef * step_state.spin[:, j]
                )
                selected_p_hit = selected_p_hit.at[:, i].set(
                    jnp.where(mask, p_hit, selected_p_hit[:, i])
                )
                killed = (
                    mask
                    & (state.alive[:, j] > 0.5)
                    & (~step_state.alive[:, j])
                    & (selected_damage[:, i] > 0.0)
                )
                selected_kill = selected_kill.at[:, i].set(selected_kill[:, i] | killed)
        damage_taken = step_state.last_damage.sum(axis=1)

        # Auto-reset finished environments and zero their recurrent state.
        def _reset_all(_):
            rk = jax.random.split(key_reset, B)
            return jax.vmap(lambda k: env.reset(k, mode=reset_mode, fixed_prob=fixed_prob))(rk)

        reset_obs, reset_state = jax.lax.cond(
            jnp.any(gd > 0.5),
            _reset_all,
            lambda _: (nobs, nstate),
            operand=None,
        )

        def _pick(new, old):
            shape = (B,) + (1,) * (new.ndim - 1)
            return jnp.where(gd.reshape(shape) > 0.5, new, old)

        nobs = jax.tree.map(_pick, reset_obs, nobs)
        nstate = jax.tree.map(_pick, reset_state, nstate)
        h2_cur = jnp.where(
            gd[:, None, None] > 0.5,
            jnp.zeros_like(h2_cur.reshape(B, N, 256)),
            h2_cur.reshape(B, N, 256),
        ).reshape(M, 256)
        h2_opp = jnp.where(
            gd[:, None, None] > 0.5,
            jnp.zeros_like(h2_opp.reshape(B, N, 256)),
            h2_opp.reshape(B, N, 256),
        ).reshape(M, 256)
        nobs_stack = _stack_obs(nobs, env.agents)
        trans = dict(
            obs=obs,
            actions=actions,
            logp=logp.reshape((B, N)),
            # Store [B,N] here; lax.scan adds the leading time axis.
            actor_train_mask=train_agents.astype(jnp.float32),
            value=value,
            reward=r,
            done=gd,
            terminated=terminated,
            truncated=truncated,
            final_value=final_value,
            target_mask=tm.reshape((B, N, 4)),
            fire_mask=fm.reshape((B, N, 2)),
            global_state=g,
            # stats describe the frame produced by this step.  step_state is
            # the raw post-step state, captured before auto-reset.
            score=step_state.score,
            outcome=step_state.outcome,
            time_left=step_state.time,
            next_global_state=step_global_state,
            aux_fire=step_state.last_fired.astype(jnp.float32),
            aux_p_hit=selected_p_hit,
            aux_damage=selected_damage,
            aux_kill=selected_kill,
            aux_target_present=(step_state.target != 0),
            aux_damage_taken=damage_taken,
            aux_kill_score=step_state.last_kill_score,
        )
        return (
            nstate,
            nobs_stack,
            h2_cur,
            h2_opp,
            ap,
            bp,
            cp,
            gd,
            fixed_prob,
            league_mask,
            league_teams,
        ), trans

    def rollout(
        actor_params,
        opponent_params,
        critic_params,
        state,
        obs,
        h0,
        h0_opp,
        key,
        fixed_prob,
        league_mask,
        league_teams,
    ):
        (
            state,
            obs,
            h,
            h_opp,
            _,
            _,
            _,
            last_done,
            _,
            _,
            _,
        ), traj = jax.lax.scan(
            _step,
            (
                state,
                obs,
                h0,
                h0_opp,
                actor_params,
                opponent_params,
                critic_params,
                jnp.zeros((B,), dtype=jnp.float32),
                fixed_prob,
                league_mask,
                league_teams,
            ),
            jax.random.split(key, T),
        )
        g = jax.vmap(env.get_global_state)(state)
        last_value = critic.apply({"params": critic_params}, g)
        traj["h0"] = h0.reshape((B, N, 256))
        return state, obs, h, h_opp, traj, last_value, last_done

    return rollout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch-envs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--critic-ckpt", type=str, default="")
    ap.add_argument("--random-reset", action="store_true")
    ap.add_argument("--reset-mode", choices=("fixed", "random", "curriculum"), default="fixed")
    ap.add_argument("--rsi-late-game", action=argparse.BooleanOptionalAction, default=True,
                    help="random episodes start with only rsi_remaining_* seconds left on the 180s clock")
    ap.add_argument("--rsi-remaining-min", type=float, default=None)
    ap.add_argument("--rsi-remaining-max", type=float, default=None)
    ap.add_argument("--rsi-time-min", type=float, default=None)
    ap.add_argument("--rsi-time-max", type=float, default=None)
    ap.add_argument("--rsi-symmetric-prob", type=float, default=None)
    ap.add_argument("--no-body-push-terminal", action="store_true",
                    help="keep body-push dense penalty but disable only its terminal foul")
    ap.add_argument("--fixed-spawn-rotate", action="store_true",
                    help="training: randomly rotate the fixed spawn triangle")
    ap.add_argument("--fixed-spawn-rotate-max", type=float, default=3.141592653589793,
                    help="max absolute triangle rotation in radians (default 180 deg)")
    ap.add_argument("--fixed-prob-start", type=float, default=1.0)
    ap.add_argument("--fixed-prob-end", type=float, default=1.0)
    ap.add_argument("--curriculum-steps", type=int, default=0,
                    help="environment steps to anneal fixed_prob_start -> fixed_prob_end")
    ap.add_argument("--curriculum-offset-steps", type=int, default=0)
    ap.add_argument("--start-iter", type=int, default=0,
                    help="resume the training schedule at this iteration (VMAS warm start)")
    ap.add_argument("--save-dir", type=str, default="")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--sync-every", type=int, default=8, help="GPU sync every N PPO minibatches (memory/speed tradeoff)")
    ap.add_argument("--scan", action="store_true", help="scan minibatches inside each epoch")
    ap.add_argument("--scan-all", action="store_true", help="scan epochs and minibatches in one jitted call")
    ap.add_argument("--timing", action="store_true", help="print per-phase timing")
    ap.add_argument("--profile-dir", type=str, default="", help="JAX profiler trace output directory")
    ap.add_argument("--profile-at", type=int, default=1, help="profile iteration offset after start")
    ap.add_argument("--profile-stage", choices=("all", "rollout", "update"), default="all", help="which phase to trace")
    ap.add_argument("--remat", action="store_true", help="checkpoint actor/critic sequence to save memory")
    ap.add_argument("--bf16", action="store_true", help="use bf16 compute for actor/critic (params/optimizer stay fp32)")
    ap.add_argument("--log-dir", type=str, default="", help="CSV/HJSON log root")
    ap.add_argument("--log-every", type=int, default=1, help="write logs every N iterations")
    ap.add_argument("--live-view", action="store_true", help="write the RM liveview JSON for the web dashboard")
    ap.add_argument("--live-dir", type=str, default="outputs/live", help="directory for rm_live.json")
    ap.add_argument("--live-every", type=int, default=1, help="write liveview every N iterations")
    ap.add_argument("--no-log", action="store_true", help="disable CSV logging")
    ap.add_argument("--resume-jax", type=str, default="", help="resume from a full JAX checkpoint")
    ap.add_argument("--rollout-lag", type=int, default=0,
                    help="use rollout actor/critic that is N optimizer updates behind training (0=off)")
    ap.add_argument("--league", action="store_true",
                    help="mix current self-play with a frozen best opponent")
    ap.add_argument("--league-opponent", type=str, default="",
                    help="actor checkpoint used as the initial frozen best")
    ap.add_argument("--league-base", type=str, default="",
                    help="actor checkpoint used as the fixed base500 regression guard")
    ap.add_argument("--league-self-play-frac", type=float, default=0.75,
                    help="fraction of environment slots kept as current-vs-current self-play")
    ap.add_argument("--league-eval-every", type=int, default=10)
    ap.add_argument("--league-eval-envs", type=int, default=32)
    ap.add_argument("--league-eval-steps", type=int, default=1800)
    ap.add_argument(
        "--league-eval-seed",
        type=int,
        default=-1,
        help="fixed reset/noise suite for best selection (default: seed + 100003)",
    )
    ap.add_argument("--league-promote-win-rate", type=float, default=0.55)
    ap.add_argument("--league-promote-min-score-diff", type=float, default=0.0)
    ap.add_argument("--league-base-min-win-rate", type=float, default=0.50)
    ap.add_argument("--league-best-out", type=str, default="")
    ap.add_argument("--league-eval-json", type=str, default="")
    ap.add_argument("--aux-heads", action="store_true",
                    help="enable training-only world-understanding auxiliary heads")
    ap.add_argument("--aux-coef", type=float, default=0.05,
                    help="outer weight of the combined auxiliary loss")
    ap.add_argument("--aux-warmup-iters", type=int, default=5,
                    help="linearly ramp the auxiliary loss weight over N iterations")
    ap.add_argument("--aux-warmup-start-iter", type=int, default=-1,
                    help="absolute iteration for aux warmup; default uses --start-iter")
    ap.add_argument("--aux-lr", type=float, default=1e-4,
                    help="learning rate for the auxiliary-head optimizer")
    ap.add_argument("--aux-track-weight", type=float, default=1.0)
    ap.add_argument("--aux-damage-weight", type=float, default=1.0)
    ap.add_argument("--aux-heat-weight", type=float, default=0.75)
    ap.add_argument("--aux-rules-weight", type=float, default=0.75)
    ap.add_argument("--aux-self-weight", type=float, default=1.0)
    ap.add_argument("--fail-on-nan", action="store_true", help="abort if actor/critic loss is non-finite")
    ap.add_argument("--disable-jit", action="store_true",
                    help="disable JAX JIT for quick smoke tests (slower per step, no compile wait)")
    ap.add_argument(
        "--jax-cache-dir", type=str, default="",
        help="persistent JAX/XLA compilation cache directory (default: <JaxMARL>/.jax_cache)",
    )
    args = ap.parse_args()
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("[RM-JAX] jax_disable_jit=True (smoke mode)", flush=True)

    # Persist XLA compilation results.  This is especially important for the
    # large scan-all update: the first GPU run pays 30-60s of compilation,
    # while later runs should hit this cache.
    cache_dir = args.jax_cache_dir or os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(ROOT / "outputs" / ".jax_cache"),
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except PermissionError:
        cache_dir = "/tmp/alphaRM_jax_cache"
        os.makedirs(cache_dir, exist_ok=True)
        print(f"cache dir fallback: {cache_dir}", flush=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    print(f"JAX compilation cache: {cache_dir}", flush=True)

    log_root = JaxScalarLogger.default_root(args.save_dir, args.log_dir)
    logger = None if args.no_log else JaxScalarLogger(log_root, vars(args))
    if logger is not None:
        print(f"logging to {log_root}", flush=True)

    cfg = dict(B=args.B, T=args.T)
    env_cfg = RmConfig(body_push_time=1e9) if args.no_body_push_terminal else RmConfig()
    reset_overrides = {}
    reset_overrides["rsi_late_game"] = bool(args.rsi_late_game)
    if args.rsi_remaining_min is not None:
        reset_overrides["rsi_remaining_min"] = args.rsi_remaining_min
    if args.rsi_remaining_max is not None:
        reset_overrides["rsi_remaining_max"] = args.rsi_remaining_max
    if args.rsi_time_min is not None:
        reset_overrides["rsi_time_min"] = args.rsi_time_min
    if args.rsi_time_max is not None:
        reset_overrides["rsi_time_max"] = args.rsi_time_max
    if args.rsi_symmetric_prob is not None:
        reset_overrides["rsi_symmetric_prob"] = args.rsi_symmetric_prob
    if args.fixed_spawn_rotate:
        reset_overrides["fixed_spawn_rotate"] = True
        reset_overrides["fixed_spawn_rotate_max"] = args.fixed_spawn_rotate_max
    if reset_overrides:
        env_cfg = replace(env_cfg, **reset_overrides)
        print(f"[RM-JAX] reset overrides: {reset_overrides}", flush=True)
    env = RmEnv(env_cfg)
    if args.no_body_push_terminal:
        print("[RM-JAX] body-push terminal disabled", flush=True)
    compute_dtype = jnp.bfloat16 if args.bf16 else jnp.float32
    actor = RmActor(dtype=compute_dtype)
    critic = RmCritic(dtype=compute_dtype)
    print(f"compute_dtype: {compute_dtype} (params/optimizer fp32)", flush=True)
    N = env.num_agents
    B, T = args.B, args.T
    M = B * N

    # ---- actor parameters ----
    if args.ckpt:
        import torch
        from parity.convert_rm_actor import convert

        ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
        actor_params = convert(ck["loss_agents"])
        print(f"loaded actor checkpoint: {args.ckpt}", flush=True)
    else:
        obs0 = jnp.zeros((1, env.observation_space(env.agents[0]).shape[0]))
        h0 = jnp.zeros((1, 256))
        actor_params = actor.init(jax.random.PRNGKey(args.seed), obs0, h0)["params"]

    # ---- critic parameters ----
    critic_params = critic.init(
        jax.random.PRNGKey(args.seed + 1), jnp.zeros((1, STATE_DIM))
    )["params"]
    critic_ckpt = args.critic_ckpt or args.ckpt
    if critic_ckpt:
        import torch
        ck_c = torch.load(critic_ckpt, map_location="cpu", mmap=True, weights_only=False)
        critic_params = convert_critic(ck_c["loss_agents"])
        print(f"loaded critic checkpoint: {critic_ckpt}", flush=True)
    actor_tx = optax.chain(optax.clip_by_global_norm(2.0), optax.adam(args.lr, eps=1e-5))
    critic_tx = optax.chain(optax.clip_by_global_norm(2.0), optax.adam(args.lr, eps=1e-5))
    actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_tx)
    critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=critic_tx)
    aux_model = RmAuxHeads(dtype=jnp.float32)
    aux_params = aux_model.init(
        jax.random.PRNGKey(args.seed + 2),
        jnp.zeros((1, 256)),
        jnp.zeros((1, 256)),
        jnp.zeros((1, 6)),
        jnp.zeros((1, env.observation_space(env.agents[0]).shape[0])),
    )["params"]
    aux_tx = optax.chain(optax.clip_by_global_norm(2.0), optax.adam(args.aux_lr, eps=1e-5))
    aux_state = TrainState.create(apply_fn=aux_model.apply, params=aux_params, tx=aux_tx)

    # ---- frozen-best league ----
    league_mask = jnp.zeros((B,), dtype=jnp.bool_)
    league_teams = jnp.zeros((B,), dtype=jnp.int32)
    best_actor_params = actor_state.params
    base_actor_params = actor_state.params
    league_eval = None
    league_best_out = args.league_best_out
    league_eval_json = args.league_eval_json
    if args.league:
        if not (0.0 < args.league_self_play_frac < 1.0):
            raise ValueError("--league-self-play-frac must be in (0, 1)")
        n_self = int(round(B * args.league_self_play_frac))
        if n_self <= 0 or n_self >= B:
            raise ValueError("league split produced an empty self-play or opponent batch")
        league_mask = jnp.arange(B) >= n_self
        league_teams = (jnp.arange(B) % 2).astype(jnp.int32)
        run_root = (
            Path(args.log_dir).parent
            if args.log_dir
            else Path(args.save_dir)
            if args.save_dir
            else ROOT / "outputs"
        )
        league_best_out = league_best_out or str(run_root / "league_best.pkl")
        league_eval_json = league_eval_json or str(run_root / "league_eval.json")
        print(
            f"league enabled: self_play={n_self}/{B} frozen={B - n_self}/{B} "
            f"best_path={league_best_out}",
            flush=True,
        )

    # Rollout lag: collect with the policy from N updates ago while the
    # optimizer is already at the newer parameters.  This mirrors the
    # one-iteration stale prefetch used by Torch/VMAS OVERLAP_COLLECTION.
    # Only lag=0/1 is supported; checkpoint stores both parameter trees.
    if args.rollout_lag not in (0, 1):
        raise ValueError("--rollout-lag currently supports only 0 or 1")
    rollout_actor_params = actor_state.params
    rollout_critic_params = critic_state.params

    # ---- initial env state / optional full resume ----
    mode = "random" if args.random_reset else args.reset_mode

    def _fixed_prob(iteration: int) -> float:
        if args.curriculum_steps <= 0:
            return float(args.fixed_prob_end)
        steps = args.curriculum_offset_steps + iteration * B * T
        frac = min(max(steps / float(args.curriculum_steps), 0.0), 1.0)
        return float(args.fixed_prob_start + (args.fixed_prob_end - args.fixed_prob_start) * frac)

    start_iter = int(args.start_iter)
    if args.resume_jax:
        with open(args.resume_jax, "rb") as f:
            ck = pickle.load(f)
        if int(ck.get("B", args.B)) != B or int(ck.get("T", args.T)) != T:
            raise ValueError(
                f"resume B/T mismatch: checkpoint B={ck.get('B')} T={ck.get('T')}, "
                f"requested B={B} T={T}"
            )
        actor_state = actor_state.replace(
            params=ck["actor_params"],
            opt_state=ck["actor_opt_state"],
            step=ck["actor_step"],
        )
        critic_state = critic_state.replace(
            params=ck["critic_params"],
            opt_state=ck["critic_opt_state"],
            step=ck["critic_step"],
        )
        if "aux_params" in ck:
            aux_state = aux_state.replace(
                params=ck["aux_params"],
                opt_state=ck["aux_opt_state"],
                step=ck["aux_step"],
            )
            print("resumed auxiliary-head state", flush=True)
        else:
            print("initialised new auxiliary heads (checkpoint predates aux heads)", flush=True)
        key = ck["rng"]
        state = ck["env_state"]
        obs_stack = ck["obs_stack"]
        h = ck["h"].astype(compute_dtype)
        if "h_opp" in ck:
            h_opp = ck["h_opp"].astype(compute_dtype)
        elif args.league:
            # Old checkpoints have no frozen-policy history. At the common
            # cold-start case current and frozen best share the same weights,
            # so the current GRU state is the best available shadow.
            h_opp = jnp.copy(h)
            print("initialised h_opp from resumed current h", flush=True)
        else:
            h_opp = jnp.zeros_like(h, dtype=compute_dtype)
        if h.shape != (M, 256) or h_opp.shape != (M, 256):
            raise ValueError(
                f"resume recurrent-state shape mismatch: h={h.shape}, "
                f"h_opp={h_opp.shape}, expected {(M, 256)}"
            )
        # Old checkpoints predate rollout_lag; start with the current policy
        # in that case.  New checkpoints keep the exact stale snapshot.
        rollout_actor_params = ck.get("rollout_actor_params", actor_state.params)
        rollout_critic_params = ck.get("rollout_critic_params", critic_state.params)
        start_iter = int(ck["iteration"])
        print(
            f"resumed JAX checkpoint {args.resume_jax} at iteration {start_iter}",
            flush=True,
        )
    else:
        key = jax.random.PRNGKey(args.seed)
        key, rk = jax.random.split(key)
        reset_keys = jax.random.split(rk, B)
        obs, state = jax.vmap(lambda k: env.reset(k, mode=mode, fixed_prob=_fixed_prob(0)))(reset_keys)
        obs_stack = _stack_obs(obs, env.agents)
        h = jnp.zeros((M, 256), dtype=compute_dtype)
        h_opp = jnp.zeros((M, 256), dtype=compute_dtype)

    # Initialise the frozen league only after a possible checkpoint resume.
    # Otherwise a resumed run without an explicit opponent would silently use
    # the command-line cold-start actor as "best".
    if args.league:
        best_file_exists = Path(league_best_out).is_file()
        if best_file_exists:
            best_actor_params = load_actor_params(league_best_out)
            best_source = league_best_out
        elif args.league_opponent:
            best_actor_params = load_actor_params(args.league_opponent)
            best_source = args.league_opponent
        else:
            best_actor_params = _freeze_params(actor_state.params)
            best_source = "current checkpoint"
        if not best_file_exists:
            # Pin the initial anchor before the first chunk can restart. In
            # particular, resuming at iter 550 must not silently fall back to
            # the older 500 regression guard as the starting best.
            save_best_actor(
                league_best_out,
                best_actor_params,
                start_iter,
                {"initialized": 1.0},
                vars(args),
            )
            print(
                f"initialised frozen best from {best_source} -> {league_best_out}",
                flush=True,
            )
        base_actor_params = (
            load_actor_params(args.league_base)
            if args.league_base
            else best_actor_params
        )
        league_eval_cfg = replace(
            env.cfg,
            robot_foul_speed=1.0e9,
            wall_foul_speed=1.0e9,
            wall_push_time=1.0e9,
            body_push_time=1.0e9,
        )
        league_eval_env = RmEnv(league_eval_cfg)
        if args.league_eval_every > 0:
            league_eval = make_crossplay_evaluator(
                actor,
                league_eval_env,
                args.league_eval_envs,
                args.league_eval_steps,
            )
        print(
            f"league frozen best initialised from {best_source}; "
            f"base_guard={args.league_base or best_source}",
            flush=True,
        )

    rollout = make_rollout(
        env, actor, critic, cfg, reset_mode=mode, league=args.league
    )

    def _gae(traj, last_value, last_done):
        return compute_gae(
            traj["value"],
            traj["reward"],
            traj["done"],
            traj["terminated"],
            traj["final_value"],
            last_value,
            last_done,
            args.gamma,
            args.lam,
        )

    def _sub_batch(traj, adv, targets, idx):
        idx = jnp.asarray(idx)
        out = {}
        for k, v in traj.items():
            if k == "h0":
                out[k] = jnp.take(v, idx, axis=0)
                continue
            out[k] = jnp.take(v, idx, axis=1)
        return out, jnp.take(adv, idx, axis=1), jnp.take(targets, idx, axis=1)

    def _permute_traj(traj, order):
        out = {}
        for k, v in traj.items():
            if k == "h0":
                out[k] = jnp.take(v, order, axis=0)
            else:
                out[k] = jnp.take(v, order, axis=1)
        return out

    def _slice_batch(traj, adv, targets, start, mb):
        out = {}
        for k, v in traj.items():
            if k == "h0":
                out[k] = v[start:start + mb]
            else:
                out[k] = v[:, start:start + mb]
        return out, adv[:, start:start + mb], targets[:, start:start + mb]

    def _minibatch_epoch(traj, adv, targets, order, mb):
        """Shuffle once and return [num_mb, T, mb, ...] pytrees."""
        B_ = order.shape[0]
        n_mb = B_ // mb
        out = {}
        for k, v in traj.items():
            if k == "h0":
                continue
            vs = jnp.take(v, order, axis=1)
            vs = vs.reshape((vs.shape[0], n_mb, mb) + vs.shape[2:])
            out[k] = jnp.swapaxes(vs, 0, 1)
        h0 = jnp.take(traj["h0"], order, axis=0)
        out["h0"] = h0.reshape((n_mb, mb * N, 256))

        def split_adv(x):
            x = jnp.take(x, order, axis=1)
            x = x.reshape((x.shape[0], n_mb, mb) + x.shape[2:])
            return jnp.swapaxes(x, 0, 1)

        return out, split_adv(adv), split_adv(targets)

    aux_weights = (
        args.aux_track_weight,
        args.aux_damage_weight,
        args.aux_heat_weight,
        args.aux_rules_weight,
        args.aux_self_weight,
    )

    def update_scan(actor_state, critic_state, aux_state, minibatches, aux_coef):
        def body(carry, batch):
            a, c, u = carry
            a, c, u, (loss, aux) = update(a, c, u, batch, aux_coef)
            return (a, c, u), (loss, aux)
        (a, c, u), (losses, auxes) = jax.lax.scan(
            body, (actor_state, critic_state, aux_state), minibatches
        )
        return a, c, u, losses, auxes

    _donate = () if (args.profile_dir or args.rollout_lag) else (0, 1, 2)
    update_scan_jit = jax.jit(update_scan, donate_argnums=_donate)

    def update_all(actor_state, critic_state, aux_state, traj, adv, targets, base_key, aux_coef):
        def epoch_body(carry, key):
            a, c, u = carry
            order = jax.random.permutation(key, B)
            minibatches = _minibatch_epoch(traj, adv, targets, order, mb)
            def body(carry2, batch):
                a2, c2, u2 = carry2
                a2, c2, u2, (loss, aux) = update(a2, c2, u2, batch, aux_coef)
                return (a2, c2, u2), (loss, aux)

            (a, c, u), (losses, auxes) = jax.lax.scan(
                body, (a, c, u), minibatches
            )
            mean_aux = tuple(jnp.mean(x) for x in auxes)
            return (a, c, u), (jnp.mean(losses), mean_aux)

        keys = jax.random.split(base_key, args.epochs)
        (a, c, u), (losses, auxes) = jax.lax.scan(
            epoch_body, (actor_state, critic_state, aux_state), keys
        )
        last_aux = tuple(x[-1] for x in auxes)
        return a, c, u, losses[-1], last_aux

    update_all_jit = jax.jit(update_all, donate_argnums=_donate)

    def _actor_seq(params, obs, actions, tm, fm, h0, done):
        # Encode all timesteps in parallel; scan only the GRU, then run the
        # stateless MLP/head over all timesteps in parallel as well.
        T_, M_ = obs.shape[:2]
        obs_flat = obs.reshape((T_ * M_, -1))
        enc_flat = actor.apply(
            {"params": params}, obs_flat, method=RmActor.encode
        )
        enc = enc_flat.reshape((T_, M_, -1))

        def body(carry, x):
            e, d = x
            h, carry = actor.apply(
                {"params": params}, e, carry, method=RmActor.gru_step
            )
            # Match rollout auto-reset: the terminal action still uses the
            # pre-reset hidden state, but the next episode starts from zero.
            carry = jnp.where(d[:, None], jnp.zeros_like(carry), carry)
            return carry, h

        _, hs = jax.lax.scan(body, h0, (enc, done))
        raw = actor.apply(
            {"params": params},
            hs.reshape((T_ * M_, -1)),
            enc_flat,
            method=RmActor.head,
        )
        lp, ent = evaluate_raw(
            raw,
            actions.reshape((T_ * M_, 6)),
            tm.reshape((T_ * M_, 4)),
            fm.reshape((T_ * M_, 2)),
        )
        return (
            lp.reshape((T_, M_)),
            ent.reshape((T_, M_)),
            enc_flat,
            hs.reshape((T_ * M_, -1)),
        )

    def _critic_seq(params, g):
        return jax.vmap(lambda x: critic.apply({"params": params}, x))(g)

    if args.remat:
        _actor_seq = jax.checkpoint(_actor_seq)
        _critic_seq = jax.checkpoint(_critic_seq)

    def _actor_loss(params, aux_params, traj, adv, aux_coef):
        T_, B_ = traj["obs"].shape[:2]
        obs = traj["obs"].reshape((T_, B_ * N, -1))
        act = traj["actions"].reshape((T_, B_ * N, 6))
        tm = traj["target_mask"].reshape((T_, B_ * N, 4))
        fm = traj["fire_mask"].reshape((T_, B_ * N, 2))
        done_flat = (traj["done"][..., None] * jnp.ones((1, 1, N), dtype=jnp.float32))
        done_flat = done_flat.reshape((T_, B_ * N))
        logp, ent, encoded, hidden = _actor_seq(
            params, obs, act, tm, fm,
            traj["h0"].reshape((B_ * N, 256)), done_flat,
        )
        logp = logp.reshape((T_, B_, N))
        ent = ent.reshape((T_, B_, N))
        valid = (1.0 - jnp.cumsum(traj["done"], axis=0) + traj["done"]).clip(0, 1)
        mask = valid[..., None] * traj["actor_train_mask"]
        ratio = jnp.exp(logp - traj["logp"])
        # Align with PyTorch: per-agent, unbiased std over the minibatch;
        # terminal/padding masking is applied to the loss, not to the
        # advantage statistics.
        a = _normalize_advantage(adv, mask)
        l1 = ratio * a
        l2 = jnp.clip(ratio, 1.0 - args.clip, 1.0 + args.clip) * a
        actor_loss = -jnp.sum(jnp.minimum(l1, l2) * mask) / (jnp.sum(mask) + 1e-8)
        actor_loss = actor_loss - args.ent_coef * jnp.sum(ent * mask) / (
            jnp.sum(mask) + 1e-8
        )
        approx_kl = jnp.sum((ratio - 1.0 - (logp - traj["logp"])) * mask) / (
            jnp.sum(mask) + 1e-8
        )
        if args.aux_heads:
            aux_out = aux_model.apply(
                {"params": aux_params},
                hidden,
                encoded,
                jax.lax.stop_gradient(act.reshape((T_ * B_ * N, 6))),
                obs.reshape((T_ * B_ * N, -1)),
            )
            aux_out = jax.tree.map(
                lambda x: x.reshape((T_, B_, N) + x.shape[1:]),
                aux_out,
            )
            aux_targets = {k: v for k, v in traj.items() if k.startswith("aux_")}
            aux_total, aux_metrics = auxiliary_loss(
                aux_out, aux_targets, aux_weights, traj["actor_train_mask"]
            )
            total_actor_loss = actor_loss + aux_coef * aux_total
        else:
            aux_total = jnp.asarray(0.0)
            aux_metrics = tuple(jnp.asarray(0.0) for _ in AUX_METRIC_NAMES)
            total_actor_loss = actor_loss
        return total_actor_loss, (
            actor_loss,
            approx_kl,
            jnp.sum(ent * mask) / (jnp.sum(mask) + 1e-8),
            aux_total,
            *aux_metrics,
        )

    def _critic_loss(params, traj, targets):
        val = _critic_seq(params, traj["global_state"])
        vclip = traj["value"] + jnp.clip(val - traj["value"], -args.clip, args.clip)
        vl = jnp.maximum((val - targets) ** 2, (vclip - targets) ** 2)
        T_, B_ = traj["obs"].shape[:2]
        valid = (1.0 - jnp.cumsum(traj["done"], axis=0) + traj["done"]).clip(0, 1)
        mask = valid[..., None] * jnp.ones((1, 1, N))
        value_loss = 0.5 * jnp.sum(vl * mask) / (jnp.sum(mask) + 1e-8)
        return value_loss, value_loss

    def update(actor_state, critic_state, aux_state, batch, aux_coef):
        sub, sub_adv, sub_targets = batch
        (al, (al_raw, akl, ent, aux_total, *aux_metrics)), (grads_a, grads_u) = jax.value_and_grad(
            _actor_loss, argnums=(0, 1), has_aux=True
        )(
            actor_state.params,
            aux_state.params,
            sub,
            sub_adv,
            aux_coef,
        )
        actor_state = actor_state.apply_gradients(grads=grads_a)
        aux_state = aux_state.apply_gradients(grads=grads_u)
        (vl, vl_raw), grads_c = jax.value_and_grad(
            _critic_loss, has_aux=True
        )(critic_state.params, sub, sub_targets)
        critic_state = critic_state.apply_gradients(grads=grads_c)
        aux_log = (al_raw, vl_raw, al_raw, akl, ent, aux_total, *aux_metrics)
        return actor_state, critic_state, aux_state, (al + vl, aux_log)

    @jax.jit
    def rollout_jit(ap, bp, cp, st, ob, hh, hh_opp, k, fp, lm, tt):
        return rollout(ap, bp, cp, st, ob, hh, hh_opp, k, fp, lm, tt)

    update_jit = jax.jit(
        update,
        donate_argnums=() if (args.profile_dir or args.rollout_lag) else (0, 1, 2),
    )

    run_t0 = time.time()
    bar = tqdm(range(start_iter, args.updates),
               initial=start_iter, total=args.updates,
               desc=f"RM-JAX B={B} T={T}", dynamic_ncols=True)
    for it in bar:
        iter_t0 = time.time()
        _profile_this = bool(args.profile_dir) and it == start_iter + args.profile_at
        _trace_rollout = _profile_this and args.profile_stage in ("all", "rollout")
        _trace_update = _profile_this and args.profile_stage in ("all", "update")
        if _trace_rollout:
            _trace_dir = os.path.join(args.profile_dir, "rollout") if args.profile_stage == "all" else args.profile_dir
            os.makedirs(_trace_dir, exist_ok=True)
            print(f"[profile] tracing rollout iteration {it + 1} -> {_trace_dir}", flush=True)
            jax.profiler.start_trace(_trace_dir, create_perfetto_trace=True)
        key, rk = jax.random.split(key)
        # Keep the environment/recurrent state across PPO updates.  Finished
        # environments are auto-reset inside make_rollout; resetting all B
        # environments here would turn the game into repeated 20 s episodes.
        t_reset = 0.0
        t0 = time.perf_counter()
        fixed_prob = _fixed_prob(it)
        aux_coef = 0.0
        if args.aux_heads:
            aux_warmup_start = (
                start_iter
                if args.aux_warmup_start_iter < 0
                else args.aux_warmup_start_iter
            )
            ramp = min(
                1.0,
                max(0.0, (it - aux_warmup_start + 1) / max(1, args.aux_warmup_iters)),
            )
            aux_coef = args.aux_coef * ramp
        # With lag=1, rollout uses the policy from before the previous update.
        # Without lag, this is exactly the current training policy.
        use_rollout_params = args.rollout_lag != 0
        r_actor_params = rollout_actor_params if use_rollout_params else actor_state.params
        r_critic_params = rollout_critic_params if use_rollout_params else critic_state.params
        prev_actor_params = actor_state.params
        prev_critic_params = critic_state.params
        state, obs_stack, h, h_opp, traj, last_value, last_done = rollout_jit(
            r_actor_params,
            best_actor_params,
            r_critic_params,
            state,
            obs_stack,
            h,
            h_opp,
            key,
            fixed_prob,
            league_mask,
            league_teams,
        )
        jax.block_until_ready((state, h_opp, traj, last_value, last_done))
        if _trace_rollout:
            jax.profiler.stop_trace()
            print("[profile] rollout trace stopped", flush=True)
        t_rollout = time.perf_counter() - t0
        t0 = time.perf_counter()
        adv, targets = _gae(traj, last_value, last_done)
        jax.block_until_ready((adv, targets))
        t_gae = time.perf_counter() - t0
        t_aux_data = 0.0
        if args.aux_heads:
            t0 = time.perf_counter()
            traj.update(build_aux_targets(env, traj))
            jax.block_until_ready(traj)
            t_aux_data = time.perf_counter() - t0
        mb = args.minibatch_envs if args.minibatch_envs > 0 else B
        if B % mb != 0:
            raise ValueError(f"minibatch-envs={mb} must divide B={B}")
        rng_np = np.random.default_rng(args.seed + it)
        last_loss_arr = jnp.asarray(0.0)
        last_aux_arr = tuple(
            jnp.asarray(0.0) for _ in range(6 + len(AUX_METRIC_NAMES))
        )
        n_minibatch = 0
        t_data = 0.0
        t_launch = 0.0
        t_sync = 0.0
        if _trace_update:
            _trace_dir = os.path.join(args.profile_dir, "update") if args.profile_stage == "all" else args.profile_dir
            os.makedirs(_trace_dir, exist_ok=True)
            print(f"[profile] tracing update iteration {it + 1} -> {_trace_dir}", flush=True)
            jax.profiler.start_trace(_trace_dir, create_perfetto_trace=True)
        if args.scan_all:
            t0 = time.perf_counter()
            actor_state, critic_state, aux_state, loss, aux = update_all_jit(
                actor_state, critic_state, aux_state, traj, adv, targets, key, aux_coef
            )
            t_launch += time.perf_counter() - t0
            last_loss_arr = loss
            last_aux_arr = aux
            n_minibatch += args.epochs * (B // mb)
            t0 = time.perf_counter()
            jax.block_until_ready(
                (actor_state.params, critic_state.params, aux_state.params, loss, aux)
            )
            t_sync += time.perf_counter() - t0
        elif args.scan:
            for ep in range(args.epochs):
                order = jnp.asarray(rng_np.permutation(B))
                t0 = time.perf_counter()
                minibatches = _minibatch_epoch(traj, adv, targets, order, mb)
                t_data += time.perf_counter() - t0
                t0 = time.perf_counter()
                actor_state, critic_state, aux_state, losses, auxes = update_scan_jit(
                    actor_state, critic_state, aux_state, minibatches, aux_coef
                )
                t_launch += time.perf_counter() - t0
                t0 = time.perf_counter()
                jax.block_until_ready(
                    (actor_state.params, critic_state.params, aux_state.params, losses, auxes)
                )
                t_sync += time.perf_counter() - t0
                n_minibatch += B // mb
                last_loss_arr = losses[-1]
                last_aux_arr = tuple(x[-1] for x in auxes)
        else:
            for ep in range(args.epochs):
                order = rng_np.permutation(B)
                for start in range(0, B, mb):
                    idx = order[start:start + mb]
                    t0 = time.perf_counter()
                    sub, sub_adv, sub_targets = _sub_batch(traj, adv, targets, idx)
                    t_data += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    actor_state, critic_state, aux_state, (loss, aux) = update_jit(
                        actor_state, critic_state, aux_state,
                        (sub, sub_adv, sub_targets), aux_coef,
                    )
                    t_launch += time.perf_counter() - t0
                    n_minibatch += 1
                    if n_minibatch % max(1, args.sync_every) == 0:
                        t0 = time.perf_counter()
                        jax.block_until_ready(
                            (actor_state.params, critic_state.params, aux_state.params)
                        )
                        t_sync += time.perf_counter() - t0
                    last_loss_arr = loss
                    last_aux_arr = aux
        jax.block_until_ready(
            (
                actor_state.params,
                critic_state.params,
                aux_state.params,
                last_loss_arr,
                *last_aux_arr,
            )
        )
        if args.rollout_lag:
            # The collection just consumed prev_*; the trainer has now moved
            # to the new parameters.  The next rollout must stay one update
            # behind, so keep the pre-update snapshot.
            rollout_actor_params = prev_actor_params
            rollout_critic_params = prev_critic_params
        if _trace_update:
            jax.profiler.stop_trace()
            print("[profile] update trace stopped", flush=True)
        last_loss = float(last_loss_arr)
        last_aux = tuple(float(x) for x in last_aux_arr)
        if args.fail_on_nan and not all(
            math.isfinite(x) for x in (last_loss, *last_aux)
        ):
            raise RuntimeError(
                f"non-finite PPO loss at iteration {it + 1}: "
                f"loss={last_loss} aux={last_aux}"
            )
        ret = float(jnp.mean(jnp.sum(traj["reward"], axis=0)))
        stats = _episode_stats(traj)
        iter_time = time.time() - iter_t0
        bar.set_postfix(loss=f"{last_loss:.3f}", ret=f"{ret:.2f}",
                        done=stats["done_count"],
                        score=f"{stats['score_red']:.1f}/{stats['score_blue']:.1f}",
                        refresh=False)
        print(
            f"iter_time={iter_time:.2f}s "
            f"update {it + 1}/{args.updates} loss={last_loss:.4f} "
            f"actor={last_aux[0]:.4f} value={last_aux[1]:.4f} "
            f"ent={last_aux[4]:.3f} aux={last_aux[5]:.3f} ret={ret:.2f} "
            f"reward_mean={float(jnp.mean(traj['reward'])):.5f} "
            f"done={stats['done_count']} score={stats['score_red']:.1f}/{stats['score_blue']:.1f}",
            flush=True,
        )
        if (it + 1) % max(1, args.log_every) == 0:
            print(f"[RM] outcomes: {_outcome_text(stats['outcome_counts'])}", flush=True)
        if args.timing:
            print(
                f"timing reset={t_reset:.2f}s rollout={t_rollout:.2f}s "
                f"gae={t_gae:.2f}s aux_data={t_aux_data:.2f}s "
                f"data={t_data:.2f}s launch={t_launch:.2f}s "
                f"sync={t_sync:.2f}s total={time.time()-iter_t0:.2f}s",
                flush=True,
            )
        league_metrics = {}
        t_league_eval = 0.0
        if (
            args.league
            and league_eval is not None
            and (it + 1) % max(1, args.league_eval_every) == 0
        ):
            t_eval0 = time.perf_counter()
            eval_seed = (
                args.league_eval_seed
                if args.league_eval_seed >= 0
                else args.seed + 100003
            )
            eval_key = jax.random.PRNGKey(eval_seed)
            eval_best = summarize_eval(
                league_eval(actor_state.params, best_actor_params, eval_key)
            )
            promote = (
                eval_best["win_rate"] >= args.league_promote_win_rate
                and eval_best["mean_score_diff"] >= args.league_promote_min_score_diff
            )
            eval_base = None
            if (
                promote
                and args.league_base
                and str(Path(args.league_base))
                != str(Path(args.league_opponent or league_best_out))
            ):
                eval_base = summarize_eval(
                    league_eval(
                        actor_state.params,
                        base_actor_params,
                        eval_key,
                    )
                )
            if (
                eval_base is not None
                and eval_base["win_rate"] < args.league_base_min_win_rate
            ):
                promote = False
            league_metrics.update(
                {
                    "league/opponent_envs_current_red": int(
                        jnp.sum(league_mask & (league_teams == 0))
                    ),
                    "league/opponent_envs_current_blue": int(
                        jnp.sum(league_mask & (league_teams == 1))
                    ),
                    "league/self_play_frac": float(args.league_self_play_frac),
                    "league/eval_seed": float(eval_seed),
                    "league/vs_best_win_rate": eval_best["win_rate"],
                    "league/vs_best_draw_rate": eval_best["draw_rate"],
                    "league/vs_best_win_or_draw_rate": eval_best["win_or_draw_rate"],
                    "league/vs_best_mean_score_diff": eval_best["mean_score_diff"],
                    "league/vs_best_mean_steps": eval_best["mean_steps"],
                    "league/vs_best_finished_rate": eval_best["finished_rate"],
                    "league/vs_best_as_red_win_rate": eval_best["as_red_win_rate"],
                    "league/vs_best_as_blue_win_rate": eval_best["as_blue_win_rate"],
                    "league/vs_best_as_red_score_diff": eval_best[
                        "as_red_mean_score_diff"
                    ],
                    "league/vs_best_as_blue_score_diff": eval_best[
                        "as_blue_mean_score_diff"
                    ],
                    "league/promoted": int(promote),
                }
            )
            for code, count in eval_best["outcome_counts"].items():
                league_metrics[f"league/vs_best_outcome_{code}"] = int(count)
            if eval_base is not None:
                league_metrics.update(
                    {
                        "league/vs_base_win_rate": eval_base["win_rate"],
                        "league/vs_base_mean_score_diff": eval_base["mean_score_diff"],
                    }
                )
            payload = {
                "iteration": it + 1,
                "eval_seed": int(eval_seed),
                "current_red_opponent_envs": int(
                    jnp.sum(league_mask & (league_teams == 0))
                ),
                "current_blue_opponent_envs": int(
                    jnp.sum(league_mask & (league_teams == 1))
                ),
                "vs_best": eval_best,
                "vs_base": eval_base,
                "promoted": bool(promote),
                "best_checkpoint": league_best_out,
            }
            if promote:
                save_best_actor(
                    league_best_out,
                    actor_state.params,
                    it + 1,
                    league_metrics,
                    vars(args),
                )
                best_actor_params = _freeze_params(actor_state.params)
                # When best changes, some league envs are still mid-episode.
                # Seed the new frozen policy's recurrent shadow from the
                # current policy's latest hidden state instead of carrying a
                # GRU state produced by the previous best.
                h_opp = jnp.where(
                    jnp.repeat(league_mask, N)[:, None], h, h_opp
                )
                print(
                    f"[league] promoted iteration {it + 1} -> {league_best_out} "
                    f"win_rate={eval_best['win_rate']:.3f} "
                    f"score_diff={eval_best['mean_score_diff']:.2f}",
                    flush=True,
                )
            else:
                print(
                    f"[league] keep best; win_rate={eval_best['win_rate']:.3f} "
                    f"score_diff={eval_best['mean_score_diff']:.2f}",
                    flush=True,
                )
            write_eval_json(league_eval_json, payload)
            t_league_eval = time.perf_counter() - t_eval0
        if logger is not None and (it + 1) % max(1, args.log_every) == 0:
            metrics = {
                "iteration": it + 1,
                "frames": (it + 1) * B * T,
                "train/curriculum/fixed_prob": fixed_prob,
                "train/agents/total_loss": last_loss,
                "train/agents/loss_objective": last_aux[0],
                "train/agents/loss_critic": last_aux[1],
                "train/agents/loss_entropy": -last_aux[4],
                "train/agents/entropy": last_aux[4],
                "train/agents/kl_approx": last_aux[3],
                "train/agents/aux_total": last_aux[5],
                "train/agents/aux_coef_effective": aux_coef,
                "train/termination/terminated_agent_fraction": float(
                    jnp.mean(traj["terminated"])
                ),
                "train/termination/truncated_agent_fraction": float(
                    jnp.mean(traj["truncated"])
                ),
                "train/termination/truncated_agent_count": float(
                    jnp.sum(traj["truncated"])
                ),
                "train/termination/terminated_agent_count": float(
                    jnp.sum(traj["terminated"])
                ),
                "collection/agents/reward/step_reward_mean": ret,
                "collection/agents/reward/robot_reward_mean": float(jnp.mean(traj["reward"])),
                "collection/agents/reward/episode_reward_mean": stats["episode_return_mean"],
                "collection/agents/reward/episode_reward_min": stats["episode_return_min"],
                "collection/agents/reward/episode_reward_max": stats["episode_return_max"],
                "collection/agents/reward/episode_len_mean": stats["episode_len_mean"],
                "collection/agents/info/score_red": stats["score_red"],
                "collection/agents/info/score_blue": stats["score_blue"],
                "collection/agents/info/last_score_red": stats["last_score_red"],
                "collection/agents/info/last_score_blue": stats["last_score_blue"],
                "collection/agents/info/time_left": stats["time_left"],
                "collection/agents/info/done_count": stats["done_count"],
                "timers/collection_time": t_rollout,
                "timers/gae_time": t_gae,
                "timers/aux_data_time": t_aux_data,
                "timers/league_eval_time": t_league_eval,
                "timers/training_time": t_data + t_launch + t_sync,
                "timers/total_time": time.time() - iter_t0,
            }
            metrics.update(league_metrics)
            for name, value in zip(AUX_METRIC_NAMES, last_aux[6:]):
                metrics[f"train/{name}"] = value
            for code, count in stats["outcome_counts"].items():
                metrics[f"collection/agents/info/outcome_{code}"] = count
            logger.log((it + 1) * B * T, metrics)
        if args.live_view and (it + 1) % max(1, args.live_every) == 0:
            write_rm_live(traj, args.live_dir, it + 1)
        if args.save_dir and ((it + 1) % args.save_every == 0 or it + 1 == args.updates):
            sd = Path(args.save_dir)
            sd.mkdir(parents=True, exist_ok=True)
            path = sd / f"jax_ckpt_{it + 1:06d}.pkl"
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(
                    {
                        "format": "rm_jax_full_v3_league",
                        "B": B,
                        "T": T,
                        "iteration": it + 1,
                        "actor_params": actor_state.params,
                        "rollout_actor_params": rollout_actor_params,
                        "rollout_critic_params": rollout_critic_params,
                        "actor_opt_state": actor_state.opt_state,
                        "actor_step": actor_state.step,
                        "critic_params": critic_state.params,
                        "critic_opt_state": critic_state.opt_state,
                        "critic_step": critic_state.step,
                        "aux_params": aux_state.params,
                        "aux_opt_state": aux_state.opt_state,
                        "aux_step": aux_state.step,
                        "rng": key,
                        "env_state": state,
                        "obs_stack": obs_stack,
                        "h": h,
                        "h_opp": h_opp,
                        "league_enabled": bool(args.league),
                        "league_best_path": league_best_out or None,
                        "args": vars(args),
                    },
                    f,
                )
            tmp_path.replace(path)
            print(f"saved {path}", flush=True)
    print(f"done in {time.time() - run_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
