"""Auxiliary world-understanding heads for the RM JAX actor.

The heads are training-only and live outside the deployable actor parameter
tree. They are conditioned on the actor's recurrent hidden state, the current
encoder output, the executed action, and the current local observation.

The supervision comes from privileged simulator state, but no privileged value
is fed into the actor. This makes the heads a representation-learning aid rather
than a deployment-time information leak.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from .config import RmConfig

N_AGENTS = 6
N_ENEMIES = 3
AGENT_DIM = 30
GLOBAL_OFFSET = N_AGENTS * AGENT_DIM

TRACK_PER_ENEMY = 4  # dx, dy, log_sigma, visible_next
TRACK_DIM = N_ENEMIES * TRACK_PER_ENEMY
DAMAGE_DIM = 4  # fire, conditional hit probability, damage, kill
HEAT_PER_ENEMY = 3  # heat margin, heat delta, overheated
HEAT_DIM = N_ENEMIES * HEAT_PER_ENEMY
RULES_DIM = 7
SELF_DIM = 14
AUX_OUTPUT_DIM = TRACK_DIM + DAMAGE_DIM + HEAT_DIM + RULES_DIM + SELF_DIM
ENEMY_OBS_START = 24 + 4 + 4 + 2 * 18
ENEMY_OBS_DIM = 17
MAX_TRACK_AGE_STEPS = 15  # 1.5 s at dt=0.1


class RmAuxHeads(nn.Module):
    """Small shared trunk with five semantic prediction heads."""

    hidden_size: int = 256
    dtype: Any = None

    @nn.compact
    def __call__(self, hidden, encoded, action, obs):
        # ``action`` is stopped by the caller. The direct observation term makes
        # legality/visibility boundaries easier to learn without changing actor
        # inference or requiring a new recurrent pass.
        x = jnp.concatenate([hidden, encoded, action, obs], axis=-1)
        x = nn.Dense(self.hidden_size, dtype=self.dtype)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
        x = nn.Dense(self.hidden_size, dtype=self.dtype)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)

        return {
            "track": nn.Dense(TRACK_DIM, name="track")(x),
            "damage": nn.Dense(DAMAGE_DIM, name="damage")(x),
            "heat": nn.Dense(HEAT_DIM, name="heat")(x),
            "rules": nn.Dense(RULES_DIM, name="rules")(x),
            "self": nn.Dense(SELF_DIM, name="self")(x),
        }


def _agent_tensor(global_state):
    shape = global_state.shape[:-1]
    return global_state[..., :GLOBAL_OFFSET].reshape(shape + (N_AGENTS, AGENT_DIM))


def _take_team(values, team):
    """values [...,2], team [N] -> values selected per agent [...,N]."""
    return jnp.take(values, team, axis=-1)


def _shift_forward(x, fill):
    """Shift the time axis so out[t] = x[t+1]; last row is ``fill``."""
    return jnp.concatenate([x[1:], jnp.full_like(x[-1:], fill)], axis=0)


def _point_in_rect(points, poly):
    """JAX/local copy of the fixed-rectangle spawn test from rm.env."""
    x, y = points[..., 0], points[..., 1]
    inside = jnp.zeros_like(x, dtype=jnp.bool_)
    for k in range(poly.shape[0]):
        x1, y1 = poly[k, 0], poly[k, 1]
        x2, y2 = poly[(k + 1) % poly.shape[0], 0], poly[(k + 1) % poly.shape[0], 1]
        crosses = (y1 > y) != (y2 > y)
        denom = jnp.where(jnp.abs(y2 - y1) < 1e-9, 1e-9, y2 - y1)
        x_int = (x2 - x1) * (y - y1) / denom + x1
        inside = inside ^ (crosses & (x < x_int))
    return inside


def build_aux_targets(
    env,
    traj: Mapping[str, jax.Array],
) -> dict[str, jax.Array]:
    """Derive compact privileged labels from a real rollout.

    The function never modifies simulator state. ``next_global_state`` is the
    raw post-step state captured before the trainer's auto-reset, so terminal
    transitions remain valid labels.
    """
    cfg: RmConfig = env.cfg
    g = traj["global_state"]
    gn = traj["next_global_state"]
    agents = _agent_tensor(g)
    agents_next = _agent_tensor(gn)
    glob = g[..., GLOBAL_OFFSET:]
    glob_next = gn[..., GLOBAL_OFFSET:]
    target_mask = traj["target_mask"]
    done = traj["done"]

    team = jnp.asarray([cfg.team(i) for i in range(N_AGENTS)], dtype=jnp.int32)
    sign = jnp.where(team == 0, 1.0, -1.0)
    enemy_idx = jnp.asarray(
        [[cfg.enemy_indices(i)[k] for k in range(N_ENEMIES)] for i in range(N_AGENTS)],
        dtype=jnp.int32,
    )

    # Relative positions in team-canonical world axes. The current enemy
    # observation uses the same convention, unlike an ego-yaw frame.
    current_pos = agents[..., :2] * jnp.asarray(
        [cfg.field_hx, cfg.field_hy], dtype=jnp.float32
    )
    next_pos = agents_next[..., :2] * jnp.asarray(
        [cfg.field_hx, cfg.field_hy], dtype=jnp.float32
    )
    enemy_pos = jnp.take(current_pos, enemy_idx, axis=2)
    enemy_pos_next = jnp.take(next_pos, enemy_idx, axis=2)
    rel = sign[None, None, :, None, None] * (
        enemy_pos - current_pos[:, :, :, None, :]
    )
    rel_next = sign[None, None, :, None, None] * (
        enemy_pos_next - next_pos[:, :, :, None, :]
    )
    rel_scale = max(2.0 * cfg.v_max * cfg.dt, 1e-6)
    rel = rel / rel_scale
    rel_next = rel_next / rel_scale

    # Actor-visible semantics, not target legality. In particular a visible
    # corpse is still an object to track, but cannot appear in target_mask.
    enemy_slots = jnp.asarray(
        [ENEMY_OBS_START + ENEMY_OBS_DIM * k for k in range(N_ENEMIES)],
        dtype=jnp.int32,
    )
    visible = jnp.take(traj["obs"], enemy_slots, axis=-1).astype(jnp.float32)
    enemy_alive = jnp.take(agents[..., 10], enemy_idx, axis=2)
    enemy_alive_next = jnp.take(agents_next[..., 10], enemy_idx, axis=2)
    t_size, b_size = done.shape
    reset = jnp.concatenate(
        [jnp.zeros((1, b_size), dtype=jnp.bool_), done[:-1] > 0.5], axis=0
    )

    n_slots = N_AGENTS * N_ENEMIES
    flat_rel = rel.reshape(t_size, b_size, n_slots, 2)
    flat_next = rel_next.reshape(t_size, b_size, n_slots, 2)
    flat_vis = visible.reshape(t_size, b_size, n_slots)
    flat_alive = enemy_alive.reshape(t_size, b_size, n_slots)

    def track_body(carry, xs):
        anchor, has_seen, age = carry
        reset_t, vis_t, rel_t, next_t, alive_t = xs
        # A reset starts a new episode; never carry an enemy anchor across it.
        anchor = jnp.where(reset_t[:, None, None], jnp.zeros_like(anchor), anchor)
        has_seen = jnp.where(reset_t[:, None], jnp.zeros_like(has_seen), has_seen)
        age = jnp.where(reset_t[:, None], jnp.zeros_like(age), age)

        vis_b = vis_t[..., None]
        base = jnp.where(vis_b, rel_t, anchor)
        available = has_seen | (vis_t > 0.5)
        # Corpses remain physical objects and respawn at their current corpse
        # position, so alive does not gate permanence. Position uncertainty
        # fades smoothly with occlusion age instead of disappearing abruptly.
        hidden = available & (vis_t <= 0.5)
        age_frac = jnp.clip(
            age.astype(jnp.float32) / max(MAX_TRACK_AGE_STEPS, 1), 0.0, 1.0
        )
        age_weight = 0.5 * (1.0 + jnp.cos(jnp.pi * age_frac))
        weight = jnp.where(available, jnp.where(vis_t > 0.5, 1.0, age_weight), 0.0)
        delta = next_t - base

        new_anchor = jnp.where(vis_b, rel_t, anchor)
        new_seen = has_seen | (vis_t > 0.5)
        new_age = jnp.where(vis_t > 0.5, 0, age + 1)
        out = (delta, weight, hidden, available, base, new_seen, age)
        return (new_anchor, new_seen, new_age), out

    xs = (reset, flat_vis, flat_rel, flat_next, flat_alive)
    anchor0 = jnp.zeros((b_size, n_slots, 2), dtype=flat_rel.dtype)
    seen0 = jnp.zeros((b_size, n_slots), dtype=jnp.bool_)
    age0 = jnp.zeros((b_size, n_slots), dtype=jnp.int32)
    _, (track_delta, track_weight, track_hidden, track_seen, track_base, seen_out, track_age) = (
        jax.lax.scan(track_body, (anchor0, seen0, age0), xs)
    )
    track_delta = track_delta.reshape(t_size, b_size, N_AGENTS, N_ENEMIES, 2)
    track_weight = track_weight.reshape(t_size, b_size, N_AGENTS, N_ENEMIES)
    track_hidden = track_hidden.reshape(t_size, b_size, N_AGENTS, N_ENEMIES)
    track_seen = track_seen.reshape(t_size, b_size, N_AGENTS, N_ENEMIES)
    track_base = track_base.reshape(t_size, b_size, N_AGENTS, N_ENEMIES, 2)
    seen_out = seen_out.reshape(t_size, b_size, N_AGENTS, N_ENEMIES)
    track_age = track_age.reshape(t_size, b_size, N_AGENTS, N_ENEMIES)
    track_visible = track_seen & ~track_hidden

    visible_next = _shift_forward(visible, 0.0)
    has_next = jnp.concatenate(
        [jnp.ones((max(t_size - 1, 0), b_size), dtype=jnp.bool_),
         jnp.zeros((1, b_size), dtype=jnp.bool_)],
        axis=0,
    )
    visible_next_valid = has_next & (done <= 0.5)

    # Enemy heat: predict the margin to overheat and the next one-step change.
    enemy_heat = jnp.take(agents[..., 7], enemy_idx, axis=2)
    enemy_heat_next = jnp.take(agents_next[..., 7], enemy_idx, axis=2)
    enemy_overheated = jnp.take(agents_next[..., 8], enemy_idx, axis=2)
    heat_valid = (enemy_alive > 0.5) & seen_out
    heat_target = jnp.stack(
        [1.0 - enemy_heat_next, enemy_heat_next - enemy_heat], axis=-1
    )

    # Damage heads target the action-selected opponent. ``p_hit`` is the
    # post-step tracking probability multiplied by the target's incoming-spin
    # reduction, not a sampled Bernoulli outcome.
    target_present = traj["aux_target_present"] > 0.5
    own_alive = agents[..., 10] > 0.5
    damage_target = jnp.stack(
        [
            traj["aux_fire"].astype(jnp.float32),
            traj["aux_p_hit"],
            traj["aux_damage"] / max(cfg.damage_per_frame, 1e-6),
            traj["aux_kill"].astype(jnp.float32),
        ],
        axis=-1,
    )

    # Rule head: occupancy and decomposed scoring. Center and kill deltas use
    # intuitive per-step scales so their gradients are not suppressed by 90.
    score_base = glob[..., 0:2]
    score_next = glob_next[..., 0:2]
    occ_next = jnp.stack([glob_next[..., 3], glob_next[..., 4]], axis=-1)
    score_own = _take_team(score_base, team)
    score_opp = _take_team(score_base[:, :, ::-1], team)
    score_own_next = _take_team(score_next, team)
    score_opp_next = _take_team(score_next[:, :, ::-1], team)
    occ_own = _take_team(occ_next, team)
    occ_opp = _take_team(occ_next[:, :, ::-1], team)
    score_delta_own = (score_own_next - score_own) * (cfg.target_score / 10.0)
    score_delta_opp = (score_opp_next - score_opp) * (cfg.target_score / 10.0)
    kill_score = traj["aux_kill_score"] / 10.0
    kill_own = _take_team(kill_score, team)
    kill_opp = _take_team(kill_score[:, :, ::-1], team)
    center_own = (score_delta_own - kill_own) / 0.03
    center_opp = (score_delta_opp - kill_opp) / 0.03
    done_per_agent = jnp.broadcast_to(done[..., None], occ_own.shape)
    rules_target = jnp.stack(
        [
            occ_own,
            occ_opp,
            center_own,
            center_opp,
            kill_own,
            kill_opp,
            done_per_agent.astype(jnp.float32),
        ],
        axis=-1,
    )

    # Self model: predict action consequences and named resource components.
    curr = agents
    nxt = agents_next
    delta_pos = (nxt[..., :2] - curr[..., :2]) * jnp.asarray(
        [cfg.field_hx, cfg.field_hy], dtype=jnp.float32
    ) / max(cfg.v_max * cfg.dt, 1e-6)
    delta_vel = (nxt[..., 2:4] - curr[..., 2:4]) * cfg.v_max / max(cfg.a_max * cfg.dt, 1e-6)
    cross = nxt[..., 4] * curr[..., 5] - nxt[..., 5] * curr[..., 4]
    dot = nxt[..., 4] * curr[..., 4] + nxt[..., 5] * curr[..., 5]
    delta_yaw = jnp.arctan2(cross, dot) / max(cfg.yaw_step_max, 1e-6)
    delta_spin = nxt[..., 9] - curr[..., 9]
    raw_damage_taken = traj["aux_damage_taken"]
    actual_damage_taken = jnp.minimum(raw_damage_taken, curr[..., 6] * cfg.hp_max)
    just_died = curr[..., 10] > 0.5
    just_died = just_died & (nxt[..., 10] <= 0.5)
    revived = (curr[..., 10] <= 0.5) & (nxt[..., 10] > 0.5)
    spawn_damage_event = jnp.zeros_like(curr[..., 10], dtype=jnp.float32)
    for i in range(N_AGENTS):
        spawn_damage_event = spawn_damage_event.at[:, :, i].set(
            _point_in_rect(
                nxt[..., i, :2],
                jnp.asarray(env.enemy_spawns[i], dtype=jnp.float32),
            ).astype(jnp.float32)
        )
    spawn_damage_event = spawn_damage_event * (
        (curr[..., 10] > 0.5) & (curr[..., 11] <= 0.5)
    ).astype(jnp.float32)
    resource_valid = ~just_died & ~revived
    after_damage_hp = jnp.maximum(
        curr[..., 6] - actual_damage_taken / cfg.hp_max
        - spawn_damage_event * cfg.spawn_damage_per_frame / cfg.hp_max,
        0.0,
    )
    heal_amount = jnp.where(
        resource_valid, jnp.maximum(nxt[..., 6] - after_damage_hp, 0.0), 0.0
    )

    home = jnp.zeros_like(curr[..., 7], dtype=jnp.bool_)
    for i in range(N_AGENTS):
        home = home.at[:, :, i].set(
            _point_in_rect(
                curr[..., i, :2] * jnp.asarray([cfg.field_hx, cfg.field_hy]),
                jnp.asarray(env.own_spawns[i], dtype=jnp.float32),
            )
        )
    cool_per_frame = jnp.where(
        home & (curr[..., 10] > 0.5),
        cfg.cool_per_frame * cfg.home_cool_mult,
        cfg.cool_per_frame,
    ) / cfg.heat_max
    heat_cool = jnp.minimum(curr[..., 7], cool_per_frame)
    after_cool = curr[..., 7] - heat_cool
    invuln_expiring = (
        (curr[..., 11] > 0.5)
        & ~revived
        & (curr[..., 12] > 0.0)
        & (curr[..., 12] <= cfg.dt + 1e-6)
        & cfg.respawn_heat_lock
    )
    resource_valid = resource_valid & ~invuln_expiring
    heat_gain = jnp.where(
        resource_valid & (traj["aux_fire"] > 0.5),
        jnp.maximum(nxt[..., 7] - after_cool, 0.0),
        0.0,
    )
    self_target = jnp.stack(
        [
            delta_pos[..., 0],
            delta_pos[..., 1],
            delta_vel[..., 0],
            delta_vel[..., 1],
            delta_yaw,
            delta_spin,
            actual_damage_taken / max(cfg.damage_per_frame, 1e-6),
            heal_amount,
            heat_cool / max(cfg.cool_per_frame * cfg.home_cool_mult / cfg.heat_max, 1e-6),
            heat_gain / max(cfg.heat_per_frame / cfg.heat_max, 1e-6),
            spawn_damage_event,
            nxt[..., 8],
            nxt[..., 10],
            nxt[..., 11],
        ],
        axis=-1,
    )

    return {
        "aux_track_target": track_delta,
        "aux_track_valid": track_weight,
        "aux_track_hidden": track_hidden,
        "aux_track_visible": track_visible,
        "aux_track_age": track_age,
        "aux_track_base": track_base,
        "aux_visible_target": visible_next,
        "aux_visible_valid": jnp.broadcast_to(
            visible_next_valid[..., None, None], visible_next.shape
        ),
        "aux_heat_target": heat_target,
        "aux_heat_valid": heat_valid,
        "aux_overheat_target": enemy_overheated,
        "aux_damage_target": damage_target,
        "aux_damage_valid": target_present,
        "aux_fire_valid": own_alive,
        "aux_rules_target": rules_target,
        "aux_self_target": self_target,
        "aux_self_valid": resource_valid,
    }


def _masked_mean(x, mask, axis=None):
    m = mask.astype(x.dtype)
    return jnp.sum(x * m, axis=axis) / jnp.maximum(jnp.sum(m, axis=axis), 1.0)


def _huber(pred, target):
    err = jnp.abs(pred - target)
    return jnp.where(err < 1.0, 0.5 * err * err, err - 0.5)


def _binary_accuracy(logits, target):
    return ((logits > 0.0) == (target > 0.5)).astype(jnp.float32)


def auxiliary_loss(
    outputs: Mapping[str, jax.Array],
    targets: Mapping[str, jax.Array],
    weights: Sequence[float],
    agent_mask: jax.Array | None = None,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Compute the five auxiliary losses and compact monitoring metrics.

    Metrics are normalized unless their name includes ``_m`` or ``_points``.
    ``agent_mask`` excludes frozen opponents in league environments so their
    privileged labels cannot backpropagate into the learner-controlled actor.
    """
    w_track, w_damage, w_heat, w_rules, w_self = [jnp.asarray(w, jnp.float32) for w in weights]
    rel_scale = 0.6  # 2 * v_max * dt for the RM configuration
    if agent_mask is None:
        learn_mask = jnp.ones_like(targets["aux_fire_valid"], dtype=jnp.float32)
    else:
        learn_mask = agent_mask.astype(jnp.float32)
    pair_mask = learn_mask[..., None]

    # ---------------------------------------------------------------- tracking
    tr = outputs["track"].reshape(outputs["track"].shape[:-1] + (N_ENEMIES, TRACK_PER_ENEMY))
    tr_delta = tr[..., :2]
    tr_logsig = jnp.clip(tr[..., 2], -4.0, 2.0)
    tr_vis = tr[..., 3]
    tr_target = targets["aux_track_target"]
    tr_position_weight = targets["aux_track_valid"] * pair_mask
    tr_hidden = targets["aux_track_hidden"]
    tr_visible = targets["aux_track_visible"]
    err2 = jnp.sum((tr_delta - tr_target) ** 2, axis=-1)
    # The clamp floor is -4, so this constant keeps the NLL monitor
    # non-negative without changing its gradient (constant shift).
    nll = 4.0 + 0.5 * (
        err2 * jnp.exp(-2.0 * tr_logsig) + 2.0 * tr_logsig
    )
    loss_track_pos = _masked_mean(nll, tr_position_weight)
    vis_logits = tr[..., 3]
    vis_bce = optax.sigmoid_binary_cross_entropy(
        vis_logits, targets["aux_visible_target"]
    )
    track_vis_valid = targets["aux_visible_valid"] * pair_mask
    loss_track_vis = _masked_mean(vis_bce, track_vis_valid)
    loss_track = loss_track_pos + 0.25 * loss_track_vis
    pos_err_m = jnp.sqrt(err2) * rel_scale

    # ------------------------------------------------------------------ damage
    out_dmg = outputs["damage"]
    fire_logit, hit_logit, damage_logit, kill_logit = (
        out_dmg[..., 0],
        out_dmg[..., 1],
        out_dmg[..., 2],
        out_dmg[..., 3],
    )
    dmg_t = targets["aux_damage_target"]
    dmg_valid = targets["aux_damage_valid"] * learn_mask
    fire_valid = targets["aux_fire_valid"] * learn_mask
    pred_hit = jax.nn.sigmoid(hit_logit)
    pred_dmg = jax.nn.sigmoid(damage_logit)
    loss_fire = _masked_mean(
        optax.sigmoid_binary_cross_entropy(fire_logit, dmg_t[..., 0]), fire_valid
    )
    loss_hit = _masked_mean(
        optax.sigmoid_binary_cross_entropy(hit_logit, dmg_t[..., 1]), dmg_valid
    )
    loss_damage = _masked_mean(_huber(pred_dmg, dmg_t[..., 2]), dmg_valid)
    loss_kill = _masked_mean(
        optax.sigmoid_binary_cross_entropy(kill_logit, dmg_t[..., 3]), dmg_valid
    )
    loss_damage_head = loss_fire + loss_hit + loss_damage + 0.5 * loss_kill

    # -------------------------------------------------------------- enemy heat
    ht = outputs["heat"].reshape(outputs["heat"].shape[:-1] + (N_ENEMIES, HEAT_PER_ENEMY))
    heat_margin_pred = jax.nn.sigmoid(ht[..., 0])
    heat_delta_pred = ht[..., 1]
    heat_event_logit = ht[..., 2]
    heat_t = targets["aux_heat_target"]
    heat_valid = targets["aux_heat_valid"] * pair_mask
    loss_heat_margin = _masked_mean(_huber(heat_margin_pred, heat_t[..., 0]), heat_valid)
    loss_heat_delta = _masked_mean(_huber(heat_delta_pred, heat_t[..., 1]), heat_valid)
    loss_heat_event = _masked_mean(
        optax.sigmoid_binary_cross_entropy(
            heat_event_logit, targets["aux_overheat_target"]
        ),
        heat_valid,
    )
    loss_heat = loss_heat_margin + loss_heat_delta + 0.5 * loss_heat_event

    # ------------------------------------------------------------------- rules
    ru = outputs["rules"]
    rule_t = targets["aux_rules_target"]
    occ_pred = jax.nn.sigmoid(ru[..., 0:2])
    score_pos_pred = ru[..., 2:6]
    done_logit = ru[..., 6]
    loss_occ = _masked_mean(
        jnp.mean(_huber(occ_pred, rule_t[..., 0:2]), axis=-1), learn_mask
    )
    loss_score_pos = _masked_mean(
        jnp.mean(_huber(score_pos_pred, rule_t[..., 2:6]), axis=-1), learn_mask
    )
    loss_done = _masked_mean(
        optax.sigmoid_binary_cross_entropy(done_logit, rule_t[..., 6]),
        learn_mask,
    )
    loss_rules = loss_occ + loss_score_pos + 0.5 * loss_done

    # -------------------------------------------------------------------- self
    sf = outputs["self"]
    self_t = targets["aux_self_target"]
    self_valid = targets["aux_self_valid"] * learn_mask
    loss_self_motion = _masked_mean(
        jnp.mean(_huber(sf[..., :6], self_t[..., :6]), axis=-1), learn_mask
    )
    loss_self_resource = _masked_mean(
        jnp.mean(_huber(sf[..., 6:10], self_t[..., 6:10]), axis=-1),
        self_valid,
    )
    self_event_logits = sf[..., 10:14]
    self_event_target = self_t[..., 10:14]
    loss_self_event = _masked_mean(
        jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                self_event_logits, self_event_target
            ),
            axis=-1,
        ),
        learn_mask,
    )
    loss_self = loss_self_motion + loss_self_resource + 0.5 * loss_self_event

    total = (
        w_track * loss_track
        + w_damage * loss_damage_head
        + w_heat * loss_heat
        + w_rules * loss_rules
        + w_self * loss_self
    )

    metrics = (
        total,
        loss_track,
        loss_track_pos,
        loss_track_vis,
        _masked_mean(pos_err_m, tr_visible * pair_mask),
        _masked_mean(pos_err_m, tr_hidden * tr_position_weight),
        _masked_mean(
            targets["aux_track_age"].astype(jnp.float32),
            tr_hidden * pair_mask,
        ),
        _masked_mean(
            _binary_accuracy(vis_logits, targets["aux_visible_target"]),
            track_vis_valid,
        ),
        loss_damage_head,
        _masked_mean(_binary_accuracy(fire_logit, dmg_t[..., 0]), fire_valid),
        _masked_mean(jnp.abs(pred_hit - dmg_t[..., 1]), dmg_valid),
        _masked_mean(jnp.abs(pred_dmg - dmg_t[..., 2]), dmg_valid) * 30.0,
        _masked_mean(_binary_accuracy(kill_logit, dmg_t[..., 3]), dmg_valid),
        loss_heat,
        _masked_mean(jnp.abs(heat_margin_pred - heat_t[..., 0]), heat_valid),
        _masked_mean(jnp.abs(heat_delta_pred - heat_t[..., 1]), heat_valid),
        _masked_mean(
            _binary_accuracy(heat_event_logit, targets["aux_overheat_target"]),
            heat_valid,
        ),
        loss_rules,
        _masked_mean(
            jnp.mean(jnp.abs(occ_pred - rule_t[..., 0:2]), axis=-1),
            learn_mask,
        ),
        _masked_mean(
            jnp.mean(
                jnp.abs(score_pos_pred[..., 0:2] - rule_t[..., 2:4]), axis=-1
            ),
            learn_mask,
        )
        * 0.03,
        _masked_mean(
            jnp.mean(
                jnp.abs(score_pos_pred[..., 2:4] - rule_t[..., 4:6]), axis=-1
            ),
            learn_mask,
        )
        * 10.0,
        _masked_mean(_binary_accuracy(done_logit, rule_t[..., 6]), learn_mask),
        loss_self,
        _masked_mean(
            jnp.mean(jnp.abs(sf[..., :6] - self_t[..., :6]), axis=-1),
            learn_mask,
        ),
        _masked_mean(
            jnp.mean(jnp.abs(sf[..., 6:10] - self_t[..., 6:10]), axis=-1),
            self_valid,
        ),
        _masked_mean(
            jnp.mean(
                _binary_accuracy(self_event_logits, self_event_target),
                axis=-1,
            ),
            learn_mask,
        ),
    )
    return total, metrics


AUX_METRIC_NAMES = (
    "aux/loss_total",
    "aux/loss_track",
    "aux/loss_track_position",
    "aux/loss_track_visibility",
    "aux/track_visible_pos_mae_m",
    "aux/track_hidden_pos_mae_m",
    "aux/track_hidden_age_mean",
    "aux/track_visibility_accuracy",
    "aux/loss_damage",
    "aux/damage_fire_accuracy",
    "aux/damage_hit_probability_mae",
    "aux/damage_realized_mae_points",
    "aux/damage_kill_accuracy",
    "aux/loss_enemy_heat",
    "aux/enemy_heat_margin_mae",
    "aux/enemy_heat_delta_mae",
    "aux/enemy_overheat_accuracy",
    "aux/loss_rules",
    "aux/rules_occupancy_mae",
    "aux/rules_center_points_mae",
    "aux/rules_kill_points_mae",
    "aux/rules_done_accuracy",
    "aux/loss_self",
    "aux/self_motion_mae",
    "aux/self_resource_mae",
    "aux/self_event_accuracy",
)


__all__ = [
    "AUX_METRIC_NAMES",
    "AUX_OUTPUT_DIM",
    "DAMAGE_DIM",
    "HEAT_DIM",
    "RULES_DIM",
    "SELF_DIM",
    "TRACK_DIM",
    "RmAuxHeads",
    "auxiliary_loss",
    "build_aux_targets",
]
