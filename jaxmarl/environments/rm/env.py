"""Pure-JAX RM 3v3 environment (single environment).

This is the first port of ``VectorizedMultiAgentSimulator/vmas/scenarios/rm.py``
to JaxMARL.  The state is intentionally single-environment; use ``jax.vmap`` or
JaxMARL's batched wrappers for throughput.  The reference implementation lives
in ``VectorizedMultiAgentSimulator/vmas/scenarios/rm_rules.py``.

Parity status: config, fixed reset, movement, lock/visibility, heat/fire,
damage, respawn, centre scoring, terminal outcomes and the 140-D observation
layout are ported.  Collision fouls, wall/body-push fouls, random/late-game
reset and the full reward shaping are still being ported (marked TODO below).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box

from .config import RmConfig

Array = jax.Array
EPS = 1e-9


# ---------------------------------------------------------------------------
# geometry helpers (JAX; same conventions as rm_geometry.py)
# ---------------------------------------------------------------------------
def wrap_angle(a: Array) -> Array:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def clip_norm(x: Array, max_norm: float = 1.0) -> Array:
    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x * jnp.minimum(1.0, max_norm / jnp.maximum(n, EPS))


def rotate(v: Array, a: Array) -> Array:
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack((v[..., 0] * c - v[..., 1] * s,
                      v[..., 0] * s + v[..., 1] * c), axis=-1)


def ray_edges(origin: Array, direction: Array, edges: Array) -> Tuple[Array, Array]:
    """origin/direction (...,2), edges (E,4) -> t (...,E), hit (...,E)."""
    q1 = edges[:, :2]
    q2 = edges[:, 2:]
    s = q2 - q1
    o = origin[..., None, :]
    d = direction[..., None, :]
    denom = d[..., 0] * s[:, 1] - d[..., 1] * s[:, 0]
    valid = jnp.abs(denom) > EPS
    q = q1 - o
    ds = jnp.where(valid, denom, 1.0)
    t = (q[..., 0] * s[:, 1] - q[..., 1] * s[:, 0]) / ds
    u = (q[..., 0] * d[..., 1] - q[..., 1] * d[..., 0]) / ds
    hit = valid & (t >= 0.0) & (u >= -EPS) & (u <= 1.0 + EPS)
    return t, hit


def segment_circle_blocked(observer: Array, points: Array, circles: Array) -> Array:
    """observer (...,2), points (...,S,2), circles (...,K,3) -> (...,S)."""
    c = circles[..., None, :, :2]
    r = circles[..., None, :, 2]
    ab = points[..., :, None, :] - observer[..., None, None, :]
    denom = jnp.sum(ab * ab, axis=-1).clip(EPS)
    t = jnp.sum((c - observer[..., None, None, :]) * ab, axis=-1) / denom
    t = jnp.clip(t, 0.0, 1.0)
    q = observer[..., None, None, :] + t[..., None] * ab
    dist = jnp.linalg.norm(c - q, axis=-1)
    return jnp.any(dist <= r + EPS, axis=-1)


def visibility_arc(observer: Array, yaw: Array, target: Array,
                   target_radius: float, circles: Array, edges: Array,
                   n_samples: int) -> Array:
    """Visible fraction of the target's front arc.

    observer/target (B,2), yaw (B,), circles (B,K,3), edges (E,4).
    """
    rel = target - observer
    d = jnp.linalg.norm(rel, axis=-1)
    d_safe = jnp.maximum(d, 2.0 * target_radius)
    full = 2.0 * jnp.arccos(jnp.clip(target_radius / d_safe, -1.0, 1.0))
    u = jnp.where((d > 1e-6)[..., None], -rel / jnp.maximum(d, 1e-6)[..., None],
                  jnp.array([1.0, 0.0]))
    phi = -full[..., None] / 2.0 + (jnp.arange(n_samples)[None, :] + 0.5) * full[..., None] / n_samples
    p = target[..., None, :] + target_radius * rotate(u[..., None, :], phi)
    blocked = segment_circle_blocked(observer, p, circles)
    # cover edges
    seg = p - observer[..., None, :]
    t, hit = ray_edges(observer[..., None, :], seg, edges)
    blocked = blocked | jnp.any(hit & (t < 1.0 - 1e-6), axis=-1)
    return (~blocked).astype(jnp.float32).mean(axis=-1)


def ray_wall(origin: Array, direction: Array, hx: float, hy: float) -> Array:
    ox, oy = origin[..., 0], origin[..., 1]
    dx, dy = direction[..., 0], direction[..., 1]
    ts = []
    for val, o, d in ((-hx, ox, dx), (hx, ox, dx), (-hy, oy, dy), (hy, oy, dy)):
        t = (val - o) / jnp.where(jnp.abs(d) < EPS, EPS, d)
        ts.append(jnp.where(t > 1e-6, t, jnp.inf))
    return jnp.min(jnp.stack(ts, axis=-1), axis=-1)


def obstacle_distances(origin: Array, edges: Array, max_range: float,
                       hx: float, hy: float, n_rays: int = 8) -> Array:
    ang = jnp.arange(n_rays) * (2.0 * jnp.pi / n_rays)
    dirs = jnp.stack((jnp.cos(ang), jnp.sin(ang)), axis=-1)
    dirs = jnp.broadcast_to(dirs[None, :, :], (origin.shape[0], n_rays, 2))
    o = jnp.broadcast_to(origin[:, None, :], (origin.shape[0], n_rays, 2))
    t, hit = ray_edges(o, dirs, edges)
    best = jnp.min(jnp.where(hit, t, jnp.inf), axis=-1)
    best = jnp.minimum(best, ray_wall(o, dirs, hx, hy))
    best = jnp.clip(best, max=max_range)
    return jnp.where(jnp.isinf(best), 1.0, best / max_range)


def point_in_rect(points: Array, poly: Array) -> Array:
    """Coarse ray cast for the fixed rectangles used by spawn/heal zones."""
    x, y = points[..., 0], points[..., 1]
    inside = jnp.zeros_like(x, dtype=jnp.bool_)
    n = poly.shape[0]
    for k in range(n):
        x1, y1 = poly[k, 0], poly[k, 1]
        x2, y2 = poly[(k + 1) % n, 0], poly[(k + 1) % n, 1]
        crosses = (y1 > y) != (y2 > y)
        denom = jnp.where(jnp.abs(y2 - y1) < EPS, EPS, y2 - y1)
        x_int = (x2 - x1) * (y - y1) / denom + x1
        inside = inside ^ (crosses & (x < x_int))
    return inside


def nearest_point_on_polygon(points: Array, poly: Array):
    """points (M,2), poly (P,2) -> nearest q, distance, outward normal, inside."""
    p = points
    best_d = jnp.full((p.shape[0],), jnp.inf, dtype=p.dtype)
    best_q = jnp.zeros_like(p)
    best_n = jnp.zeros_like(p)
    n = poly.shape[0]
    for k in range(n):
        a = poly[k]
        b = poly[(k + 1) % n]
        s = b - a
        ss = jnp.maximum(jnp.sum(s * s), EPS)
        tt = jnp.clip(jnp.sum((p - a) * s, axis=-1) / ss, 0.0, 1.0)
        q = a + tt[:, None] * s
        d = jnp.linalg.norm(p - q, axis=-1)
        upd = d < best_d
        best_d = jnp.where(upd, d, best_d)
        best_q = jnp.where(upd[:, None], q, best_q)
        nrm = jnp.stack((s[1], -s[0])) / jnp.sqrt(ss)
        best_n = jnp.where(upd[:, None], nrm, best_n)
    inside = point_in_rect(p, poly)
    out_n = (p - best_q) / jnp.maximum(best_d[:, None], EPS)
    normal = jnp.where(inside[:, None], best_n, out_n)
    return best_q, best_d, normal, inside


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
@struct.dataclass
class RmState:
    pos: Array          # (6,2)
    yaw: Array          # (6,)
    vel: Array          # (6,2)
    v_pre: Array        # (6,2)
    yaw_rate: Array     # (6,)
    yaw_cmd: Array      # (6,)
    spin: Array         # (6,)
    spin_cmd: Array
    hp: Array
    heat: Array
    overheated: Array
    alive: Array
    invuln: Array
    invuln_timer: Array
    respawn_timer: Array
    target: Array       # int
    target_cmd: Array
    fire_cmd: Array
    p_t: Array
    tau_eff: Array
    last_fired: Array
    hit_cue: Array
    hit_cue_valid: Array
    score: Array        # (2,)
    time: Array         # scalar
    episode_elapsed: Array
    total_damage: Array # (2,)
    cum_damage: Array   # (6,6)
    kill_damage: Array
    last_kill_credit: Array
    last_kill_score: Array
    last_p_final: Array
    occ: Array          # (2,)
    outcome: Array
    winner: Array
    foul_loser: Array
    foul_agent: Array
    foul_intensity: Array
    double_foul: Array
    both_200: Array
    done: Array
    just_died: Array
    step: Array
    reward_dense: Array
    reward_terminal: Array
    center_dist_prev: Array
    last_hit_damage: Array
    last_damage: Array
    last_kill_credit: Array
    last_kill_score: Array
    prev_wall_contact: Array
    prev_pair_contact: Array
    wall_contact_time: Array
    body_push_time: Array
    body_push_foul: Array
    proximity_time: Array
    episode_start_elapsed: Array
    stagnation_had_combat: Array
    stagnation_had_occ: Array
    stagnation_had_heal: Array
    stagnation_last_event: Array
    stagnation_speed: Array
    wall_push_time: Array
    wall_push_foul_intent: Array


class RmEnv(MultiAgentEnv):
    """Single-environment JAX port of the RM 3v3 game."""

    def __init__(self, cfg: RmConfig | None = None):
        self.cfg = cfg or RmConfig()
        super().__init__(self.cfg.n_agents)
        n = self.cfg.n_agents
        self.agents = [f"agent_{i}" for i in range(n)]
        self.team = jnp.array([self.cfg.team(i) for i in range(n)], dtype=jnp.int32)
        self.enemy_idx = jnp.array([self.cfg.enemy_indices(i) for i in range(n)], dtype=jnp.int32)
        self.mate_idx = jnp.array([self.cfg.teammate_indices(i) for i in range(n)], dtype=jnp.int32)
        self._enemy_obs_idx = tuple(i for i in range(n) for _ in range(3))
        self._enemy_tgt_idx = tuple(
            int(self.cfg.enemy_indices(i)[k]) for i in range(n) for k in range(3)
        )
        # target 0 is a dummy (self); target 1..3 maps to the three enemies.
        self._target_table = tuple(
            (i,) + tuple(int(x) for x in self.cfg.enemy_indices(i))
            for i in range(n)
        )
        self.red_slots = jnp.asarray(self.cfg.red_slots, dtype=jnp.float32)
        self.blue_slots = jnp.asarray(self.cfg.blue_slots, dtype=jnp.float32)
        self.red_spawn = jnp.asarray(self.cfg.red_spawn, dtype=jnp.float32)
        self.blue_spawn = jnp.asarray(self.cfg.blue_spawn, dtype=jnp.float32)
        self.own_spawns = jnp.stack([self.red_spawn] * 3 + [self.blue_spawn] * 3)
        self.enemy_spawns = jnp.stack([self.blue_spawn] * 3 + [self.red_spawn] * 3)
        # all static cover edges
        edges = []
        for poly in self.cfg.covers:
            p = jnp.asarray(poly, dtype=jnp.float32)
            for k in range(p.shape[0]):
                edges.append(jnp.concatenate([p[k], p[(k + 1) % p.shape[0]]]))
        self.cover_edges = jnp.stack(edges)
        self.cover_polys = [jnp.asarray(poly, dtype=jnp.float32) for poly in self.cfg.covers]
        # PyTorch samples random starts from a legal-centre grid (not just the
        # own spawn box).  Build the same two static pools once.
        self.spawn_pool_red = self._build_spawn_pool(0)
        self.spawn_pool_blue = self._build_spawn_pool(1)
        self.observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (self.cfg.obs_dim,), jnp.float32)
            for a in self.agents
        }
        self.action_spaces = {
            a: Box(-1.0, 1.0, (6,), jnp.float32) for a in self.agents
        }

    # ------------------------------------------------------------------
    def _build_spawn_pool(self, team: int) -> Array:
        """Legal random-spawn centres, matching PyTorch `_build_spawn_pool`."""
        cfg = self.cfg
        xs = jnp.arange(-cfg.bound_x + 0.1, cfg.bound_x - 0.1 + 1e-6, 0.2)
        ys = jnp.arange(-cfg.bound_y + 0.1, cfg.bound_y - 0.1 + 1e-6, 0.2)
        gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
        grid = jnp.stack((gx.reshape(-1), gy.reshape(-1)), axis=-1)
        valid = jnp.ones((grid.shape[0],), dtype=jnp.bool_)
        for poly in self.cover_polys:
            _, dist, _, inside = nearest_point_on_polygon(grid, poly)
            valid = valid & (~inside) & (dist >= cfg.robot_radius + 0.05)
        enemy = self.enemy_spawns[0 if team == 0 else cfg.n_per_team]
        valid = valid & (~point_in_rect(grid, enemy))
        return grid[valid]

    def _empty_reward(self, n: int = 6) -> Array:
        return jnp.zeros((n,), dtype=jnp.float32)

    def _reset_fixed(self, key: Array) -> Tuple[Dict[str, Array], RmState]:
        cfg = self.cfg
        pos = jnp.concatenate([self.red_slots, self.blue_slots], axis=0)
        k_obs = key
        if cfg.fixed_spawn_rotate:
            # Rotate the two triangles around their own centroids.  Blue uses
            # the mirrored angle so the two teams remain 180-degree symmetric.
            k_rot, k_obs = jax.random.split(key)
            theta_r = jax.random.uniform(
                k_rot, (), minval=-cfg.fixed_spawn_rotate_max,
                maxval=cfg.fixed_spawn_rotate_max,
            )
            theta_b = jax.random.uniform(
                jax.random.fold_in(k_rot, 1), (), minval=-cfg.fixed_spawn_rotate_max,
                maxval=cfg.fixed_spawn_rotate_max,
            )
            rc = self.red_slots.mean(axis=0)
            bc = self.blue_slots.mean(axis=0)

            def rotate_triangle(p, c, a):
                rel = p - c
                ca, sa = jnp.cos(a), jnp.sin(a)
                return jnp.stack((
                    rel[:, 0] * ca - rel[:, 1] * sa,
                    rel[:, 0] * sa + rel[:, 1] * ca,
                ), axis=-1) + c

            rotated = jnp.concatenate((
                rotate_triangle(self.red_slots, rc, theta_r),
                rotate_triangle(self.blue_slots, bc, theta_b),
            ), axis=0)
            limit = jnp.array([cfg.bound_x, cfg.bound_y])
            in_bounds = jnp.all(jnp.abs(rotated) <= limit)
            d = jnp.linalg.norm(rotated[:, None] - rotated[None, :], axis=-1)
            d = d + jnp.eye(cfg.n_agents) * 1e9
            separated = jnp.min(d) >= cfg.rsi_min_sep - 1e-5
            pos = jnp.where(in_bounds & separated, rotated, pos)
        yaw = jnp.arctan2(-pos[:, 1], -pos[:, 0])
        z6 = jnp.zeros((6,), dtype=jnp.float32)
        z62 = jnp.zeros((6, 2), dtype=jnp.float32)
        z66 = jnp.zeros((6, 6), dtype=jnp.float32)
        state = RmState(
            pos=pos, yaw=yaw, vel=z62, v_pre=z62, yaw_rate=z6, yaw_cmd=z6,
            spin=z6, spin_cmd=z6, hp=jnp.full((6,), cfg.hp_max, dtype=jnp.float32),
            heat=z6, overheated=jnp.zeros((6,), dtype=jnp.bool_),
            alive=jnp.ones((6,), dtype=jnp.bool_), invuln=z6, invuln_timer=z6,
            respawn_timer=z6, target=jnp.zeros((6,), dtype=jnp.int32),
            target_cmd=jnp.zeros((6,), dtype=jnp.int32),
            fire_cmd=jnp.zeros((6,), dtype=jnp.bool_),
            p_t=z6, tau_eff=jnp.full((6,), cfg.tau_base),
            last_fired=jnp.zeros((6,), dtype=jnp.bool_),
            hit_cue=z6, hit_cue_valid=jnp.zeros((6,), dtype=jnp.bool_),
            score=jnp.zeros((2,), dtype=jnp.float32),
            time=jnp.asarray(cfg.episode_time, dtype=jnp.float32),
            episode_elapsed=jnp.asarray(0.0, dtype=jnp.float32),
            total_damage=jnp.zeros((2,), dtype=jnp.float32),
            cum_damage=z66, kill_damage=z66, last_kill_credit=z6,
            last_kill_score=jnp.zeros((2,), dtype=jnp.float32), last_p_final=z6,
            occ=jnp.zeros((2,), dtype=jnp.float32), outcome=jnp.asarray(0, dtype=jnp.int32),
            winner=jnp.asarray(-1, dtype=jnp.int32), foul_loser=jnp.asarray(-1, dtype=jnp.int32),
            foul_agent=jnp.zeros((6,), dtype=jnp.bool_), foul_intensity=z6,
            double_foul=jnp.zeros((), dtype=jnp.bool_),
            both_200=jnp.zeros((), dtype=jnp.bool_), done=jnp.zeros((), dtype=jnp.bool_),
            just_died=jnp.zeros((6,), dtype=jnp.bool_), step=jnp.asarray(0, dtype=jnp.int32),
            reward_dense=z6, reward_terminal=z6,
            center_dist_prev=jnp.linalg.norm(pos, axis=-1),
            last_hit_damage=z6, last_damage=z66,
            prev_wall_contact=jnp.zeros((6,), dtype=jnp.bool_),
            prev_pair_contact=jnp.zeros((6, 6), dtype=jnp.bool_),
            wall_contact_time=z6, body_push_time=z6,
            body_push_foul=jnp.zeros((6,), dtype=jnp.bool_),
            proximity_time=z66,
            episode_start_elapsed=jnp.asarray(0.0, dtype=jnp.float32),
            stagnation_had_combat=jnp.zeros((), dtype=jnp.bool_),
            stagnation_had_occ=jnp.zeros((), dtype=jnp.bool_),
            stagnation_had_heal=jnp.zeros((), dtype=jnp.bool_),
            stagnation_last_event=jnp.asarray(0.0, dtype=jnp.float32),
            stagnation_speed=z6,
            wall_push_time=z6, wall_push_foul_intent=z6,
        )
        obs = self.get_obs(state, k_obs)
        return obs, state

    def reset(self, key: Array, mode: str = "fixed", fixed_prob: float = 1.0) -> Tuple[Dict[str, Array], RmState]:
        """Fixed spawn by default; ``mode='random'`` samples spawn boxes.

        ``mode='curriculum'`` samples fixed gameplay with probability
        ``fixed_prob`` and late-game random spawn otherwise.
        """
        if mode == "fixed":
            return self._reset_fixed(key)
        if mode == "random":
            return self._reset_random(key)
        if mode == "curriculum":
            k_choice, k_reset = jax.random.split(key)
            return jax.lax.cond(
                jax.random.uniform(k_choice) < fixed_prob,
                lambda k: self._reset_fixed(k),
                lambda k: self._reset_random(k),
                k_reset,
            )
        raise ValueError(f"unknown reset mode: {mode}")

    def _reset_random(self, key: Array) -> Tuple[Dict[str, Array], RmState]:
        """Random reset, matching the PyTorch reset distribution.

        Mirrors the reference: legal-centre pools, optional symmetric mirror,
        anchor jitter/permutation, random yaw/HP/spin/heat and either a random
        late-game clock or a random total episode horizon.
        """
        cfg = self.cfg
        _, state = self._reset_fixed(key)
        k_sym, k_pos, k_yaw, k_hp, k_spin, k_heat, k_time = jax.random.split(key, 7)
        sym = jax.random.uniform(k_sym, ()) < cfg.rsi_symmetric_prob
        n_team = cfg.n_per_team
        red_anchor_c = self.own_spawns[:n_team].mean(axis=1).mean(axis=0)
        blue_anchor_c = self.own_spawns[n_team:].mean(axis=1).mean(axis=0)

        def sample_positions(_):
            kk = jax.random.fold_in(k_pos, _)
            kr, kb = jax.random.split(kk)
            ri = jax.random.randint(kr, (n_team,), 0, self.spawn_pool_red.shape[0])
            bi = jax.random.randint(kb, (n_team,), 0, self.spawn_pool_blue.shape[0])
            r = self.spawn_pool_red[ri]
            b = self.spawn_pool_blue[bi]
            if cfg.spawn_anchor_init:
                jr = (jax.random.uniform(kr, (2,), minval=-1.0, maxval=1.0)
                      * cfg.spawn_anchor_jitter)
                jb = (jax.random.uniform(kb, (2,), minval=-1.0, maxval=1.0)
                      * cfg.spawn_anchor_jitter)
                r = r.at[0].set(red_anchor_c + jr)
                b = b.at[0].set(blue_anchor_c + jb)
            b = jnp.where(sym, -r, b)
            if cfg.spawn_anchor_init:
                b = b.at[0].set(jnp.where(sym, -r[0], blue_anchor_c + jb))
            pts = jnp.concatenate((r, b), axis=0)
            d = jnp.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
            d = d + jnp.eye(2 * n_team) * 1e9
            return r, b, jnp.min(d) >= cfg.rsi_min_sep

        def body(_, carry):
            pos, valid = carry
            r, b, ok = sample_positions(_)
            pts = jnp.concatenate((r, b), axis=0)
            return (jnp.where(valid, pos, jnp.where(ok, pts, pos)), valid | ok)

        pos, valid = jax.lax.fori_loop(
            0, 32, body, (state.pos, jnp.asarray(False))
        )
        # Valid rows already have a legal position; invalid rows keep the
        # fixed-slot fallback from `_reset_fixed`.

        # Random yaw, then anchor direction, then per-row permutation.
        yaw_r = jax.random.uniform(k_yaw, (n_team,), minval=-jnp.pi, maxval=jnp.pi)
        yaw_b_free = jax.random.uniform(
            jax.random.fold_in(k_yaw, 1), (n_team,), minval=-jnp.pi, maxval=jnp.pi
        )
        yaw_b = jnp.where(sym, wrap_angle(yaw_r + jnp.pi), yaw_b_free)
        if cfg.spawn_anchor_init:
            yaw_r = yaw_r.at[0].set(jnp.arctan2(-pos[0, 1], -pos[0, 0]))
            yaw_b = yaw_b.at[0].set(jnp.where(
                sym,
                wrap_angle(yaw_r[0] + jnp.pi),
                jnp.arctan2(-pos[n_team, 1], -pos[n_team, 0]),
            ))
        perm_r = jnp.argsort(jax.random.uniform(
            jax.random.fold_in(k_yaw, 2), (n_team,)
        ))
        perm_b = jnp.where(sym, perm_r, jnp.argsort(jax.random.uniform(
            jax.random.fold_in(k_yaw, 3), (n_team,)
        )))
        pos = jnp.concatenate((
            jnp.take(pos[:n_team], perm_r, axis=0),
            jnp.take(pos[n_team:], perm_b, axis=0),
        ), axis=0)
        yaw = jnp.concatenate((
            jnp.take(yaw_r, perm_r, axis=0),
            jnp.take(yaw_b, perm_b, axis=0),
        ), axis=0)

        hp_r = jax.random.uniform(
            k_hp, (n_team,), minval=cfg.rsi_hp_min, maxval=cfg.rsi_hp_max
        )
        hp_b_free = jax.random.uniform(
            jax.random.fold_in(k_hp, 1), (n_team,),
            minval=cfg.rsi_hp_min, maxval=cfg.rsi_hp_max,
        )
        spin_r = jax.random.uniform(k_spin, (n_team,), maxval=cfg.rsi_spin_max)
        spin_b_free = jax.random.uniform(
            jax.random.fold_in(k_spin, 1), (n_team,), maxval=cfg.rsi_spin_max
        )
        heat_r = jax.random.uniform(k_heat, (n_team,), maxval=cfg.rsi_heat_max)
        heat_b_free = jax.random.uniform(
            jax.random.fold_in(k_heat, 1), (n_team,), maxval=cfg.rsi_heat_max
        )
        hp = jnp.concatenate((hp_r, jnp.where(sym, hp_r, hp_b_free)), axis=0)
        spin = jnp.concatenate((spin_r, jnp.where(sym, spin_r, spin_b_free)), axis=0)
        heat = jnp.concatenate((heat_r, jnp.where(sym, heat_r, heat_b_free)), axis=0)

        if cfg.rsi_late_game:
            remain = jax.random.uniform(
                k_time, (), minval=cfg.rsi_remaining_min, maxval=cfg.rsi_remaining_max
            )
            time = remain
            elapsed = cfg.episode_time - remain
        else:
            time = jax.random.uniform(
                k_time, (), minval=cfg.rsi_time_min, maxval=cfg.rsi_time_max
            )
            elapsed = jnp.asarray(0.0, dtype=jnp.float32)

        state = state.replace(
            pos=pos, yaw=yaw, vel=jnp.zeros_like(state.vel),
            v_pre=jnp.zeros_like(state.v_pre), yaw_rate=jnp.zeros_like(state.yaw_rate),
            yaw_cmd=jnp.zeros_like(state.yaw_cmd), hp=hp, spin=spin,
            spin_cmd=spin, heat=heat, overheated=jnp.zeros_like(state.overheated),
            time=time, episode_elapsed=elapsed, episode_start_elapsed=elapsed,
            stagnation_last_event=elapsed,
            stagnation_speed=jnp.zeros((cfg.n_agents,), dtype=jnp.float32),
            center_dist_prev=jnp.linalg.norm(pos, axis=-1),
            target=jnp.zeros((cfg.n_agents,), dtype=jnp.int32),
            target_cmd=jnp.zeros((cfg.n_agents,), dtype=jnp.int32),
            done=jnp.asarray(False),
        )
        obs = self.get_obs(state, k_time)
        return obs, state

    # ------------------------------------------------------------------
    def _visibility_for(self, state: RmState, i: int, j: int, n_samples: int,
                        locked_target: Array | None = None,
                        pair_class: int | None = None) -> Tuple[Array, Array]:
        """visibility probability and FOV for observer i / target j."""
        cfg = self.cfg
        obs = state.pos[i]
        tgt = state.pos[j]
        rel = tgt - obs
        d = jnp.linalg.norm(rel)
        bearing = jnp.arctan2(rel[1], rel[0])
        if locked_target is None:
            half = cfg.fov_half
        else:
            prev = locked_target[i]
            narrow = (prev != 0) & (prev != pair_class)
            half = jnp.where(narrow, cfg.fov_half_narrow, cfg.fov_half)
        in_fov = jnp.abs(wrap_angle(bearing - state.yaw[i])) <= half
        circles = []
        for q in range(cfg.n_agents):
            if q == i or q == j:
                continue
            circles.append(jnp.concatenate([state.pos[q], jnp.array([cfg.occlusion_radius])]))
        circles = jnp.stack(circles)[None, :, :] if circles else jnp.zeros((1, 0, 3))
        p = visibility_arc(obs[None, :], state.yaw[i][None], tgt[None, :],
                           cfg.robot_radius, circles, self.cover_edges, n_samples)[0]
        return p, in_fov & (d < cfg.lock_range)

    # ------------------------------------------------------------------
    def _visibility_pairs(self, state: RmState, obs_idx: Array, tgt_idx: Array,
                          n_samples: int,
                          locked_target: Array | None = None,
                          pair_class: Array | None = None) -> Tuple[Array, Array]:
        """Batched visibility for (observer, target) pairs.

        The old path called `visibility_arc` once per pair and rebuilt the
        occluder circles each time.  Build all occluders once and evaluate the
        complete pair set in one vectorized call instead.

        When ``locked_target`` is supplied, the angular half-width follows
        VMAS: unlocked observers use 45 deg, while a locked observer uses the
        narrower 40 deg for every pair except the currently locked target.
        ``pair_class`` is the target class (1..3) of each pair.
        """
        cfg = self.cfg
        origin = state.pos[obs_idx]
        target = state.pos[tgt_idx]
        yaw = state.yaw[obs_idx]
        m = obs_idx.shape[0]
        circle_pos = jnp.broadcast_to(state.pos[None, :, :], (m, cfg.n_agents, 2))
        ids = jnp.arange(cfg.n_agents)[None, :]
        excluded = (ids == obs_idx[:, None]) | (ids == tgt_idx[:, None])
        # Move the observer and target circles far away instead of changing
        # their radius; this preserves the exact segment-circle arithmetic.
        circle_pos = jnp.where(excluded[..., None], 1e6, circle_pos)
        circle_r = jnp.full((m, cfg.n_agents), cfg.occlusion_radius, dtype=jnp.float32)
        circles = jnp.concatenate((circle_pos, circle_r[..., None]), axis=-1)
        p = visibility_arc(origin, yaw, target, cfg.robot_radius, circles,
                           self.cover_edges, n_samples)
        rel = target - origin
        d = jnp.linalg.norm(rel, axis=-1)
        bearing = jnp.arctan2(rel[:, 1], rel[:, 0])
        if locked_target is None:
            half = jnp.full_like(bearing, cfg.fov_half)
        else:
            if pair_class is None:
                raise ValueError("pair_class is required with locked_target")
            prev = locked_target[obs_idx]
            # VMAS: unlocked -> 45 deg; locked observer -> 40 deg for every
            # target other than the one currently locked.
            narrow = (prev != 0) & (prev != pair_class)
            half = jnp.where(narrow, cfg.fov_half_narrow, cfg.fov_half)
        in_fov = jnp.abs(wrap_angle(bearing - yaw)) <= half
        return p, in_fov & (d < cfg.lock_range)

    def _visibility_enemy_pairs(self, state: RmState, n_samples: int) -> Tuple[Array, Array]:
        """Visibility for all (agent, each of its 3 enemies), shape [6,3]."""
        obs_idx = jnp.asarray(self._enemy_obs_idx, dtype=jnp.int32)
        tgt_idx = jnp.asarray(self._enemy_tgt_idx, dtype=jnp.int32)
        pair_class = jnp.tile(
            jnp.arange(1, 4, dtype=jnp.int32), self.cfg.n_agents
        )
        p, fov = self._visibility_pairs(
            state, obs_idx, tgt_idx, n_samples,
            locked_target=state.target, pair_class=pair_class,
        )
        return p.reshape(self.cfg.n_agents, 3), fov.reshape(self.cfg.n_agents, 3)

    def _visibility_selected(self, state: RmState, targets: Array,
                             n_samples: int) -> Array:
        """Visibility for each agent's currently selected enemy, shape [6]."""
        cfg = self.cfg
        obs_idx = jnp.arange(cfg.n_agents, dtype=jnp.int32)
        table = jnp.asarray(self._target_table, dtype=jnp.int32)
        tgt_idx = table[obs_idx, targets]
        p, _ = self._visibility_pairs(
            state, obs_idx, tgt_idx, n_samples,
            locked_target=state.target, pair_class=targets,
        )
        return p

    # ------------------------------------------------------------------
    def _obstacles(self, state: RmState) -> Array:
        o = obstacle_distances(state.pos, self.cover_edges,
                               self.cfg.obstacle_max_range,
                               self.cfg.field_hx, self.cfg.field_hy,
                               self.cfg.obstacle_rays)
        return o

    def _canon_obstacles(self, o: Array) -> Array:
        blue = (self.team == 1)[:, None]
        rotated = o[:, [4, 5, 6, 7, 0, 1, 2, 3]]
        return jnp.where(blue, rotated, o)

    # ------------------------------------------------------------------
    def get_obs(self, state: RmState, key: Array | None = None,
                pvis_obs: Array | None = None, fov_obs: Array | None = None) -> Dict[str, Array]:
        cfg = self.cfg
        if key is None:
            key = jax.random.PRNGKey(0)
        n = cfg.n_agents
        sign = jnp.where(self.team == 0, 1.0, -1.0)       # (6,)
        yaw_c = wrap_angle(state.yaw + jnp.where(self.team == 0, 0.0, jnp.pi))
        obstacle = self._canon_obstacles(self._obstacles(state))
        own_centers = jnp.stack([self.own_spawns[i].mean(axis=0) for i in range(n)])
        if pvis_obs is None or fov_obs is None:
            pvis_obs, fov_obs = self._visibility_enemy_pairs(state, cfg.vis_samples_obs)
        obs = []
        for i in range(n):
            o = []
            si = sign[i]
            # self: 24
            o += [si * state.pos[i, 0] / cfg.field_hx,
                  si * state.pos[i, 1] / cfg.field_hy,
                  si * state.vel[i, 0] / cfg.v_max,
                  si * state.vel[i, 1] / cfg.v_max,
                  jnp.sin(yaw_c[i]), jnp.cos(yaw_c[i]),
                  state.yaw_rate[i] / cfg.yaw_rate_max,
                  state.hp[i] / cfg.hp_max, state.heat[i] / cfg.heat_max,
                  state.overheated[i].astype(jnp.float32), state.spin[i],
                  state.alive[i].astype(jnp.float32), state.invuln[i],
                  state.invuln_timer[i] / cfg.invuln_time,
                  state.respawn_timer[i] / cfg.respawn_time,
                  (state.target[i] != 0).astype(jnp.float32)]
            o += [(state.target[i] == k).astype(jnp.float32) for k in range(4)]
            o += [point_in_rect(state.pos[i], self.own_spawns[i]).astype(jnp.float32)]
            o += [(jnp.linalg.norm(state.pos[i]) <= cfg.center_radius).astype(jnp.float32)]
            o += [point_in_rect(state.pos[i], self.enemy_spawns[i]).astype(jnp.float32)]
            o += [jnp.linalg.norm(state.pos[i]) / cfg.lock_range]
            rel_spawn = si * (own_centers[i] - state.pos[i])
            o += [rel_spawn[0] / cfg.field_hx, rel_spawn[1] / cfg.field_hy,
                  jnp.linalg.norm(rel_spawn) / cfg.lock_range,
                  point_in_rect(state.pos[i], self.own_spawns[i]).astype(jnp.float32)]
            rel_center = si * (-state.pos[i])
            o += [rel_center[0] / cfg.field_hx, rel_center[1] / cfg.field_hy,
                  jnp.linalg.norm(state.pos[i]) / cfg.lock_range,
                  (jnp.linalg.norm(state.pos[i]) <= cfg.center_radius).astype(jnp.float32)]
            # mates: 2 * 18
            for j in self.cfg.teammate_indices(i):
                alive_j = state.alive[j]
                rel = si * (state.pos[j] - state.pos[i])
                abs_pos = si * state.pos[j]
                rvel = si * (state.vel[j] - state.vel[i])
                dyaw = wrap_angle(state.yaw[j] - state.yaw[i])
                o += [rel[0] / cfg.lock_range, rel[1] / cfg.lock_range,
                      abs_pos[0] / cfg.field_hx, abs_pos[1] / cfg.field_hy,
                      jnp.where(alive_j, rvel[0], 0.0) / (2 * cfg.v_max),
                      jnp.where(alive_j, rvel[1], 0.0) / (2 * cfg.v_max),
                      jnp.where(alive_j, state.hp[j], 0.0) / cfg.hp_max,
                      alive_j.astype(jnp.float32),
                      jnp.where(alive_j, state.invuln[j], 0.0),
                      jnp.sin(dyaw), jnp.cos(dyaw),
                      jnp.where(alive_j, state.heat[j], 0.0) / cfg.heat_max,
                      jnp.where(alive_j, state.spin[j], 0.0)]
                o += [(alive_j & (state.target[j] == k)).astype(jnp.float32) for k in range(4)]
                o += [state.respawn_timer[j] / cfg.respawn_time]
            # enemies: 3 * 17
            for k in range(3):
                j = int(self.cfg.enemy_indices(i)[k])
                pvis = pvis_obs[i, k]
                fov = fov_obs[i, k]
                vis = fov & (pvis > 0.0)
                alive_j = state.alive[j]
                rel = si * (state.pos[j] - state.pos[i])
                abs_pos = si * state.pos[j]
                rvel = si * (state.vel[j] - state.vel[i])
                dyaw = wrap_angle(state.yaw[j] - state.yaw[i])
                # Match the reference observation noise statistically, not
                # sample-for-sample: HP noise grows while the target spins;
                # yaw noise is +/-5 deg except on the target's fire frame.
                if cfg.obs_noise:
                    hp_sigma = cfg.hp_noise_base + cfg.hp_noise_spin * state.spin[j]
                    hp_noise = jax.random.normal(
                        jax.random.fold_in(key, 1000 + i * 3 + k)
                    ) * hp_sigma
                    hp_obs = jnp.round(
                        jnp.clip(state.hp[j] + hp_noise, 0.0, cfg.hp_max)
                    ) / cfg.hp_max
                    yaw_noise = (jax.random.uniform(
                        jax.random.fold_in(key, 2000 + i * 3 + k),
                        minval=-cfg.yaw_noise_deg,
                        maxval=cfg.yaw_noise_deg,
                    ) * jnp.pi / 180.0)
                    dyaw_obs = wrap_angle(
                        dyaw + jnp.where(state.last_fired[j], 0.0, yaw_noise)
                    )
                else:
                    hp_obs = state.hp[j] / cfg.hp_max
                    dyaw_obs = dyaw
                o += [vis.astype(jnp.float32),
                      jnp.where(vis, rel[0], 0.0) / cfg.lock_range,
                      jnp.where(vis, rel[1], 0.0) / cfg.lock_range,
                      jnp.where(vis, abs_pos[0], 0.0) / cfg.field_hx,
                      jnp.where(vis, abs_pos[1], 0.0) / cfg.field_hy,
                      jnp.where(vis & alive_j, rvel[0], 0.0) / (2 * cfg.v_max),
                      jnp.where(vis & alive_j, rvel[1], 0.0) / (2 * cfg.v_max),
                      jnp.where(vis & alive_j, hp_obs, 0.0),
                      jnp.where(vis & alive_j, state.invuln[j], 0.0),
                      jnp.where(vis, jnp.sin(dyaw_obs), 0.0),
                      jnp.where(vis, jnp.cos(dyaw_obs), 0.0),
                      jnp.where(vis & alive_j, state.spin[j], 0.0),
                      jnp.where(vis & alive_j, state.last_fired[j].astype(jnp.float32), 0.0),
                      jnp.where(vis, pvis, 0.0),
                      jnp.where(vis, alive_j.astype(jnp.float32), 0.0),
                      ((state.target[i] == k + 1) & vis & alive_j).astype(jnp.float32),
                      jnp.where(vis, state.respawn_timer[j], 0.0) / cfg.respawn_time]
            cue = wrap_angle(state.hit_cue[i] - state.yaw[i])
            o += [jnp.where(state.hit_cue_valid[i], jnp.sin(cue), 0.0),
                  jnp.where(state.hit_cue_valid[i], jnp.cos(cue), 0.0),
                  state.hit_cue_valid[i].astype(jnp.float32)]
            t = self.cfg.team(i)
            own_alive = jnp.sum(state.alive & (self.team == t)) / 3.0
            enemy_alive = jnp.sum(state.alive & (self.team == 1 - t)) / 3.0
            own_hp = jnp.sum(jnp.where(state.alive & (self.team == t), state.hp, 0.0)) / (3 * cfg.hp_max)
            enemy_hp = jnp.sum(jnp.where(state.alive & (self.team == 1 - t), state.hp, 0.0)) / (3 * cfg.hp_max)
            o += [state.score[t] / cfg.target_score,
                  state.score[1 - t] / cfg.target_score,
                  (state.score[t] - state.score[1 - t]) / cfg.target_score,
                  state.time / cfg.episode_time,
                  state.occ[t] / 3.0, state.occ[1 - t] / 3.0,
                  own_alive, enemy_alive, own_hp, enemy_hp]
            o += [obstacle[i, k] for k in range(cfg.obstacle_dim)]
            obs.append(jnp.stack(o))
        return {a: obs[i] for i, a in enumerate(self.agents)}

    # ------------------------------------------------------------------
    def get_global_state(self, state: RmState) -> Array:
        """Privileged 191-D state for the centralized critic."""
        cfg = self.cfg
        # The critic sees world-frame clearance, matching the reference.
        obstacle = self._obstacles(state)
        parts = []
        for i in range(cfg.n_agents):
            parts += [
                state.pos[i, 0] / cfg.field_hx,
                state.pos[i, 1] / cfg.field_hy,
                state.vel[i, 0] / cfg.v_max,
                state.vel[i, 1] / cfg.v_max,
                jnp.sin(state.yaw[i]), jnp.cos(state.yaw[i]),
                state.hp[i] / cfg.hp_max,
                state.heat[i] / cfg.heat_max,
                state.overheated[i].astype(jnp.float32),
                state.spin[i],
                state.alive[i].astype(jnp.float32),
                state.invuln[i],
                state.invuln_timer[i] / cfg.invuln_time,
                state.respawn_timer[i] / cfg.respawn_time,
                state.target[i].astype(jnp.float32) / 3.0,
                state.p_t[i],
                state.tau_eff[i] / 0.6,
                self.team[i].astype(jnp.float32),
            ]
            for k in range(3):
                j = int(cfg.enemy_indices(i)[k])
                parts.append(state.cum_damage[i, j] / cfg.hp_max)
            parts.append(state.last_p_final[i])
            parts += [obstacle[i, k] for k in range(cfg.obstacle_dim)]
        total_hp = jnp.stack([
            jnp.sum(jnp.where(state.alive & (self.team == 0), state.hp, 0.0)),
            jnp.sum(jnp.where(state.alive & (self.team == 1), state.hp, 0.0)),
        ])
        alive_count = jnp.stack([
            jnp.sum(state.alive & (self.team == 0)),
            jnp.sum(state.alive & (self.team == 1)),
        ]).astype(jnp.float32)
        parts += [
            state.score[0] / cfg.target_score,
            state.score[1] / cfg.target_score,
            state.time / cfg.episode_time,
            state.occ[0] / 3.0,
            state.occ[1] / 3.0,
            alive_count[0] / 3.0,
            alive_count[1] / 3.0,
            total_hp[0] / (3 * cfg.hp_max),
            total_hp[1] / (3 * cfg.hp_max),
            state.total_damage[0] / 1000.0,
            state.total_damage[1] / 1000.0,
        ]
        return jnp.stack(parts)

    # ------------------------------------------------------------------
    def _decode(self, state: RmState, actions: Array) -> Tuple[RmState, Array]:
        cfg = self.cfg
        raw = jnp.asarray(actions, dtype=jnp.float32)
        spin_target = jnp.clip(raw[:, 3], 0.0, 1.0)
        target_cmd = jnp.rint(jnp.clip(raw[:, 4], 0.0, 3.0)).astype(jnp.int32)
        fire_cmd = raw[:, 5] > 0.5
        yaw_cmd = jnp.clip(raw[:, 2], -1.0, 1.0)
        alive = state.alive
        spin = jnp.clip(state.spin + jnp.clip(spin_target - state.spin,
                                              -cfg.spin_follow_rate * cfg.dt,
                                              cfg.spin_follow_rate * cfg.dt), 0.0, 1.0)
        spin = jnp.where(alive, spin, 0.0)
        vmax = cfg.v_max * (1.0 - cfg.spin_speed_coef * spin)
        amax = cfg.a_max * (1.0 - cfg.spin_speed_coef * spin)
        v_body = clip_norm(raw[:, :2], 1.0)
        if cfg.relative_actions:
            c, s = jnp.cos(state.yaw), jnp.sin(state.yaw)
            v_world = jnp.stack([v_body[:, 0] * c - v_body[:, 1] * s,
                                 v_body[:, 0] * s + v_body[:, 1] * c], axis=-1)
            v_des = v_world * vmax[:, None]
        else:
            team_sign = jnp.where(self.team == 0, 1.0, -1.0)
            v_des = v_body * (team_sign * vmax)[:, None]
        # Match the reference acceleration limiter: clamp the velocity
        # difference to amax*dt (not a unit vector), then clamp the resulting
        # speed to vmax.  The previous version normalised the delta and
        # multiplied by vmax, which made tiny commands saturate every step.
        delta = clip_norm(v_des - state.vel, (amax * cfg.dt)[:, None])
        v_next = clip_norm(state.vel + delta, vmax[:, None])
        speed = jnp.linalg.norm(state.vel, axis=-1, keepdims=True)
        new_speed = jnp.maximum(speed - cfg.corpse_decel, 0.0)
        v_corpse = state.vel * (new_speed / jnp.maximum(speed, EPS))
        v_corpse = jnp.where(new_speed < cfg.corpse_stop_eps, 0.0, v_corpse)
        v_target = jnp.where(alive[:, None], v_next, v_corpse)
        return state.replace(v_pre=v_target, spin=spin, spin_cmd=spin_target,
                             target_cmd=jnp.where(alive, target_cmd, 0),
                             fire_cmd=jnp.where(alive, fire_cmd, False),
                             yaw_cmd=jnp.where(alive, yaw_cmd, 0.0)), v_target

    # ------------------------------------------------------------------
    def _surface_contact(self, state: RmState):
        """Wall/cover contact, outward normal and inward push speed."""
        cfg = self.cfg
        x, y = state.pos[:, 0], state.pos[:, 1]
        wall_x = ((x >= cfg.bound_x - cfg.robot_radius - 1e-3)
                  | (x <= -cfg.bound_x + cfg.robot_radius + 1e-3))
        wall_y = ((y >= cfg.bound_y - cfg.robot_radius - 1e-3)
                  | (y <= -cfg.bound_y + cfg.robot_radius + 1e-3))
        wall_now = wall_x | wall_y
        nx = ((x >= cfg.bound_x - cfg.robot_radius - 1e-3).astype(jnp.float32)
              - (x <= -cfg.bound_x + cfg.robot_radius + 1e-3).astype(jnp.float32))
        ny = ((y >= cfg.bound_y - cfg.robot_radius - 1e-3).astype(jnp.float32)
              - (y <= -cfg.bound_y + cfg.robot_radius + 1e-3).astype(jnp.float32))
        nn = jnp.maximum(jnp.sqrt(nx * nx + ny * ny), EPS)
        wall_normal = jnp.stack((nx / nn, ny / nn), axis=-1)
        wall_push = jnp.clip(jnp.sum(state.v_pre * wall_normal, axis=-1), 0.0) * wall_now.astype(jnp.float32)

        cover_now = jnp.zeros((cfg.n_agents,), dtype=jnp.bool_)
        cover_push = jnp.zeros((cfg.n_agents,), dtype=jnp.float32)
        for poly in self.cover_polys:
            _, dist, normal, inside = nearest_point_on_polygon(state.pos, poly)
            near = (dist <= cfg.robot_radius + 1e-3) | inside
            cover_now = cover_now | near
            push_i = jnp.clip(-jnp.sum(state.v_pre * normal, axis=-1), 0.0)
            cover_push = jnp.maximum(cover_push, jnp.where(near, push_i, 0.0))
        surface_now = wall_now | cover_now
        push = wall_push + cover_push
        return surface_now, push

    def _pair_contact(self, state: RmState):
        dmat = jnp.linalg.norm(state.pos[:, None, :] - state.pos[None, :, :], axis=-1)
        pair_now = dmat <= (2.0 * self.cfg.robot_radius + 1e-3)
        pair_now = pair_now & (~jnp.eye(self.cfg.n_agents, dtype=jnp.bool_))
        # n points from i to j (same convention as VMAS check_fouls).
        normal = (state.pos[None, :, :] - state.pos[:, None, :]) / jnp.maximum(dmat, EPS)[..., None]
        rel_v = state.v_pre[:, None, :] - state.v_pre[None, :, :]
        # Positive means the pair is closing (closing speed).
        approach = jnp.clip(jnp.sum(rel_v * normal, axis=-1), 0.0)
        return dmat, pair_now, normal, approach

    def _resolve_contacts(self, state: RmState) -> RmState:
        """Two non-foul separation rounds, matching VMAS resolve_contacts."""
        cfg = self.cfg
        r = cfg.robot_radius
        pos = state.pos
        vel = state.vel
        for round_idx in (0, 1):
            if round_idx == 0:
                vx = jnp.where(pos[:, 0] <= -cfg.bound_x + 1e-6,
                               jnp.maximum(vel[:, 0], 0.0), vel[:, 0])
                vx = jnp.where(pos[:, 0] >= cfg.bound_x - 1e-6,
                               jnp.minimum(vx, 0.0), vx)
                vy = jnp.where(pos[:, 1] <= -cfg.bound_y + 1e-6,
                               jnp.maximum(vel[:, 1], 0.0), vel[:, 1])
                vy = jnp.where(pos[:, 1] >= cfg.bound_y - 1e-6,
                               jnp.minimum(vy, 0.0), vy)
                vel = jnp.stack((vx, vy), axis=-1)
            for poly in self.cover_polys:
                q, dist, normal, inside = nearest_point_on_polygon(pos, poly)
                overlap = inside | (dist < r)
                target = q + r * normal
                pos = jnp.where(overlap[:, None], target, pos)
                if round_idx == 0:
                    vn = jnp.sum(vel * normal, axis=-1)
                    vel = jnp.where((overlap & (vn < 0))[:, None],
                                    vel - vn[:, None] * normal, vel)
            for i in range(cfg.n_agents):
                for j in range(i + 1, cfg.n_agents):
                    rel = pos[j] - pos[i]
                    d = jnp.linalg.norm(rel)
                    n = jnp.where(d > EPS, rel / jnp.maximum(d, EPS),
                                  jnp.array([1.0, 0.0]))
                    overlap = d < 2.0 * r
                    pen = jnp.maximum(2.0 * r - d, 0.0)
                    pos = pos.at[i].set(jnp.where(overlap, pos[i] - 0.5 * pen * n, pos[i]))
                    pos = pos.at[j].set(jnp.where(overlap, pos[j] + 0.5 * pen * n, pos[j]))
                    if round_idx == 0:
                        v_imp = jnp.sum((vel[i] - vel[j]) * n)
                        corr = jnp.maximum(v_imp, 0.0)
                        vel = vel.at[i].set(jnp.where(overlap, vel[i] - 0.5 * corr * n, vel[i]))
                        vel = vel.at[j].set(jnp.where(overlap, vel[j] + 0.5 * corr * n, vel[j]))
            pos = jnp.clip(pos, jnp.array([-cfg.bound_x, -cfg.bound_y]),
                           jnp.array([cfg.bound_x, cfg.bound_y]))
        return state.replace(pos=pos, vel=vel)

    def _check_collision_fouls(self, state: RmState) -> RmState:
        """Port of VMAS ``check_fouls`` (single environment).

        Fouls are based on the pre-collision velocity normal component, not
        on the agent's absolute speed.  The actual culprit(s) are determined
        by which side is closing faster; ties mark both sides.  Wall/cover
        impacts use the inward normal component.  This must run before
        ``_resolve_contacts`` so the pre-collision geometry is preserved.
        """
        cfg = self.cfg
        N = cfg.n_agents
        pos = state.pos
        vpre = state.v_pre
        r = cfg.robot_radius
        foul_team = jnp.zeros((2,), dtype=jnp.bool_)
        foul_agent = jnp.zeros((N,), dtype=jnp.bool_)
        foul_intensity = jnp.zeros((N,), dtype=jnp.float32)
        robot_denom = max(2.0 * cfg.v_max - cfg.robot_foul_speed, EPS)

        def add_team(ft, t, mask):
            return ft.at[t].set(ft[t] | jnp.any(mask & (self.team == t)))

        # 1. Wall impacts (inward normal component).
        wall_foul = jnp.zeros((N,), dtype=jnp.bool_)
        for axis, lo, hi in ((0, -cfg.bound_x, cfg.bound_x),
                             (1, -cfg.bound_y, cfg.bound_y)):
            at_lo = pos[:, axis] <= lo + 1e-6
            at_hi = pos[:, axis] >= hi - 1e-6
            imp_lo = jnp.clip(-vpre[:, axis], 0.0)
            imp_hi = jnp.clip(vpre[:, axis], 0.0)
            wall_foul = wall_foul | (at_lo & (imp_lo >= cfg.wall_foul_speed))
            wall_foul = wall_foul | (at_hi & (imp_hi >= cfg.wall_foul_speed))
        foul_agent = foul_agent | wall_foul
        for t in (0, 1):
            foul_team = add_team(foul_team, t, wall_foul)

        # 2. Cover impacts.
        cover_foul = jnp.zeros((N,), dtype=jnp.bool_)
        for poly in self.cover_polys:
            q, dist, normal, inside = nearest_point_on_polygon(pos, poly)
            overlap = inside | (dist < r)
            impact = jnp.clip(-jnp.sum(vpre * normal, axis=-1), 0.0)
            cover_foul = cover_foul | (overlap & (impact >= cfg.wall_foul_speed))
        foul_agent = foul_agent | cover_foul
        for t in (0, 1):
            foul_team = add_team(foul_team, t, cover_foul)

        # 3. Robot / corpse impacts.  Only the active (faster closing) side is
        # marked; exact ties mark both.  Dead agents can be hit but never foul.
        for i in range(N):
            for j in range(i + 1, N):
                rel = pos[j] - pos[i]
                d = jnp.linalg.norm(rel)
                n = jnp.where(d > EPS, rel / jnp.maximum(d, EPS),
                              jnp.array([1.0, 0.0]))
                v_imp = jnp.sum((vpre[i] - vpre[j]) * n)
                impact = jnp.maximum(v_imp, 0.0)
                overlap = d < 2.0 * r
                ai = state.alive[i]
                aj = state.alive[j]
                both = ai & aj
                one = ai ^ aj
                hit = overlap & both & (impact >= cfg.robot_foul_speed)
                ti = self.cfg.team(i)
                tj = self.cfg.team(j)

                # Both alive: compare each side's closing component.
                c_i = jnp.maximum(jnp.sum(vpre[i] * n), 0.0)
                c_j = jnp.maximum(jnp.sum(-vpre[j] * n), 0.0)
                tie = jnp.abs(c_i - c_j) <= cfg.foul_tie_tol
                foul_i = hit & (~tie) & (c_i > c_j)
                foul_j = hit & (~tie) & (c_j > c_i)
                foul_tie = hit & tie

                # Active side only.
                foul_agent = foul_agent.at[i].set(foul_agent[i] | foul_i)
                foul_agent = foul_agent.at[j].set(foul_agent[j] | foul_j)
                foul_team = add_team(foul_team, ti, foul_i)
                foul_team = add_team(foul_team, tj, foul_j)
                sev = jnp.clip((impact - cfg.robot_foul_speed) / robot_denom, 0.0, 1.0)
                foul_intensity = foul_intensity.at[i].set(
                    jnp.maximum(foul_intensity[i], jnp.where(foul_i, sev, 0.0)))
                foul_intensity = foul_intensity.at[j].set(
                    jnp.maximum(foul_intensity[j], jnp.where(foul_j, sev, 0.0)))

                # Exact tie: both sides pay, whether same team or not.
                foul_agent = foul_agent.at[i].set(foul_agent[i] | foul_tie)
                foul_agent = foul_agent.at[j].set(foul_agent[j] | foul_tie)
                foul_team = add_team(foul_team, ti, foul_tie)
                foul_team = add_team(foul_team, tj, foul_tie)
                foul_intensity = foul_intensity.at[i].set(
                    jnp.maximum(foul_intensity[i], jnp.where(foul_tie, sev, 0.0)))
                foul_intensity = foul_intensity.at[j].set(
                    jnp.maximum(foul_intensity[j], jnp.where(foul_tie, sev, 0.0)))

                # One alive vs a corpse: use only the living agent's closing
                # component (the corpse cannot commit a foul).
                imp_i = jnp.maximum(jnp.sum(vpre[i] * n), 0.0)
                imp_j = jnp.maximum(jnp.sum(-vpre[j] * n), 0.0)
                fi = overlap & one & ai & (imp_i >= cfg.robot_foul_speed)
                fj = overlap & one & aj & (imp_j >= cfg.robot_foul_speed)
                foul_agent = foul_agent.at[i].set(foul_agent[i] | fi)
                foul_agent = foul_agent.at[j].set(foul_agent[j] | fj)
                foul_team = add_team(foul_team, ti, fi)
                foul_team = add_team(foul_team, tj, fj)
                sev_i = jnp.clip((imp_i - cfg.robot_foul_speed) / robot_denom, 0.0, 1.0)
                sev_j = jnp.clip((imp_j - cfg.robot_foul_speed) / robot_denom, 0.0, 1.0)
                foul_intensity = foul_intensity.at[i].set(
                    jnp.maximum(foul_intensity[i], jnp.where(fi, sev_i, 0.0)))
                foul_intensity = foul_intensity.at[j].set(
                    jnp.maximum(foul_intensity[j], jnp.where(fj, sev_j, 0.0)))

        both = foul_team[0] & foul_team[1]
        red_only = foul_team[0] & (~foul_team[1])
        blue_only = foul_team[1] & (~foul_team[0])
        trigger = (foul_team[0] | foul_team[1]) & (~state.done)
        outcome = jnp.where(trigger,
                            jnp.where(both, 22, jnp.where(red_only, 10, 9)),
                            state.outcome)
        winner = jnp.where(trigger,
                           jnp.where(both, 0, jnp.where(red_only, 2, 1)),
                           state.winner)
        foul_loser = jnp.where(trigger,
                               jnp.where(both, -1, jnp.where(red_only, 0, 1)),
                               state.foul_loser)
        double_foul = jnp.where(trigger & both, True, state.double_foul)
        return state.replace(
            done=state.done | trigger,
            outcome=outcome,
            winner=winner,
            foul_loser=foul_loser,
            foul_agent=jnp.where(trigger, foul_agent, state.foul_agent),
            foul_intensity=jnp.where(trigger, foul_intensity, state.foul_intensity),
            double_foul=double_foul,
        )

    def _push_fouls(self, state: RmState) -> RmState:
        """Wall/cover and sustained robot push game-ending fouls."""
        cfg = self.cfg
        surface_now, push = self._surface_contact(state)
        _, pair_now, normal, _ = self._pair_contact(state)
        # wall/cover sustained push
        wall_push_time = jnp.where(
            surface_now & (push > cfg.wall_push_min),
            state.wall_push_time + cfg.dt, 0.0)
        wall_trigger = (wall_push_time >= cfg.wall_push_time) & (~state.done)
        red_wall = (wall_trigger & (self.team == 0)).any()
        blue_wall = (wall_trigger & (self.team == 1)).any()
        wall_row = red_wall | blue_wall
        wall_both = red_wall & blue_wall
        wall_out = jnp.where(wall_both, 26,
                             jnp.where(red_wall, 25, 24))
        state = state.replace(
            wall_push_time=wall_push_time,
            wall_push_foul_intent=jnp.where(wall_trigger, push / cfg.v_max,
                                             state.wall_push_foul_intent),
        )
        # robot-robot sustained push (living pusher, teammates/corpses included)
        other_present = state.alive[:, None] | (state.respawn_timer > 0.0)[:, None]
        v_into = jnp.sum(state.v_pre[:, None, :] * normal, axis=-1)
        body = pair_now & state.alive[:, None] & other_present & (v_into > cfg.body_push_min)
        body_any = body.any(axis=-1)
        body_time = jnp.where(body_any, state.body_push_time + cfg.dt, 0.0)
        body_trigger = (body_any & (body_time >= cfg.body_push_time)
                        & (~state.done) & (~wall_row))
        red_body = (body_trigger & (self.team == 0)).any()
        blue_body = (body_trigger & (self.team == 1)).any()
        body_row = red_body | blue_body
        body_both = red_body & blue_body
        body_out = jnp.where(body_both, 29,
                             jnp.where(red_body, 28, 27))
        # wall foul wins the opposite team; body likewise.
        outcome = jnp.where(wall_row, wall_out, state.outcome)
        outcome = jnp.where(body_row, body_out, outcome)
        winner = jnp.where(wall_row,
                           jnp.where(wall_both, 0, jnp.where(red_wall, 2, 1)),
                           state.winner)
        winner = jnp.where(body_row,
                           jnp.where(body_both, 0, jnp.where(red_body, 2, 1)),
                           winner)
        done = state.done | wall_row | body_row
        foul_loser = jnp.where(wall_row | body_row, -1, state.foul_loser)
        return state.replace(
            body_push_time=body_time,
            body_push_foul=state.body_push_foul | body_trigger,
            done=done, outcome=outcome, winner=winner, foul_loser=foul_loser)

    # ------------------------------------------------------------------
    def _proximity_penalty(self, state: RmState, dmat: Array, normal: Array):
        cfg = self.cfg
        near = dmat <= cfg.proximity_radius
        near = near & (~jnp.eye(cfg.n_agents, dtype=jnp.bool_))
        time_new = jnp.where(near, state.proximity_time + cfg.dt, 0.0)
        pen = jnp.zeros((cfg.n_agents,), dtype=jnp.float32)
        width = max(cfg.proximity_radius - 2.0 * cfg.robot_radius, EPS)
        for i in range(cfg.n_agents):
            for j in range(i + 1, cfg.n_agents):
                m = near[i, j]
                closeness = jnp.clip((cfg.proximity_radius - dmat[i, j]) / width, 0.0, 1.0)
                time_frac = jnp.clip(time_new[i, j] / max(cfg.proximity_ramp_time, EPS), 0.0, 1.0)
                total = ((cfg.k_proximity_penalty + cfg.k_proximity_ramp * time_frac)
                         * (0.2 + 0.8 * closeness) * m.astype(jnp.float32))
                n_ij = normal[i, j]
                v_i = jnp.clip(jnp.sum(state.v_pre[i] * n_ij), 0.0)
                v_j = jnp.clip(jnp.sum(state.v_pre[j] * (-n_ij)), 0.0)
                share_i = jnp.where(v_i > v_j, cfg.proximity_active_share,
                                    jnp.where(v_i == v_j, 0.5, cfg.proximity_passive_share))
                pen = pen.at[i].add(total * share_i)
                pen = pen.at[j].add(total * (1.0 - share_i))
        return pen, time_new

    # ------------------------------------------------------------------
    def get_action_mask(self, state: RmState) -> Array:
        """[N, 6] target mask: class 0 always legal, class k when enemy k visible."""
        cfg = self.cfg
        mask = jnp.zeros((cfg.n_agents, 4), dtype=jnp.bool_)
        pvis_all, fov_all = self._visibility_enemy_pairs(state, cfg.vis_samples_obs)
        for k in range(1, 4):
            for i in range(cfg.n_agents):
                j = int(cfg.enemy_indices(i)[k - 1])
                dist = jnp.linalg.norm(state.pos[j] - state.pos[i])
                ok = fov_all[i, k - 1] & (pvis_all[i, k - 1] > 0) & state.alive[j] & (dist < cfg.lock_range)
                mask = mask.at[i, k].set(ok)
        mask = jnp.where(state.alive[:, None], mask, False)
        mask = mask.at[:, 0].set(True)
        return mask

    def get_fire_mask(self, state: RmState) -> Array:
        # Releasing (0) is always legal.  Pressing (1) is only legal when a
        # target is locked AND the shooter can actually fire: alive, not in
        # respawn invulnerability, not overheated, and below max heat.
        cfg = self.cfg
        target_invuln = jnp.zeros((cfg.n_agents,), dtype=jnp.bool_)
        for i in range(cfg.n_agents):
            for k in range(3):
                j = int(cfg.enemy_indices(i)[k])
                target_invuln = target_invuln.at[i].set(
                    target_invuln[i] | ((state.target[i] == k + 1)
                                        & (state.invuln[j] > 0.0))
                )
        fire_ok = ((state.target != 0) & state.alive
                   & (state.invuln <= 0.0) & (~state.overheated)
                   & (state.heat < cfg.heat_max) & (~target_invuln))
        return jnp.stack((jnp.ones((cfg.n_agents,), dtype=jnp.bool_),
                          fire_ok), axis=-1)

    def step_env(self, key: Array, state: RmState, actions: Dict[str, Array],
                 deterministic: bool = False):
        cfg = self.cfg
        act = jnp.stack([actions[a] for a in self.agents])
        if deterministic:
            # Export/playback path: no PRNG primitives enter the traced graph.
            # Callers using this path must disable observation/cue noise.
            sub = key
        else:
            key, sub = jax.random.split(key)
        state, v = self._decode(state, act)
        score_before = state.score
        # integrate motion and keep robots inside the legal rectangle
        pos = state.pos + v * cfg.dt
        pos = jnp.clip(pos, jnp.array([-cfg.bound_x, -cfg.bound_y]),
                       jnp.array([cfg.bound_x, cfg.bound_y]))
        state = state.replace(pos=pos, vel=v)
        state = self._check_collision_fouls(state)
        # VMAS resolves contacts before finalizing the step, even when the
        # collision check just ended the episode.  Keep that ordering; the
        # post-done state matters for exact replay/parity, even though the
        # trainer masks it after the terminal transition.
        state = self._resolve_contacts(state)
        state = self._push_fouls(state)

        # target selection (candidate visibility uses the pre-turn yaw)
        prev_target = state.target
        new_target = jnp.zeros((cfg.n_agents,), dtype=jnp.int32)
        pvis_pre, fov_pre = self._visibility_enemy_pairs(state, cfg.vis_samples_obs)
        for k in range(1, 4):
            valid = jnp.zeros((cfg.n_agents,), dtype=jnp.bool_)
            for i in range(cfg.n_agents):
                j = int(cfg.enemy_indices(i)[k - 1])
                dist = jnp.linalg.norm(state.pos[j] - state.pos[i])
                valid = valid.at[i].set((state.target_cmd[i] == k) & fov_pre[i, k - 1]
                                        & (pvis_pre[i, k - 1] > 0) & state.alive[j]
                                        & (dist < cfg.lock_range))
            new_target = jnp.where(valid, k, new_target)
        state = state.replace(target=new_target)
        state = state.replace(p_t=jnp.where(new_target == prev_target, state.p_t, 0.0))

        # turret auto-track or manual yaw
        old_yaw = state.yaw
        bearing = jnp.zeros((cfg.n_agents,), dtype=jnp.float32)
        for i in range(cfg.n_agents):
            for k in range(3):
                j = int(cfg.enemy_indices(i)[k])
                rel = state.pos[j] - state.pos[i]
                bearing = bearing.at[i].set(jnp.where(state.target[i] == k + 1,
                                                      jnp.arctan2(rel[1], rel[0]), bearing[i]))
        err = wrap_angle(bearing - state.yaw)
        yaw_auto = wrap_angle(state.yaw + jnp.clip(err, -cfg.yaw_step_max, cfg.yaw_step_max))
        yaw_manual = wrap_angle(state.yaw + state.yaw_cmd * cfg.yaw_rate_max * cfg.dt)
        yaw = jnp.where((state.target != 0) & state.alive, yaw_auto, yaw_manual)
        yaw = jnp.where(state.alive, yaw, state.yaw)
        state = state.replace(yaw=yaw, yaw_rate=wrap_angle(yaw - old_yaw) / cfg.dt)

        # Post-turn lock survival (VMAS rm_rules.py): the new target only
        # survives if it is still geometrically visible after the turret turn.
        # Invulnerable targets keep the lock (pre-aim); dead targets drop it.
        pvis_post, fov_post = self._visibility_enemy_pairs(state, cfg.vis_samples_obs)
        lock_alive = jnp.zeros((cfg.n_agents,), dtype=jnp.bool_)
        for i in range(cfg.n_agents):
            keep_i = jnp.zeros((), dtype=jnp.bool_)
            for k in range(3):
                j = int(self.cfg.enemy_indices(i)[k])
                dist = jnp.linalg.norm(state.pos[j] - state.pos[i])
                keep_i = keep_i | (
                    (state.target[i] == k + 1)
                    & fov_post[i, k]
                    & (pvis_post[i, k] > 0.0)
                    & state.alive[j]
                    & (dist < cfg.lock_range)
                )
            lock_alive = lock_alive.at[i].set(keep_i)
        state = state.replace(
            target=jnp.where(lock_alive, state.target, 0).astype(jnp.int32),
            p_t=jnp.where(lock_alive, state.p_t, 0.0),
        )

        # heat cool
        home = jnp.stack([point_in_rect(state.pos[i], self.own_spawns[i]) for i in range(cfg.n_agents)])
        cool = jnp.where(state.alive & home, cfg.cool_per_frame * cfg.home_cool_mult,
                         cfg.cool_per_frame)
        heat = jnp.where(state.alive, jnp.maximum(state.heat - cool, 0.0), state.heat)
        state = state.replace(heat=heat,
                              overheated=state.overheated & ~(state.alive & (heat <= 0.0)),
                              last_fired=jnp.zeros((6,), dtype=jnp.bool_),
                              # PyTorch zeroes this at the start of every
                              # reward/fire pass; it is a one-frame display
                              # value, not a persistent "last shot" memory.
                              last_p_final=jnp.zeros((6,), dtype=jnp.float32))

        # lock survival & fire
        damage = jnp.zeros((cfg.n_agents, cfg.n_agents), dtype=jnp.float32)
        pvis_fire = self._visibility_selected(state, state.target, cfg.vis_samples_fire)
        for i in range(cfg.n_agents):
            for k in range(3):
                j = int(cfg.enemy_indices(i)[k])
                m = state.target[i] == k + 1
                # only the currently selected target needs the expensive
                # 64-sample firing visibility.
                pv = pvis_fire[i]
                d = jnp.linalg.norm(state.pos[j] - state.pos[i])
                p_dist = jnp.where(d <= 1.0, 1.0,
                                  jnp.where(d < cfg.lock_range,
                                            1.0 - ((d - 1.0) / (cfg.lock_range - 1.0)) ** 2, 0.0))
                p_self = 1.0 - 0.25 * jnp.clip(jnp.linalg.norm(state.vel[i]) / cfg.v_max, 0.0, 1.0)
                v_rel = state.vel[j] - state.vel[i]
                cross = (state.pos[j, 0] - state.pos[i, 0]) * v_rel[1] - (state.pos[j, 1] - state.pos[i, 1]) * v_rel[0]
                omega = jnp.abs(cross) / jnp.maximum(d, 2 * cfg.robot_radius) ** 2
                p_motion = 1.0 / (1.0 + (omega / 0.75) ** 2)
                p_base = p_dist * p_self * p_motion * pv
                p_base = jnp.where((d >= cfg.lock_range) | (pv <= 0.0), 0.0,
                                   jnp.maximum(p_base, cfg.p_floor))
                p_ideal = jnp.minimum(p_base, 1.0 - cfg.spin_cap_coef * state.spin[i])
                tau = cfg.tau_base * (1.0 + cfg.spin_tau_coef * state.spin[j])
                alpha = 1.0 - jnp.exp(-cfg.dt / tau)
                p_t = jnp.where(m, state.p_t[i] + alpha * (p_ideal - state.p_t[i]), state.p_t[i])
                p_final = p_t * (1.0 - cfg.spin_incoming_coef * state.spin[j])
                bearing_i = jnp.arctan2(state.pos[j, 1] - state.pos[i, 1],
                                        state.pos[j, 0] - state.pos[i, 0])
                fire = (m & state.alive[i] & (state.invuln[i] <= 0.0) & (~state.overheated[i])
                        & (heat[i] < cfg.heat_max) & state.alive[j] & (state.invuln[j] <= 0.0)
                        & (pv > 0.0) & (d < cfg.lock_range)
                        & (jnp.abs(wrap_angle(bearing_i - state.yaw[i])) < cfg.fire_angle)
                        & state.fire_cmd[i])
                dmg = jnp.where(fire, cfg.damage_per_frame * p_final, 0.0)
                damage = damage.at[i, j].set(dmg)
                new_heat = jnp.minimum(state.heat[i] + cfg.heat_per_frame, cfg.heat_max)
                state = state.replace(p_t=state.p_t.at[i].set(p_t),
                                      tau_eff=state.tau_eff.at[i].set(jnp.where(m, tau, state.tau_eff[i])),
                                      heat=state.heat.at[i].set(jnp.where(fire, new_heat, state.heat[i])),
                                      overheated=state.overheated.at[i].set(state.overheated[i] | (fire & (new_heat >= cfg.heat_max))),
                                      last_fired=state.last_fired.at[i].set(state.last_fired[i] | fire),
                                      last_p_final=state.last_p_final.at[i].set(jnp.where(fire, p_final, state.last_p_final[i])))
        # enemy spawn damage (before death resolution)
        spawn_in = jnp.stack([point_in_rect(state.pos[i], self.enemy_spawns[i]) for i in range(cfg.n_agents)])
        spawn_dmg = spawn_in & state.alive & (state.invuln <= 0.0)
        state = state.replace(hp=jnp.where(spawn_dmg,
                                           jnp.maximum(state.hp - cfg.spawn_damage_per_frame, 0.0),
                                           state.hp))
        # damage/hp/death
        hp_before = state.hp
        total_dmg = damage.sum(axis=0)
        actual_dmg = jnp.minimum(total_dmg, hp_before)
        hp = jnp.maximum(hp_before - total_dmg, 0.0)
        died = state.alive & (hp <= 0.0)
        kill_add = jnp.zeros((2,), dtype=jnp.float32)
        for j in range(cfg.n_agents):
            col = damage[:, j]
            killer = jnp.argmax(col)
            best = col[killer]
            kill = died[j] & (best > 0.0)
            kt = self.team[killer]
            kill_add = kill_add.at[kt].add(jnp.where(kill, cfg.kill_score, 0.0))
        # actual damage credit by team
        # total_damage: sum actual damage by attacker team (per victim split by raw share)
        td = state.total_damage
        for j in range(cfg.n_agents):
            denom = total_dmg[j]
            share = damage[:, j] / jnp.maximum(denom, EPS)
            # Tie-break damage must count all actual damage, not only killing
            # blows.  The previous `died[j]` gate made total_damage equal to
            # the last hit on a death and lost the timeout damage tie-break.
            for tt in (0, 1):
                td = td.at[tt].add(
                    (share * actual_dmg[j] * (self.team == tt)).sum()
                )
        kill_damage = state.kill_damage + damage
        # Team kill reward: every surviving teammate gets a fixed bonus for
        # each opponent killed.  No last-hit or damage-share credit is used.
        alive_after = state.alive & (~died)
        kill_count = kill_add / max(cfg.kill_score, EPS)
        last_credit = state.last_kill_credit
        for tt in (0, 1):
            last_credit = last_credit + (
                (alive_after & (self.team == tt)).astype(jnp.float32)
                * kill_count[tt] * cfg.k_kill_team_reward
            )
        # kill_damage is kept only for diagnostics; reset a victim's column on death.
        for j in range(cfg.n_agents):
            died_j = died[j] & (total_dmg[j] > 0.0)
            kill_damage = kill_damage.at[:, j].set(jnp.where(died_j, 0.0, kill_damage[:, j]))
        # v1.21 alignment: on the death frame, PyTorch clears heat,
        # overheat, spin, aim lock/p_t, cues and last_fired, and also clears
        # any shooter locked onto the dead target.  Without this, dead agents
        # keep target=3/heat=160 and diverge from VMAS after the first death.
        target = jnp.where(died, jnp.zeros_like(state.target), state.target)
        target_died = died
        for i in range(cfg.n_agents):
            for k in range(3):
                j = int(cfg.enemy_indices(i)[k])
                target_died = target_died.at[i].set(
                    target_died[i] | ((target[i] == k + 1) & died[j])
                )
        target = jnp.where(target_died, jnp.zeros_like(target), target)
        heat = jnp.where(died, jnp.zeros_like(state.heat), state.heat)
        p_t = jnp.where(target_died, jnp.zeros_like(state.p_t), state.p_t)
        state = state.replace(
            hp=hp, alive=state.alive & ~died, just_died=died,
            score=state.score + kill_add, total_damage=td,
            cum_damage=state.cum_damage + damage,
            kill_damage=kill_damage, last_kill_credit=last_credit,
            last_kill_score=kill_add, last_hit_damage=damage.sum(axis=1),
            last_damage=damage, heat=heat,
            overheated=state.overheated & ~died,
            spin=jnp.where(died, jnp.zeros_like(state.spin), state.spin),
            p_t=p_t, target=target,
            last_fired=state.last_fired & ~died,
            hit_cue_valid=state.hit_cue_valid & ~died,
            hit_cue=jnp.where(died, jnp.zeros_like(state.hit_cue), state.hit_cue),
            respawn_timer=jnp.where(died, cfg.respawn_time, state.respawn_timer),
            invuln=jnp.where(died, 0.0, state.invuln))
        # hit direction cue: strongest attacker, +/-45 deg, one frame
        total_dmg = damage.sum(axis=0)
        if deterministic:
            noises = jnp.zeros((cfg.n_agents,), dtype=jnp.float32)
        else:
            cue_key, sub = jax.random.split(sub)
            noises = jax.random.uniform(
                cue_key, (cfg.n_agents,),
                minval=-cfg.hit_cue_noise_deg, maxval=cfg.hit_cue_noise_deg,
            )
        noises = noises * jnp.pi / 180.0
        cue = state.hit_cue
        cue_valid = jnp.zeros((cfg.n_agents,), dtype=jnp.bool_)
        for j in range(cfg.n_agents):
            col = damage[:, j]
            atk = jnp.argmax(col)
            best = col[atk]
            valid = (total_dmg[j] > 1e-9) & (best > 1e-9)
            bearing = jnp.arctan2(state.pos[atk, 1] - state.pos[j, 1],
                                  state.pos[atk, 0] - state.pos[j, 0])
            cue = cue.at[j].set(jnp.where(valid, wrap_angle(bearing + noises[j]), cue[j]))
            cue_valid = cue_valid.at[j].set(valid)
        state = state.replace(hit_cue=cue, hit_cue_valid=cue_valid)
        # combat event for empty-game timer
        combat_event = state.last_fired.any() | (damage.sum() > 0.0)
        state = state.replace(
            stagnation_had_combat=state.stagnation_had_combat | combat_event,
            stagnation_last_event=jnp.where(combat_event, state.episode_elapsed,
                                            state.stagnation_last_event),
        )
        # respawn (match VMAS: do not tick the timer on the death frame,
        # clear all per-life combat state, and keep invuln_timer unchanged on
        # the revive frame before it starts counting down)
        dead = ~state.alive
        rt = jnp.where(state.just_died, state.respawn_timer,
                       jnp.maximum(state.respawn_timer - cfg.dt, 0.0))
        revive = dead & (~state.just_died) & (rt <= 0.0)
        revive2 = revive[:, None]
        state = state.replace(
            respawn_timer=jnp.where(revive, 0.0, jnp.where(dead, rt, 0.0)),
            alive=state.alive | revive,
            hp=jnp.where(revive, cfg.hp_revive, state.hp),
            heat=jnp.where(revive, 0.0, state.heat),
            overheated=state.overheated & (~revive),
            spin=jnp.where(revive, 0.0, state.spin),
            p_t=jnp.where(revive, 0.0, state.p_t),
            target=jnp.where(revive, 0, state.target),
            hit_cue_valid=state.hit_cue_valid & (~revive),
            vel=jnp.where(revive2, 0.0, state.vel),
            invuln=jnp.where(revive, 1.0, state.invuln),
            invuln_timer=jnp.where(revive, cfg.invuln_time, state.invuln_timer),
        )
        # invuln countdown and heat lock on expiry (reference v1.2)
        expired = (state.invuln > 0.5) & (~revive) & (state.invuln_timer > 0.0)
        inv = jnp.where(revive, cfg.invuln_time,
                        jnp.maximum(state.invuln_timer - cfg.dt, 0.0))
        over = state.overheated | (expired & (inv <= 0.0) & cfg.respawn_heat_lock)
        state = state.replace(
            invuln_timer=inv,
            invuln=(inv > 0.0).astype(jnp.float32),
            overheated=over,
            heat=jnp.where(expired & (inv <= 0.0) & cfg.respawn_heat_lock,
                           cfg.heat_max, state.heat),
        )
        # home healing (reference: heal zone in own spawn, not on revive frame)
        heal_in = jnp.stack([point_in_rect(state.pos[i], self.own_spawns[i]) for i in range(cfg.n_agents)])
        heal = state.alive & (~revive) & heal_in
        hp_before_heal = state.hp
        hp_healed = jnp.where(heal, jnp.minimum(state.hp + cfg.heal_per_frame, cfg.hp_max), state.hp)
        heal_event = (heal & (hp_healed > hp_before_heal)).any()
        state = state.replace(hp=hp_healed,
                              stagnation_had_heal=state.stagnation_had_heal | heal_event,
                              stagnation_last_event=jnp.where(heal_event, state.episode_elapsed,
                                                              state.stagnation_last_event))
        # centre occupancy and net score
        center_prev = state.center_dist_prev
        d_center = jnp.linalg.norm(state.pos, axis=-1)
        in_center = d_center <= cfg.center_radius
        occ = jnp.stack([jnp.sum(in_center & state.alive & (state.invuln <= 0.0) & (self.team == 0)),
                         jnp.sum(in_center & state.alive & (state.invuln <= 0.0) & (self.team == 1))]).astype(jnp.float32)
        net = occ[0] - occ[1]
        score = state.score + cfg.score_per_frame * jnp.stack([jnp.maximum(net, 0.0),
                                                               jnp.maximum(-net, 0.0)])
        time = jnp.maximum(state.time - cfg.dt, 0.0)
        state = state.replace(score=score, occ=occ, time=time,
                              episode_elapsed=state.episode_elapsed + cfg.dt,
                              center_dist_prev=d_center)
        # dense reward (non-contact terms)
        hit_r = damage.sum(axis=1) * cfg.k_hit_reward / cfg.hp_max
        center = jnp.linalg.norm(state.pos, axis=-1)
        center_r = cfg.k_center_reward * jnp.exp(
            -(center ** 2) / (2.0 * cfg.center_sigma ** 2)
        ) * (center > cfg.center_radius).astype(jnp.float32) * state.alive.astype(jnp.float32)
        center_progress = (
            cfg.k_center_progress * (center_prev - center)
            * state.alive.astype(jnp.float32)
        )
        in_point = in_center & state.alive & (state.invuln <= 0.0)
        occupy = cfg.k_occupy_reward * in_point.astype(jnp.float32)
        survival = cfg.k_survival_reward * (state.hp / cfg.hp_max) * state.alive.astype(jnp.float32)
        own_centers = jnp.stack([self.own_spawns[i].mean(axis=0) for i in range(6)])
        d_home = jnp.linalg.norm(state.pos - own_centers, axis=-1)
        low = (state.hp < cfg.low_hp_threshold * cfg.hp_max).astype(jnp.float32)
        home = (
            cfg.k_low_hp_home_reward
            * jnp.exp(-(d_home ** 2) / (2.0 * cfg.home_sigma ** 2))
            * low * state.alive.astype(jnp.float32)
        )
        in_own_spawn = jnp.stack([point_in_rect(state.pos[i], self.own_spawns[i]) for i in range(6)])
        vx, vy = state.vel[:, 0], state.vel[:, 1]
        outward_speed = jnp.where(
            self.team == 0, jnp.maximum(vx, -vy), jnp.maximum(-vx, vy)
        ) / cfg.v_max
        outward_speed = jnp.clip(outward_speed, -1.0, 1.0)
        outward = (
            cfg.k_outward_speed_reward * outward_speed
            * (state.hp >= cfg.hp_max - 1e-6).astype(jnp.float32)
            * in_own_spawn.astype(jnp.float32) * state.alive.astype(jnp.float32)
        )
        speed_now = jnp.linalg.norm(state.vel, axis=-1)
        elapsed = state.episode_elapsed
        camp = (
            (in_own_spawn | (center > cfg.center_radius))
            & (low < 0.5) & (elapsed > cfg.camp_grace_time)
            & (speed_now < cfg.camp_speed) & (state.target == 0) & state.alive
        )
        ramp = jnp.clip(
            (elapsed - cfg.camp_grace_time)
            / max(cfg.episode_time - cfg.camp_grace_time, 1e-6), 0.0, 1.0
        )
        camp_weight = cfg.k_camp_penalty + cfg.k_camp_ramp * ramp
        camp_pen = camp_weight * camp.astype(jnp.float32)
        death = -cfg.k_death_penalty * died.astype(jnp.float32)

        # Zero-sum centre shaping.  The old per-agent centre terms paid both
        # teams for standing in the ring, so two teams could farm a tie.  Pay
        # the team differential instead: the red share is +diff and the blue
        # share is -diff, split across that team's living agents.  The score
        # increment is also paid as a differential (scorer +, opponent -).
        team_f = self.team.astype(jnp.float32)
        alive_f = state.alive.astype(jnp.float32)
        centre_terms = center_r + center_progress + occupy
        centre_red = jnp.sum(centre_terms * alive_f * (team_f == 0.0))
        centre_blue = jnp.sum(centre_terms * alive_f * (team_f == 1.0))
        centre_diff = centre_red - centre_blue
        n_red = jnp.maximum(jnp.sum(alive_f * (team_f == 0.0)), 1.0)
        n_blue = jnp.maximum(jnp.sum(alive_f * (team_f == 1.0)), 1.0)
        centre_zero = (
            jnp.where(team_f == 0.0, centre_diff / n_red, -centre_diff / n_blue)
            * alive_f
        )
        score_delta = state.score - score_before - state.last_kill_score
        score_diff = score_delta[0] - score_delta[1]
        score_zero = (
            cfg.k_score_reward
            * jnp.where(team_f == 0.0, score_diff, -score_diff)
            * alive_f
        )
        dense = (hit_r + centre_zero + survival + home
                 + outward + death - camp_pen
                 + score_zero
                 + state.last_kill_credit
                 - cfg.k_fire_heat_penalty * state.last_fired.astype(jnp.float32))
        # contact shaping
        surface_now, push = self._surface_contact(state)
        dmat, pair_now, pair_normal, approach = self._pair_contact(state)
        new_wall = surface_now & (~state.prev_wall_contact)
        new_pair = pair_now & (~state.prev_pair_contact)
        wall_contact_time = jnp.where(surface_now, state.wall_contact_time + cfg.dt, 0.0)
        wall_frac = wall_contact_time / max(cfg.wall_contact_ramp_time, EPS)
        push_frac = jnp.clip(push / cfg.v_max, 0.0, 1.0)
        wall_income = (
            cfg.k_wall_impact * new_wall.astype(jnp.float32) * push_frac
            + cfg.k_wall_push * surface_now.astype(jnp.float32)
            * (push > cfg.wall_push_min).astype(jnp.float32) * push_frac
            + (cfg.k_wall_contact_penalty + cfg.k_wall_contact_ramp * wall_frac)
            * (0.2 + 0.8 * push_frac) * surface_now.astype(jnp.float32)
        )
        pair_pen = (
            0.5 * cfg.k_agent_impact * new_pair.astype(jnp.float32)
            * jnp.clip(approach / (2.0 * cfg.v_max), 0.0, 1.0)
        ).sum(axis=-1)
        prox_pen, prox_time = self._proximity_penalty(state, dmat, pair_normal)
        v_into = jnp.sum(state.v_pre[:, None, :] * pair_normal, axis=-1)
        body_push_any = (pair_now & state.alive[:, None]
                         & (v_into > cfg.body_push_min)).any(axis=-1)
        body_strength = jnp.max(jnp.where(pair_now, v_into, 0.0), axis=-1)
        body_pen = (
            (cfg.k_body_push_penalty + cfg.k_body_push_ramp
             * state.body_push_time / max(cfg.body_push_ramp_time, EPS))
            * (0.2 + 0.8 * jnp.clip(body_strength / (2.0 * cfg.v_max), 0.0, 1.0))
            * body_push_any.astype(jnp.float32)
        )
        contact_pen = wall_income + pair_pen + prox_pen + body_pen
        contact_now = new_pair.any(axis=-1) | new_wall
        spin = jnp.where(contact_now, 0.0, state.spin)
        dense = dense - contact_pen
        state = state.replace(done=state.done, wall_contact_time=wall_contact_time,
                              proximity_time=prox_time, prev_wall_contact=surface_now,
                              prev_pair_contact=pair_now, spin=spin)

        # empty-game timer / speed relief
        occ_event = occ.sum() > 0.0
        speed_now = jnp.clip(jnp.linalg.norm(state.vel, axis=-1) / max(cfg.v_max, EPS), 0.0, 1.0) * state.alive.astype(jnp.float32)
        alpha = min(cfg.dt / max(cfg.stagnation_speed_ema_time, EPS), 1.0)
        event_now = combat_event | heal_event | occ_event
        stagnation_speed = jnp.where(event_now,
                                     speed_now,
                                     (1.0 - alpha) * state.stagnation_speed + alpha * speed_now)
        inactive = (state.episode_elapsed - state.stagnation_last_event) >= cfg.stagnation_time
        stagnant = (~state.done) & inactive
        state = state.replace(
            stagnation_had_occ=state.stagnation_had_occ | occ_event,
            stagnation_last_event=jnp.where(occ_event, state.episode_elapsed,
                                            state.stagnation_last_event),
            stagnation_speed=stagnation_speed,
            done=state.done | stagnant,
            outcome=jnp.where(stagnant, 23, state.outcome),
            winner=jnp.where(stagnant, 0, state.winner),
            foul_loser=jnp.where(stagnant, -1, state.foul_loser),
        )
        # terminal / winner: preserve collision/push outcomes set earlier
        prior_done = state.done
        done_target = (state.score >= cfg.target_score).any() & (~prior_done)
        done_time = (time <= 0.0) & (~prior_done)
        # VMAS convention: winner 0=draw, 1=red, 2=blue.  The previous
        # 0/1 encoding made `winner - 1` point at no team for a red win,
        # so target/time wins silently received no terminal reward.
        # VMAS tie-break: score, then actual damage, then surviving HP, then draw.
        score_red = state.score[0]
        score_blue = state.score[1]
        # PyTorch special-cases simultaneous 200: draw_200, then both get +2.
        both200 = (score_red >= cfg.target_score) & (score_blue >= cfg.target_score) & (~prior_done)
        score_eq = score_red == score_blue
        dmg_rd = score_eq & (state.total_damage[0] > state.total_damage[1])
        dmg_bd = score_eq & (state.total_damage[1] > state.total_damage[0])
        dmg_eq = score_eq & (state.total_damage[0] == state.total_damage[1])
        team_f = self.team.astype(jnp.float32)
        hp_red = jnp.sum(jnp.where(state.alive & (team_f == 0.0), state.hp, 0.0))
        hp_blue = jnp.sum(jnp.where(state.alive & (team_f == 1.0), state.hp, 0.0))
        hp_rd = dmg_eq & (hp_red > hp_blue)
        hp_bd = dmg_eq & (hp_blue > hp_red)
        target_winner = jnp.where(both200, 0,
                                   jnp.where(score_red > score_blue, 1,
                                             jnp.where(score_blue > score_red, 2, 0)))
        target_outcome = jnp.where(
            done_target,
            jnp.where(both200, 20,
                       jnp.where(target_winner == 1, 1,
                                 jnp.where(target_winner == 2, 2, 20))),
            jnp.where(done_time, jnp.where(score_red > score_blue, 3,
                                            jnp.where(score_blue > score_red, 4,
                                                       jnp.where(dmg_rd, 5,
                                                                 jnp.where(dmg_bd, 6,
                                                                           jnp.where(hp_rd, 7,
                                                                                     jnp.where(hp_bd, 8, 20)))))), 0),
        )
        time_winner = jnp.where(score_red > score_blue, 1,
                                 jnp.where(score_blue > score_red, 2,
                                           jnp.where(dmg_rd, 1,
                                                     jnp.where(dmg_bd, 2,
                                                               jnp.where(hp_rd, 1,
                                                                         jnp.where(hp_bd, 2, 0))))))
        winner = jnp.where(prior_done, state.winner,
                           jnp.where(done_target, target_winner,
                                     jnp.where(done_time, time_winner, state.winner)))
        outcome = jnp.where(prior_done, state.outcome, target_outcome)
        done = prior_done | done_target | done_time

        # VMAS returns early on a target-score finish, so the clock is NOT
        # decremented again on that terminal frame.  JAX previously always
        # subtracted dt first, making the final time and time-based win
        # reward differ by one tick.
        time = jnp.where(done_target & (~prior_done),
                         jnp.minimum(state.time + cfg.dt, cfg.episode_time),
                         state.time)
        state = state.replace(time=time, both_200=state.both_200 | both200)

        # terminal reward
        team = self.team
        wall_foul = (outcome == 24) | (outcome == 25) | (outcome == 26)
        body_foul = (outcome == 27) | (outcome == 28) | (outcome == 29)
        collision_foul = prior_done & state.foul_agent.any() & (~wall_foul) & (~body_foul)
        special = wall_foul | body_foul | collision_foul | stagnant
        win_side = team == (winner - 1)
        time_frac = jnp.clip(time / max(cfg.episode_time, EPS), 0.0, 1.0)
        winner_score = jnp.where(winner == 1, state.score[0], state.score[1])
        loser_score = jnp.where(winner == 1, state.score[1], state.score[0])
        margin_frac = jnp.clip((winner_score - loser_score) / max(cfg.target_score, EPS), 0.0, 1.0)
        win_r = cfg.terminal_win + cfg.k_terminal_time * time_frac + cfg.k_terminal_margin * margin_frac
        normal = done & (winner != 0) & (~special)
        term = jnp.where(normal,
                         jnp.where(win_side, win_r, cfg.terminal_loss), 0.0)
        term = jnp.where(done & (winner == 0) & (~special), cfg.terminal_draw, term)
        zero_timeout = (
            done
            & (winner == 0)
            & (outcome == 20)
            & (state.score[0] == 0.0)
            & (state.score[1] == 0.0)
        )
        term = jnp.where(
            zero_timeout,
            cfg.terminal_zero_draw,
            term,
        )
        term = jnp.where(done & both200, cfg.terminal_both_200, term)
        # collision: only the culprit pays -10 - 3*intensity
        foul_pen = cfg.terminal_foul - cfg.k_foul_intensity * state.foul_intensity
        culprit = collision_foul & state.foul_agent
        term = jnp.where(culprit, foul_pen, term)
        # wall/body push: pusher -10, same-team non-pusher -0.5, opponent 0
        pusher = jnp.where(body_foul, state.body_push_foul,
                           jnp.where(wall_foul, state.wall_push_foul_intent > 0.0, False))
        pusher_team = jnp.stack([
            (pusher & (team == 0)).any(), (pusher & (team == 1)).any()
        ])
        teammate_pen = cfg.terminal_push_foul * 0.0 - cfg.body_push_teammate_penalty
        term = jnp.where((wall_foul | body_foul) & pusher,
                         cfg.terminal_push_foul, term)
        term = jnp.where((wall_foul | body_foul) & (~pusher)
                         & pusher_team[team], teammate_pen, term)
        # keep opponent at zero for push fouls
        term = jnp.where((wall_foul | body_foul) & (~pusher) & (~pusher_team[team]),
                         0.0, term)
        # empty-game terminal: -stagnation_penalty with clearance/spawn/centre/speed relief
        clearance = jnp.clip(
            jnp.min(self._obstacles(state), axis=-1) * cfg.obstacle_max_range
            / max(cfg.stagnation_clearance_ref, EPS), 0.0, 1.0)
        spawn_far = jnp.clip(
            jnp.linalg.norm(state.pos - jnp.stack([self.own_spawns[i].mean(axis=0) for i in range(6)]), axis=-1)
            / max(cfg.stagnation_spawn_ref, EPS), 0.0, 1.0)
        alive_f = state.alive.astype(jnp.float32)
        centers = []
        for tt in (0, 1):
            mask = (self.team == tt).astype(jnp.float32) * alive_f
            count = mask.sum()
            cent = (state.pos * mask[:, None]).sum(axis=0) / jnp.maximum(count, 1.0)
            centers.append(jnp.where(count > 0.5, jnp.linalg.norm(cent), 1e3))
        d_red, d_blue = centers[0], centers[1]
        denom = jnp.maximum(d_red + d_blue, EPS)
        red_adv = jnp.clip((d_blue - d_red) / denom, 0.0, 1.0)
        blue_adv = jnp.clip((d_red - d_blue) / denom, 0.0, 1.0)
        team_factor = jnp.where(self.team == 0, 1.0 - cfg.stagnation_center_discount * red_adv,
                                1.0 - cfg.stagnation_center_discount * blue_adv)
        speed_factor = 1.0 - cfg.stagnation_speed_discount * jnp.clip(state.stagnation_speed, 0.0, 1.0)
        stag_pen = (cfg.stagnation_penalty
                    * (1.0 - cfg.stagnation_clearance_discount * clearance)
                    * (1.0 - cfg.stagnation_spawn_discount * spawn_far)
                    * team_factor * speed_factor)
        term = jnp.where(stagnant, -stag_pen, term)
        # v1.21 alignment: PyTorch zeroes the entire dense reward for dead
        # agents (including kill credit and contact shaping) and gives only
        # -k_death_penalty on the death frame.
        dense = jnp.where(state.alive, dense, 0.0) + jnp.where(
            died, -cfg.k_death_penalty, 0.0)
        state = state.replace(done=done, outcome=outcome, winner=winner,
                              reward_dense=dense, reward_terminal=term,
                              last_kill_credit=jnp.zeros_like(state.last_kill_credit))
        pvis_obs, fov_obs = self._visibility_enemy_pairs(state, cfg.vis_samples_obs)
        obs = self.get_obs(state, sub, pvis_obs=pvis_obs, fov_obs=fov_obs)
        rewards = {a: dense[i] + term[i] for i, a in enumerate(self.agents)}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        info = {"outcome": outcome, "score": state.score, "time": time,
                "winner": winner, "kill_score": state.last_kill_score}
        return obs, state, rewards, dones, info
