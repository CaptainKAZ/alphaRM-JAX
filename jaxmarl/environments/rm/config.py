"""Static configuration for the RM 3v3 environment.

The values mirror ``VectorizedMultiAgentSimulator/vmas/scenarios/rm_config.py``
(the PyTorch reference).  This file intentionally contains no simulation logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Tuple


def rect(x0: float, y0: float, x1: float, y1: float) -> Tuple[Tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


@dataclass(frozen=True)
class RmConfig:
    # time / field
    dt: float = 0.1
    episode_time: float = 180.0
    field_hx: float = 6.0
    field_hy: float = 4.0
    bound_x: float = 5.7
    bound_y: float = 3.7

    # robots
    n_agents: int = 6
    n_per_team: int = 3
    robot_radius: float = 0.30
    occlusion_radius: float = 0.35
    v_max: float = 3.0
    a_max: float = 5.0
    hp_max: float = 325.0
    relative_actions: bool = False

    # turret / spin
    yaw_rate_max: float = 270.0 * pi / 180.0
    yaw_step_max: float = 27.0 * pi / 180.0
    spin_time: float = 1.0
    spin_follow_rate: float = 1.0
    spin_speed_coef: float = 0.5
    spin_cap_coef: float = 0.1
    spin_incoming_coef: float = 0.4
    spin_tau_coef: float = 0.5

    # perception / fire
    fov_half: float = 45.0 * pi / 180.0
    # Locked-target FOV is slightly narrower (80 deg total), matching VMAS.
    fov_half_narrow: float = 40.0 * pi / 180.0
    lock_range: float = 12.0
    vis_samples_obs: int = 16
    vis_samples_fire: int = 64
    obstacle_rays: int = 8
    obstacle_max_range: float = 16.0
    hp_noise_base: float = 10.0
    hp_noise_spin: float = 5.0
    yaw_noise_deg: float = 5.0
    hit_cue_noise_deg: float = 45.0
    obs_noise: bool = True
    tau_base: float = 0.4
    p_floor: float = 0.02

    # weapon / heat
    heat_max: float = 160.0
    heat_per_frame: float = 15.0
    cool_per_frame: float = 2.0
    home_cool_mult: float = 2.0
    damage_per_frame: float = 30.0
    fire_angle: float = 1.0 * pi / 180.0

    # death / respawn / heal
    respawn_time: float = 5.0
    respawn_invuln: bool = True
    respawn_heat_lock: bool = True
    invuln_time: float = 3.0
    hp_revive: float = 65.0
    heal_per_frame: float = 8.125
    spawn_damage_per_frame: float = 8.125

    # contest
    wall_foul_speed: float = 2.5
    robot_foul_speed: float = 2.0
    # random / late-game reset
    rsi_symmetric_prob: float = 0.5
    rsi_min_sep: float = 1.0
    rsi_time_min: float = 30.0
    rsi_time_max: float = 180.0
    rsi_late_game: bool = True
    # Full 180 s horizon; random episodes keep 80–150 s on the clock.
    rsi_remaining_min: float = 80.0
    rsi_remaining_max: float = 150.0
    rsi_hp_min: float = 65.0
    rsi_hp_max: float = 325.0
    rsi_spin_max: float = 0.5
    rsi_heat_max: float = 80.0
    spawn_anchor_init: bool = True
    spawn_anchor_jitter: float = 0.45
    foul_tie_tol: float = 0.1
    corpse_decel: float = 0.5
    corpse_stop_eps: float = 0.01

    # score
    center_radius: float = 1.5
    score_per_frame: float = 0.1
    kill_score: float = 10.0
    target_score: float = 90.0

    # fixed spawns
    red_slots: Tuple[Tuple[float, float], ...] = (
        (-5.35, 3.30), (-5.35, 2.20), (-4.50, 2.75),
    )
    blue_slots: Tuple[Tuple[float, float], ...] = (
        (5.35, -3.30), (5.35, -2.20), (4.50, -2.75),
    )
    # Training-only rotation of the fixed three-agent triangle around its
    # centroid.  Defaults to off so strict eval remains unchanged.
    fixed_spawn_rotate: bool = False
    fixed_spawn_rotate_max: float = pi

    # static cover (CCW polygons)
    red_l: Tuple[Tuple[float, float], ...] = (
        (-3.50, -1.00), (-3.30, -1.00), (-3.30, 2.86),
        (-1.50, 2.86), (-1.50, 3.06), (-3.50, 3.06),
    )
    blue_l: Tuple[Tuple[float, float], ...] = (
        (3.50, 1.00), (3.30, 1.00), (3.30, -2.86),
        (1.50, -2.86), (1.50, -3.06), (3.50, -3.06),
    )
    red_short: Tuple[Tuple[float, float], ...] = rect(-1.65, -0.10, -1.50, 1.70)
    blue_short: Tuple[Tuple[float, float], ...] = rect(1.50, -1.70, 1.65, 0.10)
    red_spawn: Tuple[Tuple[float, float], ...] = rect(-6.00, 1.50, -3.50, 4.00)
    blue_spawn: Tuple[Tuple[float, float], ...] = rect(3.50, -4.00, 6.00, -1.50)

    # training-only reward scales
    k_score_reward: float = 0.01
    terminal_win: float = 6.0
    terminal_loss: float = -6.0
    terminal_foul: float = -10.0
    terminal_foul_win: float = 0.0
    k_foul_intensity: float = 3.0
    terminal_push_foul: float = -10.0
    terminal_draw: float = 0.0
    terminal_zero_draw: float = -5.0   # 0:0 timeout: team total -5 (split across 3 agents)
    terminal_both_200: float = 2.0
    terminal_both_foul: float = -10.0
    k_terminal_time: float = 3.0
    k_terminal_margin: float = 3.0
    k_hit_reward: float = 0.05
    k_center_reward: float = 0.003
    k_center_progress: float = 0.02
    center_sigma: float = 1.5
    k_camp_penalty: float = 0.0005
    k_camp_ramp: float = 0.0015
    camp_grace_time: float = 8.0
    camp_speed: float = 0.25
    k_wall_contact_penalty: float = 0.001
    k_wall_contact_ramp: float = 0.004
    wall_contact_ramp_time: float = 2.0
    k_body_push_penalty: float = 0.001
    k_body_push_ramp: float = 0.004
    body_push_ramp_time: float = 2.0
    body_push_min: float = 0.3
    body_push_time: float = 0.5
    body_push_teammate_penalty: float = 0.5
    proximity_radius: float = 0.65
    k_proximity_penalty: float = 0.0025
    k_proximity_ramp: float = 0.0025
    proximity_ramp_time: float = 2.0
    proximity_active_share: float = 0.6
    proximity_passive_share: float = 0.4
    stagnation_time: float = 10.0
    stagnation_penalty: float = 5.0
    stagnation_clearance_discount: float = 0.5
    stagnation_clearance_ref: float = 4.0
    stagnation_speed_discount: float = 0.5
    stagnation_speed_ema_time: float = 2.0
    stagnation_center_discount: float = 0.5
    stagnation_spawn_discount: float = 0.5
    stagnation_spawn_ref: float = 6.0
    k_outward_speed_reward: float = 0.002
    wall_push_threshold: float = 0.05
    wall_push_time: float = 1.0
    wall_push_teammate_penalty: float = 0.5
    k_survival_reward: float = 0.00005
    k_death_penalty: float = 0.2
    k_wall_impact: float = 0.02
    k_wall_push: float = 0.001
    k_agent_impact: float = 0.3
    wall_push_min: float = 0.2
    k_fire_heat_penalty: float = 0.001
    k_occupy_reward: float = 0.002
    k_kill_team_reward: float = 0.1          # per kill, paid to every surviving teammate
    k_kill_reward_pool: float = 0.0           # legacy: damage-split kill pool (disabled)
    k_low_hp_home_reward: float = 0.0005
    low_hp_threshold: float = 0.5
    home_sigma: float = 2.0

    # observation/state dimensions of the reference implementation
    obs_dim: int = 140
    global_dim: int = 191
    obstacle_dim: int = 8

    @property
    def covers(self):
        return (self.red_l, self.blue_l, self.red_short, self.blue_short)

    def team(self, i: int) -> int:
        return 0 if i < self.n_per_team else 1

    def teammate_indices(self, i: int):
        if i < self.n_per_team:
            return tuple(j for j in range(self.n_per_team) if j != i)
        return tuple(j for j in range(self.n_per_team, self.n_agents) if j != i)

    def enemy_indices(self, i: int):
        if i < self.n_per_team:
            return tuple(range(self.n_per_team, self.n_agents))
        return tuple(range(self.n_per_team))


DEFAULT_CONFIG = RmConfig()
