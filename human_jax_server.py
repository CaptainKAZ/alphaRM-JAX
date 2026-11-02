#!/usr/bin/env python3
"""Playable browser RM game backed by the latest JAX actor checkpoint.

Run inside the GPU container:
  PYTHONPATH=/workspace/alphaRM-JAX \
    python /workspace/alphaRM-JAX/human_jax_server.py --port 8767

The browser UI is demo/human_game.html. The human controls one robot;
the other five are driven by the loaded policy. Hidden opponents are not sent
to the browser at all.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.actor import RmActor, act
from jaxmarl.environments.rm.config import RmConfig
from jaxmarl.environments.rm.env import RmEnv

HTML_PATH = ROOT / "demo" / "human_game.html"
NAMES = ("R1", "R2", "R3", "B1", "B2", "B3")
OUTCOME_NAMES = {
    0: "进行中", 1: "红方先到目标分", 2: "蓝方先到目标分",
    3: "红方时间胜", 4: "蓝方时间胜", 5: "红方伤害tiebreak",
    6: "蓝方伤害tiebreak", 7: "红方血量tiebreak",
    8: "蓝方血量tiebreak", 9: "红方胜(蓝方碰撞犯规)",
    10: "蓝方胜(红方碰撞犯规)", 20: "平局(时间tiebreak)",
    21: "平局(双方同帧200)", 22: "平局(双方同帧犯规)",
    23: "空局(10秒无事件)", 24: "红方胜(蓝方推墙/掩体犯规)",
    25: "蓝方胜(红方推墙/掩体犯规)", 26: "平局(双方推墙/掩体犯规)",
    27: "红方胜(蓝方推人犯规)", 28: "蓝方胜(红方推人犯规)",
    29: "平局(双方推人犯规)",
}
ENEMY_OBS_START = 24 + 4 + 4 + 2 * 18
ENEMY_OBS_DIM = 17


class Game:
    def __init__(self, ckpt: str, seed: int = 0, mode: str = "fixed", human: int = 0):
        with open(ckpt, "rb") as f:
            ck = pickle.load(f)
        if "actor_params" not in ck:
            raise ValueError(f"checkpoint has no actor_params: {ckpt}")
        self.ckpt_iteration = int(ck.get("iteration", -1))
        self.params = jax.device_put(ck["actor_params"])
        self.actor = RmActor()
        cfg = RmConfig(
            robot_foul_speed=1.0e9,
            wall_foul_speed=1.0e9,
            wall_push_time=1.0e9,
            body_push_time=1.0e9,
        )
        self.env = RmEnv(cfg)
        self.cfg = cfg
        self.seed = int(seed)
        self.mode = mode
        self.human = max(0, min(5, int(human)))
        self.keys = {"w": False, "a": False, "s": False, "d": False,
                     "q": False, "e": False, "shift": False, "f": False}
        self.ai = True
        self.mouse_down = False
        self.lock_down = False
        self.fire_down = False
        self.aim_angle = 0.0
        self.target = 0
        self.paused = False
        self.running = True
        self.done = False
        self.events = deque(maxlen=30)
        self.lock = threading.Lock()
        self.key = jax.random.PRNGKey(self.seed)
        self.obs = None
        self.state = None
        self.h = jnp.zeros((6, 256), dtype=jnp.float32)
        self.action_masks = None
        self.fire_masks = None
        self._snapshot = {}
        self._reset_locked()
        self._step_jit = jax.jit(self._step_core)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def _reset_locked(self):
        self.key, rk = jax.random.split(self.key)
        mode = self.mode
        if mode not in ("fixed", "random", "curriculum"):
            mode = "fixed" if mode == "fixed" else "random"
        obs_d, state = self.env.reset(rk, mode=mode)
        self.obs = jnp.stack([obs_d[a] for a in self.env.agents], axis=0)
        self.state = state
        self.h = jnp.zeros((6, 256), dtype=jnp.float32)
        self.action_masks = self.env.get_action_mask(state)
        self.fire_masks = self.env.get_fire_mask(state)
        self.target = 0
        self.mouse_down = False
        self.lock_down = False
        self.fire_down = False
        self.done = False
        self.paused = False
        self.events.clear()
        self._event("开始", f"JAX ckpt {self.ckpt_iteration} · {mode} · {NAMES[self.human]}")
        self._build_snapshot_locked()

    def _event(self, kind, text):
        self.events.appendleft({
            "t": round(float(self.state.time), 1),
            "kind": kind, "text": text,
        })

    def _human_action(self, masks):
        a = np.zeros(6, dtype=np.float32)
        s = self.state
        i = self.human
        sign = 1.0 if self.cfg.team(i) == 0 else -1.0
        vx = (1.0 if self.keys["d"] else 0.0) - (1.0 if self.keys["a"] else 0.0)
        vy = (1.0 if self.keys["w"] else 0.0) - (1.0 if self.keys["s"] else 0.0)
        n = math.hypot(vx, vy)
        if n > 1.0:
            vx, vy = vx / n, vy / n
        # Decode applies the team sign; compensate so WASD is world-oriented.
        a[0], a[1] = vx * sign, vy * sign
        a[3] = 1.0 if self.keys["shift"] else 0.0
        # Right mouse selects the visible enemy whose bearing is closest to
        # the pointer. The lock persists after release. Left mouse is fire.
        if self.target > 0 and not bool(masks[i, self.target]):
            self.target = 0
        if self.lock_down:
            best = 0
            best_err = math.pi + 1.0
            p = np.asarray(s.pos[i], dtype=np.float32)
            for slot, enemy in enumerate(self.cfg.enemy_indices(i), start=1):
                if not bool(masks[i, slot]):
                    continue
                rel = np.asarray(s.pos[enemy], dtype=np.float32) - p
                angle = math.atan2(float(rel[1]), float(rel[0]))
                err = abs((angle - float(self.aim_angle) + math.pi)
                          % (2.0 * math.pi) - math.pi)
                if err < best_err:
                    best_err = err
                    best = slot
            self.target = best
        if self.target > 0:
            a[2] = 0.0
            a[4] = float(self.target)
            a[5] = 1.0 if self.fire_down else 0.0
        else:
            err = (float(self.aim_angle) - float(s.yaw[i]) + math.pi) \
                % (2.0 * math.pi) - math.pi
            a[2] = float(np.clip(err / (self.cfg.yaw_rate_max * self.cfg.dt), -1.0, 1.0))
            a[4] = 0.0
            a[5] = 0.0
        return jnp.asarray(a)

    def _step_core(self, params, obs, h, state, human_action, human_idx,
                   ai_enabled, key):
        masks = self.env.get_action_mask(state)
        fire = self.env.get_fire_mask(state)
        ai_actions, h_next = act(
            self.actor, params, h, obs, jax.random.split(key, 6),
            masks, fire, deterministic=True,
        )
        actions = jnp.where(ai_enabled, ai_actions, jnp.zeros_like(ai_actions))
        actions = actions.at[human_idx].set(human_action)
        action_dict = {
            agent: actions[i] for i, agent in enumerate(self.env.agents)
        }
        obs_d, nstate, _, dones, _ = self.env.step_env(key, state, action_dict)
        nobs = jnp.stack([obs_d[a] for a in self.env.agents], axis=0)
        return (
            nstate,
            nobs,
            h_next,
            self.env.get_action_mask(nstate),
            self.env.get_fire_mask(nstate),
            dones["__all__"],
        )

    def step_once(self, force: bool = False):
        with self.lock:
            if self.done or (self.paused and not force):
                return
            try:
                masks = self.action_masks
                self.key, sk = jax.random.split(self.key)
                human_action = self._human_action(np.asarray(masks))
                nstate, nobs, h_next, next_masks, next_fire, done = self._step_jit(
                    self.params, self.obs, self.h, self.state,
                    human_action, jnp.asarray(self.human, dtype=jnp.int32),
                    jnp.asarray(self.ai), sk,
                )
                jax.block_until_ready((nstate, h_next, done))
                self.state = nstate
                self.h = h_next
                self.obs = nobs
                self.action_masks = next_masks
                self.fire_masks = next_fire
                self.done = bool(np.asarray(done).item())
                if self.done:
                    self._event("终局", self.outcome_name())
                self._build_snapshot_locked()
            except Exception:
                self.done = True
                tb = traceback.format_exc()
                print(tb, flush=True)
                self._event("错误", tb.splitlines()[-1])

    def loop(self):
        while self.running:
            t0 = time.time()
            self.step_once()
            time.sleep(max(0.0, self.cfg.dt - (time.time() - t0)))

    def reset(self, mode=None, seed=None, human=None):
        with self.lock:
            if mode is not None:
                self.mode = mode
            if seed is not None:
                self.seed = int(seed)
                self.key = jax.random.PRNGKey(self.seed)
            if human is not None:
                self.human = max(0, min(5, int(human)))
            self._reset_locked()

    def outcome_name(self):
        return OUTCOME_NAMES.get(int(np.asarray(self.state.outcome).item()), "进行中")

    def _build_snapshot_locked(self):
        s = self.state
        mask = np.asarray(self.action_masks, dtype=bool)
        h = self.human
        robots = []
        obs_view = np.asarray(self.obs)
        for i in range(6):
            same_team = int(self.cfg.team(i)) == int(self.cfg.team(h))
            if i == h or same_team:
                visible = True
            else:
                slot = self.cfg.enemy_indices(h).index(i)
                # Display visibility follows the FOV/corridor observation,
                # not the target mask. A visible corpse is still rendered.
                visible = bool(
                    obs_view[h, ENEMY_OBS_START + ENEMY_OBS_DIM * slot] > 0.5
                )
            if not visible:
                # Security boundary: hidden opponents never reach the page.
                robots.append({
                    "name": NAMES[i], "team": int(self.cfg.team(i)),
                    "visible": False,
                })
                continue
            if i != h and not same_team:
                base = ENEMY_OBS_START + ENEMY_OBS_DIM * slot
                sign = 1.0 if self.cfg.team(h) == 0 else -1.0
                abs_obs = obs_view[h, base + 3:base + 5] * np.asarray(
                    [self.cfg.field_hx, self.cfg.field_hy], dtype=np.float32
                )
                world_abs = sign * abs_obs
                rvel = obs_view[h, base + 5:base + 7] * (2.0 * self.cfg.v_max)
                dyaw = math.atan2(float(obs_view[h, base + 9]),
                                  float(obs_view[h, base + 10]))
                robots.append({
                    "name": NAMES[i],
                    "team": int(self.cfg.team(i)),
                    "x": round(float(world_abs[0]), 4),
                    "y": round(float(world_abs[1]), 4),
                    "vx": round(float(s.vel[h, 0] + sign * rvel[0]), 4),
                    "vy": round(float(s.vel[h, 1] + sign * rvel[1]), 4),
                    "yaw": round(float(s.yaw[h] + dyaw), 5),
                    "hp": round(float(obs_view[h, base + 7] * self.cfg.hp_max), 2),
                    "heat": None,
                    "heat_known": False,
                    "overheated": False,
                    "spin": round(float(obs_view[h, base + 11]), 3),
                    "alive": bool(obs_view[h, base + 14] > 0.5),
                    "invuln": bool(obs_view[h, base + 8] > 0.5),
                    "invuln_timer": 0.0,
                    "respawn_timer": round(
                        float(obs_view[h, base + 16] * self.cfg.respawn_time), 3
                    ),
                    "target": 0,
                    "p_t": 0.0,
                    "fired": bool(obs_view[h, base + 12] > 0.5),
                    "visible": True,
                    "observed": True,
                })
                continue
            robots.append({
                "name": NAMES[i],
                "team": int(self.cfg.team(i)),
                "x": round(float(s.pos[i, 0]), 4),
                "y": round(float(s.pos[i, 1]), 4),
                "vx": round(float(s.vel[i, 0]), 4),
                "vy": round(float(s.vel[i, 1]), 4),
                "yaw": round(float(s.yaw[i]), 5),
                "hp": round(float(s.hp[i]), 2),
                "heat": round(float(s.heat[i]), 2),
                "overheated": bool(s.overheated[i]),
                "spin": round(float(s.spin[i]), 3),
                "alive": bool(s.alive[i]),
                "invuln": bool(s.invuln[i]),
                "invuln_timer": round(float(s.invuln_timer[i]), 3),
                "respawn_timer": round(float(s.respawn_timer[i]), 3),
                "target": int(np.rint(float(s.target[i]))),
                "p_t": round(float(s.p_t[i]), 4),
                "fired": bool(s.last_fired[i]),
                "visible": True,
                "heat_known": True,
                "observed": False,
            })
        self._snapshot = {
                "ok": True,
                "mode": self.mode,
                "seed": self.seed,
                "human": h,
                "human_name": NAMES[h],
                "ai": self.ai,
                "terminal_fouls": False,
                "paused": self.paused,
                "done": self.done,
                "time": round(float(s.time), 2),
                "score": [round(float(s.score[0]), 3), round(float(s.score[1]), 3)],
                "occ": [int(np.asarray(s.occ[0])), int(np.asarray(s.occ[1]))],
                "outcome": int(np.asarray(s.outcome).item()),
                "outcome_name": self.outcome_name(),
                "winner": int(np.asarray(s.winner).item()),
                "robots": robots,
                "mask": mask[h].astype(int).tolist(),
                "mouse_down": self.mouse_down,
                "lock_down": self.lock_down,
                "fire_down": self.fire_down,
                "target": int(self.target),
                "events": list(self.events),
            }

    def state_dict(self):
        with self.lock:
            out = dict(self._snapshot)
            out["paused"] = self.paused
            out["mouse_down"] = self.mouse_down
            out["lock_down"] = self.lock_down
            out["fire_down"] = self.fire_down
            out["target"] = int(self.target)
            out["events"] = list(self.events)
            return out


GAME: Game | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "RmJaxHuman/1.0"

    def log_message(self, *_):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PATH.read_text(encoding="utf-8"),
                       "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, json.dumps(GAME.state_dict(), ensure_ascii=False))
        elif self.path == "/api/meta":
            self._send(200, json.dumps({
                "ok": True, "mode": GAME.mode, "human": GAME.human,
                "terminal_fouls": False,
                "actor_iteration": GAME.ckpt_iteration,
            }))
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, json.dumps({"ok": False, "error": "bad json"}))
            return
        try:
            if self.path == "/api/input":
                with GAME.lock:
                    for k in GAME.keys:
                        if k in data.get("keys", {}):
                            GAME.keys[k] = bool(data["keys"][k])
                    if "aim_angle" in data:
                        GAME.aim_angle = float(data["aim_angle"])
                    if "mouse_down" in data:
                        GAME.mouse_down = bool(data["mouse_down"])
                    if "lock_down" in data:
                        GAME.lock_down = bool(data["lock_down"])
                    if "fire_down" in data:
                        GAME.fire_down = bool(data["fire_down"])
                    if "human" in data:
                        GAME.human = max(0, min(5, int(data["human"])))
                        GAME.target = 0
                        GAME.mouse_down = False
                        GAME.lock_down = False
                        GAME.fire_down = False
                        GAME.h = jnp.zeros((6, 256), dtype=jnp.float32)
                        GAME._build_snapshot_locked()
                self._send(200, json.dumps({"ok": True}))
            elif self.path == "/api/reset":
                GAME.reset(data.get("mode"), data.get("seed"), data.get("human"))
                self._send(200, json.dumps({"ok": True}))
            elif self.path == "/api/pause":
                with GAME.lock:
                    GAME.paused = bool(data.get("paused", not GAME.paused))
                self._send(200, json.dumps({"ok": True, "paused": GAME.paused}))
            elif self.path == "/api/step":
                with GAME.lock:
                    GAME.paused = True
                GAME.step_once(force=True)
                self._send(200, json.dumps({"ok": True}))
            else:
                self._send(404, json.dumps({"ok": False, "error": "not found"}))
        except Exception:
            self._send(500, json.dumps({"ok": False, "error": traceback.format_exc()}))


def main():
    global GAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="fixed")
    ap.add_argument("--human", type=int, default=0)
    ap.add_argument(
        "--ckpt",
        default=str(
            ROOT
            / "outputs"
            / "rm_jax_league900_1200"
            / "jax_ckpt_001200.pkl"
        ),
    )
    args = ap.parse_args()
    GAME = Game(args.ckpt, seed=args.seed, mode=args.mode, human=args.human)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[rm-jax-human] http://{args.host}:{args.port}/ "
          f"actor={GAME.ckpt_iteration} human={NAMES[GAME.human]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
