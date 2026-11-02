"""Small CSV logger for the JAX RM trainer.

The layout mirrors the useful part of BenchMARL/torchrl's CSVLogger:

    <root>/scalars/<metric>.csv      # one file per scalar, rows: step,value
    <root>/progress.csv              # wide table, easy to inspect
    <root>/metrics.jsonl             # one JSON object per iteration
    <root>/hparams.json              # CLI/run configuration

It deliberately has no dependency on torch/torchrl so it can run in the
lightweight JAX test container.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

OUTCOME_NAMES = {
    0: "进行中",
    1: "红方先到目标分",
    2: "蓝方先到目标分",
    3: "红方时间胜",
    4: "蓝方时间胜",
    5: "红方伤害tiebreak",
    6: "蓝方伤害tiebreak",
    7: "红方血量tiebreak",
    8: "蓝方血量tiebreak",
    9: "红方胜(蓝方碰撞犯规)",
    10: "蓝方胜(红方碰撞犯规)",
    20: "平局(时间tiebreak)",
    21: "平局(双方同帧200)",
    22: "平局(双方同帧犯规)",
    23: "空局(10秒无事件)",
    24: "红方胜(蓝方推墙/掩体犯规)",
    25: "蓝方胜(红方推墙/掩体犯规)",
    26: "平局(双方推墙/掩体犯规)",
    27: "红方胜(蓝方推人犯规)",
    28: "蓝方胜(红方推人犯规)",
    29: "平局(双方推人犯规)",
}
GLOBAL_SCORE_OFFSET = 6 * 30
TARGET_SCORE = 90.0


def write_rm_live(traj: Mapping[str, Any], out_dir: str, it: int,
                  n_clips: int = 10) -> None:
    """Write the same ``rm_live.json`` shape consumed by ``rm_live.html``.

    JAX stores the full post-step global state, so the live view has the
    complete 191-D state without adding a separate rendering path.
    """
    state = np.asarray(traj["next_global_state"], dtype=np.float32)
    done = np.asarray(traj["done"], dtype=bool)
    codes = np.asarray(traj["outcome"])
    if codes.ndim == 3:
        codes = codes[..., 0]
    codes = codes.astype(np.int32)
    if state.ndim != 3 or done.ndim != 2:
        return
    T, B = done.shape
    n = min(int(n_clips), B)
    if n <= 0 or T <= 0:
        return

    is_init = np.zeros((T, B), dtype=bool)
    is_init[0] = True
    if T > 1:
        is_init[1:] = done[:-1]

    red = state[:, :, GLOBAL_SCORE_OFFSET] * TARGET_SCORE
    blue = state[:, :, GLOBAL_SCORE_OFFSET + 1] * TARGET_SCORE
    red_delta = np.diff(red, axis=0, prepend=red[:1])
    blue_delta = np.diff(blue, axis=0, prepend=blue[:1])
    red_delta[is_init] = 0.0
    blue_delta[is_init] = 0.0
    total_delta = red_delta + blue_delta
    final_total = red[-1] + blue[-1]

    counts: dict[int, int] = {}
    n_done = 0
    for t in range(T):
        for e in range(B):
            if done[t, e] and int(codes[t, e]) > 0:
                c = int(codes[t, e])
                counts[c] = counts.get(c, 0) + 1
                n_done += 1

    clips = []
    for e in range(n):
        clips.append({
            "env": e,
            "n": T,
            "state": [[round(float(v), 5) for v in row]
                      for row in state[:, e]],
            "is_init": is_init[:, e].tolist(),
            "done": done[:, e].tolist(),
            "codes": codes[:, e].tolist(),
        })

    payload = {
        "iter": int(it),
        "n_clips": len(clips),
        "n": T,
        "state_dim": int(state.shape[-1]),
        "outcome_names": {str(k): v for k, v in OUTCOME_NAMES.items()},
        "clips": clips,
        "all_n_envs": B,
        "all_n_done": n_done,
        "all_outcome_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "trajectory_score_deltas": [round(float(v), 3) for v in total_delta.sum(axis=0).tolist()],
        "trajectory_score_delta_red": [round(float(v), 3) for v in red_delta.sum(axis=0).tolist()],
        "trajectory_score_delta_blue": [round(float(v), 3) for v in blue_delta.sum(axis=0).tolist()],
        "trajectory_score_delta_mean": round(float(total_delta.sum(axis=0).mean()), 3),
        "trajectory_score_delta_max": round(float(total_delta.sum(axis=0).max()), 3),
        "trajectory_score_final": [round(float(v), 3) for v in final_total.tolist()],
        "trajectory_score_final_mean": round(float(final_total.mean()), 3),
        "trajectory_score_final_max": round(float(final_total.max()), 3),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rm_live.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def _scalar(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


class JaxScalarLogger:
    def __init__(self, root: str, hparams: Optional[Mapping[str, Any]] = None):
        self.root = Path(root)
        self.scalars_dir = self.root / "scalars"
        self.scalars_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.root / "progress.csv"
        self.metrics_path = self.root / "metrics.jsonl"
        self._fields: Optional[list[str]] = None
        if self.progress_path.is_file() and self.progress_path.stat().st_size > 0:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            if header:
                self._fields = header.split(",")
        if hparams is not None:
            with open(self.root / "hparams.json", "w", encoding="utf-8") as f:
                json.dump({k: str(v) if isinstance(v, Path) else v for k, v in hparams.items()},
                          f, ensure_ascii=False, indent=2)

    @staticmethod
    def _key(key: str) -> str:
        return key.replace("/", "_").replace(" ", "_").replace(":", "_")

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        clean: dict[str, float] = {}
        for key, value in metrics.items():
            v = _scalar(value)
            if v is not None:
                clean[self._key(key)] = v
        if not clean:
            return
        for key, value in clean.items():
            path = self.scalars_dir / f"{key}.csv"
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{int(step)},{value:.10g}\n")
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": int(step), **clean}, ensure_ascii=False) + "\n")

        if self._fields is None:
            self._fields = ["step"] + sorted(clean)
            with open(self.progress_path, "w", encoding="utf-8") as f:
                f.write(",".join(self._fields) + "\n")
        # Keep a stable schema; new keys still live in metrics.jsonl.
        row = [str(int(step))] + [
            f"{clean.get(k, ''):.10g}" if k in clean else ""
            for k in self._fields[1:]
        ]
        with open(self.progress_path, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")

    @staticmethod
    def default_root(save_dir: str = "", log_dir: str = "") -> str:
        if log_dir:
            return log_dir
        if save_dir:
            return str(Path(save_dir) / "logs")
        return str(Path("outputs") / time.strftime("jax_rm_%Y%m%d_%H%M%S"))
