#!/usr/bin/env python3
"""Feasibility spike: export RmActor step to ONNX and compare with JAX."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
from jax2onnx import to_onnx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.actor import RmActor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    actor = RmActor()
    params = jax.device_put(ck["actor_params"])

    def actor_step(obs, hidden):
        return actor.apply({"params": params}, obs, hidden)

    inputs = [
        jax.ShapeDtypeStruct((6, 140), jnp.float32),
        jax.ShapeDtypeStruct((6, 256), jnp.float32),
    ]
    model = to_onnx(
        actor_step,
        inputs,
        model_name="rm_actor_1200",
        opset=18,
        input_names=["obs", "hidden"],
        output_names=["raw", "hidden_next"],
        export_mode="web",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)

    rng = np.random.default_rng(args.seed)
    obs = rng.normal(0.0, 0.5, size=(6, 140)).astype(np.float32)
    hidden = rng.normal(0.0, 0.5, size=(6, 256)).astype(np.float32)
    raw_j, h_j = actor_step(jnp.asarray(obs), jnp.asarray(hidden))

    sess = ort.InferenceSession(
        str(out), providers=["CPUExecutionProvider"]
    )
    feeds = {
        sess.get_inputs()[0].name: obs,
        sess.get_inputs()[1].name: hidden,
    }
    raw_o, h_o = sess.run(None, feeds)
    errors = {
        "raw_max_abs": float(np.max(np.abs(np.asarray(raw_j) - raw_o))),
        "hidden_max_abs": float(np.max(np.abs(np.asarray(h_j) - h_o))),
    }
    print(f"model={out} size={out.stat().st_size} bytes inputs={[x.name for x in sess.get_inputs()]} outputs={[x.name for x in sess.get_outputs()]}")
    print(errors)
    if errors["raw_max_abs"] > 1e-3 or errors["hidden_max_abs"] > 1e-3:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
