#!/usr/bin/env python3
"""Feasibility spike: export one RM environment step to ONNX."""
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
from onnx import TensorProto, helper, numpy_helper
from jax2onnx import to_onnx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxmarl.environments.rm.config import RmConfig
from jaxmarl.environments.rm.env import RmEnv


def make_web_compatible(model: onnx.ModelProto) -> dict[str, int]:
    """Patch kernels missing from ORT CPU/WASM and make outputs unique by name."""
    graph = model.graph
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            list(graph.input) + list(graph.output) + list(graph.value_info)
        )
        if value.type.HasField("tensor_type")
    }
    shapes = {
        value.name: tuple(
            dim.dim_value if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        )
        for value in list(graph.output) + list(graph.value_info)
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.HasField("shape")
    }
    for initializer in graph.initializer:
        types.setdefault(initializer.name, initializer.data_type)

    nodes = []
    count = {"bool_where": 0, "bool_eyelike": 0, "renamed_outputs": 0}
    for node in graph.node:
        if (
            node.op_type == "Where"
            and len(node.input) == 3
            and all(types.get(name) == TensorProto.BOOL for name in node.input)
        ):
            cond, x, y = node.input
            out = node.output[0]
            base = f"{node.name}_boolfix"
            not_cond = f"{base}_not"
            x_and = f"{base}_and_x"
            y_and = f"{base}_and_y"
            nodes.extend(
                [
                    helper.make_node("Not", [cond], [not_cond], name=f"{base}_Not"),
                    helper.make_node("And", [cond, x], [x_and], name=f"{base}_AndX"),
                    helper.make_node("And", [not_cond, y], [y_and], name=f"{base}_AndY"),
                    helper.make_node("Or", [x_and, y_and], [out], name=f"{base}_Or"),
                ]
            )
            count["bool_where"] += 1
            continue

        if (
            node.op_type == "EyeLike"
            and node.output
            and types.get(node.output[0]) == TensorProto.BOOL
        ):
            shape = shapes.get(node.output[0])
            if shape and len(shape) == 2 and all(d is not None for d in shape):
                k = next(
                    (
                        int(helper.get_attribute_value(attr))
                        for attr in node.attribute
                        if attr.name == "k"
                    ),
                    0,
                )
                value = np.eye(shape[0], shape[1], k=k, dtype=np.bool_)
                nodes.append(
                    helper.make_node(
                        "Constant",
                        [],
                        [node.output[0]],
                        name=f"{node.name}_const",
                        value=numpy_helper.from_array(value),
                    )
                )
                count["bool_eyelike"] += 1
                continue

        nodes.append(node)

    output_names = [value.name for value in graph.output]
    if len(set(output_names)) != len(output_names):
        original_outputs = list(graph.output)
        for i, value in enumerate(original_outputs):
            old_name = value.name
            new_name = f"web_output_{i}"
            nodes.append(
                helper.make_node(
                    "Identity",
                    [old_name],
                    [new_name],
                    name=f"web_output_identity_{i}",
                )
            )
            value.name = new_name
            count["renamed_outputs"] += 1

    del graph.node[:]
    graph.node.extend(nodes)
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="validate an existing ONNX file without exporting it again",
    )
    ap.add_argument(
        "--save-patched",
        action="store_true",
        help="with --check-only, overwrite the input with its web-compatible graph",
    )
    args = ap.parse_args()
    cfg = RmConfig(
        robot_foul_speed=1.0e9,
        wall_foul_speed=1.0e9,
        wall_push_time=1.0e9,
        body_push_time=1.0e9,
        obs_noise=False,
        hit_cue_noise_deg=0.0,
    )
    env = RmEnv(cfg)
    key = jax.random.PRNGKey(args.seed)
    _, state = env.reset(key, mode="fixed")
    state_leaves, state_tree = jax.tree_util.tree_flatten(state)
    state_specs = tuple(
        jax.ShapeDtypeStruct(x.shape, x.dtype) for x in state_leaves
    )
    key_fixed = jax.random.PRNGKey(args.seed + 17)

    def step_core(*args):
        leaf_args = args[:-1]
        action = args[-1]
        state_in = jax.tree_util.tree_unflatten(state_tree, tuple(leaf_args))
        action_d = {
            agent: action[i] for i, agent in enumerate(env.agents)
        }
        obs_d, state_out, _, dones, _ = env.step_env(
            key_fixed, state_in, action_d, deterministic=True
        )
        obs = jnp.stack([obs_d[a] for a in env.agents], axis=0)
        return (
            tuple(jax.tree_util.tree_leaves(state_out))
            + (obs, dones["__all__"])
        )

    input_specs = state_specs + (
        jax.ShapeDtypeStruct((6, 6), jnp.float32),
    )
    input_names = [f"state_{i}" for i in range(len(state_specs))] + ["actions"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.check_only:
        model = to_onnx(
            step_core,
            input_specs,
            model_name="rm_env_step",
            opset=18,
            input_names=input_names,
            export_mode="web",
        )
        patch_counts = make_web_compatible(model)
        print(f"web_patch={patch_counts}", flush=True)
        onnx.save(model, out)

    if not out.is_file():
        raise FileNotFoundError(f"ONNX file does not exist: {out}")
    model = onnx.load(out, load_external_data=False)
    if not args.check_only:
        patch_counts = {"bool_where": 0, "bool_eyelike": 0, "renamed_outputs": 0}
    else:
        patch_counts = make_web_compatible(model)
        print(f"web_patch={patch_counts}", flush=True)
        if args.save_patched:
            onnx.save(model, out)
    onnx.checker.check_model(model)

    rng = np.random.default_rng(args.seed)
    actions = rng.uniform(-0.25, 0.25, size=(6, 6)).astype(np.float32)
    jax_out = step_core(
        *[jnp.asarray(x) for x in state_leaves],
        jnp.asarray(actions),
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    print(
        f"model={out} size={out.stat().st_size} "
        f"inputs={len(sess.get_inputs())} outputs={len(sess.get_outputs())}",
        flush=True,
    )
    feeds = {
        sess.get_inputs()[i].name: np.asarray(x)
        for i, x in enumerate(state_leaves)
    }
    feeds[sess.get_inputs()[-1].name] = actions
    ort_out = sess.run(None, feeds)
    jax_leaves = jax.tree_util.tree_leaves(jax_out)
    if len(jax_leaves) != len(ort_out):
        raise RuntimeError(
            f"output count mismatch: jax={len(jax_leaves)} ort={len(ort_out)}"
        )
    max_err = 0.0
    for i, (a, b) in enumerate(zip(jax_leaves, ort_out)):
        a_np = np.asarray(a)
        if a_np.shape != b.shape:
            raise RuntimeError(
                f"output {i} shape mismatch: jax={a_np.shape} ort={b.shape}"
            )
        if a_np.dtype == np.bool_:
            err = np.logical_xor(a_np, b)
        else:
            err = np.abs(a_np - b)
        max_err = max(max_err, float(np.max(err)))
    print(f"outputs_checked={len(ort_out)}", flush=True)
    print(f"max_abs_error={max_err}", flush=True)
    if max_err > 1e-3:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
