#!/usr/bin/env python3
"""Convert the PyTorch RM actor in a BenchMARL checkpoint to Flax parameters.

Checkpoint layout (v64):
  ckpt["loss_agents"]["actor_network_params..."] -> torch state dict
The encoder lives under ``...models.module.0`` and GRU+head under ``...models.module.1``.
Only the inference actor is converted; critic/optimizer are ignored.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxmarl.environments.rm.actor import RmActor

PREFIX = "actor_network_params.module.0.module.0.module.0.models.module."


def _ln(dst, torch_sd, src, flax):
    dst[flax]["scale"] = np.asarray(torch_sd[src + ".weight"], dtype=np.float32)
    dst[flax]["bias"] = np.asarray(torch_sd[src + ".bias"], dtype=np.float32)


def _linear(dst, torch_sd, src, flax):
    # torch Linear stores [out, in]; Flax Dense stores [in, out]
    dst[flax]["kernel"] = np.asarray(torch_sd[src + ".weight"].T, dtype=np.float32)
    dst[flax]["bias"] = np.asarray(torch_sd[src + ".bias"], dtype=np.float32)


def _token_mlp(dst, sd, src, flax):
    _linear(dst[flax], sd, src + ".0", "Dense_0")
    _ln(dst[flax], sd, src + ".2", "LayerNorm_0")


def convert(torch_sd):
    model = RmActor()
    dummy_obs = jnp.zeros((1, 140), dtype=jnp.float32)
    dummy_carry = jnp.zeros((1, 256), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_obs, dummy_carry)["params"]

    e = params["RmEntityEncoder_0"]
    token_names = ["self_proj", "spawn_proj", "center_proj", "mate_proj",
                   "enemy_proj", "hud_proj", "cue_proj", "obstacle_proj"]
    for i, name in enumerate(token_names):
        _token_mlp(e, torch_sd, PREFIX + f"0.{name}", f"_TokenMLP_{i}")
    e["type_emb"]["embedding"] = np.asarray(torch_sd[PREFIX + "0.type_emb.weight"], dtype=np.float32)
    e["slot_emb"] = np.asarray(torch_sd[PREFIX + "0.slot_emb"], dtype=np.float32)
    _ln(e, torch_sd, PREFIX + "0.out_norm", "LayerNorm_4")

    for layer in range(2):
        pre = PREFIX + f"0.transformer.layers.{layer}."
        qkv = np.asarray(torch_sd[pre + "self_attn.in_proj_weight"], dtype=np.float32)
        qkv_b = np.asarray(torch_sd[pre + "self_attn.in_proj_bias"], dtype=np.float32)
        q, k, v = np.split(qkv, 3, axis=0)
        qb, kb, vb = np.split(qkv_b, 3, axis=0)
        sa = f"SelfAttention_{layer}"
        for name, mat, bias in [("query", q, qb), ("key", k, kb), ("value", v, vb)]:
            # [out, in] -> [in, out] -> [in, heads, head_dim]
            e[sa][name]["kernel"] = mat.T.reshape(128, 4, 32)
            e[sa][name]["bias"] = bias.reshape(4, 32)
        out_w = np.asarray(torch_sd[pre + "self_attn.out_proj.weight"], dtype=np.float32)
        out_b = np.asarray(torch_sd[pre + "self_attn.out_proj.bias"], dtype=np.float32)
        e[sa]["out"]["kernel"] = out_w.T.reshape(4, 32, 128)
        e[sa]["out"]["bias"] = out_b
        _ln(e, torch_sd, pre + "norm1", f"LayerNorm_{2 * layer}")
        _linear(e, torch_sd, pre + "linear1", f"Dense_{2 * layer}")
        _linear(e, torch_sd, pre + "linear2", f"Dense_{2 * layer + 1}")
        _ln(e, torch_sd, pre + "norm2", f"LayerNorm_{2 * layer + 1}")

    _linear(e, torch_sd, PREFIX + "0.map_neck.0", "Dense_4")
    _ln(e, torch_sd, PREFIX + "0.map_neck.2", "LayerNorm_5")
    _linear(e, torch_sd, PREFIX + "0.mate_neck.0", "Dense_5")
    _ln(e, torch_sd, PREFIX + "0.mate_neck.2", "LayerNorm_6")
    e["Dense_6"]["kernel"] = np.asarray(torch_sd[PREFIX + "0.enemy_slot_proj.weight"].T, dtype=np.float32)
    e["enemy_absent"] = np.asarray(torch_sd[PREFIX + "0.enemy_absent"], dtype=np.float32)
    _linear(e, torch_sd, PREFIX + "0.out_proj", "Dense_7")

    # GRU: torch gates are r,z,n, Flax GRUCell names are ir/iz/in + hr/hz/hn.
    gru = params["GRUCell_0"]
    w_ih = np.asarray(torch_sd[PREFIX + "1.gru.params.grus.0.weight_ih_l0"], dtype=np.float32)
    w_hh = np.asarray(torch_sd[PREFIX + "1.gru.params.grus.0.weight_hh_l0"], dtype=np.float32)
    b_ih = np.asarray(torch_sd[PREFIX + "1.gru.params.grus.0.bias_ih_l0"], dtype=np.float32)
    b_hh = np.asarray(torch_sd[PREFIX + "1.gru.params.grus.0.bias_hh_l0"], dtype=np.float32)
    for idx, gate in enumerate(["r", "z", "n"]):
        i_name = {"r": "ir", "z": "iz", "n": "in"}[gate]
        h_name = {"r": "hr", "z": "hz", "n": "hn"}[gate]
        gru[i_name]["kernel"] = w_ih[idx * 256:(idx + 1) * 256].T
        # Flax has no hidden bias on the r/z recurrent Dense layers; fold
        # torch bias_hh into the input bias there (constant term).  The
        # candidate gate has an explicit ``hn.bias`` and must keep the
        # input/hidden biases separate.
        if gate == "n":
            gru[i_name]["bias"] = b_ih[idx * 256:(idx + 1) * 256]
            gru[h_name]["bias"] = b_hh[idx * 256:(idx + 1) * 256]
        else:
            gru[i_name]["bias"] = (
                b_ih[idx * 256:(idx + 1) * 256]
                + b_hh[idx * 256:(idx + 1) * 256]
            )
        gru[h_name]["kernel"] = w_hh[idx * 256:(idx + 1) * 256].T

    # MLP after [h, encoder(256)] = 512.
    _linear(params, torch_sd, PREFIX + "1.mlp.params.0", "Dense_0")
    _ln(params, torch_sd, PREFIX + "1.mlp.params.1", "LayerNorm_0")
    _linear(params, torch_sd, PREFIX + "1.mlp.params.3", "Dense_1")
    _ln(params, torch_sd, PREFIX + "1.mlp.params.4", "LayerNorm_1")
    _linear(params, torch_sd, PREFIX + "1.mlp.params.6", "Dense_2")
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="rm_actor_params.pkl")
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt["loss_agents"]
    params = convert(sd)
    with open(args.out, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    print(f"saved {args.out}")
    # hard shape check against the model
    m = RmActor()
    obs = jnp.zeros((1, 140), dtype=jnp.float32)
    carry = jnp.zeros((1, 256), dtype=jnp.float32)
    out, nc = m.apply({"params": params}, obs, carry)
    print("forward", out.shape, nc.shape)


if __name__ == "__main__":
    main()
