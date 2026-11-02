#!/usr/bin/env python3
"""Convert the PyTorch RM centralized critic to Flax parameters.

The v64 checkpoint stores the critic under
``ckpt["loss_agents"]["critic_network_params.*"]``.  The architecture is the
no-map, team-canonical critic in ``benchmarl/models/rm_critic.py``.
"""
from __future__ import annotations

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxmarl.environments.rm.critic import RmCritic

PREFIX = "critic_network_params."


def _linear(dst, sd, src, flax):
    dst[flax]["kernel"] = np.asarray(sd[src + ".weight"].T, dtype=np.float32)
    dst[flax]["bias"] = np.asarray(sd[src + ".bias"], dtype=np.float32)


def _ln(dst, sd, src, flax):
    dst[flax]["scale"] = np.asarray(sd[src + ".weight"], dtype=np.float32)
    dst[flax]["bias"] = np.asarray(sd[src + ".bias"], dtype=np.float32)


def convert(sd):
    model = RmCritic()
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, 191), dtype=jnp.float32)
    )["params"]

    _linear(params, sd, PREFIX + "agent_proj", "Dense_0")
    params["slot_emb"]["embedding"] = np.asarray(
        sd[PREFIX + "slot_emb.weight"], dtype=np.float32
    )
    # The JAX critic uses a constant 1-D team index to avoid nn.Embed's
    # atomic scatter-add gradient; the parameter layout matches the old one.
    params["team_emb"]["embedding"] = np.asarray(
        sd[PREFIX + "team_emb.weight"], dtype=np.float32
    )
    _linear(params, sd, PREFIX + "global_proj", "global_proj")
    params["type_emb"]["embedding"] = np.asarray(
        sd[PREFIX + "type_emb.weight"], dtype=np.float32
    )

    for layer in range(2):
        pre = PREFIX + f"transformer.layers.{layer}."
        sa = f"SelfAttention_{layer}"
        qkv = np.asarray(sd[pre + "self_attn.in_proj_weight"], dtype=np.float32)
        qkv_b = np.asarray(sd[pre + "self_attn.in_proj_bias"], dtype=np.float32)
        q, k, v = np.split(qkv, 3, axis=0)
        qb, kb, vb = np.split(qkv_b, 3, axis=0)
        for name, mat, bias in (("query", q, qb), ("key", k, kb), ("value", v, vb)):
            # PyTorch Linear [out,in] -> Flax [in,out] -> [in,heads,head_dim]
            params[sa][name]["kernel"] = mat.T.reshape(128, 4, 32)
            params[sa][name]["bias"] = bias.reshape(4, 32)
        out_w = np.asarray(sd[pre + "self_attn.out_proj.weight"], dtype=np.float32)
        out_b = np.asarray(sd[pre + "self_attn.out_proj.bias"], dtype=np.float32)
        params[sa]["out"]["kernel"] = out_w.T.reshape(4, 32, 128)
        params[sa]["out"]["bias"] = out_b

        _ln(params, sd, pre + "norm1", f"LayerNorm_{2 * layer}")
        _linear(params, sd, pre + "linear1", f"Dense_{2 * layer + 1}")
        _linear(params, sd, pre + "linear2", f"Dense_{2 * layer + 2}")
        _ln(params, sd, pre + "norm2", f"LayerNorm_{2 * layer + 1}")

    _ln(params, sd, PREFIX + "out_norm", "LayerNorm_4")
    _linear(params, sd, PREFIX + "value_head.0", "Dense_5")
    _linear(params, sd, PREFIX + "value_head.2", "Dense_6")
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="rm_critic_params.pkl")
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt["loss_agents"]
    params = convert(sd)
    with open(args.out, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
    print(f"saved {args.out}")
    m = RmCritic()
    out = m.apply({"params": params}, jnp.zeros((2, 191), dtype=jnp.float32))
    print("forward", out.shape, float(jnp.abs(out).mean()))


if __name__ == "__main__":
    main()
