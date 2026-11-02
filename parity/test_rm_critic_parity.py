#!/usr/bin/env python3
"""Numerical parity for the converted PyTorch RM critic and Flax port."""
from __future__ import annotations

import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from jaxmarl.environments.rm.critic import RmCritic

N_AGENTS = 6
AGENT_DIM = 30
GLOBAL_DIM = 11
STATE_DIM = 191
D = 128
HEADS = 4
HEAD_DIM = 32
PREFIX = "critic_network_params."


def canonicalize(agents, glob, team):
    if team == 1:
        obstacles = agents[..., 22:30][..., [4, 5, 6, 7, 0, 1, 2, 3]]
        a = torch.cat([-agents[..., 0:6], agents[..., 6:22], obstacles], -1)
    else:
        a = agents
    own = slice(0, 3) if team == 0 else slice(3, 6)
    opp = slice(3, 6) if team == 0 else slice(0, 3)
    ca = torch.cat([a[:, own], a[:, opp]], 1)
    flags = torch.cat(
        [
            torch.zeros(ca.shape[0], 3, 1, device=ca.device),
            torch.ones(ca.shape[0], 3, 1, device=ca.device),
        ],
        1,
    )
    ca = torch.cat([ca[..., :17], flags, ca[..., 18:]], -1)
    g = glob
    if team == 1:
        order = torch.tensor([1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9])
        g = glob[:, order]
    return ca, g


def torch_forward(sd, state):
    x = state
    m = x.shape[0]
    agents = x[:, : N_AGENTS * AGENT_DIM].reshape(m, N_AGENTS, AGENT_DIM)
    glob = x[:, N_AGENTS * AGENT_DIM:]
    ca = torch.cat([canonicalize(agents, glob, 0)[0], canonicalize(agents, glob, 1)[0]], 0)
    cg = torch.cat([canonicalize(agents, glob, 0)[1], canonicalize(agents, glob, 1)[1]], 0)
    team = torch.cat([torch.zeros(m), torch.ones(m)]).long()
    agent = F.linear(ca, sd[PREFIX + "agent_proj.weight"], sd[PREFIX + "agent_proj.bias"])
    slot = sd[PREFIX + "slot_emb.weight"]
    agent = agent + slot[None] + sd[PREFIX + "team_emb.weight"][team][:, None, :]
    g = F.linear(cg, sd[PREFIX + "global_proj.weight"], sd[PREFIX + "global_proj.bias"])
    tokens = torch.cat([g.unsqueeze(1), agent], 1)
    type_ids = torch.tensor([0] + [1] * N_AGENTS)
    tokens = tokens + sd[PREFIX + "type_emb.weight"][type_ids][None]

    for layer in range(2):
        pre = PREFIX + f"transformer.layers.{layer}."
        h = F.layer_norm(tokens, (D,), sd[pre + "norm1.weight"], sd[pre + "norm1.bias"])
        qkv = F.linear(h, sd[pre + "self_attn.in_proj_weight"], sd[pre + "self_attn.in_proj_bias"])
        q, k, v = qkv.chunk(3, -1)
        # [B, T, d] -> [B, heads, T, head_dim]
        q = q.reshape(q.shape[0], q.shape[1], HEADS, HEAD_DIM).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], HEADS, HEAD_DIM).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], HEADS, HEAD_DIM).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(att.shape[0], att.shape[2], D)
        att = F.linear(att, sd[pre + "self_attn.out_proj.weight"], sd[pre + "self_attn.out_proj.bias"])
        tokens = tokens + att
        h = F.layer_norm(tokens, (D,), sd[pre + "norm2.weight"], sd[pre + "norm2.bias"])
        h = F.linear(h, sd[pre + "linear1.weight"], sd[pre + "linear1.bias"])
        h = F.gelu(h)
        h = F.linear(h, sd[pre + "linear2.weight"], sd[pre + "linear2.bias"])
        tokens = tokens + h

    tokens = F.layer_norm(tokens, (D,), sd[PREFIX + "out_norm.weight"], sd[PREFIX + "out_norm.bias"])
    x = tokens[:, 1:, :]
    x = F.linear(x, sd[PREFIX + "value_head.0.weight"], sd[PREFIX + "value_head.0.bias"])
    x = F.gelu(x)
    x = F.linear(x, sd[PREFIX + "value_head.2.weight"], sd[PREFIX + "value_head.2.bias"])
    x = x[..., 0]
    return torch.cat([x[:m, 0:3], x[m:, 0:3]], 1)


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    params_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/rm_critic_params.pkl"
    if ckpt_path is None:
        raise SystemExit("usage: test_rm_critic_parity.py CHECKPOINT PARAMS.pkl")
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt["loss_agents"]
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    rng = np.random.default_rng(0)
    state = rng.normal(size=(8, STATE_DIM)).astype(np.float32)
    with torch.no_grad():
        pt = torch_forward(sd, torch.from_numpy(state)).numpy()
    jax_out = RmCritic().apply({"params": params}, jnp.asarray(state))
    diff = np.abs(pt - jax_out)
    print("pt", pt[0])
    print("jax", np.asarray(jax_out)[0])
    print("max_abs", float(diff.max()), "mean_abs", float(diff.mean()))


if __name__ == "__main__":
    main()
