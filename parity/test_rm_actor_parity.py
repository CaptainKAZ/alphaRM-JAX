#!/usr/bin/env python3
"""Numeric parity: PyTorch RM actor checkpoint vs Flax RmActor."""
from __future__ import annotations

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxmarl.environments.rm.actor import RmActor
from convert_rm_actor import PREFIX, convert

D = 128
N = 11
SCALAR = 132


def _ln(x, sd, name):
    return F.layer_norm(x, (D,), sd[name + ".weight"], sd[name + ".bias"], 1e-5)


def _lin(x, sd, name):
    return F.linear(x, sd[name + ".weight"], sd[name + ".bias"])


def _proj(x, sd, name):
    x = _lin(x, sd, name + ".0")
    x = F.gelu(x, approximate="none")
    return _ln(x, sd, name + ".2")


def _head_ln(x, sd, name):
    return F.layer_norm(x, (256,), sd[name + ".weight"], sd[name + ".bias"], 1e-5)


def torch_actor(sd, obs, h0=None):
    # obs [B,140]
    scalar, obstacle = obs[:, :SCALAR], obs[:, SCALAR:]
    s = 0
    self_f = scalar[:, s:s+24]; s += 24
    spawn_f = scalar[:, s:s+4]; s += 4
    center_f = scalar[:, s:s+4]; s += 4
    mates_f = scalar[:, s:s+36].reshape(-1, 2, 18); s += 36
    enemy_f = scalar[:, s:s+51].reshape(-1, 3, 17); s += 51
    cue_f = scalar[:, s:s+3]; s += 3
    hud_f = scalar[:, s:s+10]; s += 10
    assert s == SCALAR
    tokens = torch.cat([
        _proj(self_f, sd, "self_proj").unsqueeze(1),
        _proj(spawn_f, sd, "spawn_proj").unsqueeze(1),
        _proj(center_f, sd, "center_proj").unsqueeze(1),
        _proj(mates_f.reshape(-1, 18), sd, "mate_proj").reshape(-1, 2, D),
        _proj(enemy_f.reshape(-1, 17), sd, "enemy_proj").reshape(-1, 3, D),
        _proj(hud_f, sd, "hud_proj").unsqueeze(1),
        _proj(cue_f, sd, "cue_proj").unsqueeze(1),
        _proj(obstacle, sd, "obstacle_proj").unsqueeze(1),
    ], 1)
    type_ids = torch.tensor([0,1,2,3,3,4,4,4,5,6,7], device=obs.device)
    tokens = tokens + F.embedding(type_ids, sd["type_emb.weight"]).unsqueeze(0)
    tokens = tokens + sd["slot_emb"].unsqueeze(0)
    valid = torch.ones((obs.shape[0], N), dtype=torch.bool, device=obs.device)
    valid[:, 5:8] = enemy_f[..., 0] > 0.5
    valid[:, 9] = cue_f[..., 2] > 0.5

    # Load nn.TransformerEncoder exactly.
    layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=4, dim_feedforward=256, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )
    trans = nn.TransformerEncoder(layer, num_layers=2)
    tsd = {}
    for k, v in sd.items():
        if k.startswith("transformer."):
            tsd[k[len("transformer."):]] = v
    trans.load_state_dict(tsd, strict=True)
    mixed = trans(tokens, src_key_padding_mask=~valid)
    mixed = _ln(mixed, sd, "out_norm")

    map_vec = _proj(torch.cat([mixed[:, 1], mixed[:, 2], mixed[:, 10]], -1), sd, "map_neck")
    mate_vec = _proj(torch.cat([mixed[:, 3], mixed[:, 4]], -1), sd, "mate_neck")
    enemy_mask = valid[:, 5:8].unsqueeze(-1).float()
    enemy_vec = (F.linear(mixed[:, 5:8], sd["enemy_slot_proj.weight"]) * enemy_mask).sum(1)
    has_enemy = enemy_mask.any(1).float()
    enemy_vec = enemy_vec + (1.0 - has_enemy) * sd["enemy_absent"]
    hud_vec = mixed[:, 8]
    cue_vec = mixed[:, 9] * valid[:, 9:10].float()
    enc = _lin(torch.cat([mixed[:, 0], map_vec, mate_vec, enemy_vec, hud_vec, cue_vec], -1), sd, "out_proj")

    # GRU, manual standard-gate convention.
    w_ih = sd["gru.params.grus.0.weight_ih_l0"]
    w_hh = sd["gru.params.grus.0.weight_hh_l0"]
    b_ih = sd["gru.params.grus.0.bias_ih_l0"]
    b_hh = sd["gru.params.grus.0.bias_hh_l0"]
    h = torch.zeros((obs.shape[0], 256)) if h0 is None else h0
    r = torch.sigmoid(enc @ w_ih[:256].T + b_ih[:256] + h @ w_hh[:256].T + b_hh[:256])
    z = torch.sigmoid(enc @ w_ih[256:512].T + b_ih[256:512] + h @ w_hh[256:512].T + b_hh[256:512])
    n = torch.tanh(enc @ w_ih[512:].T + b_ih[512:] + r * (h @ w_hh[512:].T + b_hh[512:]))
    h = (1 - z) * n + z * h
    x = torch.cat([h, enc], -1)
    x = _lin(x, sd, "mlp.params.0"); x = F.gelu(x, approximate="none"); x = _head_ln(x, sd, "mlp.params.1")
    x = _lin(x, sd, "mlp.params.3"); x = F.gelu(x, approximate="none"); x = _head_ln(x, sd, "mlp.params.4")
    return _lin(x, sd, "mlp.params.6"), h, enc


def make_obs(b=4):
    torch.manual_seed(1234)
    obs = torch.randn((b, 140)) * 0.2
    # enemy visible: first component of each enemy 17-vector
    s = 24 + 4 + 4 + 36
    for j in range(3):
        obs[:, s + j * 17] = torch.tensor([0.2, 0.8, 1.2, 0.0])[:b]
    s += 51
    obs[:, s + 2] = 1.0  # cue valid
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    sd_raw = ck["loss_agents"]
    jp = convert(sd_raw)
    sd = {}
    for k, v in sd_raw.items():
        if k.startswith(PREFIX + "0."):
            sd[k[len(PREFIX + "0."):]] = v
        elif k.startswith(PREFIX + "1."):
            sd[k[len(PREFIX + "1."):]] = v
    np_obs = make_obs().numpy()
    obs = torch.from_numpy(np_obs)
    ref, h, enc_ref = torch_actor(sd, obs)
    model = RmActor()
    jraw, jh = model.apply({"params": jp}, jnp.asarray(np_obs), jnp.zeros((4, 256)))
    from jaxmarl.environments.rm.actor import RmEntityEncoder
    enc_model = RmEntityEncoder()
    enc_jax = enc_model.apply({"params": jp["RmEntityEncoder_0"]}, jnp.asarray(np_obs))
    print("enc max abs", float(np.max(np.abs(enc_ref.detach().numpy() - np.asarray(enc_jax)))))
    print("ref shape", tuple(ref.shape), "jax shape", tuple(jraw.shape))
    print("raw max abs", float(np.max(np.abs(ref.detach().numpy() - np.asarray(jraw)))))
    print("carry max abs", float(np.max(np.abs(h.detach().numpy() - np.asarray(jh)))))
    print("ref", ref.detach().numpy()[:1, :4])
    print("jax", np.asarray(jraw)[:1, :4])


if __name__ == "__main__":
    main()
