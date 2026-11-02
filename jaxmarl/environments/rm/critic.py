"""JAX/Flax port of the RM team-canonical centralized critic.

Input:  global state [B, 191] = 6 agents * 30 + 11 global features.
Output: six values [B, 6], aligned with the environment's agent order.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

N_AGENTS = 6
BASE_AGENT_DIM = 22
OBSTACLE_DIM = 8
AGENT_DIM = BASE_AGENT_DIM + OBSTACLE_DIM
GLOBAL_DIM = 11
STATE_DIM = N_AGENTS * AGENT_DIM + GLOBAL_DIM
OWN = 3


def _canonicalize(agents, glob, team: int):
    if team == 1:
        obstacles = agents[..., BASE_AGENT_DIM:AGENT_DIM][..., [4, 5, 6, 7, 0, 1, 2, 3]]
        a = jnp.concatenate(
            [-agents[..., 0:6], agents[..., 6:BASE_AGENT_DIM], obstacles], axis=-1
        )
    else:
        a = agents
    own = slice(0, 3) if team == 0 else slice(3, 6)
    opp = slice(3, 6) if team == 0 else slice(0, 3)
    ca = jnp.concatenate([a[:, own], a[:, opp]], axis=1)
    zeros = jnp.zeros((ca.shape[0], OWN, 1), dtype=ca.dtype)
    ones = jnp.ones((ca.shape[0], OWN, 1), dtype=ca.dtype)
    ca = jnp.concatenate(
        [ca[..., :17], jnp.concatenate([zeros, ones], axis=1), ca[..., 18:]],
        axis=-1,
    )
    g = glob
    if team == 1:
        order = jnp.asarray([1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9])
        g = jnp.take(glob, order, axis=-1)
    return ca, g


class RmCritic(nn.Module):
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 3
    ffn_dim: int = 384
    head_hidden: int = 384
    dtype: object = None

    @nn.compact
    def __call__(self, state):
        m = state.shape[0]
        agents = state[:, : N_AGENTS * AGENT_DIM].reshape(m, N_AGENTS, AGENT_DIM)
        glob = state[:, N_AGENTS * AGENT_DIM:]
        with jax.named_scope("rm_critic/canonicalize"):
            ca0, cg0 = _canonicalize(agents, glob, 0)
            ca1, cg1 = _canonicalize(agents, glob, 1)
            ca = jnp.concatenate([ca0, ca1], axis=0)
            cg = jnp.concatenate([cg0, cg1], axis=0)
            # critic inputs are data, not trainable parameters; do not build a
            # backward scatter graph through the canonicalization/indexing ops
            ca = jax.lax.stop_gradient(ca)
            cg = jax.lax.stop_gradient(cg)
        with jax.named_scope("rm_critic/forward"):
            vals = self._forward_team(ca, cg)
        return jnp.concatenate([vals[:m, 0:3], vals[m:, 0:3]], axis=1).astype(jnp.float32)

    @nn.compact
    def _forward_team(self, agent, glob):
        d = self.d_model
        compute_dtype = self.dtype or jnp.float32
        agent = agent.astype(compute_dtype)
        glob = glob.astype(compute_dtype)
        # Match PyTorch RmCritic: team_emb is the own/opponent flag of the
        # canonical team, NOT the red/blue view id.  _canonicalize() always
        # produces [0,0,0,1,1,1], so use a constant 1-D index.  A per-token
        # index would make XLA emit a very expensive atomic scatter-add into
        # the tiny 2xd table during backward.
        team_ids = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int32)
        agent_token = (
            nn.Dense(d, dtype=compute_dtype)(agent)
            + nn.Embed(
                N_AGENTS, d, name="slot_emb", dtype=compute_dtype
            )(jnp.arange(N_AGENTS))[None]
            + nn.Embed(2, d, name="team_emb", dtype=compute_dtype)(team_ids)[
                None
            ]
        )
        global_token = nn.Dense(
            d, name="global_proj", dtype=compute_dtype
        )(glob)[:, None, :]
        tokens = jnp.concatenate([global_token, agent_token], axis=1)
        type_ids = jnp.asarray([0] + [1] * N_AGENTS)
        tokens = tokens + nn.Embed(
            2, d, name="type_emb", dtype=compute_dtype
        )(type_ids)[None]

        for _ in range(self.n_layers):
            x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(tokens)
            attn = nn.SelfAttention(
                num_heads=self.n_heads,
                qkv_features=d,
                out_features=d,
                deterministic=True,
                dtype=self.dtype,
                param_dtype=jnp.float32,
            )(x)
            tokens = tokens + attn
            x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(tokens)
            x = nn.Dense(self.ffn_dim, dtype=self.dtype)(x)
            x = nn.gelu(x, approximate=False)
            x = nn.Dense(d, dtype=self.dtype)(x)
            tokens = tokens + x
        tokens = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(tokens)
        x = tokens[:, 1:, :]
        x = nn.Dense(self.head_hidden, dtype=self.dtype)(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(1, dtype=self.dtype)(x)
        return x[..., 0]


__all__ = ["RmCritic", "STATE_DIM"]
