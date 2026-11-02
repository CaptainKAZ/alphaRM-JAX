"""JAX/Flax actor for the RM 3v3 environment.

This mirrors the current PyTorch actor used by ``BenchMARL/benchmarl/models/rm_encoder.py``
and ``gru_rm.yaml``:

    obs(140) -> 11 entity tokens -> 2-layer pre-norm Transformer (d=128, 4 heads)
             -> six semantic pools -> 14 policy parameters
             -> 1-layer GRU(256), input passthrough
             -> 2x256 MLP -> [cont(8), target(4), fire(2)]

The module deliberately returns raw policy parameters.  Distribution
construction (bounded tanh normal, target/fire masks) lives in ``act``.
"""

from __future__ import annotations

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .config import RmConfig

N_TOKENS = 11
SELF_DIM = 24
SPAWN_DIM = 4
CENTER_DIM = 4
MATE_DIM = 18
ENEMY_DIM = 17
CUE_DIM = 3
HUD_DIM = 10
OBSTACLE_DIM = 8
SCALAR_DIM = (
    SELF_DIM + SPAWN_DIM + CENTER_DIM
    + 2 * MATE_DIM + 3 * ENEMY_DIM + CUE_DIM + HUD_DIM
)
OBS_DIM = SCALAR_DIM + OBSTACLE_DIM
PARAM_DIM = 14


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


class _TokenMLP(nn.Module):
    d_model: int = 128
    dtype: object = None

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model, dtype=self.dtype)(x)
        x = _gelu(x)
        return nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)


class RmEntityEncoder(nn.Module):
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 256
    dtype: object = None

    @nn.compact
    def __call__(self, obs):
        # obs: [B, 140]
        scalar = obs[:, :SCALAR_DIM]
        obstacle = obs[:, SCALAR_DIM:]
        s = 0
        self_f = scalar[:, s:s + SELF_DIM]
        s += SELF_DIM
        spawn_f = scalar[:, s:s + SPAWN_DIM]
        s += SPAWN_DIM
        center_f = scalar[:, s:s + CENTER_DIM]
        s += CENTER_DIM
        mates_f = scalar[:, s:s + 2 * MATE_DIM].reshape(-1, 2, MATE_DIM)
        s += 2 * MATE_DIM
        enemy_f = scalar[:, s:s + 3 * ENEMY_DIM].reshape(-1, 3, ENEMY_DIM)
        s += 3 * ENEMY_DIM
        cue_f = scalar[:, s:s + CUE_DIM]
        s += CUE_DIM
        hud_f = scalar[:, s:s + HUD_DIM]
        s += HUD_DIM
        assert s == SCALAR_DIM

        self_t = _TokenMLP(self.d_model, dtype=self.dtype)(self_f)[:, None, :]
        spawn_t = _TokenMLP(self.d_model, dtype=self.dtype)(spawn_f)[:, None, :]
        center_t = _TokenMLP(self.d_model, dtype=self.dtype)(center_f)[:, None, :]
        mate_t = _TokenMLP(self.d_model, dtype=self.dtype)(mates_f.reshape(-1, MATE_DIM))
        mate_t = mate_t.reshape(-1, 2, self.d_model)
        enemy_t = _TokenMLP(self.d_model, dtype=self.dtype)(enemy_f.reshape(-1, ENEMY_DIM))
        enemy_t = enemy_t.reshape(-1, 3, self.d_model)
        hud_t = _TokenMLP(self.d_model, dtype=self.dtype)(hud_f)[:, None, :]
        cue_t = _TokenMLP(self.d_model, dtype=self.dtype)(cue_f)[:, None, :]
        obs_t = _TokenMLP(self.d_model, dtype=self.dtype)(obstacle)[:, None, :]

        tokens = jnp.concatenate(
            [self_t, spawn_t, center_t, mate_t, enemy_t, hud_t, cue_t, obs_t],
            axis=1,
        )
        d = self.d_model
        compute_dtype = self.dtype or jnp.float32
        type_ids = jnp.asarray([0, 1, 2, 3, 3, 4, 4, 4, 5, 6, 7])
        type_emb = nn.Embed(
            num_embeddings=8, features=d, name="type_emb", dtype=compute_dtype
        )(type_ids)
        slot_emb = self.param(
            "slot_emb", nn.initializers.normal(stddev=0.02), (N_TOKENS, d)
        )
        tokens = (
            tokens
            + type_emb[None, :, :]
            + slot_emb.astype(compute_dtype)[None, :, :]
        )

        valid = jnp.ones((obs.shape[0], N_TOKENS), dtype=jnp.bool_)
        valid = valid.at[:, 5:8].set(enemy_f[..., 0] > 0.5)
        valid = valid.at[:, 9].set(cue_f[..., 2] > 0.5)
        attn_mask = valid[:, None, None, :]

        for _ in range(self.n_layers):
            x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(tokens)
            attn = nn.SelfAttention(
                num_heads=self.n_heads,
                qkv_features=d,
                out_features=d,
                deterministic=True,
                dtype=self.dtype,
                param_dtype=jnp.float32,
            )(x, mask=attn_mask)
            tokens = tokens + attn
            x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(tokens)
            x = nn.Dense(self.ffn_dim, dtype=self.dtype)(x)
            x = _gelu(x)
            x = nn.Dense(d, dtype=self.dtype)(x)
            tokens = tokens + x
        tokens = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(tokens)

        self_vec = tokens[:, 0]
        map_vec = nn.Dense(d, dtype=self.dtype)(jnp.concatenate([tokens[:, 1], tokens[:, 2], tokens[:, 10]], axis=-1))
        map_vec = _gelu(map_vec)
        map_vec = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(map_vec)
        mate_vec = nn.Dense(d, dtype=self.dtype)(jnp.concatenate([tokens[:, 3], tokens[:, 4]], axis=-1))
        mate_vec = _gelu(mate_vec)
        mate_vec = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(mate_vec)

        enemy_mask = valid[:, 5:8][..., None].astype(compute_dtype)
        enemy_vec = nn.Dense(d, use_bias=False, dtype=self.dtype)(tokens[:, 5:8]) * enemy_mask
        enemy_vec = enemy_vec.sum(axis=1)
        has_enemy = enemy_mask.any(axis=1).astype(compute_dtype)
        absent = self.param("enemy_absent", nn.initializers.zeros, (d,)).astype(
            compute_dtype
        )
        enemy_vec = enemy_vec + (jnp.ones_like(has_enemy) - has_enemy) * absent

        hud_vec = tokens[:, 8]
        cue_vec = tokens[:, 9] * valid[:, 9:10].astype(compute_dtype)
        out = jnp.concatenate(
            [self_vec, map_vec, mate_vec, enemy_vec, hud_vec, cue_vec], axis=-1
        )
        return nn.Dense(256, dtype=self.dtype)(out)


class RmActor(nn.Module):
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ffn_dim: int = 256
    hidden_size: int = 256
    dtype: object = None

    @nn.compact
    def encode(self, obs):
        return RmEntityEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ffn_dim=self.ffn_dim,
            dtype=self.dtype,
        )(obs)

    @nn.compact
    def gru_step(self, enc, carry):
        """Only the recurrent state transition; used inside lax.scan."""
        return nn.GRUCell(features=self.hidden_size, dtype=self.dtype, param_dtype=jnp.float32)(carry, enc)

    @nn.compact
    def head(self, h, enc):
        """Stateless 2x256 MLP + policy head, run over all timesteps in parallel."""
        x = jnp.concatenate([h, enc], axis=-1)
        x = nn.Dense(256, dtype=self.dtype)(x)
        x = _gelu(x)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
        x = nn.Dense(256, dtype=self.dtype)(x)
        x = _gelu(x)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)(x)
        return nn.Dense(PARAM_DIM, dtype=self.dtype)(x)

    @nn.compact
    def step(self, enc, carry):
        h, new_carry = self.gru_step(enc, carry)
        return self.head(h, enc), new_carry

    def __call__(self, obs, carry):
        return self.step(self.encode(obs), carry)

def split_params(raw):
    """Split raw actor output into the three action heads."""
    return raw[..., :8], raw[..., 8:12], raw[..., 12:14]


def bounded_params(raw):
    """PyTorch ``BoundedNormalParamExtractor`` equivalent for 4 continuous dims."""
    loc_raw, scale_raw = raw[..., :4], raw[..., 4:8]
    loc = 3.0 * jnp.tanh(loc_raw / 3.0)
    scale = 0.01 + 0.99 * jax.nn.sigmoid(scale_raw)
    return loc, scale


def act(model, params, carry, obs, key, action_mask=None, fire_mask=None,
        deterministic: bool = True, low=-1.0, high=1.0):
    """Sample/derive a 6-D VMAS action and return the new GRU carry."""
    raw, new_carry = model.apply({"params": params}, obs, carry)
    raw = raw.astype(jnp.float32)
    cont_raw, target_logits, fire_logits = split_params(raw)
    loc, scale = bounded_params(cont_raw)

    if deterministic:
        cont = low + (high - low) * 0.5 * (jnp.tanh(loc) + 1.0)
        if action_mask is not None:
            target_logits = jnp.where(action_mask, target_logits, -1e9)
        if fire_mask is not None:
            fire_logits = jnp.where(fire_mask, fire_logits, -1e9)
        target = jnp.argmax(target_logits, axis=-1)
        fire = jnp.argmax(fire_logits, axis=-1)
    else:
        if key.ndim >= 2:
            kk = jax.vmap(lambda k: jax.random.split(k, 2))(key)
            k1, k2 = kk[:, 0], kk[:, 1]
            z = jax.vmap(lambda k: jax.random.normal(k, loc.shape[1:]))(k1)
        else:
            k1, k2 = jax.random.split(key)
            z = jax.random.normal(k1, loc.shape)
        cont = low + (high - low) * 0.5 * (jnp.tanh(z) + 1.0)
        if action_mask is not None:
            target_logits = jnp.where(action_mask, target_logits, -1e9)
        if fire_mask is not None:
            fire_logits = jnp.where(fire_mask, fire_logits, -1e9)
        if key.ndim >= 2:
            target = jax.vmap(lambda k, l: jax.random.categorical(k, l))(k2, target_logits)
            fire = jax.vmap(lambda k, l: jax.random.categorical(k, l))(k2, fire_logits)
        else:
            target = jax.random.categorical(k2, target_logits, axis=-1)
            fire = jax.random.categorical(k2, fire_logits, axis=-1)

    return jnp.concatenate(
        [cont, target.astype(jnp.float32)[:, None], fire.astype(jnp.float32)[:, None]],
        axis=-1,
    ), new_carry


__all__ = [
    "OBS_DIM", "PARAM_DIM", "RmActor", "RmEntityEncoder",
    "split_params", "bounded_params", "act", "evaluate_raw",
]


def _masked_logits(logits, mask):
    if mask is None:
        return logits
    return jnp.where(mask, logits, -1e9)


def _continuous_logprob_entropy(loc, scale, action):
    # action = tanh(z), z ~ Normal(loc, scale), dims 0..3
    z = jnp.arctanh(jnp.clip(action, -0.999999, 0.999999))
    logdet = jnp.log(1.0 - jnp.tanh(z) ** 2 + 1e-6)
    logp = -0.5 * ((z - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)
    logp = jnp.sum(logp - logdet, axis=-1)
    ent = jnp.sum(jnp.log(scale) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) - logdet, axis=-1)
    return logp, ent


def sample_actions(model, params, obs, carry, key, target_mask=None, fire_mask=None):
    """Vectorised stochastic action + joint log-prob/entropy.

    obs: [M,140], carry: [M,256], key: [M,2].
    Returns action [M,6], logp [M], entropy [M], new_carry [M,256].
    """
    raw, new_carry = model.apply({"params": params}, obs, carry)
    raw = raw.astype(jnp.float32)
    cont_raw, target_logits, fire_logits = split_params(raw)
    loc, scale = bounded_params(cont_raw)
    target_logits = _masked_logits(target_logits, target_mask)
    fire_logits = _masked_logits(fire_logits, fire_mask)
    kk = jax.vmap(lambda k: jax.random.split(k, 3))(key)
    k0, k1, k2 = kk[:, 0], kk[:, 1], kk[:, 2]
    eps = jax.vmap(lambda k: jax.random.normal(k, (4,)))(k0)
    z = loc + scale * eps
    cont = jnp.tanh(z)
    target = jax.vmap(lambda k, l: jax.random.categorical(k, l))(k1, target_logits)
    fire = jax.vmap(lambda k, l: jax.random.categorical(k, l))(k2, fire_logits)
    action = jnp.concatenate(
        [cont, target.astype(jnp.float32)[:, None], fire.astype(jnp.float32)[:, None]],
        axis=-1,
    )
    logp_c, ent_c = _continuous_logprob_entropy(loc, scale, cont)
    logp_t = jnp.take_along_axis(
        jax.nn.log_softmax(target_logits, axis=-1), target[:, None], axis=-1
    )[:, 0]
    logp_f = jnp.take_along_axis(
        jax.nn.log_softmax(fire_logits, axis=-1), fire[:, None], axis=-1
    )[:, 0]
    ent_t = -jnp.sum(
        jax.nn.softmax(target_logits, axis=-1) * jax.nn.log_softmax(target_logits, axis=-1),
        axis=-1,
    )
    ent_f = -jnp.sum(
        jax.nn.softmax(fire_logits, axis=-1) * jax.nn.log_softmax(fire_logits, axis=-1),
        axis=-1,
    )
    logp = logp_c + logp_t + logp_f
    entropy = ent_c + ent_t + ent_f
    return action, logp, entropy, new_carry


def evaluate_raw(raw, action, target_mask=None, fire_mask=None):
    """Recompute log-prob/entropy from an already-computed raw actor output."""
    raw = raw.astype(jnp.float32)
    cont_raw, target_logits, fire_logits = split_params(raw)
    loc, scale = bounded_params(cont_raw)
    target_logits = _masked_logits(target_logits, target_mask)
    fire_logits = _masked_logits(fire_logits, fire_mask)
    cont = action[:, :4]
    logp_c, ent_c = _continuous_logprob_entropy(loc, scale, cont)
    target = jnp.rint(jnp.clip(action[:, 4], 0.0, 3.0)).astype(jnp.int32)
    fire = (action[:, 5] > 0.5).astype(jnp.int32)
    logp_t = jnp.take_along_axis(
        jax.nn.log_softmax(target_logits, axis=-1), target[:, None], axis=-1
    )[:, 0]
    logp_f = jnp.take_along_axis(
        jax.nn.log_softmax(fire_logits, axis=-1), fire[:, None], axis=-1
    )[:, 0]
    ent_t = -jnp.sum(
        jax.nn.softmax(target_logits, axis=-1) * jax.nn.log_softmax(target_logits, axis=-1),
        axis=-1,
    )
    ent_f = -jnp.sum(
        jax.nn.softmax(fire_logits, axis=-1) * jax.nn.log_softmax(fire_logits, axis=-1),
        axis=-1,
    )
    return logp_c + logp_t + logp_f, ent_c + ent_t + ent_f


def evaluate_actions(model, params, obs, carry, action, target_mask=None, fire_mask=None):
    """Recompute log-prob/entropy of stored actions for PPO."""
    raw, new_carry = model.apply({"params": params}, obs, carry)
    raw = raw.astype(jnp.float32)
    cont_raw, target_logits, fire_logits = split_params(raw)
    loc, scale = bounded_params(cont_raw)
    target_logits = _masked_logits(target_logits, target_mask)
    fire_logits = _masked_logits(fire_logits, fire_mask)
    cont = action[:, :4]
    logp_c, ent_c = _continuous_logprob_entropy(loc, scale, cont)
    target = jnp.rint(jnp.clip(action[:, 4], 0.0, 3.0)).astype(jnp.int32)
    fire = (action[:, 5] > 0.5).astype(jnp.int32)
    logp_t = jnp.take_along_axis(
        jax.nn.log_softmax(target_logits, axis=-1), target[:, None], axis=-1
    )[:, 0]
    logp_f = jnp.take_along_axis(
        jax.nn.log_softmax(fire_logits, axis=-1), fire[:, None], axis=-1
    )[:, 0]
    ent_t = -jnp.sum(
        jax.nn.softmax(target_logits, axis=-1) * jax.nn.log_softmax(target_logits, axis=-1),
        axis=-1,
    )
    ent_f = -jnp.sum(
        jax.nn.softmax(fire_logits, axis=-1) * jax.nn.log_softmax(fire_logits, axis=-1),
        axis=-1,
    )
    return logp_c + logp_t + logp_f, ent_c + ent_t + ent_f, new_carry
