"""Train-stage-only JAX benchmark on a fixed synthetic batch.

This deliberately skips the environment.  It runs the same actor/critic
architecture and optimizer-update shape as the JAX trainer, on data generated
by ``.opencode/gen_train_bench.py``.  It is a compute benchmark, not a PPO
correctness test.
"""
import argparse
import ctypes
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxmarl.environments.rm.actor import RmActor
from jaxmarl.environments.rm.critic import RmCritic

N = 6
H = 256


def make_batches(data, mb):
    obs = data["obs"]          # [T,B,N,140]
    state = data["state"]
    target = data["target"]
    target_v = data["target_v"]
    T, B = obs.shape[:2]
    n_mb = B // mb
    assert n_mb * mb == B
    def pack(x, dim):
        # [T,B,N,...] -> [n_mb,T,mb*N,...]
        x = x[:, : n_mb * mb].reshape((T, n_mb, mb) + x.shape[2:])
        return np.transpose(x, (1, 0, 2) + tuple(range(3, x.ndim)))
    obs_b = pack(obs, 0).reshape((n_mb, T, mb * N, 140))
    state_b = pack(state, 0).reshape((n_mb, T, mb * N, 191))
    target_b = pack(target, 0).reshape((n_mb, T, mb * N, 14))
    target_v_b = pack(target_v, 0).reshape((n_mb, T, mb * N, 6))
    h0 = np.zeros((n_mb, mb * N, H), dtype=np.float32)
    return obs_b, state_b, target_b, target_v_b, h0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="parity/bench_train_data.npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--mb", type=int, default=8)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--part", choices=("both", "actor", "critic"), default="both")
    p.add_argument("--remat", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--json", default="")
    p.add_argument("--ncu-profile", action="store_true",
                   help="start/stop CUDA profiling around the selected run for ncu --profile-from-start off")
    a = p.parse_args()

    data = np.load(a.data)
    obs_b, state_b, target_b, target_v_b, h0 = make_batches(data, a.mb)
    T, B = data["obs"].shape[:2]
    n_mb = obs_b.shape[0]
    print(f"data T={T} B={B} agents={N} mb={a.mb} n_mb={n_mb} epochs={a.epochs}", flush=True)

    key = jax.random.PRNGKey(0)
    model_dtype = jnp.bfloat16 if a.bf16 else None
    actor = RmActor(dtype=model_dtype)
    k1, k2 = jax.random.split(key)
    actor_params = actor.init(k1, jnp.zeros((N, 140), jnp.float32), jnp.zeros((N, H), jnp.float32))
    critic = RmCritic(dtype=model_dtype)
    critic_params = critic.init(k2, jnp.zeros((N, 191), jnp.float32))
    opt = optax.adam(3e-4)
    params = (actor_params, critic_params)
    opt_state = opt.init(params)

    def actor_seq(params, obs, h0):
        # obs [T,MB,140], h0 [MB,256]
        T_, MB = obs.shape[:2]
        enc = actor.apply(params, obs.reshape((T_ * MB, -1)), method=RmActor.encode)
        enc = enc.reshape((T_, MB, -1))
        def body(carry, x):
            e, h = x
            new_h, carry = actor.apply(params, e, carry, method=RmActor.gru_step)
            carry = jnp.where(h[:, None] > 0.5, jnp.zeros_like(carry), carry)
            return carry, new_h
        _, hs = jax.lax.scan(body, h0, (enc, jnp.zeros((T_, MB), jnp.float32)))
        raw = actor.apply(
            params, hs.reshape((T_ * MB, -1)), enc.reshape((T_ * MB, -1)),
            method=RmActor.head,
        )
        return raw

    def loss(params, batch):
        ap, cp = params
        obs, state, target, target_v, h = batch
        actor_loss = jnp.array(0.0, dtype=jnp.float32)
        critic_loss = jnp.array(0.0, dtype=jnp.float32)
        if a.part in ("both", "actor"):
            raw = actor_seq(ap, obs, h).reshape((T * a.mb * N, 14))
            actor_loss = jnp.mean((raw - target.reshape((-1, 14))) ** 2)
        if a.part in ("both", "critic"):
            def critic_eval(cp, st):
                return critic.apply(cp, st)
            if a.remat:
                critic_eval = jax.checkpoint(critic_eval)
            val = critic_eval(cp, state.reshape((T * a.mb * N, 191)))
            critic_loss = jnp.mean((val - target_v.reshape((-1, 6))) ** 2)
        return actor_loss + critic_loss, (actor_loss, critic_loss)

    @jax.jit
    def update_step(params, opt_state, batch):
        (loss_value, aux), grads = jax.value_and_grad(loss, has_aux=True)(params, batch)
        updates, new_opt = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, new_opt, loss_value, aux

    @jax.jit
    def run_epochs(params, opt_state, obs, state, target, target_v, h0):
        def epoch(carry, _):
            p, o = carry
            def mb(carry2, batch):
                p2, o2 = carry2
                p2, o2, loss, aux = update_step(p2, o2, batch)
                return (p2, o2), (loss, aux)
            (p, o), scan_out = jax.lax.scan(
                mb, (p, o),
                (obs, state, target, target_v, h0),
            )
            return (p, o), jnp.mean(scan_out[0])
        return jax.lax.scan(epoch, (params, opt_state), None, length=a.epochs)

    batch = tuple(jnp.asarray(x) for x in (obs_b, state_b, target_b, target_v_b, h0))
    # Warm up the exact full-scan executable before timing it.
    t0 = time.perf_counter()
    (_, _), warm_losses = run_epochs(params, opt_state, *batch)
    jax.block_until_ready(warm_losses)
    compile_one = time.perf_counter() - t0
    cuda = None
    if a.ncu_profile:
        cuda = ctypes.CDLL("libcuda.so.1")
        # ncu --profile-from-start off will only collect this region.
        cuda.cuProfilerStart()
    t0 = time.perf_counter()
    (_, _), losses = run_epochs(params, opt_state, *batch)
    jax.block_until_ready(losses)
    total = time.perf_counter() - t0
    if cuda is not None:
        cuda.cuProfilerStop()
    print(f"warmup={compile_one:.3f}s selected_run={total:.3f}s "
          f"updates={a.epochs*n_mb} mean_loss={float(jnp.mean(losses)):.6f}", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"framework": "jax", "part": a.part, "T": int(T), "B": int(B),
                       "epochs": a.epochs, "mb": a.mb, "updates": a.epochs*n_mb,
                       "compile_one_sec": compile_one, "selected_run_sec": total,
                       "mean_loss": float(jnp.mean(losses))}, f, indent=2)


if __name__ == "__main__":
    main()
