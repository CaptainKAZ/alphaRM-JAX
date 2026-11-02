"""Chunked JAX critic benchmark.

This is diagnostic code, not part of the training path.  It measures the
critic with an explicit state chunk, mirroring the PyTorch RmCritic chunk=2048
path.  Run inside the JaxMARL checkout with the GPU container.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import optax

from jaxmarl.environments.rm.critic import RmCritic

N = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=9600)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--remat", action="store_true")
    args = p.parse_args()

    state = jnp.zeros((args.batch, 191), dtype=jnp.float32)
    target = jnp.zeros((args.batch, 6), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    critic = RmCritic()
    params = critic.init(key, state[:1])
    opt = optax.adam(3e-4)
    opt_state = opt.init(params)

    n_chunks = (args.batch + args.chunk - 1) // args.chunk
    padded = n_chunks * args.chunk
    if padded != args.batch:
        state = jnp.pad(state, ((0, padded - args.batch), (0, 0)))
        target = jnp.pad(target, ((0, padded - args.batch), (0, 0)))
    state_chunks = state.reshape(n_chunks, args.chunk, 191)
    target_chunks = target.reshape(n_chunks, args.chunk, N)

    def value_chunk(p, x):
        return critic.apply(p, x)

    @jax.checkpoint
    def value_chunk_ckpt(p, x):
        return critic.apply(p, x)

    def loss(p, x, y):
        def body(_, chunk_x):
            if args.remat:
                return None, value_chunk_ckpt(p, chunk_x)
            return None, value_chunk(p, chunk_x)

        _, vals = jax.lax.scan(body, None, x)
        values = vals.reshape((-1, N))[: args.batch]
        return jnp.mean((values - target[: args.batch]) ** 2)

    @jax.jit
    def update(p, o, x):
        loss_value, grads = jax.value_and_grad(loss)(p, x, target_chunks)
        updates, o = opt.update(grads, o, p)
        return optax.apply_updates(p, updates), o, loss_value

    @jax.jit
    def run(p, o, x):
        def body(carry, _):
            p, o = carry
            p, o, l = update(p, o, x)
            return (p, o), l

        return jax.lax.scan(body, (p, o), None, length=args.steps)

    (warm_p, warm_o), warm_losses = run(params, opt_state, state_chunks)
    jax.block_until_ready(warm_losses)

    t0 = time.perf_counter()
    (_, _), losses = run(warm_p, warm_o, state_chunks)
    jax.block_until_ready(losses)
    dt = time.perf_counter() - t0
    print(
        f"batch={args.batch} chunk={args.chunk} chunks={n_chunks} "
        f"steps={args.steps} time={dt:.3f}s "
        f"per_step={dt / args.steps:.3f}s "
        f"mean_loss={float(jnp.mean(losses)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
