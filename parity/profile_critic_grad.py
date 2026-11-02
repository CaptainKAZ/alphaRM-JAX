"""Critic chunk-gradient-accumulation benchmark.

Unlike profile_critic_jax.py, this does a forward+backward per chunk and
accumulates gradients, so the full 9600-state activation graph is never live.
This is the shape we would need in the real PPO update.
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
    p.add_argument("--chunk", type=int, default=1600)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--precision", choices=("default", "float32", "tensorfloat32", "bfloat16"), default="default")
    p.add_argument("--trace", default="")
    args = p.parse_args()
    if args.precision != "default":
        jax.config.update("jax_default_matmul_precision", args.precision)
    model_dtype = jnp.bfloat16 if args.precision == "bfloat16" else None

    if args.batch % args.chunk != 0:
        raise SystemExit("batch must be divisible by chunk for this benchmark")

    n_chunks = args.batch // args.chunk
    state = jnp.zeros((args.batch, 191), dtype=jnp.float32)
    target = jnp.zeros((args.batch, N), dtype=jnp.float32)
    state_chunks = state.reshape(n_chunks, args.chunk, 191)
    target_chunks = target.reshape(n_chunks, args.chunk, N)

    critic = RmCritic(dtype=None if model_dtype is None else model_dtype)
    params = critic.init(jax.random.PRNGKey(0), state[:1])
    opt = optax.adam(3e-4)
    opt_state = opt.init(params)

    def loss_chunk(p, x, y):
        value = critic.apply(p, x)
        return jnp.mean((value - y) ** 2)

    def update(p, o, x, y):
        def body(carry_grads, chunk):
            xc, yc = chunk
            l, g = jax.value_and_grad(loss_chunk)(p, xc, yc)
            carry_grads = jax.tree.map(jnp.add, carry_grads, g)
            return carry_grads, l

        init_grads = jax.tree.map(jnp.zeros_like, p)
        grads, losses = jax.lax.scan(
            body, init_grads, (x, y)
        )
        grads = jax.tree.map(lambda g: g / n_chunks, grads)
        updates, o = opt.update(grads, o, p)
        return optax.apply_updates(p, updates), o, jnp.mean(losses)

    @jax.jit
    def run(p, o, x, y):
        def body(carry, _):
            p, o = carry
            p, o, l = update(p, o, x, y)
            return (p, o), l

        return jax.lax.scan(body, (p, o), None, length=args.steps)

    # Warmup: compile the full scan before timing.  Without this, the first
    # timing includes XLA compilation and is not a steady-state measurement.
    (warm_p, warm_o), warm_losses = run(params, opt_state, state_chunks, target_chunks)
    jax.block_until_ready(warm_losses)

    if args.trace:
        with jax.profiler.trace(args.trace, create_perfetto_trace=True):
            (_, _), traced_losses = run(
                warm_p, warm_o, state_chunks, target_chunks
            )
            jax.block_until_ready(traced_losses)

    t0 = time.perf_counter()
    (_, _), losses = run(warm_p, warm_o, state_chunks, target_chunks)
    jax.block_until_ready(losses)
    dt = time.perf_counter() - t0
    print(
        f"batch={args.batch} chunk={args.chunk} chunks={n_chunks} "
        f"steps={args.steps} time={dt:.3f}s per_step={dt / args.steps:.3f}s "
        f"mean_loss={float(jnp.mean(losses)):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
