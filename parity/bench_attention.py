"""Microbenchmark Flax SelfAttention vs jax.nn.dot_product_attention.

Diagnostic only.  Run in the GPU container.
"""

import argparse
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def timed(fn, *args):
    fn(*args)[0].block_until_ready()
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=19200)
    p.add_argument("--seq", type=int, default=7)
    p.add_argument("--d", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    args = p.parse_args()

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (args.batch, args.seq, args.d), jnp.float32)
    target = jax.random.normal(key, (args.batch, args.seq, args.d), jnp.float32)

    layer = nn.SelfAttention(
        num_heads=args.heads,
        qkv_features=args.d,
        out_features=args.d,
        deterministic=True,
    )
    params = layer.init(key, x)

    def flax_loss(params, x):
        y = layer.apply(params, x)
        return jnp.mean((y - target) ** 2)

    @jax.jit
    def flax_step(params, x):
        loss, grads = jax.value_and_grad(flax_loss)(params, x)
        return loss, grads

    # Warm up the exact executable.
    flax_step(params, x)
    t0 = time.perf_counter()
    loss, _ = flax_step(params, x)
    jax.block_until_ready(loss)
    flax_time = time.perf_counter() - t0
    print(f"flax SelfAttention B={args.batch} T={args.seq} d={args.d}: "
          f"fwd+bwd={flax_time:.3f}s loss={float(loss):.6f}", flush=True)

    # Same q/k/v shape directly through jax.nn.dot_product_attention.
    q = jax.random.normal(key, (args.batch, args.heads, args.seq, args.d // args.heads))
    k = jax.random.normal(key, q.shape)
    v = jax.random.normal(key, q.shape)

    def direct_loss(q, k, v):
        y = jax.nn.dot_product_attention(q, k, v)
        return jnp.mean(y ** 2)

    for impl in (None, "xla", "cudnn"):
        try:
            fn = jax.jit(jax.value_and_grad(direct_loss))
            fn(q, k, v)[0].block_until_ready()
            t0 = time.perf_counter()
            out, _ = fn(q, k, v)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            print(f"jax.nn.dot_product_attention impl={impl!r}: "
                  f"fwd+bwd={dt:.3f}s", flush=True)
        except Exception as e:
            print(f"jax.nn.dot_product_attention impl={impl!r}: "
                  f"ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
