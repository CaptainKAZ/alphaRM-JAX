import time
import jax, jax.numpy as jnp
from jaxmarl.environments.rm.env import RmEnv
from jaxmarl.environments.rm.actor import RmActor
from jaxmarl.environments.rm.critic import RmCritic
from baselines.MAPPO.train_rm_jax import make_rollout

B, T = 512, 200
env = RmEnv(); N = env.num_agents
key = jax.random.PRNGKey(0)
key, *rk = jax.random.split(key, B + 1)
obs, state = jax.vmap(lambda k: env.reset(k, mode='fixed'))(jnp.stack(rk))
obs = jnp.stack([obs[a] for a in env.agents], axis=1)
h0 = jnp.zeros((B*N, 256), dtype=jnp.float32)
actor = RmActor(); critic = RmCritic()
ap = actor.init(jax.random.PRNGKey(1), jnp.zeros((1, 140)), jnp.zeros((1, 256)))['params']
cp = critic.init(jax.random.PRNGKey(2), jnp.zeros((1, 191)))['params']
rollout = make_rollout(env, actor, critic, {'B': B, 'T': T}, reset_mode='fixed')
league_mask = jnp.zeros((B,), dtype=jnp.bool_)
rj = jax.jit(
    lambda ap, bp, cp, st, ob, hh, hh_opp, k, fp, lm, tt:
    rollout(ap, bp, cp, st, ob, hh, hh_opp, k, fp, lm, tt)
)

def timeit(fn, *args, n=3):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); ts.append(time.perf_counter()-t)
    return ts

print('backend', jax.default_backend(), 'devices', jax.devices())
print('full rollout times', timeit(
    rj, ap, ap, cp, state, obs, h0, h0, key,
    jnp.asarray(1.0), league_mask, jnp.asarray(0, dtype=jnp.int32),
))

def env_scan(st, ob, keys):
    def body(carry, k):
        st, ob = carry
        actions = jnp.zeros((B, N, 6), dtype=st.pos.dtype)
        ad = {a: actions[:, i] for i, a in enumerate(env.agents)}
        stepl = jax.random.split(k, B)
        nobs, nst, rew, done, info = jax.vmap(env.step_env)(stepl, st, ad)
        nob = jnp.stack([nobs[a] for a in env.agents], axis=1)
        return (nst, nob), rew
    return jax.lax.scan(body, (st, ob), keys)

keys = jax.random.split(key, T)
es = jax.jit(env_scan)
print('env scan times', timeit(es, state, obs, keys))

def step_one(st, ob, k):
    actions = jnp.zeros((B, N, 6), dtype=st.pos.dtype)
    ad = {a: actions[:, i] for i, a in enumerate(env.agents)}
    nobs, nst, rew, done, info = jax.vmap(env.step_env)(jax.random.split(k, B), st, ad)
    nob = jnp.stack([nobs[a] for a in env.agents], axis=1)
    return nst, nob
sj = jax.jit(step_one)
st, ob = sj(state, obs, key)
for _ in range(3): st, ob = sj(st, ob, key); jax.block_until_ready(ob)
ts=[]; 
for _ in range(200):
    t=time.perf_counter(); st,ob=sj(st,ob,key); jax.block_until_ready(ob); ts.append(time.perf_counter()-t)
print('python-loop env total', sum(ts), 'per-step', sum(ts)/len(ts))
