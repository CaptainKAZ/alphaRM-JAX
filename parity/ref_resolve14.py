import copy, torch, numpy as np
from ref_common import output_path, vmas_root
vmas_root()
from vmas import make_env
from vmas.scenarios import rm_rules as R
torch.set_grad_enabled(False)
env=make_env('rm', num_envs=1, device='cpu', seed=0); env.reset()
raw=torch.tensor([[0.3,0.1,0.0,0.0,0.0,0.0]],dtype=torch.float32)
for _ in range(13):
    env.step([raw.clone() for _ in range(6)])
s=env.scenario.s
pre=copy.deepcopy(s)
v=torch.stack([R.decode_agent_action(s,i,raw,s.vel[:,i].clone()) for i in range(6)],dim=1)
pre.pos = s.pos + v*env.scenario.cfg.dt
pre.vel = v
pre.v_pre = v
post=copy.deepcopy(pre); R.resolve_contacts(post)
np.savez(output_path('ref_resolve14.npz'),
         pre_pos=pre.pos.cpu().numpy(), pre_vel=pre.vel.cpu().numpy(),
         post_pos=post.pos.cpu().numpy(), post_vel=post.vel.cpu().numpy())
print('pre pos',pre.pos[0].numpy().round(5))
print('pre vel',pre.vel[0].numpy().round(5))
print('post pos',post.pos[0].numpy().round(5))
print('post vel',post.vel[0].numpy().round(5))
