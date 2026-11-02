import numpy as np, torch
from ref_common import output_path, vmas_root
vmas_root()
from vmas import make_env
from vmas.scenarios import rm_rules as R
torch.set_grad_enabled(False)
env=make_env('rm',num_envs=1,device='cpu',seed=0)
env.reset()
s=env.scenario.s
pos=torch.tensor([[[-0.05,0.0],[0.05,0.0],[-4.,3.],[4.,3.],[4.,-3.],[-4.,-3.]]],dtype=torch.float32)
s.pos[:]=pos
s.vel[:]=0; s.v_pre[:]=0; s.time[:]=180; s.episode_elapsed[:]=0
s.prev_pair_contact[:]=False; s.prev_wall_contact[:]=False; s.hp[:]=s.cfg.hp_max
s.alive[:]=True; s.invuln[:]=False; s.invuln_timer[:]=0; s.heat[:]=0; s.overheated[:]=False
s.spin[:]=0; s.target[:]=0; s.target_cmd[:]=0; s.fire_cmd[:]=False
# sync VMAS world agents, otherwise env.step overwrites s.pos from the old agents
for i, ag in enumerate(env.world.agents):
    ag.state.pos[:] = s.pos[:, i]
    ag.state.vel[:] = s.vel[:, i]
    ag.controller.reset()
raw=torch.zeros((1,6),dtype=torch.float32); raw[0,0]=1.0; raw[0,1]=-1.0
# Per-agent action: only red0 and red1 move toward each other.
actions=[]
for i in range(6):
    a=torch.zeros((1,6),dtype=torch.float32)
    if i==0: a[0,0]=1.0
    if i==1: a[0,1]=-1.0
    actions.append(a)
env.step(actions)
s=env.scenario.s
out={k:getattr(s,k).detach().cpu().numpy() for k in ['pos','vel','v_pre','hp','reward_dense','reward_terminal','outcome','winner','foul_agent','foul_intensity','done']}
np.savez(output_path('ref_collision_full.npz'),**out)
print('pos',out['pos']); print('vel',out['vel']); print('vpre',out['v_pre']); print('outcome',out['outcome'])
