import numpy as np, torch
from ref_common import output_path, vmas_root
vmas_root()
from vmas.scenarios.rm_config import RmConfig as TCfg
from vmas.scenarios import rm_rules as R
torch.set_grad_enabled(False)
cfg=TCfg(); s=R.make_state(1,torch.device('cpu'),cfg); R.reset_state(s,mode='fixed')
pos=torch.tensor([[[-0.05,0.0],[0.05,0.0],[-4.,3.],[4.,3.],[4.,-3.],[-4.,-3.]]],dtype=torch.float32)
s.pos[:]=pos; s.vel[:]=0; s.v_pre[:]=0; s.time[:]=180; s.episode_elapsed[:]=0
s.prev_pair_contact[:]=False; s.prev_wall_contact[:]=False; s.hp[:]=cfg.hp_max
s.alive[:]=True; s.invuln[:]=False; s.heat[:]=0; s.overheated[:]=False; s.spin[:]=0; s.target[:]=0
raw=torch.zeros((1,6),dtype=torch.float32); raw[0,0]=1.0; raw[0,1]=-1.0
for i in range(6):
    s.vel[:,i]=R.decode_agent_action(s,i,raw,s.vel[:,i])
s.pos=torch.clamp(s.pos+s.vel*cfg.dt,-torch.tensor([cfg.bound_x,cfg.bound_y]),torch.tensor([cfg.bound_x,cfg.bound_y]))
R.step_post(s,generator=torch.Generator().manual_seed(0))
out={k:(getattr(s,k).detach().cpu().numpy() if torch.is_tensor(getattr(s,k)) else np.asarray(getattr(s,k))) for k in ['pos','vel','v_pre','hp','reward_dense','reward_terminal','outcome','winner','foul_agent','foul_intensity','done']}
np.savez(output_path('ref_collision_move.npz'),**out)
print('pos',out['pos']); print('outcome',out['outcome'],'foul',out['foul_agent'])
