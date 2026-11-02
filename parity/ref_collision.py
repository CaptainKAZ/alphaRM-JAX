import numpy as np, torch
from ref_common import output_path, vmas_root
vmas_root()
from vmas.scenarios.rm_config import RmConfig as TCfg
from vmas.scenarios import rm_rules as R

torch.set_grad_enabled(False)
cfg=TCfg(); s=R.make_state(1,torch.device('cpu'),cfg); R.reset_state(s,mode='fixed')
pos=torch.tensor([[[0.0,0.0],[0.50,0.0],[-4.0,3.0],[4.0,3.0],[4.0,-3.0],[-4.0,-3.0]]],dtype=torch.float32)
s.pos[:]=pos
s.vel[:]=0.0
s.v_pre[:]=0.0
s.prev_pair_contact[:]=False
s.prev_wall_contact[:]=False
s.time[:]=180.0
s.episode_elapsed[:]=0.0
R.step_post(s, generator=torch.Generator().manual_seed(0))
out={}
for k in ['pos','vel','v_pre','hp','reward_dense','reward_terminal','outcome','winner','foul_agent','foul_intensity','done']:
    v=getattr(s,k)
    if torch.is_tensor(v): v=v.detach().cpu().numpy()
    out[k]=np.asarray(v)
np.savez(output_path('ref_collision.npz'),**out)
print('ref pos',out['pos'])
print('ref outcome',out['outcome'])
