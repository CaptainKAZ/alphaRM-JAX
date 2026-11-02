import numpy as np, torch
from ref_common import output_path, vmas_root
vmas_root()
from vmas.scenarios.rm_config import RmConfig as TCfg
from vmas.scenarios import rm_rules as R

torch.set_grad_enabled(False)
cfg=TCfg(); s=R.make_state(1,torch.device('cpu'),cfg); R.reset_state(s,mode='fixed')
raw=torch.tensor([[0.5,0.2,0.0,0.0,0.0,0.0]],dtype=torch.float32).repeat(6,1) # (6,6) but decode expects (B,6)
# decode each agent with its own raw; B=1
for i in range(6):
    v=R.decode_agent_action(s,i,raw[i:i+1],s.vel[:,i])
    s.vel[:,i]=v
# VMAS integrates holonomic velocity before step_post.
s.pos = torch.clamp(s.pos + s.vel * cfg.dt, -torch.tensor([cfg.bound_x,cfg.bound_y]), torch.tensor([cfg.bound_x,cfg.bound_y]))
R.step_post(s)
out={}
for k in ['pos','yaw','vel','v_pre','hp','heat','overheated','spin','alive','invuln','invuln_timer','respawn_timer','target','score','time','episode_elapsed','total_damage','occ','outcome','winner','global_state','reward_dense','reward_terminal']:
    v=getattr(s,k)
    if torch.is_tensor(v): v=v.detach().cpu().numpy()
    out[k]=np.asarray(v)
out['obs']=s.obs.detach().cpu().numpy()
np.savez(output_path('ref_fixed_step.npz'),**out)
print({k:out[k].shape for k in out})
