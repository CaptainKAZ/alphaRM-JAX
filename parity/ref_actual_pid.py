import numpy as np, torch
from ref_common import output_path, vmas_root
vmas_root()
from vmas import make_env
torch.set_grad_enabled(False)
env=make_env('rm', num_envs=1, device='cpu', seed=0)
env.reset()
raw=torch.tensor([[0.3,0.1,0.0,0.0,0.0,0.0]],dtype=torch.float32)
out={k:[] for k in ['pos','vel','v_pre','yaw','hp','heat','target','score','time','episode_elapsed','outcome','done']}
for t in range(100):
    env.step([raw.clone() for _ in range(6)])
    s=env.scenario.s
    for k in out:
        v=getattr(s,k)
        out[k].append(v.detach().cpu().numpy().copy())
for k in out:
    out[k]=np.stack(out[k])
np.savez(output_path('ref_actual_pid.npz'), **out)
print({k:out[k].shape for k in out})
