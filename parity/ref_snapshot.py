import numpy as np, torch
from ref_common import output_path, vmas_root
vmas_root()
from vmas.scenarios.rm_config import RmConfig as TCfg
from vmas.scenarios import rm_rules as R

torch.set_grad_enabled(False)
cfg = TCfg()
s = R.make_state(1, torch.device('cpu'), cfg)
R.reset_state(s, mode='fixed')
out = {}
for k in ['pos','yaw','vel','v_pre','hp','heat','overheated','spin','alive','invuln','invuln_timer','respawn_timer','target','score','time','episode_elapsed','total_damage','occ','outcome','winner','global_state']:
    v = getattr(s, k)
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    out[k] = np.asarray(v)
out['obs'] = s.obs.detach().cpu().numpy()
np.savez(output_path('ref_fixed_reset.npz'), **out)
print({k: out[k].shape for k in out})
