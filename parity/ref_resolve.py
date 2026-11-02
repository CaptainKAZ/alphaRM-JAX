"""Reference snapshots for VMAS resolve_contacts, for JAX parity."""
import numpy as np
import torch
from ref_common import output_path, vmas_root

vmas_root()
from vmas.scenarios.rm_config import RmConfig
from vmas.scenarios import rm_rules as R

torch.set_grad_enabled(False)
cfg = RmConfig()
s = R.make_state(1, torch.device("cpu"), cfg)
R.reset_state(s, mode="fixed")

cases = {
    "robot": (
        [[-0.05, 0.0], [0.05, 0.0], [-4.0, 3.0], [4.0, 3.0], [4.0, -3.0], [-4.0, -3.0]],
        [[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ),
    "cover": (
        [[-3.40, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, -3.0], [-4.0, -3.0], [0.0, -3.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ),
    "wall": (
        [[5.99, 0.0], [-4.0, 3.0], [4.0, 3.0], [4.0, -3.0], [-4.0, -3.0], [0.0, -3.0]],
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ),
}

out = {}
for name, (pos, vel) in cases.items():
    s.pos[:] = torch.tensor([pos], dtype=torch.float32)
    s.vel[:] = torch.tensor([vel], dtype=torch.float32)
    out[name + "_in_pos"] = s.pos.cpu().numpy().copy()
    out[name + "_in_vel"] = s.vel.cpu().numpy().copy()
    R.resolve_contacts(s)
    out[name + "_pos"] = s.pos.cpu().numpy().copy()
    out[name + "_vel"] = s.vel.cpu().numpy().copy()

np.savez(output_path("ref_resolve.npz"), **out)
for k, v in out.items():
    print(k, v)
