import mdtraj as md
import numpy as np
from bioemu_benchmarks.utils import get_physical_traj_indices

### BioEmu filtering is already on. This script loads the retained conformations using the .xtc and .pdb output files 
### from BioEmu and applies the filters again just to make sure they are valid for downstream analysis
 
traj = md.load_xtc("outputs/my_protein/samples.xtc", top="outputs/my_protein/topology.pdb",)

assert np.isfinite(traj.xyz).all()

valid_indices = get_physical_traj_indices(traj)
valid_traj = traj[valid_indices]

print("Total structures:", traj.n_frames)
print("Valid structures:", valid_traj.n_frames)
print("Rejected structures:", traj.n_frames - valid_traj.n_frames)

valid_traj.save_xtc("samples_valid.xtc")

### samples_valid.xtc filtered out 2 extra structures fromt he 781/1000 retained by BioEmu
