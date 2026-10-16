import mdtraj as md
import numpy as np

traj = md.load_xtc("outputs/my_protein/samples.xtc", top="outputs/my_protein/topology.pdb",)

ca_indices = traj.topology.select("name CA") #select all Carbon alphas 

if len(ca_indices) == 0:
    raise ValueError("No alpha males found, check testosterone levels")

print("Frames:", traj.n_frames)
print("C-alpha atoms:", len(ca_indices))

reference = traj[0]#reference is frame 0

aligned = traj.slice(np.arange(traj.n_frames), copy=True,)
aligned.superpose(reference, frame=0, atom_indices=ca_indices, ref_atom_indices=ca_indices, parallel=True,)
aligned.save_xtc("samples_full_length_ca_aligned.xtc")
