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
aligned.save_xtc("outputs/my_protein/samples_full_length_ca_aligned.xtc")

n = traj.n_frames
rmsd_matrix = np.empty((n, n), dtype=np.float32)

for reference_frame in range(n):
    rmsd_matrix[reference_frame] = md.rmsd(traj, traj, frame=reference_frame, atom_indices=ca_indices, parallel=True,)

# MDTraj RMSD = nanometers. To convert to angstroms:
rmsd_matrix *= 10.0

rmsd_matrix = 0.5 * (rmsd_matrix + rmsd_matrix.T)
np.fill_diagonal(rmsd_matrix, 0.0)

np.save("outputs/my_protein/full_length_ca_pairwise_rmsd_angstrom.npy", rmsd_matrix)