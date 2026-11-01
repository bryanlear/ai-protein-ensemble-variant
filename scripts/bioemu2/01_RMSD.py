"""
RMSD from 8 trajectories against the shared starting frame
Plots all trajectoriy lines dimmed and solid line for the mean RMSD 🤫
e.g., python scripts/bioemu2/01_RMSD.py outputs/bioemu2/output_bioemu2/trajectory_output_1_WT
"""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bioemu2-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/bioemu2/output_bioemu2/trajectory_output_1_WT"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: input_dir/rmsd")
    parser.add_argument("--selection", default="protein and name CA",
                        help="MDTraj atom selection for both alignment and RMSD")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir or input_dir / "rmsd"
    paths = [input_dir / f"trajectory_{i}" for i in range(8)]
    trajectories = []
    for path in paths:
        trajectory = md.load(str(path / "samples.xtc"), top=str(path / "topology.cif"))
        if trajectory.n_frames != 101:
            raise ValueError(f"{path.name}: expected 101 frames; found {trajectory.n_frames}")
        if not np.isfinite(trajectory.xyz).all():
            raise ValueError(f"{path.name}: coordinates contain non-finite values")
        trajectories.append(trajectory)

    reference = trajectories[0][0]
    indices = reference.topology.select(args.selection)
    if len(indices) < 3:
        raise ValueError("Select at least three atoms for alignment")
    for path, trajectory in zip(paths, trajectories):
        if trajectory.topology != reference.topology:
            raise ValueError(f"{path.name}: topology differs from trajectory_0")
        
        # XTC coordinates = nm
        
        if not np.allclose(trajectory.xyz[0], reference.xyz[0], atol=0.001, rtol=0):
            raise ValueError(f"{path.name}: frame 0 differs from the shared reference")

    # Superpose each frame on same reference using carbon alphas
    # and calculate residuals 
    
    rmsd = []
    
    for trajectory in trajectories:
        aligned = trajectory.atom_slice(indices)
        selected_reference = reference.atom_slice(indices)
        aligned.superpose(selected_reference, frame=0)
        delta = aligned.xyz.astype(np.float64) - selected_reference.xyz.astype(np.float64)
        rmsd.append(10.0 * np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=1)))
    rmsd = np.asarray(rmsd)
    mean_rmsd = rmsd.mean(axis=0)
    time_ns = np.arange(101, dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(output_dir / "rmsd.csv", np.column_stack([time_ns, rmsd.T, mean_rmsd]),
               delimiter=",", header="time_ns," + ",".join(
                   f"trajectory_{i}_rmsd_angstrom" for i in range(8))
               + ",mean_rmsd_angstrom", comments="")

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    
    for i, values in enumerate(rmsd):
        ax.plot(time_ns, values, label=f"Trajectory {i}", linewidth=1.1, alpha=0.2)
    ax.plot(time_ns, mean_rmsd, label="Mean (8 trajectories)",
            color="black", linewidth=2.5, alpha=1.0, zorder=10)
    label = "Cα" if args.selection == "protein and name CA" else args.selection
    ax.set(title=f"{input_dir.name} — {label} RMSD",
           xlabel="Time (ns)", ylabel="RMSD (Å)", xlim=(0, 100), ylim=(0, None))
    ax.text(0.01, 0.98, "Reference: trajectory 0, frame 0 • aligned on selected atoms",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(output_dir / "rmsd.png", dpi=200)
    fig.savefig(output_dir / "rmsd.pdf")
    plt.close(fig)
    print(f"Checked 8 trajectories: 101 frames each, shared frame 0, {len(indices)} selected atoms.")
    print(f"Frame 0 RMSD range: {rmsd[:, 0].min():.6f}–{rmsd[:, 0].max():.6f} Å")
    print(f"Saved plot and CSV data to {output_dir}")

if __name__ == "__main__":
    main()
