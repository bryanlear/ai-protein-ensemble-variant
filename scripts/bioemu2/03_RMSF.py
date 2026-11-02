"""
 python scripts/bioemu2/03_RMSF.py outputs/bioemu2/output_bioemu2/trajectory_output_1_WT
"""
import argparse
import csv
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
SELECTION = "protein and name CA"
N_REPLICAS = 8
N_FRAMES = 101

def calculate_rmsf(coordinates_nm):
    coordinates = np.asarray(coordinates_nm, dtype=np.float64)
    delta = coordinates - coordinates.mean(axis=0)
    return 10.0 * np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=0))

def analyze(input_dir):
    curves = []
    reference = None
    topology = None
    for replica in range(N_REPLICAS):
        path = input_dir / f"trajectory_{replica}"
        for name in ("samples.xtc", "topology.cif"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"Input file not found: {path / name}")
        trajectory = md.load(str(path / "samples.xtc"), top=str(path / "topology.cif"))
        if trajectory.n_frames != N_FRAMES:
            raise ValueError(f"{path.name}: expected {N_FRAMES} frames; found {trajectory.n_frames}")
        if not np.isfinite(trajectory.xyz).all():
            raise ValueError(f"{path.name}: coordinates contain non-finite values")
        if topology is None:
            topology = trajectory.topology
            initial_xyz = trajectory.xyz[0].copy()
            indices = topology.select(SELECTION)
            if len(indices) < 3:
                raise ValueError("At least three C-alpha atoms are required for alignment")
        elif trajectory.topology != topology:
            raise ValueError(f"{path.name}: topology differs from trajectory_0")
        if not np.allclose(trajectory.xyz[0], initial_xyz, atol=0.001, rtol=0):
            raise ValueError(f"{path.name}: frame 0 differs from the shared starting position")
        ca = trajectory.atom_slice(indices)
        if reference is None:
            reference = ca[0]
        ca.superpose(reference, frame=0)
        coordinates = ca.xyz.astype(np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError(f"{path.name}: alignment produced non-finite coordinates")
        curves.append(calculate_rmsf(coordinates))
    return reference.topology, indices, np.asarray(curves)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: input_dir/rmsf")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "rmsf").expanduser().resolve()
    topology, indices, rmsf = analyze(input_dir)
    mean_rmsf = rmsf.mean(axis=0)
    sd_rmsf = rmsf.std(axis=0, ddof=1)
    atoms = list(topology.atoms)
    residue_indices = np.array([atom.residue.index + 1 for atom in atoms])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "rmsf_mean"
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residue_index_1based", "chain_index_0based", "residue_number",
                         "residue_name", "source_atom_index_0based"]
                        + [f"trajectory_{i}_rmsf_angstrom" for i in range(N_REPLICAS)]
                        + ["mean_rmsf_angstrom", "sd_rmsf_angstrom"])
        for i, atom in enumerate(atoms):
            residue = atom.residue
            writer.writerow([residue_indices[i], residue.chain.index, residue.resSeq,
                             residue.name, indices[i], *rmsf[:, i], mean_rmsf[i], sd_rmsf[i]])

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for replica, values in enumerate(rmsf):
        ax.plot(residue_indices, values, label=f"Trajectory {replica}", linewidth=1.1, alpha=0.2)
    ax.plot(residue_indices, mean_rmsf, label="Mean (8 trajectories)",
            color="black", linewidth=1.1, zorder=10)
    ax.set(title=f"{input_dir.name} — Cα RMSF", xlabel="Residue index (1-based)",
           ylabel="RMSF (Å)", xlim=(residue_indices[0], residue_indices[-1]), ylim=(0, None))
    ax.text(0.01, 0.98, "Fit: frame 0 | RMSF: each residue's mean position within each replica | Frames: 0–100 ns",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(output_dir / f"{stem}.png", dpi=200)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
