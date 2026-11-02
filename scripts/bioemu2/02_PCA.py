"""
Align Cα coordinates, save all Cα atoms, and fit one combined PCA.

e.g., python scripts/bioemu2/02_PCA.py outputs/bioemu2/output_bioemu2 
or just the pca with saved alignment: python scripts/bioemu2/02_PCA.py --pca-only
"""
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/bioemu2/output_bioemu2"
PROTEINS = (
    ("WT", "trajectory_output_1_WT"),
    ("MUTANT_1", "trajectory_output_2_MUTANT"),
    ("MUTANT_2", "trajectory_output_3_MUTANT"),
)
SELECTION = "protein and name CA"
FIT_INDICES = {
    "WT": np.arange(428),
    "MUTANT_1": np.arange(428),
    "MUTANT_2": np.r_[np.arange(81), np.arange(86, 433)],
}


def run_pca(all_ca_by_protein, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bioemu2-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    blocks = []
    for protein, _ in PROTEINS:
        coordinates = all_ca_by_protein[protein]
        expected_atoms = 433 if protein == "MUTANT_2" else 428
        if coordinates.shape != (8, 101, expected_atoms, 3):
            raise ValueError(f"{protein}: unexpected coordinate shape {coordinates.shape}")
        if not np.isfinite(coordinates).all():
            raise ValueError(f"{protein}: non-finite coordinates")
        blocks.append(coordinates[:, :, FIT_INDICES[protein], :].reshape(808, 1284))
    matrix = np.concatenate(blocks, axis=0).astype(np.float64)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    largest = np.argmax(np.abs(components), axis=1)
    components *= np.sign(components[np.arange(len(components)), largest])[:, None]
    variance = singular_values**2 / (len(matrix) - 1)
    if variance.sum() <= 0:
        raise ValueError("PCA requires nonzero coordinate variance")
    ratio = variance / variance.sum()
    scores = centered @ components[:3].T
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "pca_model.npz", mean_coordinates_nm=mean,
                        components=components, explained_variance_nm2=variance,
                        explained_variance_ratio=ratio, singular_values=singular_values,
                        wt_residue_number=np.arange(1, 429),
                        feature_order=np.asarray("CA1_x, CA1_y, CA1_z, CA2_x, ..."))
    with (output_dir / "pca_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein", "trajectory_id", "frame_index", "time_ns", "PC1_nm", "PC2_nm", "PC3_nm"])
        for group, (protein, _) in enumerate(PROTEINS):
            for trajectory_id in range(8):
                for frame in range(101):
                    row = group * 808 + trajectory_id * 101 + frame
                    writer.writerow([protein, trajectory_id, frame, frame, *scores[row]])
    colors = np.asarray(["#0072B2", "#D55E00", "#009E73"])
    labels = ["WT", "Mutant 1", "Mutant 2"]
    group_ids = np.repeat(np.arange(3), 808)
    order = np.random.default_rng(42).permutation(len(scores))
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.scatter(scores[order, 0], scores[order, 1], c=colors[group_ids[order]],
               s=15, alpha=0.6, linewidths=0)
    ax.set(title="Combined Cα PCA — 24 trajectories",
           xlabel=f"PC1 ({ratio[0]:.1%} variance; nm)",
           ylabel=f"PC2 ({ratio[1]:.1%} variance; nm)")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=color, label=label)
                       for color, label in zip(colors, labels)], frameon=False)
    ax.grid(alpha=0.15)
    fig.savefig(output_dir / "pca_pc1_pc2.png", dpi=300)
    fig.savefig(output_dir / "pca_pc1_pc2.pdf")
    plt.close(fig)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.12, top=0.90)
    ax.scatter(scores[order, 0], scores[order, 1], scores[order, 2],
               c=colors[group_ids[order]], s=15, alpha=0.6, linewidths=0,
               depthshade=False)
    ax.set(title="Combined Cα PCA — 24 trajectories",
           xlabel=f"PC1 ({ratio[0]:.1%} variance; nm)",
           ylabel=f"PC2 ({ratio[1]:.1%} variance; nm)",
           zlabel=f"PC3 ({ratio[2]:.1%} variance; nm)")
    ax.view_init(elev=25, azim=45)
    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 12
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=color, label=label)
                       for color, label in zip(colors, labels)], frameon=False)
    fig.canvas.draw()
    axis_labels = (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label)
    fig.savefig(output_dir / "pca_pc1_pc2_pc3.png", dpi=300, bbox_inches="tight",
                bbox_extra_artists=axis_labels, pad_inches=0.4)
    fig.savefig(output_dir / "pca_pc1_pc2_pc3.pdf", bbox_inches="tight",
                bbox_extra_artists=axis_labels, pad_inches=0.4)
    plt.close(fig)
    summary = {
        "samples": 2424, "samples_per_protein": 808, "features": 1284,
        "corresponding_ca_atoms": 428, "fit_count": 1,
        "centering": "One feature mean across all 2424 frames",
        "scaling": "None", "solver": "numpy.linalg.svd, full_matrices=False",
        "coordinate_units": "nm", "frame_zero_included": True,
        "pc1_explained_variance_ratio": float(ratio[0]),
        "pc2_explained_variance_ratio": float(ratio[1]),
        "pc3_explained_variance_ratio": float(ratio[2]),
        "mutant_2_duplicate": "Residues 82–86 retained in the aligned array; not used as corresponding PCA features",
        "plot_order_seed": 42,
    }
    (output_dir / "pca_metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved one combined PCA and 2424 frame scores to {output_dir}")

def load_ca(path):
    trajectory = md.load(str(path / "samples.xtc"), top=str(path / "topology.cif"))
    if trajectory.n_frames != 101:
        raise ValueError(f"{path}: expected 101 frames, found {trajectory.n_frames}")
    indices = trajectory.topology.select(SELECTION)
    ca = trajectory.atom_slice(indices)
    atoms = list(ca.topology.atoms)
    if len({a.residue.chain.index for a in atoms}) != 1:
        raise ValueError(f"{path}: expected one protein chain")
    if [a.residue.resSeq for a in atoms] != list(range(1, len(atoms) + 1)):
        raise ValueError(f"{path}: expected consecutive mature-chain residue numbers")
    if not np.isfinite(ca.xyz).all():
        raise ValueError(f"{path}: Cα coordinates contain non-finite values")
    return ca, indices

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, help="Default: input_dir/pca/aligned")
    parser.add_argument("--pca-only", action="store_true", help="Use the saved aligned arrays")
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "pca/aligned").expanduser().resolve()
    if args.pca_only:
        coordinates = {
            protein: np.load(output_dir / f"{protein}_all_ca_nm.npy", allow_pickle=False)
            for protein, _ in PROTEINS
        }
        run_pca(coordinates, output_dir)
        return

    reference_path = input_dir / PROTEINS[0][1] / "trajectory_0"
    wt, _ = load_ca(reference_path)
    reference = wt[0]
    sequence = reference.topology.to_fasta()[0]
    if len(sequence) != 428 or sequence[76:81] != "HGGVP":
        raise ValueError("WT sequence does not match the documented mature-chain mapping")

    expected_sequences = {
        "WT": sequence,
        "MUTANT_1": sequence[:76] + "P" + sequence[77:],
        "MUTANT_2": sequence[:81] + "HGGVP" + sequence[81:],
    }
    sources = {}
    all_ca_by_protein, mapping_rows = {}, []
    for protein, folder in PROTEINS:
        all_ca = []
        first_topology = None
        mapped_indices = FIT_INDICES[protein]
        wt_position_by_ca_index = {
            int(ca_index): wt_index + 1
            for wt_index, ca_index in enumerate(mapped_indices)
        }
        sources[protein] = []
        for trajectory_id in range(8):
            path = input_dir / folder / f"trajectory_{trajectory_id}"
            ca, original_indices = load_ca(path)
            if ca.topology.to_fasta()[0] != expected_sequences[protein]:
                raise ValueError(f"{path}: sequence differs from the recorded variant")
            if first_topology is None:
                first_topology = ca.topology
            elif ca.topology != first_topology:
                raise ValueError(f"{path}: Cα topology differs within the protein group")

            ca.superpose(reference, frame=0, atom_indices=mapped_indices,
                         ref_atom_indices=np.arange(428))
            if not np.isfinite(ca.xyz).all():
                raise ValueError(f"{path}: alignment produced non-finite coordinates")
            all_ca.append(ca.xyz.copy())
            sources[protein].append(str(path))
            for ca_index, atom in enumerate(ca.topology.atoms):
                wt_position = wt_position_by_ca_index.get(ca_index, "")
                is_duplicate = protein == "MUTANT_2" and 81 <= ca_index < 86
                duplicate_of = ca_index - 4 if is_duplicate else ""
                mapping_rows.append((protein, trajectory_id, ca_index, wt_position,
                                     atom.residue.resSeq, atom.residue.name,
                                     int(original_indices[ca_index]),
                                     ca_index in wt_position_by_ca_index, duplicate_of))
        all_ca_by_protein[protein] = np.stack(all_ca)
        print(f"Aligned {protein}: 8 trajectories, 101 frames, "
              f"{all_ca_by_protein[protein].shape[2]} Cα atoms retained")

    if not np.allclose(all_ca_by_protein["WT"][0, 0], reference.xyz[0], atol=1e-5, rtol=0):
        raise ValueError("Aligned WT reference differs from its input coordinates")
    output_dir.mkdir(parents=True, exist_ok=True)
    for protein, coordinates in all_ca_by_protein.items():
        np.save(output_dir / f"{protein}_all_ca_nm.npy", coordinates)
    np.save(output_dir / "WT_reference_ca_nm.npy", reference.xyz[0])
    with (output_dir / "atom_mapping.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein", "trajectory_id", "saved_ca_index_0based",
                         "wt_residue_number", "source_residue_number",
                         "source_residue_name", "source_atom_index_0based",
                         "used_for_fit", "duplicate_of_wt_residue_number"])
        writer.writerows(mapping_rows)
    metadata = {
        "scope": "Aligned coordinate files before PCA",
        "reference": str(reference_path), "reference_frame": 0,
        "selection": SELECTION, "coordinate_units": "nm",
        "coordinate_files": {
            protein: {"file": f"{protein}_all_ca_nm.npy", "shape": list(coordinates.shape)}
            for protein, coordinates in all_ca_by_protein.items()
        },
        "trajectory_ids": list(range(8)),
        "source_directories": sources,
        "frame_index": list(range(101)),
        "time_ns": list(range(101)),
        "all_ca_array_axes": ["trajectory_within_protein", "frame", "source_CA_atom", "xyz"],
        "time": "Frame 0–100 at 1 ns intervals; all starting frames retained",
        "mapping_sources": ["outputs/bioemu2/mutant_1/README.md",
                            "outputs/bioemu2/mutant_2/METHODOLOGY.md",
                            "outputs/bioemu2/mutant_2/residue_mapping.tsv"],
        "alignment": "Least-squares fit to 428 corresponding WT Cα atoms; apply the transform to all Cα atoms",
        "mutant_2_duplicate": "HGGVP at residues 82–86; all five coordinates retained in each saved frame",
        "atom_mapping": "One row per saved Cα atom per trajectory; duplicate residues have no one-to-one WT position",
        "saved_atom_policy": "All Cα atoms in source order; no truncation, padding, or common coordinate array",
        "pca_performed": False,
        "pca_centering_performed": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved all shit Cα coordinates in nm to {output_dir}")
    run_pca(all_ca_by_protein, output_dir)

if __name__ == "__main__":
    main()
