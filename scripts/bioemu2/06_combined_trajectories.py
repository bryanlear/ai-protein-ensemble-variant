"""
run this bitch: python scripts/bioemu2/06_combined_trajectories.py
"""

import argparse
import os
from pathlib import Path
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bioemu2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTEINS = (
    ("WT", "trajectory_output_1_WT", "#0072B2"),
    ("Mutant 1", "trajectory_output_2_MUTANT", "#FF0000"),
    ("Mutant 2", "trajectory_output_3_MUTANT", "#FFA500"),
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", type=Path,
                        default=ROOT / "outputs/bioemu2/output_bioemu2")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "combined_trajectories").expanduser().resolve()
    plots = (
        ("rmsd", "rmsd", "time_ns", "Time (ns)"),
        ("rmsf", "rmsf_mean", "residue_index_1based",
         "Residue index within each protein (1-based)"),
    )
    curves = {}
    for metric, stem, x_column, _ in plots:
        curves[metric] = []
        for label, folder, color in PROTEINS:
            path = input_dir / folder / metric / f"{stem}.csv"
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            x = np.atleast_1d(data[x_column])
            y = np.array([np.atleast_1d(data[f"trajectory_{i}_{metric}_angstrom"])
                          for i in range(8)])
            if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Invalid data: {path}")
            curves[metric].append((x, y, label, color))

    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, _, _, xlabel in plots:
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        for replica in range(8):
            for x, y, label, color in curves[metric]:
                ax.plot(x, y[replica], color=color, linewidth=1.1, alpha=0.5,
                        label=label if replica == 0 else None)
        ax.set(title=f"Cα {metric.upper()} — all 24 trajectories", xlabel=xlabel,
               ylabel=f"{metric.upper()} (Å)", ylim=(0, None))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"combined_trajectories_{metric}.{extension}", dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    main()
