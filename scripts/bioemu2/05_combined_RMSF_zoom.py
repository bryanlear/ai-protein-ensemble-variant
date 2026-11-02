"""
Run: python scripts/bioemu2/05_combined_RMSF_zoom.py
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
    ("Mutant 1", "trajectory_output_2_MUTANT", "#D55E00"),
    ("Mutant 2", "trajectory_output_3_MUTANT", "#009E73"),
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", type=Path,
                        default=ROOT / "outputs/bioemu2/output_bioemu2")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.output_dir or input_dir / "combined_means").expanduser().resolve()
    curves = []
    for label, folder, color in PROTEINS:
        path = input_dir / folder / "rmsf" / "rmsf_mean.csv"
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        x = np.atleast_1d(data["residue_index_1based"])
        y = np.atleast_1d(data["mean_rmsf_angstrom"])
        region = (x >= 50) & (x <= 110)
        x, y = x[region], y[region]
        if not np.array_equal(x, np.arange(50, 111)) or not np.isfinite(y).all():
            raise ValueError(f"Invalid data for residues 50–110: {path}")
        curves.append((x, y, label, color))

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    
    for x, y, label, color in curves:
        ax.plot(x, y, label=label, color=color, linewidth=1.1)
    ax.set(title="Mean Cα RMSF — residues 50–110",
           xlabel="Residue index within each protein (1-based)", ylabel="RMSF (Å)",
           xlim=(50, 110), ylim=(0, None))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"combined_mean_rmsf_mean_50_110.{extension}", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
