""" Run BioEmu2 trajectories

e.g., python scripts/bioemu2/run_trajectory.py outputs/bioemu2/mutant_1

BUT MUST CHANGE FILE NAME IF DIFFERENT THAN mature_chain_relaxed.yaml AND mature_chain_relaxed.cif!!!

References:
https://www.science.org/doi/10.1126/science.adv9817
https://github.com/microsoft/bioemu2
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run BioEmu2 trajectories for a protein input directory.")
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent,
        help=(
            "Directory with mature_chain.yaml and mature_chain.cif "
            "(default: this script's directory). Output is saved in "
            "the trajectory_output subdirectory."),)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    yaml_path = input_dir / "mature_chain_relaxed.yaml"
    cif_path = input_dir / "mature_chain_relaxed.cif"
    for path in (yaml_path, cif_path):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")

    from bioemu2 import (
        System,
        load_model,
        map_cif_atom_order,
        save_frames,
        simulate_trajectory,
    )

    system = System.from_yaml(yaml_path)
    model = load_model()
    x0 = map_cif_atom_order(system, cif_path)  # Initial coordinates.

    num_trajectories = 8
    repeated_x0 = x0[[0] * num_trajectories]

    trajectories = simulate_trajectory(
        system,
        model=model,
        x0=repeated_x0,
        lag_time_ns=1.0,           # 1 ns between frames
        num_trajectory_steps=100, # total = 100 ns
        seed=42,
    )

    assert len(trajectories) == 8
    assert all(t.n_frames == 101 for t in trajectories)

    output_dir = input_dir / "trajectory_output"
    for i, trajectory in enumerate(trajectories):
        path = save_frames(trajectory, output_dir / f"trajectory_{i}")
        print(f"Saved trajectory {i} ({trajectory.n_frames} frames) to {path}")

if __name__ == "__main__":
    main()
