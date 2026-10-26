from pathlib import Path
from bioemu2 import (System, load_model, map_cif_atom_order, simulate_trajectory,)

input_dir = Path(__file__).resolve().parent
yaml_path = input_dir / "mature_chain.yaml"
cif_path = input_dir / "mature_chain.cif"

system = System.from_yaml(yaml_path)
model = load_model()
x0 = map_cif_atom_order(system, cif_path)  #start coord.

num_trajectories = 4
repeated_x0 = x0[[0] * num_trajectories]

trajectories = simulate_trajectory(
    system,
    model=model,
    x0=repeated_x0,
    lag_time_ns=1.0,          # 1 ns between frames
    num_trajectory_steps=100, # total = 100 ns
    seed=42,)

assert len(trajectories) == 4
assert all(t.n_frames == 101 for t in trajectories)
