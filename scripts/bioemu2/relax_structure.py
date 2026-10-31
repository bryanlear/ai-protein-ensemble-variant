""" Relax protein CIF -- PyRosetta Cartesian FastRelax

python relax_structure.py \
  ../../outputs/bioemu2/mutant_1/mature_chain.cif \
  --max-iter 1000 \
  --output ../../outputs/bioemu2/mutant_1/mature_chain_relaxed_1000.cif
### USING 1000 iterations, default is 200 

Output is <input_stem>_relaxed.cif

Rosetta may add missing atoms and hydrogens.
Reference:
https://graylab.jhu.edu"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

LOGGER = logging.getLogger("relax_structure")
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "outputs/bioemu2/mutant_1/mature_chain.cif")

def positive_int(value: str) -> int:
    number = int(value)
    if number < 1 or number > 2_147_483_647:
        raise argparse.ArgumentTypeError("Use integer from 1 to 2147483647.")
    return number

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help="Input .cif or .mmcif file. Default: mutant_1/mature_chain.cif.")
    parser.add_argument("--output", type=Path, help="Output CIF. Default: <input_stem>_relaxed.cif.")
    parser.add_argument("--repeats", type=positive_int, default=5, help="FastRelax cycles. Default: 5.")
    parser.add_argument("--max-iter", type=positive_int, default=200, help="Maximum iterations per minimization step. Default: 200.")
    parser.add_argument("--seed", type=positive_int, default=42, help="Random seed. Default: 42.")
    return parser.parse_args(argv)

def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    if input_path.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError("Input must be a .cif or .mmcif file.")
    output_path = (
        args.output.expanduser().absolute()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_relaxed.cif"))
    
    if output_path.resolve() == input_path:
        raise ValueError("Output must differ from the input file.")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError(f"Output already exists: {output_path}")
    if output_path.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError("Output must be a .cif or .mmcif file.")
    return input_path, output_path

def relax_structure(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    try:
        import pyrosetta
        from pyrosetta import rosetta
    except ImportError as error:
        raise RuntimeError(
            "PyRosetta not available. Run this script in a Python environment "
            "with PyRosetta installed. See https://www.pyrosetta.org/downloads."
        ) from error

    pyrosetta.init(
        options=f"-constant_seed -jran {args.seed} -ex1 -ex2aro -use_input_sc")
    
    pose = pyrosetta.pose_from_file(str(input_path))
    if pose.total_residue() == 0:
        raise ValueError("The input structure has no residues.")
    if any(not pose.residue(i).is_protein() for i in range(1, pose.total_residue() + 1)):
        raise ValueError("This script requires a protein-only structure.")
    sequence = pose.sequence()
    residue_count = pose.total_residue()

    scorefxn = pyrosetta.create_score_function("ref2015_cart")
    scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 0.5)
    scorefxn.set_weight(rosetta.core.scoring.pro_close, 0.0)
    movemap = rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    movemap.set_jump(False)

    # Side-chain packing but NO sequence design
    
    tasks = rosetta.core.pack.task.TaskFactory()
    tasks.push_back(rosetta.core.pack.task.operation.InitializeFromCommandline())
    tasks.push_back(rosetta.core.pack.task.operation.RestrictToRepacking())
    relax = rosetta.protocols.relax.FastRelax(scorefxn, args.repeats)
    relax.set_movemap(movemap)
    relax.set_task_factory(tasks)
    relax.cartesian(True)
    relax.max_iter(args.max_iter)
    relax.min_type("lbfgs_armijo_nonmonotone")

    before = float(scorefxn(pose))
    if not math.isfinite(before):
        raise RuntimeError("The input score is not finite.")
    LOGGER.info("Relaxing %s (%d residues). Initial score: %.3f REU.", input_path, residue_count, before)
    relax.apply(pose)
    after = float(scorefxn(pose))
    if not math.isfinite(after):
        raise RuntimeError("The relaxed score is not finite.")
    if pose.sequence() != sequence or pose.total_residue() != residue_count:
        raise RuntimeError("Relaxation changed the sequence or residue count.")
    LOGGER.info("Final score: %.3f REU. Change: %+.3f REU.", after, after - before)
    if after > before:
        LOGGER.warning("The final score is higher than the initial score. Check the output structure.")

    # Flag H97P risk without assuming a specific input sequence
    
    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).name3() == "PRO" and not pose.residue(i).is_lower_terminus() and pose.phi(i) > 0:
            label = pose.pdb_info().pose2pdb(i) if pose.pdb_info() else str(i)
            LOGGER.warning("Proline %s has positive phi (%.2f degrees). Check local geometry before simulation.", label.strip(), pose.phi(i))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".relax_", dir=output_path.parent) as directory:
        temporary_path = Path(directory) / "relaxed.cif"
        pose.dump_cif(str(temporary_path))
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("PyRosetta did not write a CIF file.")
        saved_pose = pyrosetta.pose_from_file(str(temporary_path))
        if saved_pose.sequence() != sequence or saved_pose.total_residue() != residue_count:
            raise RuntimeError("The saved CIF has a different sequence or residue count.")
 
        os.link(temporary_path, output_path)
    LOGGER.info("Saved: %s", output_path)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        input_path, output_path = validate_paths(args)
        relax_structure(input_path, output_path, args)
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
