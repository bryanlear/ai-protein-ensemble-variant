from __future__ import annotations
from tempfile import TemporaryDirectory
from typing import Sequence
from pathlib import Path
import argparse
import logging
import subprocess
import sys

LOGGER = logging.getLogger("reconstruct_side_chains")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct side chains in BioEmu ensemble")
    parser.add_argument("input_dir", type=Path, help="Directory with topology.pdb and samples.xtc.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Default: <input>_sidechains.")
    parser.add_argument("--md", action="store_true", help="Run short NVT MD equilibration.")
    parser.add_argument("--md-frames", nargs="+", metavar="FRAME", help="Zero-based MD frames e.g., 0 5 10-14.")
    return parser.parse_args(argv)

def parse_frames(values: Sequence[str], frame_count: int) -> list[int]:
    selected: set[int] = set()
    for value in values:
        for token in value.split(","):
            if "-" in token:
                parts = token.split("-")
                if len(parts) != 2 or not all(part.isdigit() for part in parts):
                    raise ValueError(f"Bad frame range: {token}")
                start, stop = map(int, parts)
                if start > stop:
                    raise ValueError(f"Bad frame range: {token}")
                selected.update(range(start, stop + 1))
            elif token.isdigit():
                selected.add(int(token))
            else:
                raise ValueError(f"Bad frame: {token}")
    if not selected:
        raise ValueError("No MD frames selected.")
    if frame_count < 1 or max(selected) >= frame_count:
        raise ValueError(f"Frame range is 0-{frame_count - 1}.")
    return sorted(selected)

def run_bioemu(pdb_path: Path, xtc_path: Path, output_dir: Path, run_md: bool) -> None:
    command = [sys.executable, "-m", "bioemu.sidechain_relax", "--pdb-path", str(pdb_path), "--xtc-path", str(xtc_path), "--outpath", str(output_dir)]
    command.extend(["--md-protocol", "nvt_equil"] if run_md else ["--no-md-equil"])
    LOGGER.info("Running BioEmu.")
    subprocess.run(command, check=True)

def load_trajectory(pdb_path: Path, xtc_path: Path):
    try:
        import mdtraj
    except ImportError as error:
        raise RuntimeError("MDTraj is not installed.") from error
    return mdtraj.load_xtc(str(xtc_path), top=str(pdb_path))

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.md_frames and not args.md:
            raise ValueError("--md-frames requires --md.")
        input_dir = args.input_dir.expanduser().resolve()
        pdb_path, xtc_path = input_dir / "topology.pdb", input_dir / "samples.xtc"
        if not input_dir.is_dir():
            raise ValueError(f"Missing directory: {input_dir}")
        if not pdb_path.is_file() or not xtc_path.is_file():
            raise ValueError("Missing topology.pdb or samples.xtc.")
        output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_dir.with_name(f"{input_dir.name}_sidechains")
        if output_dir == input_dir:
            raise ValueError("Output must differ from input.")
        output_dir.mkdir(parents=True, exist_ok=True)

        if not args.md:
            run_bioemu(pdb_path, xtc_path, output_dir, False)
        elif not args.md_frames:
            run_bioemu(pdb_path, xtc_path, output_dir, True)
        else:
            trajectory = load_trajectory(pdb_path, xtc_path)
            frames = parse_frames(args.md_frames, len(trajectory))
            run_bioemu(pdb_path, xtc_path, output_dir, False)
            md_output_dir = output_dir / "md_selected_frames"
            md_output_dir.mkdir(parents=True, exist_ok=True)
            (md_output_dir / "frame_indices.txt").write_text("\n".join(map(str, frames)) + "\n", encoding="utf-8")
            with TemporaryDirectory(prefix="bioemu_md_") as temp_dir:
                selected_xtc = Path(temp_dir) / "samples.xtc"
                trajectory[frames].save_xtc(str(selected_xtc))
                run_bioemu(pdb_path, selected_xtc, md_output_dir, True)
        LOGGER.info("Done: %s", output_dir)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        LOGGER.error("%s", error)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
