from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import argparse
import inspect
import logging
import math

#the script uses sequential monte carlo as steering system

LOGGER = logging.getLogger("run_bioemu")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BioEmu ensemble from FASTA file. Physical steering --> Sequential Monte Carlo (SMC).")
    parser.add_argument("fasta", type=Path, help="Input protein FASTA file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: outputs/bioemu/<FASTA stem>.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Total number of structures to generate. Default: 1000.")
    parser.add_argument("--batch-size-100", type=int, default=None, help="BioEmu batch size for 100-residue protein. By default, script selects smallest value compatible with SMC particle count.")
    parser.add_argument("--model-name", choices=("bioemu-v1.0", "bioemu-v1.1", "bioemu-v1.2"), default="bioemu-v1.1", help="BioEmu checkpoint. Default: bioemu-v1.1.")
    parser.add_argument("--num-particles", type=int, default=5, help="Number of SMC particles in each group. Default: 5.")
    parser.add_argument("--ess-threshold", type=float, default=0.5, help="Normalized effective-sample-size resampling threshold. Default: 0.5.")
    parser.add_argument("--base-seed", type=int, default=None, help="Optional random seed for reproducible sampling.")
    parser.add_argument("--cache-embeds-dir", type=Path, default=None, help="Optional directory for ColabFold embedding files.")
    parser.add_argument("--cache-so3-dir", type=Path, default=None, help="Optional directory for SO(3) cache files.")
    parser.add_argument("--msa-host-url", default=None, help="Optional ColabFold MSA server URL.")
    parser.add_argument("--keep-unphysical", action="store_true", help="Keep samples that fail the final physical-structure filter.")
    return parser.parse_args(argv)

def read_single_fasta(fasta_path: Path) -> str:
    fasta_path = fasta_path.expanduser().resolve()
    if not fasta_path.is_file():
        raise ValueError(f"FASTA file does not exist: {fasta_path}")
    if fasta_path.suffix.lower() not in {".fa", ".faa", ".fasta"}:
        raise ValueError("Input file must have a .fa, .faa, or .fasta extension ")

    records: list[str] = []
    sequence_parts: list[str] = []
    saw_header = False
#check fasta
    with fasta_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if not line[1:].strip():
                    raise ValueError(f"FASTA header on line {line_number} is empty.")
                if saw_header:
                    records.append("".join(sequence_parts))
                    sequence_parts = []
                saw_header = True
                continue
            if not saw_header:
                raise ValueError("FASTA file must start with a '>' header line.")
            sequence_parts.append("".join(line.split()).upper())

    if saw_header:
        records.append("".join(sequence_parts))
    if len(records) != 1:
        raise ValueError(f"Expected one FASTA record, but found {len(records)}.")
    if not records[0]:
        raise ValueError("The FASTA record has no sequence.")
    return records[0]

def make_smc_config(num_particles: int, ess_threshold: float) -> dict[str, Any]:
    return {
        "_target_": "bioemu.steering.dpm_smc.dpm_solver_smc",
        "_partial_": True,
        "eps_t": 0.001,
        "max_t": 0.99,
        "N": 100,
        "noise": 0.5,
        "fk_potentials": [
            {"_target_": "bioemu.steering.UmbrellaPotential", "cv": {"_target_": "bioemu.steering.CaCaDistance"}, "target": 0.38, "flatbottom": 0.1, "slope": 10.0, "order": 1, "linear_from": 0.1, "weight": 1.0},
            {"_target_": "bioemu.steering.UmbrellaPotential", "cv": {"_target_": "bioemu.steering.PairwiseClash", "min_dist": 0.41, "offset": 3}, "target": 0.0, "flatbottom": 0.0, "slope": 30.0, "weight": 1.0},
        ],
        "steering_config": {"num_particles": num_particles, "ess_threshold": ess_threshold, "start": 0.1, "end": 0.0},}

def effective_batch_size(batch_size_100: int, sequence_length: int, num_samples: int) -> int:
    scaled_size = max(1, int(batch_size_100 * (100 / sequence_length) ** 2))
    return min(scaled_size, num_samples)

def select_batch_size_100(sequence_length: int, num_samples: int, num_particles: int) -> int:
    minimum = max(1, math.ceil(num_particles * (sequence_length / 100) ** 2))
    candidate = minimum
    while True:
        batch_size = effective_batch_size(candidate, sequence_length, num_samples)
        if batch_size >= num_particles and batch_size % num_particles == 0:
            return candidate
        candidate += 1

def validate_positive_options(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be greater than zero.")
    if args.num_particles < 2:
        raise ValueError("--num-particles must be at least 2 for SMC steering.")
    if args.num_samples < args.num_particles:
        raise ValueError("--num-samples must be at least --num-particles.")
    if args.num_samples % args.num_particles != 0:
        raise ValueError("--num-samples must be a multiple of --num-particles.")
    if args.batch_size_100 is not None and args.batch_size_100 < 1:
        raise ValueError("--batch-size-100 must be greater than zero.")
    if not 0.0 <= args.ess_threshold <= 1.0:
        raise ValueError("--ess-threshold must be between 0 and 1.")

def load_bioemu_sampler():
    try:
        import torch
        from bioemu.sample import main as sample
        from bioemu.steering.dpm_smc import dpm_solver_smc  # noqa: F401
    except ImportError as error:
        raise RuntimeError("BioEmu with steering support required --> pip install 'bioemu[cuda]>=1.4.0'") from error

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if "denoiser_config" not in inspect.signature(sample).parameters:
        raise RuntimeError("This BioEmu release doesn't support the steering API. You need BioEmu 1.4.0 or later, homeboy.")

    LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))
    return sample

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        validate_positive_options(args)
        fasta_path = args.fasta.expanduser().resolve()
        sequence = read_single_fasta(fasta_path)

        batch_size_100 = args.batch_size_100
        if batch_size_100 is None:
            batch_size_100 = select_batch_size_100(len(sequence), args.num_samples, args.num_particles)

        batch_size = effective_batch_size(batch_size_100, len(sequence), args.num_samples)
        if batch_size < args.num_particles or batch_size % args.num_particles != 0:
            required = select_batch_size_100(len(sequence), args.num_samples, args.num_particles)
            raise ValueError(f"--batch-size-100={batch_size_100} gives an effective batch size of {batch_size}. SMC requires complete groups of {args.num_particles}. Use --batch-size-100 {required} or omit this option.")

        output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else (Path("outputs") / "bioemu" / fasta_path.stem).resolve()

        LOGGER.info("Input FASTA: %s", fasta_path)
        LOGGER.info("Sequence length: %d residues", len(sequence))
        LOGGER.info("Output directory: %s", output_dir)
        LOGGER.info("Samples: %d", args.num_samples)
        LOGGER.info("SMC particles: %d", args.num_particles)
        LOGGER.info("Effective BioEmu batch size: %d", batch_size)

        sample = load_bioemu_sampler()
        sample(sequence=fasta_path, num_samples=args.num_samples, output_dir=output_dir, batch_size_100=batch_size_100, model_name=args.model_name, denoiser_config=make_smc_config(args.num_particles, args.ess_threshold), cache_embeds_dir=args.cache_embeds_dir, cache_so3_dir=args.cache_so3_dir, msa_host_url=args.msa_host_url, filter_samples=not args.keep_unphysical, base_seed=args.base_seed)
    except (RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
#run:
#python scripts/02_run_bioemu.py input.fasta \
#  --num-samples 100 \
#  --output-dir outputs/my_protein