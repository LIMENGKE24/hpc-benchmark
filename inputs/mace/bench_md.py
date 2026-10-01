#!/usr/bin/env python3
"""Fixed-work MACE NVT benchmark: untimed warm-up, then timed steps. Writes one result JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from mace.calculators import MACECalculator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="unit cell or supercell (any ASE format)")
    p.add_argument("--supercell", type=int, nargs=3, default=(1, 1, 1))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=600.0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: all)")
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--units-used", type=float, required=True, help="charged cores or GPUs (see pricing.yaml)")
    p.add_argument("--tag", required=True, help="e.g. china/v100/W2-S/rep1")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if args.threads:
        torch.set_num_threads(args.threads)

    atoms = read(args.input) * tuple(args.supercell)
    atoms.calc = MACECalculator(model_paths=str(args.model), device=args.device, default_dtype="float32")

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, 1.0 * units.fs, temperature_K=args.temperature, friction=0.01 / units.fs, rng=rng)

    dyn.run(args.warmup)
    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    dyn.run(args.steps)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    steps_per_s = args.steps / elapsed
    result = {
        "tag": args.tag,
        "workload": "mace_md",
        "units_used": args.units_used,
        "natoms": len(atoms),
        "steps": args.steps,
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "ns_per_day": steps_per_s * 1e-6 * 86400,
        "time_to_1ns_s": 1e6 / steps_per_s,
        "final_epot_eV": float(atoms.get_potential_energy()),
        "host": socket.gethostname(),
        "device": args.device,
        "gpu": gpu_name(),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "model_md5": subprocess.run(["md5sum", str(args.model)], capture_output=True, text=True).stdout.split()[0],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
