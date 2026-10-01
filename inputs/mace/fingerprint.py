#!/usr/bin/env python3
"""Print env versions + a reference MACE energy/force so sites can be compared before benchmarking."""

import hashlib
import json
import os
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import ase
import mace
import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

structure, model = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else "cpu"
atoms = read(structure)
atoms.calc = MACECalculator(model_paths=model, device=device, default_dtype="float64")
forces = atoms.get_forces()

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    "cuda_arch_list": torch.cuda.get_arch_list(),
    "mace": mace.__version__,
    "ase": ase.__version__,
    "numpy": np.__version__,
    "model_md5": hashlib.md5(open(model, "rb").read()).hexdigest(),
    "natoms": len(atoms),
    "energy_eV": float(atoms.get_potential_energy()),
    "fmax_eV_A": float(np.abs(forces).max()),
}, indent=2))
