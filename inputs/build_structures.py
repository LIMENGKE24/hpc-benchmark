#!/usr/bin/env python3
"""Build all benchmark structures from the c-Na3PS4 unit cell and write MD5SUMS.

Outputs (relative to inputs/):
  vasp/POSCAR                  W1: 2x2x2 supercell, 128 atoms, species order Na P S
  mace/Na3PS4_unit.extxyz      W2: unit cell; bench_md.py applies --supercell
  mace/fingerprint.extxyz      2x2x2, rattled (fixed seed) so forces are non-zero
"""

import hashlib
from pathlib import Path

from ase.io import read, write

HERE = Path(__file__).resolve().parent
CIF = HERE / "structures" / "c-Na3PS4_experimental_ordered.cif"
ORDER = ["Na", "P", "S"]


def sort_species(atoms):
    idx = sorted(range(len(atoms)), key=lambda i: ORDER.index(atoms[i].symbol))
    return atoms[idx]


def main() -> None:
    unit = sort_species(read(CIF))
    assert unit.get_chemical_formula() == "Na6P2S8", unit.get_chemical_formula()

    w1 = sort_species(unit * (2, 2, 2))
    write(HERE / "vasp" / "POSCAR", w1, format="vasp", direct=True, sort=False,
          label="c-Na3PS4 2x2x2 (128 atoms), a=6.9965 A")

    write(HERE / "mace" / "Na3PS4_unit.extxyz", unit)

    fp = unit * (2, 2, 2)
    fp.rattle(stdev=0.05, seed=42)
    write(HERE / "mace" / "fingerprint.extxyz", fp)

    files = [CIF, HERE / "vasp" / "POSCAR", HERE / "mace" / "Na3PS4_unit.extxyz", HERE / "mace" / "fingerprint.extxyz"]
    lines = [f"{hashlib.md5(f.read_bytes()).hexdigest()}  {f.relative_to(HERE)}" for f in files]
    (HERE / "MD5SUMS").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"W1 POSCAR: {len(w1)} atoms, {w1.get_chemical_formula()}")


if __name__ == "__main__":
    main()
