#!/usr/bin/env python3
"""Turn one finished W1 run directory (OUTCAR) into a result JSON for analysis/pareto.py.

Usage: parse_vasp.py RUN_DIR --tag site/partition/W1/repN --units-used N --out results/.../repN.json
"""

import argparse
import json
import re
import socket
from pathlib import Path


def grab(pattern: str, text: str, cast=float, last=True):
    hits = re.findall(pattern, text)
    if not hits:
        return None
    return cast(hits[-1] if last else hits[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--units-used", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    outcar = (args.run_dir / "OUTCAR").read_text(errors="replace")
    if "Voluntary context switches" not in outcar:
        raise SystemExit(f"{args.run_dir}/OUTCAR is incomplete")

    natoms = grab(r"NIONS =\s+(\d+)", outcar, int)
    energy = grab(r"free  energy   TOTEN  =\s+(-?\d+\.\d+)", outcar)
    result = {
        "tag": args.tag,
        "workload": "vasp_scf",
        "units_used": args.units_used,
        "elapsed_s": grab(r"Elapsed time \(sec\):\s+([\d.]+)", outcar),
        "loop_s_per_scf": grab(r"LOOP:\s+cpu time\s+[\d.]+:\s+real time\s+([\d.]+)", outcar),
        "n_scf": len(re.findall(r"LOOP:\s+cpu time", outcar)),
        "natoms": natoms,
        "toten_eV": energy,
        "toten_eV_per_atom": energy / natoms if energy is not None and natoms else None,
        "vasp_version": grab(r"(vasp\.\d+\.\d+\.\d+)", outcar, str, last=False),
        "mpi_ranks": grab(r"running\s+(?:on\s+)?(\d+)\s+total cores", outcar, int, last=False),
        "host": socket.gethostname(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
