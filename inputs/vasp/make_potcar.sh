#!/bin/bash
# Build POTCAR (Na_pv P S, PBE) in POSCAR species order. Run once on vanda; copy the
# resulting POTCAR to the other sites and check `md5sum POTCAR` against POTCAR.md5.
# POTCARs are licensed, so they are never committed to git.
set -euo pipefail
POT_DIR=${POT_DIR:-/scratch/limengke/potpaw_PBE/POT_GGA_PAW_PBE}
cd "$(dirname "$0")"
: > POTCAR
for el in Na_pv P S; do
    cat "$POT_DIR/$el/POTCAR" >> POTCAR
done
grep TITEL POTCAR
md5sum POTCAR | tee POTCAR.md5
