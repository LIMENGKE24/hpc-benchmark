#!/bin/bash
# fornax CPU production (VASP only). genoa queue is split by CPU type; c15 excluded (anomalously slow).
set -euo pipefail
pin() { echo "select=1:ncpus=64:mpiprocs=64:vnode=fornax-c$1"; }
r=1; for n in 16 18 17; do
  qsub -l "$(pin $n)" -o logs/ -v W=W1,REP=$r,PART=genoa_9354 jobs/fornax/cpu.pbs
  r=$((r+1))
done
for r in 1 2 3; do
  qsub -l "$(pin 13)" -o logs/ -v W=W1,REP=$r,PART=genoa_7543 jobs/fornax/cpu.pbs
  qsub -q largemem -l select=1:ncpus=128:mpiprocs=128 -o logs/ -v W=W1,REP=$r,PART=largemem jobs/fornax/cpu.pbs
done
