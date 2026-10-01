#!/bin/bash
# Production: 3 reps of every workload on vanda. Run from repo root.
set -euo pipefail
for r in 1 2 3; do
  qsub -o logs/ -v W=W1,REP=$r jobs/vanda/cpu.pbs
  qsub -o logs/ -v W=W2,REP=$r jobs/vanda/cpu.pbs
  for s in S L; do
    qsub -o logs/ -v SIZE=$s,REP=$r jobs/vanda/gpu.pbs
    qsub -q gpu -o logs/ -v SIZE=$s,REP=$r,PART=gpu jobs/vanda/gpu.pbs            # free queue
  done
done
