#!/bin/bash
# GPU part only; CPU (genoa/largemem) waits for the per-node diagnosis (c13 = EPYC 7543, c15-20 = EPYC 9354).
set -euo pipefail
for r in 1 2 3; do
  qsub -o logs/ -v W=W1,REP=$r jobs/fornax/gpu.pbs
  for s in S L; do qsub -o logs/ -v W=W2,SIZE=$s,REP=$r jobs/fornax/gpu.pbs; done
done
