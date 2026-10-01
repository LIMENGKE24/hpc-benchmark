#!/bin/bash
set -euo pipefail
for r in 1 2 3; do
  for pc in p1:40 9242:96 48cp1:48 48cp2:48 48cp3:48; do
    p=${pc%:*}; n=${pc#*:}
    sbatch -p $p --ntasks-per-node=$n --export=ALL,W=W1,REP=$r jobs/china/cpu.slurm
    sbatch -p $p --ntasks-per-node=$n --export=ALL,W=W2,REP=$r jobs/china/cpu.slurm
  done
  for g in v100 v100g32; do
    sbatch -p $g --export=ALL,W=W1,REP=$r jobs/china/gpu.slurm
    for s in S L; do sbatch -p $g --export=ALL,W=W2,SIZE=$s,REP=$r jobs/china/gpu.slurm; done
  done
done
