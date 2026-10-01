#!/bin/bash
set -euo pipefail
for r in 1 2 3; do for s in S L; do
  qsub -o logs/ -v SIZE=$s,REP=$r,PART=h200 jobs/hopper/gpu.pbs
  qsub -l select=1:ncpus=8:ngpus=1:mem=32gb:gpu_model=H100 -o logs/ -v SIZE=$s,REP=$r,PART=h100 jobs/hopper/gpu.pbs
done; done
