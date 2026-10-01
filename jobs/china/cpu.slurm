#!/bin/bash
#SBATCH -J bench-china-cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -o logs/%j.out
# From repo root, one per partition (cores/node: p1 40, 9242 96, 48cp1-3 48):
#   sbatch -p p1 --ntasks-per-node=40 --export=ALL,W=W1,REP=1 jobs/china/cpu.slurm
: "${W:?--export=ALL,W=W1|W2}" "${REP:?REP=N}"
REPO=$SLURM_SUBMIT_DIR SITE=china PART=$SLURM_JOB_PARTITION
PY=/data/home/limengke/envs/mace-bench/bin/python
source "$REPO/jobs/common.sh"
N=$SLURM_NTASKS
case "$W" in
  W1) modload vasp-intel2024.2/6.5.1; export OMP_NUM_THREADS=1
      run_w1 "$N" 4 mpirun -np "$N" vasp_std ;;
  W2) run_w2 S cpu "$N" "$N" ;;
esac
