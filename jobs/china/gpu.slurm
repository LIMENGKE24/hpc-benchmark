#!/bin/bash
#SBATCH -J bench-china-gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH -t 04:00:00
#SBATCH -o logs/%j.out
# sbatch -p v100|v100g32 --export=ALL,W=W1|W2|FP,SIZE=S|L,REP=N jobs/china/gpu.slurm
# (3 cores per GPU = the node's 24 cores / 8 GPUs, which is the partition default)
: "${W:?W=W1|W2|FP}" "${REP:=1}"
REPO=$SLURM_SUBMIT_DIR SITE=china PART=$SLURM_JOB_PARTITION
PY=/data/home/limengke/envs/mace-bench/bin/python
source "$REPO/jobs/common.sh"
case "$W" in
  W1) modload vasp/6.5.1-nvhpc; export OMP_NUM_THREADS=1
      run_w1 1 1 mpirun -np 1 vasp_std ;;
  W2) run_w2 "${SIZE:?SIZE=S|L}" cuda 3 1 ;;
  FP) run_fingerprint "china_$PART" ;;
esac
