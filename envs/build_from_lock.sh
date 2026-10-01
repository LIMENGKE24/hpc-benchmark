#!/bin/bash
# Rebuild the exact vanda mace-bench env on a site WITH internet (fornax, hopper).
# China HPC has no internet: use the conda-packed tarball instead (see README §3).
#   bash envs/build_from_lock.sh /path/to/envs/mace-bench
set -euo pipefail
PREFIX=${1:?usage: build_from_lock.sh PREFIX}
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda create -q -y -p "$PREFIX" -c conda-forge --override-channels python=3.11.16 pip conda-pack
PIP_NO_CACHE_DIR=1 "$PREFIX/bin/python" -m pip install -q \
    --extra-index-url https://download.pytorch.org/whl/cu128 -r "$HERE/pip-lock.txt"
"$PREFIX/bin/python" -c "import torch, mace, ase; print(torch.__version__, mace.__version__, ase.__version__)"
