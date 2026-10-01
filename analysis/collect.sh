#!/bin/bash
# Pull results/ from every site into the local repo (needs the 4 ssh aliases).
set -euo pipefail
cd "$(dirname "$0")/.."
while read -r host path; do
  ssh -n "$host" "cd $path && tar czf - ." 2>/dev/null | tar xzf - -C results
  echo "$host: $(find results/$host -name '*.json' 2>/dev/null | wc -l | tr -d ' ') files"
done <<'SITES'
vanda  /scratch/projects/CFP05/CFP05-CF-241/limengke/hpc-benchmark/results
hopper /scratch/Projects/CFP-05/CFP05-CF-241/limengke/hpc-benchmark/results
fornax /home/li.mengke/hpc-benchmark/results
china  /ssd/work/limengke/hpc-benchmark/results
SITES
