# Shared benchmark logic, sourced by every site's job script.
# The caller sets: REPO SITE PART PY REP [SMOKE=0]
#   run_w1 UNITS NCORE LAUNCH...         VASP fixed 25-SCF single point (3 SCF if SMOKE=1)
#   run_w2 SIZE DEVICE THREADS UNITS     MACE NVT MD; SIZE = S (1024 atoms) | L (3456 atoms)

set -euo pipefail
SMOKE=${SMOKE:-0}
SUF=$([ "$SMOKE" = 1 ] && echo -smoke || true)

# Lmod/Environment Modules scripts reference unset variables, so relax `set -u` around them
modload() { set +u; module load "$@"; local rc=$?; set -u; return $rc; }

_rundir() {  # $1 = workload label
    TAG=$SITE/$PART/$1$SUF/rep$REP
    RUN=$REPO/runs/$TAG
    mkdir -p "$RUN" "$REPO/results/$(dirname "$TAG")"
    cd "$RUN"
    { echo "tag=$TAG"; date; hostname; lscpu | grep -E "Model name|^CPU\(s\)|Socket|Thread"; \
      command -v nvidia-smi >/dev/null && nvidia-smi -L || true; } > env.log
}

run_w1() {
    local units=$1 ncore=$2; shift 2
    _rundir W1
    cp "$REPO"/inputs/vasp/{INCAR,KPOINTS,POSCAR,POTCAR} .
    sed -i -E "s/^NCORE += *[0-9]+.*/NCORE   = $ncore/" INCAR
    [ "$SMOKE" = 1 ] && sed -i -E 's/^(NELM|NELMIN) += *[0-9]+/\1 = 3/' INCAR
    md5sum INCAR KPOINTS POSCAR POTCAR >> env.log
    (module list 2>&1 || true) >> env.log
    "$@" > vasp.out 2>&1
    "$PY" "$REPO/analysis/parse_vasp.py" . --tag "$TAG" --units-used "$units" --out "$REPO/results/$TAG.json"
}

run_w2() {
    local size=$1 device=$2 threads=$3 units=$4 sc warm steps
    case "$size" in
        S) sc="4 4 4";    warm=200; steps=2000 ;;
        L) sc="6 6 6";    warm=100; steps=1000 ;;   # 3456 atoms: largest that fits a 16 GB V100
        *) echo "bad SIZE $size" >&2; exit 2 ;;
    esac
    if [ "$SMOKE" = 1 ]; then warm=5; steps=20; fi
    _rundir W2-$size
    export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    "$PY" -u "$REPO/inputs/mace/bench_md.py" --input "$REPO/inputs/mace/Na3PS4_unit.extxyz" --supercell $sc \
        --model "$REPO/models/mace-mpa-0-medium.model" --warmup $warm --steps $steps \
        --device "$device" --threads "$threads" --units-used "$units" \
        --tag "$TAG" --out "$REPO/results/$TAG.json" > md.out 2>&1
    tail -25 md.out
}

run_fingerprint() {  # $1 = label, e.g. vanda_a40
    mkdir -p "$REPO/results/fingerprint"
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    "$PY" "$REPO/inputs/mace/fingerprint.py" "$REPO/inputs/mace/fingerprint.extxyz" \
        "$REPO/models/mace-mpa-0-medium.model" 2>/dev/null | tee "$REPO/results/fingerprint/$1.json"
}
