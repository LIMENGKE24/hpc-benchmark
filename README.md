# HPC Benchmark — time vs. cost Pareto front

Benchmark every CPU and GPU node type we can use on **China HPC (超算云平台)**, **vanda**, **fornax** and
**hopper** (MACE only)
with *identical* VASP and MACE-MD workloads, then plot **time-to-solution vs. cost per job** to find the
Pareto-optimal nodes (fastest for a given cost / cheapest for a given speed).

> Status: **plan v2 (2026-10-01)** — decisions and prices fixed (§4, §7); nothing has been run yet.

---

## 1. Node inventory (what gets benchmarked)

Collected 2026-10-01 from `sinfo` / `pbsnodes` / `qstat -Q`. Full machine-readable list: [`config/nodes.yaml`](config/nodes.yaml).

| Site | Partition / queue | Hardware (per node) | VASP | MACE |
|---|---|---|---|---|
| China HPC (Inner Mongolia 2) | `p1` | 2× Xeon Gold 6138, 40 c, 190 GB | CPU | CPU |
| China HPC | `9242` | 2× Xeon Platinum 9242, 96 c, 378 GB | CPU | CPU |
| China HPC | `48cp1` | Xeon Platinum 8163, 48 c, 190 GB | CPU | CPU |
| China HPC | `48cp2` | Xeon Platinum 8255C, 48 c, 190 GB | CPU | CPU |
| China HPC | `48cp3` | Xeon Platinum 8168, 48 c, 190 GB | CPU | CPU |
| China HPC | `v100` | 8× V100 16 GB, 24 c | GPU (nvhpc) | GPU |
| China HPC | `v100g32` | 8× V100 32 GB, 24 c | GPU (nvhpc) | GPU |
| vanda | `batch_cpu` (cn-*) | 72 c, 512 GB | CPU | CPU |
| vanda | `batch_gpu` (gn-a40-*) | 2× A40 48 GB | — (no GPU VASP build) | GPU |
| fornax | `genoa` | AMD EPYC (Genoa), 64 c, 384–512 GB | CPU | CPU |
| fornax | `largemem` | 128 c, 512 GB | CPU | CPU |
| fornax | `rtx5090` | 1× RTX 5090 32 GB, 32 c, 64 GB | GPU (nvhpc) | GPU |
| hopper | `h100` (4 nodes) | 8× H100, 112 c, 2 TB | — (no VASP on hopper) | GPU |
| hopper | `h200` (40 nodes) | 8× H200, 112 c, 2 TB | — (no VASP on hopper) | GPU |

Not included (no access from our account / disabled): China `p1shr`, `fat`, `fat2`, `v100g32fat`
(we are not in those Slurm groups); fornax `v100` (queue stopped), `dev`; vanda `large_mem` (ACL).
Tianjin Region 1 is **excluded**. vanda A40 has no GPU VASP build, so it runs MACE only.

Exact CPU/GPU models are **recorded automatically** by every job (`lscpu`, `nvidia-smi`), so the table
above will be corrected from real job output.

---

## 2. Benchmark workloads

All workloads use the **same structure family: cubic Na₃PS₄** (already used in the group's MACE work on
fornax, so results are directly relevant).

### W1 — VASP single-point SCF (fixed work)

| Setting | Value |
|---|---|
| Structure | c-Na₃PS₄ 2×2×2 supercell, **128 atoms** |
| Version | **VASP 6.5.1** on every site (only version present on all three) |
| Functional / PP | PBE, PAW `Na_pv P S` — **one POTCAR file copied to all sites** (md5 checked) |
| ENCUT / PREC | 520 eV / Normal, `ALGO = Normal`, `LREAL = Auto` |
| k-points | Γ-centred 2×2×2 |
| Fixed work | `NSW = 0`, `NELMIN = NELM = 25`, `EDIFF = 1E-10` → exactly 25 SCF steps on every machine |
| Output | `LWAVE = LCHARG = .FALSE.` (no I/O noise) |
| Parallel | CPU: 1 full node, all physical cores, `KPAR = 1`, `NCORE = 4`. GPU (China `v100`/`v100g32`, fornax `rtx5090`): **1 GPU, 1 MPI rank**, `NCORE = 1`, `KPAR = 1` |
| Metric | `Elapsed time` from OUTCAR + energy check (all sites must agree to < 1 meV/atom) |

Optional **W1-8GPU**: full China V100 node (8 ranks, `KPAR = 4`) to see multi-GPU scaling.
Optional **W1-large**: 3×3×3 supercell (432 atoms, Γ-only) to test scaling on fat nodes / multi-GPU.

### W2 — MACE MD (fixed work)

| Setting | Value |
|---|---|
| Model | **MACE-MPA-0 medium** (`mace-mpa-0-medium.model`, md5 pinned) |
| Code | **mace-torch 0.3.15**, float32, identical conda-packed env on all sites (see §3) |
| Ensemble | NVT Langevin, 600 K, 1 fs, seed fixed |
| Sizes | **S**: 4×4×4 = 1 024 atoms; **L**: 10×10×10 = 16 000 atoms (GPU only) |
| Steps | 200 warm-up (untimed) + 2 000 timed (GPU); 20 + 100 timed (CPU, size S only) |
| Parallel | GPU: 1 GPU, 8 CPU threads; CPU: full node, `torch.set_num_threads(cores)` |
| GPUs | China V100 ×2 types, vanda A40, fornax RTX 5090, **hopper H100 and H200** |
| Variant | **W2-cueq** (optional): same run with cuEquivariance kernels on GPUs that support them (A40, RTX 5090, H100, H200; V100 likely unsupported) |
| Metric | timed steps/s → **ns/day** and **time for 1 ns** |

### Repetition & fairness rules

- **3 replicates** per (partition × workload), on different nodes where possible; report the median and
  min–max.
- Exclusive nodes (`--exclusive` / `place=excl`) so neighbours don't affect timings.
- Every job logs hostname, `lscpu`, `nvidia-smi`, module list, binary md5, env fingerprint.
- Queue wait time is recorded but **not** included in time-to-solution (reported separately).

---

## 3. Keeping software identical across clusters

### VASP

- **Track A (primary — "what users actually get")**: the site-provided VASP 6.5.1 build:
  - vanda: `~/bin/vasp.6.5.1/` (Intel 2023b + OpenMPI, self-compiled)
  - fornax: `module load vasp/vasp.6.5.1` (CPU), `vasp/vasp.6.5.1_nvhpc` (RTX 5090)
  - China: `module load vasp-intel2024.2/6.5.1` (CPU), `vasp/6.5.1-nvhpc` (V100)
- **Track B (optional — "same binary recipe")**: compile the same VASP 6.5.1 source
  (`vasp.6.5.1_intel2023b.tar.gz` from vanda) with the same `makefile.include` on each site, to separate
  hardware speed from compiler/MPI differences. Only if Track A shows surprising gaps.
- Same INCAR / KPOINTS / POSCAR / POTCAR everywhere (md5 listed in `inputs/vasp/MD5SUMS`).

### MACE

- One environment spec ([`envs/mace-bench.yaml`](envs/mace-bench.yaml)): Python 3.11, **torch 2.8.0 + cu128**,
  mace-torch 0.3.15, ase 3.25.
  - torch 2.8 cu128 wheels cover V100 (sm_70), A40 (sm_86) and RTX 5090 (sm_120). CUDA 13 wheels dropped
    Volta, so we must **not** use the cu130 build fornax currently uses. H100/H200 (sm_90) are also covered. **TODO: verify on every GPU type.**
- **China HPC has no outbound internet** (no PyPI access) → build once on vanda, `conda-pack` it, `scp`
  the tarball to fornax, hopper and China, unpack to the same relative path.
- `inputs/mace/fingerprint.py` prints versions + model md5 + reference energy/forces for a fixed
  structure; the outputs must agree across sites before any timing runs.

---

## 4. Cost model

`cost_per_job = rate × units_used × wall_hours`, converted to **SGD**. Rates:
[`config/pricing.yaml`](config/pricing.yaml). FX (2026-09-13): 1 CNY = 0.189 SGD, 1 USD = 1.267 SGD.

| Site | Resource | Rate | ≈ SGD | Source |
|---|---|---|---|---|
| China HPC | CPU (all partitions) | 0.1 CNY / core-h | 0.0189 / core-h | provider price |
| China HPC | V100 (16 / 32 GB) | 1 CNY / GPU-h | 0.189 / GPU-h | provider price |
| vanda | CPU | 0.01 SGD / core-h | 0.010 / core-h | NUS IT chargeback (Jul 2026 briefing) |
| vanda | A40 | 0.6 SGD / GPU-h | 0.60 / GPU-h | NUS IT chargeback |
| vanda | free queues (`cpu_parallel`, `gpu`) | 0 | 0 | plotted as "free" |
| hopper | H100 / H200 | 2.5 SGD / GPU-h | 2.50 / GPU-h | NUS IT chargeback |
| fornax | CPU (`genoa`, `largemem`) | 0.0513 USD / core-h | 0.065 / core-h | market avg: AWS c7a (EPYC Genoa) on-demand |
| fornax | RTX 5090 | 0.43 USD / GPU-h | 0.545 / GPU-h | market median of RTX 5090 cloud rentals |

Rules:
- CPU jobs are charged for **all cores of the node** we request (full node, exclusive).
- GPU jobs are charged **per GPU only**; host cores are included (as on every price list above).
- fornax is group-owned, so it is priced at the **average market rental price for the same hardware**.
  Cloud CPU prices are much higher than university rates, so fornax CPU will look expensive on the plot.
  This is a known effect of the pricing method, not of the hardware.

---

## 5. Pareto front

- One panel per workload (W1 VASP, W2-S, W2-L).
- x = time-to-solution (s, log scale), y = cost per job (SGD, log scale).
- Non-dominated points (nothing is both faster and cheaper) are joined as the Pareto front and labelled.
- Script: [`analysis/pareto.py`](analysis/pareto.py) reads `results/**/*.json`.

---

## 6. Plan / milestones

| # | Step | Where | Status |
|---|---|---|---|
| 0 | Access + inventory of all partitions | all | ✅ done (2026-10-01) |
| 1 | Build structures, POTCAR, INCAR; record md5s | vanda | ⬜ |
| 2 | Build `mace-bench` env, conda-pack, distribute | vanda → fornax, hopper, China | ⬜ |
| 3 | Run `fingerprint.py` on one node per GPU type + CPU; confirm identical energies | all | ⬜ |
| 4 | Smoke test: 1 short run per partition (W1 with NELM=3, W2 with 50 steps) | all | ⬜ |
| 5 | Fill in `pricing.yaml` | — | ✅ done (2026-10-01) |
| 6 | Production: 3 replicates × all partitions × W1/W2 | all | ⬜ |
| 7 | Collect results, plot Pareto front, write summary | local | ⬜ |

Rough compute budget: W1 ≈ 0.2–1 node-h per run → ~12 partitions × 3 reps ≈ 20 node-h;
W2 ≈ < 0.5 GPU-h per run. Small enough to finish in about 1 week including queue time.

---

## 7. Decisions (2026-10-01)

1. System: **c-Na₃PS₄** ✅
2. fornax cost: **average market rental price for the same hardware** ✅
3. VASP GPU: **yes**, on fornax RTX 5090 and China V100 / V100-32G ✅
4. Tianjin Region 1: **excluded** ✅
5. hopper H100 / H200: **included for MACE only** (no VASP) ✅
6. Still open: multi-node scaling (2–4 nodes) — out of scope for v1 unless requested.

---

## Repository layout

```
config/nodes.yaml        partition inventory + scheduler resource strings
config/pricing.yaml      cost rates + sources
envs/mace-bench.yaml     pinned MACE environment
inputs/vasp/             INCAR, KPOINTS (POSCAR/POTCAR generated in step 1)
inputs/mace/             bench_md.py, fingerprint.py
analysis/pareto.py       collect results + Pareto plot
results/<site>/<partition>/<workload>/<run>.json
```
