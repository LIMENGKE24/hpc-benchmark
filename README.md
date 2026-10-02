# HPC Benchmark — time vs. cost Pareto front

Benchmark every CPU and GPU node type we can use on **China HPC (超算云平台)**, **vanda**, **fornax** and
**hopper** (MACE only)
with *identical* VASP and MACE-MD workloads, then plot **time-to-solution vs. cost per job** to find the
Pareto-optimal nodes (fastest for a given cost / cheapest for a given speed).

> Status (2026-10-02): **complete** — all partitions benchmarked, 3 runs each. Results in §9, setup findings in §8.

---

## 1. Node inventory (what gets benchmarked)

Collected 2026-10-01 from `sinfo` / `pbsnodes` / `qstat -Q`. Full machine-readable list: [`config/nodes.yaml`](config/nodes.yaml).

| Site | Partition / queue | Hardware (per node) | VASP | MACE |
|---|---|---|---|---|
| China HPC (Inner Mongolia 2) | `p1` | 2× Xeon Gold 6138, 40 c, 190 GB | CPU | — |
| China HPC | `9242` | 2× Xeon Platinum 9242, 96 c, 378 GB | CPU | — |
| China HPC | `48cp1` | Xeon Platinum 8163, 48 c, 190 GB | CPU | — |
| China HPC | `48cp2` | Xeon Platinum 8255C, 48 c, 190 GB | CPU | — |
| China HPC | `48cp3` | Xeon Platinum 8168, 48 c, 190 GB | CPU | — |
| China HPC | `v100` | 8× V100 16 GB, 24 c | GPU (nvhpc) | GPU |
| China HPC | `v100g32` | 8× V100 32 GB, 24 c | GPU (nvhpc) | GPU |
| vanda | `batch_cpu` (cn-*) | 72 c, 512 GB | CPU | — |
| vanda | `batch_gpu` (gn-a40-*) | 2× A40 48 GB | — (no GPU VASP build) | GPU |
| fornax | `genoa_9354` (c16–c20; c15 excluded) | 2× AMD EPYC 9354, 64 c, 384 GB | CPU | — |
| fornax | `genoa_7543` (c13) | 2× AMD EPYC 7543, 64 c, 512 GB | CPU | — |
| fornax | `largemem` | 2× AMD EPYC 7742, 128 c, 512 GB | CPU | — |
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
| Metric | **`LOOP+` real time** from OUTCAR (excludes startup/IO; `Elapsed time` kept for reference) + energy check (all sites must agree to < 1 meV/atom) |

Optional **W1-8GPU**: full China V100 node (8 ranks, `KPAR = 4`) to see multi-GPU scaling.
Optional **W1-large**: 3×3×3 supercell (432 atoms, Γ-only) to test scaling on fat nodes / multi-GPU.

### W2 — MACE MD (fixed work)

| Setting | Value |
|---|---|
| Model | **MACE-MPA-0 medium** (`mace-mpa-0-medium.model`, md5 pinned) |
| Code | **mace-torch 0.3.15**, float32, identical conda-packed env on all sites (see §3) |
| Ensemble | NVT Langevin, 600 K, 1 fs, seed fixed |
| Sizes | **S**: 4×4×4 = 1 024 atoms; **L**: 6×6×6 = 3 456 atoms (GPU only; 16 000 atoms ran out of memory on the 48 GB A40, and L must fit a 16 GB V100) |
| Steps | S: 200 warm-up + 2 000 timed; L: 100 + 1 000 timed |
| Parallel | 1 GPU, 8 CPU threads (China V100: 3 = 24 cores / 8 GPUs). **GPU only — no CPU MACE benchmark** |
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
| 1 | Build structures, POTCAR, INCAR; record md5s | vanda | ✅ |
| 2 | Build `mace-bench` env, conda-pack, distribute | vanda → fornax, hopper, China | ✅ (same tarball, md5 `eb73c579…`) |
| 3 | Run `fingerprint.py` on one node per GPU type + CPU; confirm identical energies | all | ✅ A40/RTX5090/H100/H200/V100 agree to 1e-13 eV |
| 4 | Smoke test: 1 short run per partition (W1 with NELM=3, W2 with 20 steps) | all | ✅ |
| 5 | Fill in `pricing.yaml` | — | ✅ done (2026-10-01) |
| 6 | Production: 3 replicates × all partitions × W1/W2 | all | ✅ 2026-10-02 |
| 7 | Collect results, plot Pareto front, write summary | local | ✅ (§9) |

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

## 8. Findings during setup

- **Project code** for vanda/hopper is now `CFP05-CF-241` (CFP03-CF-027 expired).
- **fornax GPU nodes do not mount `/scratch`** → fornax runs from `/home/li.mengke/hpc-benchmark`.
- **fornax `genoa` queue is heterogeneous**: `fornax-c13` = AMD EPYC 7543 (Milan), `fornax-c15…c20` = EPYC 9354 (Genoa).
  Stock `vasp/vasp.6.5.1` (Intel 2019): 3.7–4.2 s/SCF on c16/c18 (fastest CPU nodes measured), 6.4 s on c13,
  but 22–34 s on **c15** (3 separate runs, alone on the node; `MKL_DEBUG_CPU_TYPE=5` no help) → c15 excluded; report to fornax admin.
  On the plot genoa is split into `genoa_9354` and `genoa_7543`.
- **fornax node pinning**: use `vnode=fornax-cNN`; `host=` never matches CPU nodes (they register as `fornax-cNN-ib0`).
- **vanda free `gpu` queue**: access denied for our account → dropped from the plan.
- **hopper login node** kills background processes (incl. tmux) at logout; build envs elsewhere or in a job.
- **China HPC → no internet**: env copied as a conda-pack tarball (4.2 GB, ~2 h over 4 parallel streams).
- **VASP timing** uses the `LOOP+` real time (excludes ~14 s startup/IO per run).
- VASP on consumer RTX 5090 is FP64-limited (~115 s/SCF vs ~6 s on a 72-core CPU node).

---

## 9. Results (final, 2026-10-02)

![Pareto front](analysis/pareto.png)

Medians of 3 runs. ★ = Pareto-optimal (no other partition is both faster and cheaper).
Regenerate: `analysis/collect.sh && python analysis/pareto.py && python analysis/summary.py`.

**W1** — time in s (25 SCF, LOOP+)

| Partition | Time | Cost / job (SGD) | Runs | Pareto |
|---|---|---|---|---|
| `fornax/genoa_9354` | 104 | 0.121 | 3 | ★ |
| `china/v100` | 138 | 0.00724 | 3 | ★ |
| `china/v100g32` | 145 | 0.00762 | 3 |  |
| `vanda/batch_cpu` | 148 | 0.0297 | 3 |  |
| `china/9242` | 161 | 0.0813 | 3 |  |
| `fornax/genoa_7543` | 174 | 0.201 | 3 |  |
| `fornax/rtx5090` | 189 | 0.0286 | 3 |  |
| `fornax/largemem` | 198 | 0.458 | 3 |  |
| `china/48cp2` | 319 | 0.0805 | 3 |  |
| `china/48cp3` | 326 | 0.0821 | 3 |  |
| `china/48cp1` | 334 | 0.0841 | 3 |  |
| `china/p1` | 377 | 0.0791 | 3 |  |

**W2-L** — time in h per ns (3 456 atoms)

| Partition | Time | Cost / ns (SGD) | Runs | Pareto |
|---|---|---|---|---|
| `fornax/rtx5090` | 55.7 | 30.3 | 3 | ★ |
| `hopper/h200` | 74.6 | 186 | 3 |  |
| `hopper/h100` | 77.4 | 193 | 3 |  |
| `china/v100g32` | 145 | 27.3 | 3 | ★ |
| `china/v100` | 145 | 27.5 | 3 |  |
| `vanda/batch_gpu` | 157 | 94.4 | 3 |  |

**W2-S** — time in h per ns (1 024 atoms)

| Partition | Time | Cost / ns (SGD) | Runs | Pareto |
|---|---|---|---|---|
| `fornax/rtx5090` | 17.3 | 9.42 | 3 | ★ |
| `hopper/h200` | 17.8 | 44.4 | 3 |  |
| `hopper/h100` | 18.8 | 47.1 | 3 |  |
| `china/v100` | 41.6 | 7.86 | 3 | ★ |
| `china/v100g32` | 43.3 | 8.18 | 3 |  |
| `vanda/batch_gpu` | 47 | 28.2 | 3 |  |

**Takeaways**
- **VASP**: two Pareto points. **Fastest: fornax `genoa_9354`** (EPYC 9354, 64 c; 104 s) — but it is priced at
  AWS on-demand rates (0.12 SGD/job). **Cheapest: China `v100`** (1 GPU, nvhpc build; 138 s, 0.007 SGD/job, ~17× cheaper).
  Best university-rate CPU option: vanda (148 s, 0.03 SGD). China CPU partitions are 2–3× slower at ~0.08 SGD.
  fornax `largemem` (EPYC 7742, 128 c) is slower than the 64-core genoa nodes (198 s).
- **MACE**: fornax RTX 5090 is fastest at both sizes; China V100 is cheapest.
  H100/H200 are about as fast as the RTX 5090 but ~5× the cost per ns (no cuEquivariance; small systems).
- MACE MD is benchmarked on GPUs only (CPU nodes dropped from W2 on 2026-10-01).
- Energies agree across sites: VASP −526.017345 eV on every CPU/GPU build; MACE fingerprint to 1e-13 eV.

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
