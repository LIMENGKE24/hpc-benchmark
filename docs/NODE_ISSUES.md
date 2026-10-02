# Node issues found during the benchmark (fornax)

Two fornax nodes were consistently far slower than their identical siblings. Both were excluded from the
benchmark results; the evidence is kept here (raw records in `results/fornax/diag_*/`) so it can be passed
to the fornax administrator. Status: **open — not yet reported** (2026-10-02).

## 1. `fornax-c15` (genoa queue, 2× AMD EPYC 9354) — ~7× slower for VASP

Same binary (`module load vasp/vasp.6.5.1`, Intel 2019 build), same 64-rank MPI job, same inputs
(128-atom Na₃PS₄, Γ-centred 2×2×2, `NCORE = 4`), node allocated exclusively (`place=excl`).
Seconds per SCF step (`LOOP:` real time in OUTCAR):

| Node | CPU | Date / job | SCF steps 1–3 (s) | Record |
|---|---|---|---|---|
| c16 | EPYC 9354 | 2026-10-01 19:05, 42662 | 3.7 · 3.8 · 4.2 | `results/fornax/diag_c16/` |
| c18 | EPYC 9354 | 2026-10-01 19:05, 42664 | 3.8 · 4.0 · 4.4 | `results/fornax/diag_c18/` |
| c17 | EPYC 9354 | 2026-10-02 00:23 (production rep 3) | 3.9 · 4.0 · 4.4 | `results/fornax/genoa_9354/W1/rep3.json` |
| c13 | EPYC 7543 (older) | 2026-10-01 23:54, 42660 | 6.4 · 6.6 · 7.4 | `results/fornax/diag_c13/` |
| **c15** | EPYC 9354 | 2026-10-01 14:00, 42624 | **32.6 · 30.2 · 38.7** | `results/fornax/genoa/W1-smoke/` |
| **c15** | EPYC 9354 | 2026-10-01 15:50, 42632 (`MKL_DEBUG_CPU_TYPE=5`) | **28.7 · 31.7 · 27.7** | `results/fornax/genoa_mkl5/` |
| **c15** | EPYC 9354 | 2026-10-02 03:16, 42661 | **22.3 · 21.4 · 34.1** | `results/fornax/diag_c15/` |

c15 is 6–9× slower than c16/c17/c18 in three runs spread over 13 hours, each with the node to itself.
`lscpu` reports the same model (`AMD EPYC 9354 32-Core Processor`, 64 CPUs, 2 sockets) on all four nodes.
The MKL/AMD code-path setting made no difference, so it is not a library issue.

Things worth checking on c15: leftover processes from earlier jobs (`top`, `ps -eo pid,user,pcpu,etime,cmd --sort=-pcpu`),
CPU frequency / power cap (`cpupower frequency-info`, `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`),
NUMA / memory configuration (`numactl -H`, a DIMM missing or running in a degraded mode would fit a uniform slowdown),
thermal throttling (`dmesg | grep -i -E "throttl|mce|edac"`).

## 2. `fornax-g01` (rtx5090 queue, RTX 5090 + Ryzen 9 9950X) — 7–14× slower for VASP and MACE

Same GPU model and host CPU as g03–g07 (`nvidia-smi -L`: GeForce RTX 5090; `lscpu`: Ryzen 9 9950X), exclusive node.

| Node | Date / job | VASP, s per SCF step (nvhpc build, 1 GPU) | MACE MD, steps/s (1,024 atoms) | Record |
|---|---|---|---|---|
| g03 | 2026-10-01 16:33 | 6.9 · 7.0 · 7.7 | — | `results/fornax/rtx5090/W1/rep2.json` |
| g04 | 2026-10-02 11:00, 42691 | 6.9 · 7.1 · 7.8 | 16.0 (2026-10-01) | `results/fornax/rtx5090/W1/rep1.json`, `W2-S/rep1.json` |
| g05 | 2026-10-01 16:34 | 6.9 · 7.0 · 7.7 | 16.8 | `results/fornax/rtx5090/W1/rep3.json` |
| g06, g07 | 2026-10-01 | — | 16.1, 16.2 | `results/fornax/rtx5090/W2-S/rep2,3.json` |
| **g01** | 2026-10-01 16:09, 42633 | **96.5 · 100.7 · 111.5** | — | `results/fornax/rtx5090/W1-smoke/rep2.json` |
| **g01** | 2026-10-01 16:33 | **96.5 · 100.7 · 111.4 …** (2,610 s for 25 steps vs 188 s) | — | `results/fornax/diag_g01/W1/rep1.json` |
| **g01** | 2026-10-02 11:00, 42693 / 42692 | **96.5 · 100.7 · 111.5** | **2.29** | `results/fornax/rtx5090/W1-smoke/rep9.json`, `W2-S-smoke/rep9.json` |

g01 is 14× slower for VASP (a double-precision, PCIe-transfer-heavy code) and 7× slower for MACE (single precision,
compute-bound), in three runs over 19 hours. The step times on g01 are identical to the second across runs, which points
to a fixed hardware limitation rather than contention. Things worth checking on g01: PCIe link width/speed
(`nvidia-smi -q | grep -A3 "PCIe Generation"` and `"Link Width"` — a card negotiated at x1/x4 or Gen1 would give this
pattern), power cap and clocks (`nvidia-smi -q -d POWER,CLOCK`), ECC/Xid errors (`dmesg | grep -i xid`),
and whether the card is seated properly / a riser is in use.

## How to reproduce

From the repo on fornax (`/home/li.mengke/hpc-benchmark`), a 3-SCF VASP run takes ~1 min on a good node:

```bash
qsub -l select=1:ncpus=64:mpiprocs=64:vnode=fornax-c15 -o logs/ -v W=W1,REP=1,SMOKE=1,PART=diag_c15 jobs/fornax/cpu.pbs
qsub -l select=1:ncpus=32:vnode=fornax-g01            -o logs/ -v W=W1,REP=1,SMOKE=1 jobs/fornax/gpu.pbs
qsub -l select=1:ncpus=32:vnode=fornax-g01            -o logs/ -v W=W2,SIZE=S,REP=1,SMOKE=1 jobs/fornax/gpu.pbs
```

Compare `grep LOOP: runs/fornax/<tag>/OUTCAR` (VASP) or `steps_per_s` in the result JSON (MACE) with the tables above.

## Message for the fornax administrator (ready to send)

> Hi, while benchmarking VASP and MACE across the fornax nodes (Oct 1–2, 2026) we found two nodes that are
> consistently much slower than their identical siblings, with the node allocated exclusively each time:
>
> - **fornax-c15** (genoa, 2× EPYC 9354): VASP 6.5.1 takes 22–39 s per SCF step in three separate runs, versus
>   3.7–4.4 s on c16, c17 and c18 with the same binary and inputs (6–9× slower). Same `lscpu` output on all four.
> - **fornax-g01** (rtx5090): VASP (nvhpc) takes ~100 s per SCF step versus ~7 s on g03/g04/g05 (14×), and MACE MD
>   runs at 2.3 steps/s versus ~16 on g04–g07 (7×), again in three runs. Identical timings each time suggest a fixed
>   hardware limit — possibly a degraded PCIe link, power cap, or memory configuration.
>
> Evidence, raw outputs and reproduction commands are in https://github.com/LIMENGKE24/hpc-benchmark/blob/main/docs/NODE_ISSUES.md.
> Could you have a look at these two nodes? Happy to rerun the test after any change. Thanks, Mengke
