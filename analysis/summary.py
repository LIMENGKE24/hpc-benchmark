#!/usr/bin/env python3
"""Markdown results tables (one per workload): median time with min–max, cost, Pareto flag.

Usage: python analysis/summary.py   (prints Markdown; paste into README)
"""

import yaml

from pareto import BENCH_ORDER, LABELS, ROOT, aggregate, pareto_front

TITLE = {"W1": "VASP — 128-atom Na₃PS₄, 25 SCF steps", "W2-S": "MACE MD — 1,024 atoms", "W2-L": "MACE MD — 3,456 atoms"}
HARDWARE = {
    "china/p1": "2× Xeon Gold 6138 (40 c)", "china/9242": "2× Xeon Platinum 9242 (96 c)",
    "china/48cp1": "Xeon Platinum 8163 (48 c)", "china/48cp2": "Xeon Platinum 8255C (48 c)",
    "china/48cp3": "Xeon Platinum 8168 (48 c)", "china/v100": "1× V100 SXM2 16 GB", "china/v100g32": "1× V100 SXM2 32 GB",
    "vanda/batch_cpu": "2× Xeon Platinum 8452Y (72 c)", "vanda/batch_gpu": "1× A40 48 GB",
    "fornax/genoa_9354": "2× EPYC 9354 (64 c)", "fornax/genoa_7543": "2× EPYC 7543 (64 c)",
    "fornax/largemem": "2× EPYC 7742 (128 c)", "fornax/rtx5090": "1× RTX 5090 32 GB",
    "hopper/h100": "1× H100 80 GB", "hopper/h200": "1× H200 141 GB",
}


def fmt(v: float) -> str:
    return f"{v:.3g}" if v < 100 else f"{v:,.0f}"


def main() -> None:
    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text())
    data = aggregate(pricing)
    for bench in [b for b in BENCH_ORDER if b in data]:
        pts = data[bench]
        front = {p[2] for p in pareto_front([(p["t"], p["c"], p["label"]) for p in pts])}
        div, tunit, cunit = (1, "s", "job") if bench == "W1" else (3600, "h / ns", "ns")
        print(f"\n**{TITLE[bench]}** — sorted by time; ★ = Pareto-optimal\n")
        print(f"| Node | Hardware (charged) | Time ({tunit}) median [min–max] | Cost per {cunit} (SGD) | |")
        print("|---|---|---|---|---|")
        for p in sorted(pts, key=lambda p: p["t"]):
            t = f"{fmt(p['t'] / div)} [{fmt(p['t_lo'] / div)}–{fmt(p['t_hi'] / div)}]"
            cost = f"{p['c']:.3g}" + ("\\*" if p["site"] == "fornax" else "")  # * = cloud-proxy price, see README footnote
            print(f"| {LABELS[p['label']]} | {HARDWARE[p['label']]} | {t} | {cost} | {'★' if p['label'] in front else ''} |")


if __name__ == "__main__":
    main()
