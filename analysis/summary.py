#!/usr/bin/env python3
"""Markdown table per workload: median time, cost per job, n runs, Pareto flag. Usage: python analysis/summary.py"""

from collections import defaultdict
from statistics import median

import yaml

from pareto import ROOT, cost_sgd, load_results, pareto_front

UNITS = {"W1": "s (25 SCF, LOOP+)", "W2-S": "h per ns (1 024 atoms)", "W2-L": "h per ns (3 456 atoms)"}


def main() -> None:
    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text())
    groups = defaultdict(list)
    for r in load_results():
        r["cost"] = cost_sgd(r, pricing)
        groups[(r["bench"], f'{r["site"]}/{r["partition"]}')].append(r)

    for bench in sorted({b for b, _ in groups}):
        rows = []
        for (b, label), runs in groups.items():
            if b != bench:
                continue
            t = median(r["time_s"] for r in runs)
            c = median(r["cost"] for r in runs if r["cost"] is not None)
            rows.append((t, c, label, len(runs)))
        front = {p[2] for p in pareto_front([(t, c, lab) for t, c, lab, _ in rows])}
        scale = 1 if bench == "W1" else 3600
        print(f"\n**{bench}** — time in {UNITS.get(bench, 's')}\n")
        cost_hdr = "Cost / job (SGD)" if bench == "W1" else "Cost / ns (SGD)"
        print(f"| Partition | Time | {cost_hdr} | Runs | Pareto |\n|---|---|---|---|---|")
        for t, c, label, n in sorted(rows):
            print(f"| `{label}` | {t / scale:.3g} | {c:.3g} | {n} | {'★' if label in front else ''} |")


if __name__ == "__main__":
    main()
