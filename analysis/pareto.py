#!/usr/bin/env python3
"""Collect results/**/*.json, attach cost from config/pricing.yaml, plot time vs cost Pareto fronts.

Each result JSON needs: tag ("site/partition/workload/rep"), time_s (VASP, LOOP+ real time) or time_to_1ns_s (MACE),
and units_used (cores or GPUs charged). Usage: python analysis/pareto.py [--out pareto.png]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_results() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "results").rglob("*.json")):
        r = json.loads(path.read_text())
        site, partition, workload, _rep = r["tag"].split("/")
        r.update(site=site, partition=partition, bench=workload)
        # MACE: time for 1 ns; VASP: LOOP+ real time (set by parse_vasp.py)
        r["time_s"] = r["time_to_1ns_s"] if "time_to_1ns_s" in r else r["time_s"]
        rows.append(r)
    return rows


def cost_sgd(row: dict, pricing: dict) -> float | None:
    p = pricing.get(row["site"], {}).get(row["partition"])
    fx = pricing["fx_to_sgd"].get(p["currency"]) if p else None
    if not p or p["rate"] is None or fx is None:
        return None
    return p["rate"] * fx * row["units_used"] * row["time_s"] / 3600


def pareto_front(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Non-dominated set when minimising both time and cost."""
    front, best_cost = [], float("inf")
    for t, c, label in sorted(points):
        if c < best_cost:
            front.append((t, c, label))
            best_cost = c
    return front


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "pareto.png")
    args = ap.parse_args()

    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text())
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in load_results():
        r["cost"] = cost_sgd(r, pricing)
        groups[(r["bench"], r["site"], r["partition"])].append(r)

    by_bench: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for (bench, site, part), runs in groups.items():
        costs = [r["cost"] for r in runs if r["cost"] is not None]
        if not costs:
            print(f"skip {bench} {site}/{part}: no price")
            continue
        by_bench[bench].append((median(r["time_s"] for r in runs), median(costs), f"{site}/{part}"))

    if not by_bench:
        raise SystemExit("no priced results yet")
    fig, axes = plt.subplots(1, len(by_bench), figsize=(5.5 * len(by_bench), 4.5), squeeze=False)
    for ax, (bench, pts) in zip(axes[0], sorted(by_bench.items())):
        # free queues (cost 0) can't sit on a log axis: pin them below the cheapest paid point
        positive = [c for _, c, _ in pts if c > 0]
        floor = min(positive) / 3 if positive else 1e-3
        free = {label for _, c, label in pts if c == 0}
        pts = [(t, c if c > 0 else floor, label) for t, c, label in pts]
        front = pareto_front(pts)
        on_front = {p[2] for p in front}
        for t, c, label in pts:
            ax.scatter(t, c, color="C3" if label in on_front else "0.6", marker="v" if label in free else "o", zorder=3)
            ax.annotate(label + (" (free)" if label in free else ""), (t, c), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="C3", lw=1.2)
        ax.set(xscale="log", yscale="log", title=bench, xlabel="time to solution (s)", ylabel="cost per job (SGD)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
