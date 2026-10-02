#!/usr/bin/env python3
"""Time-vs-cost Pareto fronts from results/**/*.json and config/pricing.yaml.

One panel per workload. x = wall time, y = cost (both log). Colour = CPU/GPU, marker = site,
dark outline + step line = Pareto-optimal (no partition is both faster and cheaper).
Medians of the runs are plotted; whiskers span min–max.

Usage: python analysis/pareto.py [--out analysis/pareto.png]   (also writes a .pdf next to it)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]

BENCH_ORDER = ["W1", "W2-S", "W2-L"]
PANEL = {  # title, x label, y label, divide time_s by this
    "W1": ("(a) VASP 6.5.1 — Na$_3$PS$_4$, 128 atoms, 25 SCF steps", "Wall time for 25 SCF steps (s)", "Cost per job (SGD)", 1),
    "W2-S": ("(b) MACE-MPA-0 MD — 1,024 atoms", "Wall time per ns of MD (h)", "Cost per ns of MD (SGD)", 3600),
    "W2-L": ("(c) MACE-MPA-0 MD — 3,456 atoms", "Wall time per ns of MD (h)", "Cost per ns of MD (SGD)", 3600),
}
LABELS = {  # site/partition -> short label used on the figure and in the tables
    "china/p1": "China p1 (40c)", "china/9242": "China 9242 (96c)",
    "china/48cp1": "China 48cp1", "china/48cp2": "China 48cp2", "china/48cp3": "China 48cp3",
    "china/v100": "China V100 16 GB", "china/v100g32": "China V100 32 GB",
    "vanda/batch_cpu": "vanda CPU (72c)", "vanda/batch_gpu": "vanda A40",
    "fornax/genoa_9354": "fornax EPYC 9354 (64c)", "fornax/genoa_7543": "fornax EPYC 7543 (64c)",
    "fornax/largemem": "fornax EPYC 7742 (128c)", "fornax/rtx5090": "fornax RTX 5090",
    "hopper/h100": "hopper H100", "hopper/h200": "hopper H200",
}
SITE_MARKER = {"china": "o", "vanda": "s", "fornax": "^", "hopper": "D"}
SITE_NAME = {"china": "China HPC", "vanda": "vanda", "fornax": "fornax", "hopper": "hopper"}
KIND_COLOR = {"gpu": "#2a78d6", "cpu": "#eb6834"}  # validated categorical slots 1–2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e3e0", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.fontsize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def load_results() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "results").rglob("*.json")):
        if path.parent.name == "fingerprint":
            continue
        r = json.loads(path.read_text())
        if "tag" not in r:
            continue
        site, partition, workload, _rep = r["tag"].split("/")
        if workload.endswith("-smoke") or partition.startswith(("diag_", "genoa_mkl5")):
            continue
        if r.get("workload") == "mace_md" and r.get("device") == "cpu":  # MACE is benchmarked on GPUs only
            continue
        r.update(site=site, partition=partition, bench=workload)
        # MACE: time for 1 ns; VASP: LOOP+ real time (set by parse_vasp.py)
        r["time_s"] = r["time_to_1ns_s"] if "time_to_1ns_s" in r else r["time_s"]
        r["kind"] = "gpu" if (r.get("gpu") or r.get("workload") == "mace_md" or r.get("units_used") == 1) else "cpu"
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


def aggregate(pricing: dict) -> dict[str, list[dict]]:
    """bench -> list of {label, site, kind, t, t_lo, t_hi, c, c_lo, c_hi, n}."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in load_results():
        r["cost"] = cost_sgd(r, pricing)
        if r["cost"] is None:
            print(f"skip {r['tag']}: no price")
            continue
        groups[(r["bench"], r["site"], r["partition"])].append(r)
    out: dict[str, list[dict]] = defaultdict(list)
    for (bench, site, part), runs in groups.items():
        ts, cs = [r["time_s"] for r in runs], [r["cost"] for r in runs]
        out[bench].append(dict(label=f"{site}/{part}", site=site, kind=runs[0]["kind"], n=len(runs),
                               t=median(ts), t_lo=min(ts), t_hi=max(ts), c=median(cs), c_lo=min(cs), c_hi=max(cs)))
    return out


def _log_axis(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=20))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "pareto.png")
    args = ap.parse_args()
    from adjustText import adjust_text  # noqa: E402  (optional dependency, only needed for plotting)

    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text())
    data = aggregate(pricing)
    benches = [b for b in BENCH_ORDER if b in data]
    if not benches:
        raise SystemExit("no priced results yet")

    fig, axes = plt.subplots(1, len(benches), figsize=(5.6 * len(benches), 5.9))
    for ax, bench in zip(axes, benches):
        title, xlab, ylab, div = PANEL[bench]
        pts = data[bench]
        front = {p[2] for p in pareto_front([(p["t"], p["c"], p["label"]) for p in pts])}
        texts, xs, ys = [], [], []
        for p in sorted(pts, key=lambda p: p["t"]):
            x, c = p["t"] / div, KIND_COLOR[p["kind"]]
            ax.plot([p["t_lo"] / div, p["t_hi"] / div], [p["c"], p["c"]], color="#9a9a96", lw=0.9, zorder=2)
            ax.plot([x, x], [p["c_lo"], p["c_hi"]], color="#9a9a96", lw=0.9, zorder=2)
            opt = p["label"] in front
            ax.scatter(x, p["c"], s=120 if opt else 90, marker=SITE_MARKER[p["site"]], color=c,
                       edgecolor=INK if opt else SURFACE, linewidth=1.8 if opt else 1.2, zorder=4)
            xs.append(x); ys.append(p["c"])
            texts.append(ax.text(x, p["c"], LABELS.get(p["label"], p["label"]), fontsize=10,
                                 color=INK, fontweight="bold" if opt else "normal", zorder=5))
        fp = sorted((p["t"] / div, p["c"]) for p in pts if p["label"] in front)
        ax.step([f[0] for f in fp], [f[1] for f in fp], where="post", color=INK2, lw=1.4, zorder=3)
        ax.set(xscale="log", yscale="log", xlabel=xlab, ylabel=ylab)
        ax.set_title(title, loc="left", pad=10)
        _log_axis(ax)
        # fixed log-space padding: room for labels, and an empty lower-left corner for the "better" arrow
        ax.set_xlim(min(xs) / 2.2, max(xs) * 2.4)
        ax.set_ylim(min(ys) / 2.6, max(ys) * 2.4)
        ax.annotate("better", xy=(0.025, 0.025), xytext=(0.13, 0.09), xycoords="axes fraction",
                    textcoords="axes fraction", fontsize=10, color=INK2, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
        adjust_text(texts, x=xs, y=ys, ax=ax, expand=(1.15, 1.35), force_text=(0.3, 0.5), force_static=(0.3, 0.5),
                    only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    ensure_inside_axes=True, expand_axes=False,
                    arrowprops=dict(arrowstyle="-", color="#9a9a96", lw=0.6), min_arrow_len=10)

    handles = [Line2D([], [], marker="o", ls="", ms=9, color=KIND_COLOR["cpu"], label="CPU node (all cores)"),
               Line2D([], [], marker="o", ls="", ms=9, color=KIND_COLOR["gpu"], label="GPU (1 card)")]
    handles += [Line2D([], [], marker=m, ls="", ms=9, color="#ffffff", markeredgecolor=INK2, markeredgewidth=1.2,
                       label=SITE_NAME[s]) for s, m in SITE_MARKER.items()]
    handles += [Line2D([], [], marker="o", ls="-", ms=9, color=INK2, markerfacecolor="#ffffff",
                       markeredgecolor=INK, markeredgewidth=1.8, lw=1.4, label="Pareto front (optimal nodes)")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.075), handletextpad=0.5, columnspacing=1.6)
    fig.text(0.5, 0.004,
             "Points: median of 3 runs; whiskers: min–max. Cost = charged rate × resources × wall time, in SGD. "
             "China HPC and NUS (vanda, hopper) use their chargeback rates.\n"
             "fornax is the group's own server and is priced at public-cloud on-demand rates for equivalent "
             "hardware (AWS c7a for the EPYC CPUs; median RTX 5090 rental) — see README, Cost model.",
             ha="center", va="bottom", fontsize=9.5, color=INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
