"""
Generate the final Stage 3 closeout Pareto figures.

This aligns Stage 1 / Stage 2 / Stage 3 operating points on the same
"avg_steps vs avg_f1" plane using:
  - Stage 2 probe tables for Oracle / Global-Weitzman
  - Stage 3 main experiment JSON for Probe / Probe+E-value / Probe+CP

Usage:
  python scripts/stage3_make_final_pareto.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

DATASETS = ("hotpotqa", "musique", "2wiki")
LABELS = {
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "2wiki": "2Wiki",
}

STYLE = {
    "Oracle": {"color": "#d62728", "marker": "*", "size": 230},
    "Global-Weitzman": {"color": "#ff7f0e", "marker": "^", "size": 125},
    "Probe": {"color": "#1f77b4", "marker": "o", "size": 105},
    "Probe+E-value": {"color": "#2ca02c", "marker": "D", "size": 105},
    "Probe+CP": {"color": "#9467bd", "marker": "s", "size": 105},
}


def _load_stage2_table(dataset: str) -> List[Dict[str, str]]:
    path = RESULTS_DIR / f"stage2_probe_table_{dataset}_pdopt_best.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_stage3_json(dataset: str) -> Dict[str, object]:
    path = RESULTS_DIR / f"stage3_evalue_{dataset}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _pick_stage2_row(rows: Iterable[Dict[str, str]], strategy: str) -> Dict[str, float]:
    for row in rows:
        if row.get("strategy") == strategy:
            return {
                "avg_steps": float(row["avg_steps"]),
                "avg_f1": float(row["avg_f1"]),
            }
    raise KeyError(f"Missing strategy {strategy!r}")


def _load_dataset_points(dataset: str) -> Dict[str, Dict[str, float]]:
    stage2_rows = _load_stage2_table(dataset)
    stage3 = _load_stage3_json(dataset)
    alpha_block = stage3["per_gamma"]["0.5"]["per_alpha"]["0.1"]
    summaries = alpha_block["summaries"]
    return {
        "Oracle": _pick_stage2_row(stage2_rows, "Oracle"),
        "Global-Weitzman": _pick_stage2_row(stage2_rows, "Global-Weitzman"),
        "Probe": {
            "avg_steps": float(summaries["probe"]["avg_steps"]),
            "avg_f1": float(summaries["probe"]["avg_f1"]),
        },
        "Probe+E-value": {
            "avg_steps": float(summaries["probe_evalue"]["avg_steps"]),
            "avg_f1": float(summaries["probe_evalue"]["avg_f1"]),
        },
        "Probe+CP": {
            "avg_steps": float(summaries["probe_conformal_quantile"]["avg_steps"]),
            "avg_f1": float(summaries["probe_conformal_quantile"]["avg_f1"]),
        },
    }


def _pareto_nondominated(points: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in points]
    nd: List[Tuple[float, float]] = []
    for i, (cx, fy) in enumerate(pts):
        dominated = False
        for j, (ox, oy) in enumerate(pts):
            if i == j:
                continue
            if (ox <= cx and oy >= fy) and (ox < cx or oy > fy):
                dominated = True
                break
        if not dominated:
            nd.append((cx, fy))
    return sorted(nd, key=lambda item: (item[0], -item[1]))


def _annotate(ax: plt.Axes, name: str, x: float, y: float) -> None:
    offsets = {
        "Oracle": (5, 8),
        "Global-Weitzman": (5, -12),
        "Probe": (5, 6),
        "Probe+E-value": (5, 6),
        "Probe+CP": (5, -12),
    }
    dx, dy = offsets.get(name, (5, 5))
    ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8)


def _plot_single(dataset: str, points: Dict[str, Dict[str, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    xy = []
    for name, point in points.items():
        style = STYLE[name]
        x = point["avg_steps"]
        y = point["avg_f1"]
        xy.append((x, y))
        ax.scatter(x, y, c=style["color"], marker=style["marker"], s=style["size"])
        _annotate(ax, name, x, y)
    nd = _pareto_nondominated(xy)
    if len(nd) >= 2:
        ax.plot(
            [p[0] for p in nd],
            [p[1] for p in nd],
            color="#444444",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )
    ax.set_title(f"Stage1-3 Pareto Alignment - {LABELS[dataset]}")
    ax.set_xlabel("Avg retrieval steps")
    ax.set_ylabel("Avg F1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_combined(all_points: Dict[str, Dict[str, Dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        points = all_points[dataset]
        xy = []
        for name, point in points.items():
            style = STYLE[name]
            x = point["avg_steps"]
            y = point["avg_f1"]
            xy.append((x, y))
            ax.scatter(x, y, c=style["color"], marker=style["marker"], s=style["size"])
            _annotate(ax, name, x, y)
        nd = _pareto_nondominated(xy)
        if len(nd) >= 2:
            ax.plot(
                [p[0] for p in nd],
                [p[1] for p in nd],
                color="#444444",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
        ax.set_title(LABELS[dataset])
        ax.set_xlabel("Avg retrieval steps")
        ax.set_ylabel("Avg F1")
        ax.grid(alpha=0.25)
    fig.suptitle("Pandora-RAG Final Pareto Alignment (Stage 1-3)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    all_points = {dataset: _load_dataset_points(dataset) for dataset in DATASETS}
    for dataset, points in all_points.items():
        _plot_single(dataset, points, RESULTS_DIR / f"stage3_final_pareto_{dataset}.png")
    _plot_combined(all_points, RESULTS_DIR / "stage3_final_pareto_all.png")


if __name__ == "__main__":
    main()
