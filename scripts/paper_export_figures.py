"""
Export camera-ready PDF figures for the paper (paper/figs/).

Reads metrics from paper/results/ (fallback: results/) and writes:
  - stage3_final_pareto_all.pdf
  - evalue_hotpotqa_triple.pdf
  - stop_rag_pareto_1x3.pdf

Usage:
  python scripts/paper_export_figures.py
  python scripts/paper_export_figures.py --results-dir results --out-dir paper/figs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATASETS = ("hotpotqa", "musique", "2wiki")
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "2wiki": "2Wiki",
}

PARETO_ORDER = ("Oracle", "Global-Weitzman", "Probe", "Probe+E-value", "Probe+CP")

PARETO_STYLE: Dict[str, Dict[str, Any]] = {
    "Oracle": {
        "color": "#b0b0b0",
        "edgecolor": "#888888",
        "marker": "*",
        "size": 160,
        "zorder": 4,
        "hollow": True,
        "legend": "Oracle (structural ref.)",
    },
    "Global-Weitzman": {
        "color": "#ff7f0e",
        "edgecolor": "#ff7f0e",
        "marker": "^",
        "size": 80,
        "zorder": 3,
        "hollow": False,
        "legend": "Global-Weitzman",
    },
    "Probe": {"color": "#1f77b4", "marker": "o", "size": 70, "zorder": 5, "hollow": False, "legend": "Probe"},
    "Probe+E-value": {
        "color": "#2ca02c",
        "marker": "D",
        "size": 70,
        "zorder": 5,
        "hollow": False,
        "legend": "Probe+E-value",
    },
    "Probe+CP": {
        "color": "#9467bd",
        "marker": "s",
        "size": 70,
        "zorder": 5,
        "hollow": False,
        "legend": "Probe+CP",
    },
}

# Manual text offsets to reduce overlap (xytext points)
ANNOT_PAD: Dict[str, Dict[str, Tuple[int, int]]] = {
    "hotpotqa": {"Probe+CP": (8, 12), "Oracle": (6, 10), "Global-Weitzman": (8, -14)},
    "musique": {"Probe+CP": (8, 12), "Oracle": (6, 10)},
    "2wiki": {"Probe+CP": (-48, 14), "Probe": (6, 8), "Probe+E-value": (6, -14)},
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


def _load_stage2_table(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_stage2_row(rows: Iterable[Dict[str, str]], strategy: str) -> Dict[str, float]:
    for row in rows:
        if row.get("strategy") == strategy:
            return {"avg_steps": float(row["avg_steps"]), "avg_f1": float(row["avg_f1"])}
    raise KeyError(f"Missing strategy {strategy!r}")


def _load_dataset_points(results_dir: Path, dataset: str) -> Dict[str, Dict[str, float]]:
    stage2_rows = _load_stage2_table(results_dir / f"stage2_probe_table_{dataset}_pdopt_best.csv")
    with (results_dir / f"stage3_evalue_{dataset}.json").open(encoding="utf-8") as f:
        stage3 = json.load(f)
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


def _scatter_method(ax: plt.Axes, name: str, x: float, y: float, label: Optional[str] = None) -> None:
    st = PARETO_STYLE[name]
    kwargs: Dict[str, Any] = {
        "marker": st["marker"],
        "s": st["size"],
        "zorder": st["zorder"],
        "label": label,
    }
    if st.get("hollow"):
        kwargs["facecolors"] = "none"
        kwargs["edgecolors"] = st.get("edgecolor", st["color"])
        kwargs["linewidths"] = 0.9
    else:
        kwargs["c"] = st["color"]
    ax.scatter(x, y, **kwargs)


def _plot_pareto_combined(
    all_points: Dict[str, Dict[str, Dict[str, float]]], out_path: Path, dpi: int
) -> None:
    _set_pub_rc()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=False)
    plt.subplots_adjust(left=0.075, right=0.995, bottom=0.34, top=0.88, wspace=0.28)

    for col, dataset in enumerate(DATASETS):
        ax = axes[col]
        points = all_points[dataset]
        xy: List[Tuple[str, float, float]] = []
        for name in PARETO_ORDER:
            p = points[name]
            x, y = p["avg_steps"], p["avg_f1"]
            xy.append((name, x, y))
        # Focus y on deployable + GW (Oracle 单独 clip 处理)
        focus_names = ("Global-Weitzman", "Probe", "Probe+E-value", "Probe+CP")
        fys = [points[n]["avg_f1"] for n in focus_names]
        y_min, y_max = min(fys) - 0.02, max(fys) + 0.02
        o_y = points["Oracle"]["avg_f1"]
        y_max = max(y_max, o_y + 0.015)

        # Legend only from first column
        for name, x, y in xy:
            lab = PARETO_STYLE[name]["legend"] if col == 0 else None
            _scatter_method(ax, name, x, y, label=lab)

        nd = _pareto_nondominated([(x, y) for _, x, y in xy])
        if len(nd) >= 2:
            ax.plot(
                [p[0] for p in nd],
                [p[1] for p in nd],
                color="#666666",
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
            )

        ax.set_ylim(y_min, y_max)

        # Pareto preference arrow (axes coords, lower-left to upper-left semantics)
        ax.annotate(
            "",
            xy=(0.12, 0.88),
            xycoords="axes fraction",
            xytext=(0.32, 0.28),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.5", color="0.75", lw=0.6),
        )

        ax.set_title(DATASET_LABELS[dataset], fontsize=9.5, fontweight="normal", pad=3)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=8)

    leg = fig.legend(
        handles=axes[0].get_legend_handles_labels()[0],
        labels=axes[0].get_legend_handles_labels()[1],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=True,
        fontsize=6.9,
        handletextpad=0.4,
        columnspacing=0.7,
    )
    leg.get_frame().set_linewidth(0.4)
    fig.supxlabel("Average retrieval steps (lower is better)", fontsize=8.5, y=0.17)
    fig.supylabel("Answer F1 (higher is better)", fontsize=8.5, x=0.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(out_path.with_suffix(".png"), dpi=max(dpi, 300), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E-value Hotpot triple
# ---------------------------------------------------------------------------

def _plot_evalue_triple(
    path_noshift: Path,
    path_sudden: Path,
    out_path: Path,
    dpi: int,
) -> None:
    _set_pub_rc()
    with path_noshift.open(encoding="utf-8") as f:
        j0 = json.load(f)
    with path_sudden.open(encoding="utf-8") as f:
        j1 = json.load(f)

    def wealth_alpha(block: Dict[str, Any]) -> Tuple[List[float], float]:
        a = block["0.1"]
        return [float(x) for x in a["wealth_trace"]], float(a["wealth_cap"])

    w0, cap0 = wealth_alpha(j0["per_gamma"]["0.5"]["per_alpha"])
    w1, cap1 = wealth_alpha(j1["per_gamma"]["0.5"]["per_alpha"])
    assert abs(cap0 - cap1) < 1e-6
    cap = cap0
    alpha = 0.1

    cur0 = {k: v for k, v in j0["per_gamma"]["0.5"]["per_alpha"]["0.1"]["cumulative_error_curves"].items()}
    cur1 = {k: v for k, v in j1["per_gamma"]["0.5"]["per_alpha"]["0.1"]["cumulative_error_curves"].items()}

    y_lo = 1e-3
    w_min = min(min(w0), min(w1), y_lo)
    w_max = max(max(w0), max(w1), cap * 1.05)
    log_min = math.log10(w_min + 1e-12)
    log_max = math.log10(w_max + 1e-12)
    for extra in (w0, w1):
        for v in extra:
            log_min = min(log_min, math.log10(v + 1e-12))
            log_max = max(log_max, math.log10(v + 1e-12))
    w_shared = (10**log_min, 10**log_max)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.78))
    plt.subplots_adjust(left=0.07, right=0.995, bottom=0.34, wspace=0.32, top=0.86)

    # Left: no shift
    ax0 = axes[0]
    ax0.plot(np.arange(len(w0)), w0, color="#1f77b4", linewidth=0.9, label="Probe+E-value")
    ax0.axhline(cap, color="#c62828", linestyle="-", linewidth=1.0, zorder=0, label="Alarm boundary ($1/\\alpha=10$)")
    ax0.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=0.7, alpha=0.7)
    ax0.set_yscale("log")
    ax0.set_ylim(w_shared)
    ax0.set_ylabel("E-wealth (log scale)", fontsize=8.5)
    ax0.set_title("No shift", fontsize=9.5, pad=3)
    ax0.grid(alpha=0.3, linewidth=0.4)
    ax0.tick_params(labelsize=7.5)

    # Middle: sudden
    ax1 = axes[1]
    ax1.plot(np.arange(len(w1)), w1, color="#1f77b4", linewidth=0.9, label="Probe+E-value")
    ax1.axhline(cap, color="#c62828", linestyle="-", linewidth=1.0, zorder=0)
    ax1.axhline(1.0, color="#9e9e9e", linestyle=":", linewidth=0.7, alpha=0.7)
    ax1.set_yscale("log")
    ax1.set_ylim(w_shared)
    ax1.set_title("Sudden shift", fontsize=9.5, pad=3)
    ax1.grid(alpha=0.3, linewidth=0.4)
    ax1.tick_params(labelsize=7.5)

    # Right: cumulative error difference vs Probe
    ax2 = axes[2]
    probe = np.asarray(cur1["Probe"], dtype=np.float64)
    for label, key, color, ls in [
        ("Probe", "Probe", "#1f77b4", "-"),
        ("Probe+E-value", "Probe+E-value", "#2ca02c", "-"),
        ("Probe+CP", "Probe+CP-quantile", "#9467bd", "-"),
    ]:
        arr = np.asarray(cur1[key], dtype=np.float64) - probe
        ax2.plot(np.arange(1, len(arr) + 1), arr, color=color, linewidth=0.9, linestyle=ls, label=label)
    ax2.axhline(0.0, color="#9e9e9e", linestyle=":", linewidth=0.7)
    ax2.set_ylabel("Cumulative error $-$ Probe", fontsize=8.5)
    ax2.set_title("Cumulative error (sudden shift)", fontsize=9.5, pad=3)
    ax2.grid(alpha=0.3, linewidth=0.4)
    ax2.tick_params(labelsize=7.5)
    # Shared legend for row 0-1
    h0, l0 = ax0.get_legend_handles_labels()
    fig.legend(
        h0,
        l0,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.28, 0.02),
        fontsize=6.8,
        frameon=True,
    )
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(
        h2,
        l2,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.75, 0.02),
        fontsize=6.8,
        frameon=True,
        columnspacing=0.75,
        handlelength=1.8,
    )
    fig.supxlabel("Stream index", fontsize=8.5, y=0.17)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(out_path.with_suffix(".png"), dpi=max(dpi, 300), bbox_inches="tight")
    plt.close(fig)


def _set_pub_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 0.9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Stop-RAG 1x3 (inline from stop_rag_make_pareto data loading)
# ---------------------------------------------------------------------------


def _parse_stop_metrics_name(path: Path) -> Optional[Dict[str, Any]]:
    import re

    match = re.match(
        r"(?P<dataset>.+)_test_ckpt(?P<ckpt>[^_]+)_thr(?P<threshold>.+)\.metrics\.json$", path.name
    )
    if not match:
        return None
    out = match.groupdict()
    try:
        out["threshold"] = float(out["threshold"])
    except ValueError:
        return None
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _flatten(values: Sequence[Sequence[float]]) -> List[float]:
    return [float(item) for sub in values for item in sub]


def _read_stop_logs(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    logs: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def _stopping_from_logs(logs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter

    steps = [int(log["stop_iter"]) for log in logs if "stop_iter" in log]
    dist = Counter(steps)
    return {
        "n": len(steps),
        "avg_steps": _mean(steps),
        "stop_distribution": {str(k): int(dist[k]) for k in sorted(dist)},
    }


def _load_online_point(path: Path, root: Path) -> Optional[Dict[str, Any]]:
    parsed = _parse_stop_metrics_name(path)
    if parsed is None:
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    answer = data["metrics"]["answer"]
    f1_values = _flatten(answer["f1"])
    em_values = _flatten(answer["em"])
    stopping = data.get("stopping") or {}
    if not stopping or not stopping.get("n"):
        stopping = _stopping_from_logs(_read_stop_logs(path.with_suffix("").with_suffix(".stop_log.jsonl")))
    if not stopping or not stopping.get("n"):
        from collections import Counter

        steps = []
        tr = path.with_suffix("").with_suffix(".jsonl")
        if tr.exists():
            with tr.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("history_indices"):
                        steps.append(len(row["history_indices"]))
            dist = Counter(steps)
            stopping = {
                "n": len(steps),
                "avg_steps": _mean(steps),
                "stop_distribution": {str(k): int(dist[k]) for k in sorted(dist)},
            }
    avg_steps = float(stopping.get("avg_steps", float("nan")))
    if math.isnan(avg_steps):
        return None
    return {
        "dataset": parsed["dataset"],
        "avg_f1": _mean(f1_values),
        "avg_em": _mean(em_values),
        "avg_steps": avg_steps,
        "threshold": float(parsed["threshold"]),
    }


def _load_pandora_f1(results_dir: Path, pandora_dataset: str) -> Optional[Tuple[float, float, float]]:
    path = results_dir / f"stage3_evalue_{pandora_dataset}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    s = data["per_gamma"]["0.5"]["per_alpha"]["0.1"]["summaries"]["probe_evalue"]
    return float(s["avg_f1"]), float(s["avg_em"]), float(s["avg_steps"])


def _pareto_frontier_rows(rows: List[Dict[str, Any]], y_key: str) -> List[Dict[str, Any]]:
    clean = [r for r in rows if not math.isnan(float(r["avg_steps"]))]
    frontier: List[Dict[str, Any]] = []
    for i, cur in enumerate(clean):
        dominated = False
        for j, other in enumerate(clean):
            if i == j:
                continue
            if (
                float(other["avg_steps"]) <= float(cur["avg_steps"])
                and float(other[y_key]) >= float(cur[y_key])
                and (
                    float(other["avg_steps"]) < float(cur["avg_steps"])
                    or float(other[y_key]) > float(cur[y_key])
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(cur)
    return sorted(frontier, key=lambda row: (float(row["avg_steps"]), -float(row[y_key])))


def _plot_stop_rag_1x3(
    input_root: Path,
    results_dir: Path,
    out_path: Path,
    dpi: int,
) -> None:
    _set_pub_rc()
    stop_root = input_root
    if not stop_root.is_dir():
        return

    metrics_paths = sorted(stop_root.glob("*/online_test/*.metrics.json"))
    all_rows: List[Dict[str, Any]] = []
    for path in metrics_paths:
        p = _load_online_point(path, input_root)
        if p is not None:
            all_rows.append(p)

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_dataset[str(row["dataset"])].append(row)

    DATASET_TO_PANDORA = {
        "hotpotqa": "hotpotqa",
        "musique": "musique",
        "2wikimultihopqa": "2wiki",
    }
    order = ("hotpotqa", "musique", "2wikimultihopqa")
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), sharey=False)
    plt.subplots_adjust(left=0.07, right=0.995, bottom=0.30, wspace=0.28, top=0.86)

    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, order):
        rows = by_dataset.get(ds, [])
        if not rows:
            ax.set_title(DATASET_TO_PANDORA.get(ds, ds), fontsize=9.5)
            continue
        frontier = _pareto_frontier_rows(list(rows), "avg_f1")
        xs = np.array([float(r["avg_steps"]) for r in rows], dtype=np.float64)
        ys = np.array([float(r["avg_f1"]) for r in rows], dtype=np.float64)
        jitter = (rng.random(len(xs)) - 0.5) * 0.04
        ax.scatter(
            xs + jitter,
            ys,
            s=20,
            alpha=0.45,
            color="#b0c4de",
            edgecolors="#7a8faf",
            linewidths=0.2,
            label=None,
        )
        if frontier:
            fx = [float(r["avg_steps"]) for r in frontier]
            fy = [float(r["avg_f1"]) for r in frontier]
            ax.plot(fx, fy, color="#0d3b66", linewidth=1.0, zorder=3, label="Stop-RAG threshold sweep (frontier)")

        pandora = _load_pandora_f1(results_dir, DATASET_TO_PANDORA[ds])
        if pandora is not None:
            f1, _em, st = pandora
            ax.scatter(
                [st],
                [f1],
                color="#1b9e3e",
                marker="D",
                s=45,
                edgecolors="#0d5a22",
                linewidths=0.35,
                zorder=5,
                label="Pandora Probe+E-value" if ds == "hotpotqa" else None,
            )

        short = DATASET_TO_PANDORA[ds]
        p_row = [r for r in rows if float(r["avg_f1"])]
        if p_row:
            y_vals = [float(r["avg_f1"]) for r in p_row]
            y_lo = min(y_vals) - 0.02
            y_hi = max(y_vals) + 0.02
            if pandora is not None:
                y_hi = max(y_hi, pandora[0] + 0.02)
                y_lo = min(y_lo, pandora[0] - 0.02)
            ax.set_ylim(y_lo, y_hi)
        ax.set_title(DATASET_LABELS.get(short, short), fontsize=9.5, pad=3)
        if ax is axes[0]:
            ax.set_ylabel("Answer F1", fontsize=8.5)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7.5)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=7.0,
        frameon=True,
    )
    fig.supxlabel("Average retrieval steps", fontsize=8.5, y=0.13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=dpi)
    fig.savefig(out_path.with_suffix(".png"), dpi=max(dpi, 300), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--results-dir", type=Path, default=None, help="Default: <root>/paper/results or <root>/results")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <root>/paper/figs")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _resolve_results_dir(root: Path, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    for cand in (root / "paper" / "results", root / "results"):
        if (cand / "stage3_evalue_hotpotqa.json").exists():
            return cand
    return root / "results"


def main() -> None:
    args = parse_args()
    root = args.root
    res = _resolve_results_dir(root, args.results_dir)
    out_dir = args.out_dir or (root / "paper" / "figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    print(f"Using results from {res}")

    all_points = {d: _load_dataset_points(res, d) for d in DATASETS}
    _plot_pareto_combined(all_points, out_dir / "stage3_final_pareto_all.pdf", dpi)

    p_ns = res / "stage3_evalue_hotpotqa.json"
    p_su = res / "stage3_evalue_hotpotqa_sudden.json"
    if p_ns.is_file() and p_su.is_file():
        a0 = json.loads(p_ns.read_text(encoding="utf-8"))["per_gamma"]["0.5"]["per_alpha"]["0.1"]
        if "cumulative_error_curves" in a0:
            _plot_evalue_triple(p_ns, p_su, out_dir / "evalue_hotpotqa_triple.pdf", dpi)
        else:
            print("Skip E-value triple: run stage3 to refresh JSON (missing cumulative_error_curves).")
    else:
        print("Skip E-value triple: missing hotpot JSONs.")

    stop_in = root / "baselines" / "stop-rag-pandora" / "results"
    if not stop_in.is_dir():
        stop_in = root / "papar" / "code" / "baselines" / "stop-rag-pandora" / "results"
    if stop_in.is_dir():
        _plot_stop_rag_1x3(stop_in, res, out_dir / "stop_rag_pareto_1x3.pdf", dpi)
    else:
        print(f"Skip Stop-RAG 1x3: no sweep dir at {stop_in}")

    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
