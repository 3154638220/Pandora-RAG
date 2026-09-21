"""Summarize true-online Stop-RAG threshold sweeps and plot Pareto frontiers.

The script intentionally reads only online-test artifacts produced by
``baselines/stop-rag-pandora/scripts/stop_rag_test.sh``:

* ``*.metrics.json`` for answer metrics
* the embedded ``stopping`` block added by this repo, or the adjacent
  ``*.stop_log.jsonl`` sidecar, for average retrieval steps

It does not consume Stop-RAG offline replay files from ``compute_scores``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STOP_RAG_ROOT = ROOT / "paper" / "baselines" / "stop-rag-pandora"
STOP_RAG_RESULTS = STOP_RAG_ROOT / "results"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs" / "reports" / "baselines"

DATASET_TO_PANDORA = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihopqa": "2wiki",
}
DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "2wikimultihopqa": "2Wiki",
}
PANDORA_METHOD_KEY = "probe_evalue"
PANDORA_METHOD_LABEL = "Pandora Probe+E-value"


def _flatten(values: Sequence[Sequence[float]]) -> List[float]:
    return [float(item) for sub in values for item in sub]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _parse_metrics_name(path: Path) -> Optional[Dict[str, Any]]:
    match = re.match(r"(?P<dataset>.+)_test_ckpt(?P<ckpt>[^_]+)_thr(?P<threshold>.+)\.metrics\.json$", path.name)
    if not match:
        return None
    out = match.groupdict()
    try:
        out["threshold"] = float(out["threshold"])
    except ValueError:
        return None
    return out


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
    steps = [int(log["stop_iter"]) for log in logs if "stop_iter" in log]
    dist = Counter(steps)
    return {
        "n": len(steps),
        "avg_steps": _mean(steps),
        "stop_distribution": {str(k): int(dist[k]) for k in sorted(dist)},
    }


def _stopping_from_trace(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"n": 0, "avg_steps": float("nan"), "stop_distribution": {}}
    steps = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            history_indices = row.get("history_indices")
            if history_indices:
                steps.append(len(history_indices))
            elif isinstance(row.get("history"), list):
                steps.append(len(row["history"]))
    dist = Counter(steps)
    return {
        "n": len(steps),
        "avg_steps": _mean(steps),
        "stop_distribution": {str(k): int(dist[k]) for k in sorted(dist)},
    }


def _load_online_point(path: Path) -> Optional[Dict[str, Any]]:
    parsed = _parse_metrics_name(path)
    if parsed is None:
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    answer = data["metrics"]["answer"]
    f1_values = _flatten(answer["f1"])
    em_values = _flatten(answer["em"])
    acc_values = _flatten(answer.get("acc", []))

    stopping = data.get("stopping") or {}
    if not stopping or not stopping.get("n"):
        stopping = _stopping_from_logs(_read_stop_logs(path.with_suffix("").with_suffix(".stop_log.jsonl")))
    if not stopping or not stopping.get("n"):
        stopping = _stopping_from_trace(path.with_suffix("").with_suffix(".jsonl"))

    avg_steps = float(stopping.get("avg_steps", float("nan")))
    row = {
        "dataset": parsed["dataset"],
        "method": "Stop-RAG",
        "checkpoint": parsed["ckpt"],
        "threshold": float(parsed["threshold"]),
        "n": len(f1_values),
        "avg_f1": _mean(f1_values),
        "avg_em": _mean(em_values),
        "avg_acc": _mean(acc_values),
        "avg_steps": avg_steps,
        "stop_distribution": json.dumps(stopping.get("stop_distribution", {}), sort_keys=True),
        "metrics_path": str(path.relative_to(ROOT)),
    }
    if math.isnan(avg_steps):
        row["missing_steps"] = True
    else:
        row["missing_steps"] = False
    return row


def _pareto_frontier(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _pareto_frontier_metric(rows, "avg_f1")


def _load_pandora_point(stop_dataset: str) -> Optional[Dict[str, Any]]:
    pandora_dataset = DATASET_TO_PANDORA.get(stop_dataset)
    if pandora_dataset is None:
        return None
    path = RESULTS_DIR / f"stage3_evalue_{pandora_dataset}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    summary = data["per_gamma"]["0.5"]["per_alpha"]["0.1"]["summaries"][PANDORA_METHOD_KEY]
    return {
        "dataset": stop_dataset,
        "method": PANDORA_METHOD_LABEL,
        "checkpoint": "",
        "threshold": "",
        "n": int(summary["n"]),
        "avg_f1": float(summary["avg_f1"]),
        "avg_em": float(summary["avg_em"]),
        "avg_acc": "",
        "avg_steps": float(summary["avg_steps"]),
        "stop_distribution": "",
        "metrics_path": str(path.relative_to(ROOT)),
        "missing_steps": False,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _matched_budget_rows(stop_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in stop_rows:
        if not row.get("missing_steps"):
            by_dataset[str(row["dataset"])].append(row)

    out: List[Dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        pandora = _load_pandora_point(dataset)
        if pandora is None:
            continue
        budget = float(pandora["avg_steps"])
        target_f1 = float(pandora["avg_f1"])
        under_budget = [r for r in rows if float(r["avg_steps"]) <= budget]
        enough_f1 = [r for r in rows if float(r["avg_f1"]) >= target_f1]
        best_under = max(under_budget, key=lambda r: float(r["avg_f1"])) if under_budget else None
        cheapest_enough = min(enough_f1, key=lambda r: float(r["avg_steps"])) if enough_f1 else None
        out.append(
            {
                "dataset": dataset,
                "pandora_f1": target_f1,
                "pandora_em": float(pandora["avg_em"]),
                "pandora_steps": budget,
                "stop_rag_best_f1_at_pandora_budget": "" if best_under is None else float(best_under["avg_f1"]),
                "stop_rag_best_em_at_pandora_budget": "" if best_under is None else float(best_under["avg_em"]),
                "stop_rag_steps_at_budget_match": "" if best_under is None else float(best_under["avg_steps"]),
                "stop_rag_threshold_at_budget_match": "" if best_under is None else float(best_under["threshold"]),
                "stop_rag_steps_to_reach_pandora_f1": "" if cheapest_enough is None else float(cheapest_enough["avg_steps"]),
                "stop_rag_threshold_to_reach_pandora_f1": "" if cheapest_enough is None else float(cheapest_enough["threshold"]),
            }
        )
    return out


def _plot_dataset(
    dataset: str,
    rows: Sequence[Dict[str, Any]],
    frontier: Sequence[Dict[str, Any]],
    out_path: Path,
    y_key: str,
    y_label: str,
    pandora_y_key: str,
) -> None:
    rows = [r for r in rows if not r.get("missing_steps")]
    if not rows:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    xs = [float(r["avg_steps"]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    ax.scatter(xs, ys, color="#4c78a8", s=70, alpha=0.8, label="Stop-RAG sweep")
    if frontier:
        fx = [float(r["avg_steps"]) for r in frontier]
        fy = [float(r[y_key]) for r in frontier]
        ax.plot(fx, fy, color="#1f4e79", linewidth=1.8, marker="o", label="Stop-RAG Pareto")
    pandora = _load_pandora_point(dataset)
    if pandora is not None and pandora_y_key in pandora and pandora[pandora_y_key] != "":
        ax.scatter(
            [float(pandora["avg_steps"])],
            [float(pandora[pandora_y_key])],
            color="#2ca02c",
            marker="D",
            s=95,
            label=PANDORA_METHOD_LABEL,
        )
    ax.set_title(f"Stop-RAG Online Threshold Sweep - {DATASET_LABELS.get(dataset, dataset)}")
    ax.set_xlabel("Avg retrieval steps")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _pareto_frontier_metric(rows: Iterable[Dict[str, Any]], y_key: str) -> List[Dict[str, Any]]:
    """Non-dominated frontier in (avg_steps, y) with lower steps and higher y better."""
    clean = [r for r in rows if not r.get("missing_steps") and not math.isnan(float(r["avg_steps"]))]
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


def _write_report(
    path: Path,
    all_rows: Sequence[Dict[str, Any]],
    frontier_rows: Sequence[Dict[str, Any]],
    frontier_em_rows: Sequence[Dict[str, Any]],
    matched_rows: Sequence[Dict[str, Any]],
    plot_paths: Sequence[Path],
) -> None:
    lines = [
        "# Stop-RAG Threshold Sweep / Pareto Frontier",
        "",
        "This report is generated from true-online Stop-RAG test artifacts under `baselines/stop-rag-pandora/results/*/online_test/`.",
        "Offline `compute_scores` replay files are intentionally excluded.",
        "",
        "## Coverage",
        "",
    ]
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_dataset[str(row["dataset"])].append(row)
    for dataset, rows in sorted(by_dataset.items()):
        usable = [r for r in rows if not r.get("missing_steps")]
        lines.append(f"- {DATASET_LABELS.get(dataset, dataset)}: {len(usable)}/{len(rows)} online points have avg-step metadata.")
    lines.extend(["", "## Matched Budget", ""])
    if matched_rows:
        lines.append("| Dataset | Pandora F1 | Pandora steps | Stop-RAG F1 @ Pandora budget | Stop-RAG steps to reach Pandora F1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in matched_rows:
            budget_f1 = row["stop_rag_best_f1_at_pandora_budget"]
            reach_steps = row["stop_rag_steps_to_reach_pandora_f1"]
            lines.append(
                f"| {DATASET_LABELS.get(str(row['dataset']), row['dataset'])} | "
                f"{float(row['pandora_f1']):.4f} | {float(row['pandora_steps']):.3f} | "
                f"{budget_f1 if budget_f1 == '' else f'{float(budget_f1):.4f}'} | "
                f"{reach_steps if reach_steps == '' else f'{float(reach_steps):.3f}'} |"
            )
    else:
        lines.append("No matched-budget rows were produced because no Stop-RAG sweep point had step metadata.")
    lines.extend(["", "## Pareto Points", ""])
    if frontier_rows:
        lines.append("| Dataset | Checkpoint | Threshold | F1 | EM | Avg steps | N |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in frontier_rows:
            lines.append(
                f"| {DATASET_LABELS.get(str(row['dataset']), row['dataset'])} | "
                f"{row['checkpoint']} | {float(row['threshold']):.2f} | "
                f"{float(row['avg_f1']):.4f} | {float(row['avg_em']):.4f} | "
                f"{float(row['avg_steps']):.3f} | {int(row['n'])} |"
            )
    else:
        lines.append("No Pareto frontier was produced because avg-step metadata is missing.")
    lines.extend(["", "## Pareto Points (EM objective)", ""])
    if frontier_em_rows:
        lines.append("| Dataset | Checkpoint | Threshold | F1 | EM | Avg steps | N |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in frontier_em_rows:
            lines.append(
                f"| {DATASET_LABELS.get(str(row['dataset']), row['dataset'])} | "
                f"{row['checkpoint']} | {float(row['threshold']):.2f} | "
                f"{float(row['avg_f1']):.4f} | {float(row['avg_em']):.4f} | "
                f"{float(row['avg_steps']):.3f} | {int(row['n'])} |"
            )
    else:
        lines.append("No EM Pareto frontier was produced because avg-step metadata is missing.")
    if plot_paths:
        lines.extend(["", "## Figures", ""])
        for plot_path in plot_paths:
            lines.append(f"- `{plot_path.relative_to(ROOT)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=STOP_RAG_RESULTS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_paths = sorted(args.input_root.glob("*/online_test/*.metrics.json"))
    all_rows = [row for path in metrics_paths if (row := _load_online_point(path)) is not None]
    fieldnames = [
        "dataset",
        "method",
        "checkpoint",
        "threshold",
        "n",
        "avg_f1",
        "avg_em",
        "avg_acc",
        "avg_steps",
        "stop_distribution",
        "missing_steps",
        "metrics_path",
    ]
    _write_csv(args.results_dir / "stop_rag_online_threshold_sweep.csv", all_rows, fieldnames)

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_dataset[str(row["dataset"])].append(row)

    frontier_rows: List[Dict[str, Any]] = []
    frontier_em_rows: List[Dict[str, Any]] = []
    plot_paths: List[Path] = []
    for dataset, rows in sorted(by_dataset.items()):
        frontier = _pareto_frontier(rows)
        frontier_rows.extend(frontier)
        frontier_em = _pareto_frontier_metric(rows, "avg_em")
        frontier_em_rows.extend(frontier_em)
        short = DATASET_TO_PANDORA.get(dataset, dataset)
        plot_f1 = args.results_dir / f"stop_rag_pareto_{short}.png"
        plot_em = args.results_dir / f"stop_rag_pareto_em_{short}.png"
        _plot_dataset(dataset, rows, frontier, plot_f1, "avg_f1", "Avg F1", "avg_f1")
        _plot_dataset(dataset, rows, frontier_em, plot_em, "avg_em", "Avg EM", "avg_em")
        if plot_f1.exists():
            plot_paths.append(plot_f1)
        if plot_em.exists():
            plot_paths.append(plot_em)

    _write_csv(args.results_dir / "stop_rag_online_pareto_frontier.csv", frontier_rows, fieldnames)
    _write_csv(args.results_dir / "stop_rag_online_pareto_frontier_em.csv", frontier_em_rows, fieldnames)
    matched_rows = _matched_budget_rows(all_rows)
    _write_csv(
        args.results_dir / "stop_rag_matched_budget.csv",
        matched_rows,
        [
            "dataset",
            "pandora_f1",
            "pandora_em",
            "pandora_steps",
            "stop_rag_best_f1_at_pandora_budget",
            "stop_rag_best_em_at_pandora_budget",
            "stop_rag_steps_at_budget_match",
            "stop_rag_threshold_at_budget_match",
            "stop_rag_steps_to_reach_pandora_f1",
            "stop_rag_threshold_to_reach_pandora_f1",
        ],
    )
    _write_report(args.docs_dir / "stop_rag_threshold_sweep.md", all_rows, frontier_rows, frontier_em_rows, matched_rows, plot_paths)
    print(f"Wrote {args.results_dir / 'stop_rag_online_threshold_sweep.csv'}")
    print(f"Wrote {args.results_dir / 'stop_rag_online_pareto_frontier.csv'}")
    print(f"Wrote {args.results_dir / 'stop_rag_online_pareto_frontier_em.csv'}")
    print(f"Wrote {args.results_dir / 'stop_rag_matched_budget.csv'}")
    print(f"Wrote {args.docs_dir / 'stop_rag_threshold_sweep.md'}")


if __name__ == "__main__":
    main()
