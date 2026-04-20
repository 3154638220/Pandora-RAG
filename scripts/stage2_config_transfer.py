"""Summarize Stage2 shared-default and config-transfer robustness.

This script answers the NeurIPS P1 question:

    Is the probe robust, or does it require dataset-specific tuning?

It consumes existing Stage2 artifacts only.  For each candidate Stage2
configuration suffix, it reads:

  - artifacts/probe/<dataset>/stage2_train_meta{suffix}.json for dev metrics
  - results/stage2_probe_table_<dataset>{suffix}.csv for test metrics

It then reports:

  1. A shared default selected by macro dev F1 across datasets.
  2. Transfer rows where the config selected on one source dataset is applied
     to the other target datasets.
  3. Deltas against the per-dataset optimal suffix, by default pdopt_best.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

DATASETS: Tuple[str, ...] = ("hotpotqa", "musique", "2wiki")
DEFAULT_EXCLUDED_PREFIXES: Tuple[str, ...] = ("pdopt", "probe_lite", "p2_", "smoke")


def _tag(suffix: str) -> str:
    suffix = suffix.strip()
    return f"_{suffix}" if suffix else ""


def _meta_path(root: Path, dataset: str, suffix: str) -> Path:
    return root / "artifacts" / "probe" / dataset / f"stage2_train_meta{_tag(suffix)}.json"


def _table_path(root: Path, dataset: str, suffix: str) -> Path:
    return root / "results" / f"stage2_probe_table_{dataset}{_tag(suffix)}.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_probe_row(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("strategy") == "Probe":
                return row
    raise ValueError(f"No Probe row found in {path}")


def _float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    if val is None or val == "":
        return default
    return float(val)


def _complete_for_all(root: Path, suffix: str, datasets: Sequence[str]) -> bool:
    return all(_meta_path(root, ds, suffix).is_file() and _table_path(root, ds, suffix).is_file() for ds in datasets)


def _discover_suffixes(root: Path, datasets: Sequence[str]) -> List[str]:
    suffixes: set[str] = set()
    first = datasets[0]
    prefix = f"stage2_probe_table_{first}"
    for path in (root / "results").glob(f"{prefix}*.csv"):
        stem = path.stem
        rest = stem[len(prefix):]
        suffix = rest[1:] if rest.startswith("_") else ""
        suffixes.add(suffix)
    return sorted(s for s in suffixes if _complete_for_all(root, s, datasets))


def _default_candidate_filter(suffixes: Iterable[str]) -> List[str]:
    out: List[str] = []
    for suffix in suffixes:
        if any(suffix.startswith(p) for p in DEFAULT_EXCLUDED_PREFIXES):
            continue
        out.append(suffix)
    return out


def _config_signature(root: Path, dataset: str, suffix: str) -> Tuple[Any, ...]:
    """Fields that must stay fixed for a suffix to count as a shared config."""
    meta = _read_json(_meta_path(root, dataset, suffix))
    train_info = meta.get("train_info") or {}
    train_stats = meta.get("train_stats") or {}
    return (
        str(meta.get("probe_feature_mode", "full")),
        str(meta.get("hidden_state_key", "last_token")),
        str(meta.get("probe_target", train_info.get("probe_target", ""))),
        round(_float(train_info, "train_margin_min_abs", _float(train_stats, "margin_min_abs_filter")), 6),
        bool(meta.get("hidden_branch_residual", False)),
        bool(meta.get("seq_history_features", False)),
        bool(meta.get("sequence_gru", False)),
    )


def _shared_signature_suffixes(root: Path, suffixes: Iterable[str], datasets: Sequence[str]) -> List[str]:
    out: List[str] = []
    for suffix in suffixes:
        sigs = {_config_signature(root, ds, suffix) for ds in datasets}
        if len(sigs) == 1:
            out.append(suffix)
        else:
            LOGGER.info(
                "Skip non-shared suffix %s with per-dataset signatures: %s",
                _display_suffix(suffix),
                sorted(sigs),
            )
    return out


def _metric_bundle(root: Path, dataset: str, suffix: str) -> Dict[str, Any]:
    meta = _read_json(_meta_path(root, dataset, suffix))
    probe = _read_probe_row(_table_path(root, dataset, suffix))
    best_dev = meta.get("best_dev_row") or {}
    train_info = meta.get("train_info") or {}
    threshold_policy = meta.get("best_dev_threshold_policy_type", "global")
    per_step_thresholds = meta.get("best_dev_per_step_thresholds")
    return {
        "dataset": dataset,
        "suffix": suffix,
        "dev_f1": _float(best_dev, "avg_f1"),
        "dev_em": _float(best_dev, "avg_em"),
        "dev_steps": _float(best_dev, "avg_steps"),
        "test_f1": _float(probe, "avg_f1"),
        "test_em": _float(probe, "avg_em"),
        "test_steps": _float(probe, "avg_steps"),
        "test_cost": _float(probe, "avg_cost"),
        "probe_target": str(meta.get("probe_target", train_info.get("probe_target", ""))),
        "probe_feature_mode": str(meta.get("probe_feature_mode", "full")),
        "train_margin_min_abs": _float(train_info, "train_margin_min_abs", _float(meta.get("train_stats", {}), "margin_min_abs_filter")),
        "hidden_branch_residual": bool(meta.get("hidden_branch_residual", False)),
        "threshold_policy_type": str(threshold_policy),
        "per_step_thresholds": per_step_thresholds,
    }


def _macro_score(rows: Sequence[Dict[str, Any]], metric: str) -> float:
    vals = [_float(r, metric) for r in rows]
    return mean(vals) if vals else 0.0


def _pick_shared_default(
    root: Path,
    suffixes: Sequence[str],
    datasets: Sequence[str],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: List[Dict[str, Any]] = []
    by_suffix_rows: Dict[str, List[Dict[str, Any]]] = {}
    for suffix in suffixes:
        rows = [_metric_bundle(root, ds, suffix) for ds in datasets]
        by_suffix_rows[suffix] = rows
        summaries.append(
            {
                "suffix": suffix,
                "macro_dev_f1": _macro_score(rows, "dev_f1"),
                "macro_dev_steps": _macro_score(rows, "dev_steps"),
                "macro_test_f1": _macro_score(rows, "test_f1"),
                "macro_test_steps": _macro_score(rows, "test_steps"),
            }
        )
    if not summaries:
        raise ValueError("No complete candidate suffixes found for shared-default selection.")
    summaries.sort(key=lambda r: (float(r["macro_dev_f1"]), -float(r["macro_dev_steps"])), reverse=True)
    chosen = str(summaries[0]["suffix"])
    return chosen, by_suffix_rows[chosen], summaries


def _pick_source_configs(
    root: Path,
    suffixes: Sequence[str],
    datasets: Sequence[str],
) -> Dict[str, str]:
    chosen: Dict[str, str] = {}
    for source in datasets:
        rows = [_metric_bundle(root, source, suffix) for suffix in suffixes]
        rows.sort(key=lambda r: (_float(r, "dev_f1"), -_float(r, "dev_steps")), reverse=True)
        chosen[source] = str(rows[0]["suffix"])
    return chosen


def _baseline_rows(root: Path, datasets: Sequence[str], baseline_suffix: str) -> Dict[str, Dict[str, Any]]:
    if not _complete_for_all(root, baseline_suffix, datasets):
        raise FileNotFoundError(
            f"Baseline suffix {baseline_suffix!r} is incomplete; expected Stage2 meta/table for {datasets}."
        )
    return {ds: _metric_bundle(root, ds, baseline_suffix) for ds in datasets}


def _with_delta(row: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["baseline_suffix"] = baseline.get("suffix", "")
    out["baseline_test_f1"] = _float(baseline, "test_f1")
    out["baseline_test_steps"] = _float(baseline, "test_steps")
    out["delta_test_f1_vs_baseline"] = _float(row, "test_f1") - _float(baseline, "test_f1")
    out["delta_test_steps_vs_baseline"] = _float(row, "test_steps") - _float(baseline, "test_steps")
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def _display_suffix(suffix: str) -> str:
    return suffix if suffix else "(base default)"


def _write_report(
    path: Path,
    candidate_summary: Sequence[Dict[str, Any]],
    shared_suffix: str,
    shared_rows: Sequence[Dict[str, Any]],
    transfer_rows: Sequence[Dict[str, Any]],
    baseline_suffix: str,
    candidate_csv: Path,
    shared_csv: Path,
    transfer_csv: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Stage2 Shared Default / Config Transfer")
    lines.append("")
    lines.append(
        "This report is generated from existing Stage2 artifacts. Candidate configurations are selected on dev metrics; test metrics are only used for reporting."
    )
    lines.append("")
    lines.append(f"- Per-dataset optimal baseline: `{baseline_suffix}`")
    lines.append(f"- Shared default selected by macro dev F1: `{_display_suffix(shared_suffix)}`")
    lines.append(
        "- Strict shared-config filter: candidate suffixes whose config signature differs across datasets are excluded."
    )
    lines.append(
        f"- Default excluded prefixes: `{', '.join(DEFAULT_EXCLUDED_PREFIXES)}`"
    )
    lines.append(f"- Candidate summary CSV: `{candidate_csv}`")
    lines.append(f"- Shared-default CSV: `{shared_csv}`")
    lines.append(f"- Transfer CSV: `{transfer_csv}`")
    lines.append("")

    lines.append("## Shared Default")
    lines.append("")
    lines.append("| Dataset | Shared F1 / steps | Per-dataset optimal F1 / steps | ΔF1 | Δsteps |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in shared_rows:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['test_f1'])} / {_fmt(row['test_steps'], 3)} | "
            f"{_fmt(row['baseline_test_f1'])} / {_fmt(row['baseline_test_steps'], 3)} | "
            f"{_fmt(row['delta_test_f1_vs_baseline'])} | {_fmt(row['delta_test_steps_vs_baseline'], 3)} |"
        )
    lines.append("")
    shared_macro_f1 = _macro_score(shared_rows, "test_f1")
    shared_macro_steps = _macro_score(shared_rows, "test_steps")
    opt_macro_f1 = mean(float(r["baseline_test_f1"]) for r in shared_rows)
    opt_macro_steps = mean(float(r["baseline_test_steps"]) for r in shared_rows)
    lines.append(
        f"Macro: shared `{_fmt(shared_macro_f1)}` F1 / `{_fmt(shared_macro_steps, 3)}` steps; "
        f"per-dataset optimal `{_fmt(opt_macro_f1)}` F1 / `{_fmt(opt_macro_steps, 3)}` steps."
    )
    lines.append("")

    lines.append("## Source-Selected Transfer")
    lines.append("")
    lines.append("| Source | Selected suffix | Target | Transfer F1 / steps | Per-dataset optimal F1 / steps | ΔF1 | Δsteps |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in transfer_rows:
        lines.append(
            f"| {row['source_dataset']} | `{_display_suffix(str(row['source_selected_suffix']))}` | {row['dataset']} | "
            f"{_fmt(row['test_f1'])} / {_fmt(row['test_steps'], 3)} | "
            f"{_fmt(row['baseline_test_f1'])} / {_fmt(row['baseline_test_steps'], 3)} | "
            f"{_fmt(row['delta_test_f1_vs_baseline'])} | {_fmt(row['delta_test_steps_vs_baseline'], 3)} |"
        )
    lines.append("")

    lines.append("## Candidate Dev Ranking")
    lines.append("")
    lines.append("| Rank | Suffix | Macro dev F1 | Macro dev steps | Macro test F1 | Macro test steps |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for idx, row in enumerate(candidate_summary[:20], start=1):
        lines.append(
            f"| {idx} | `{_display_suffix(str(row['suffix']))}` | {_fmt(row['macro_dev_f1'])} | "
            f"{_fmt(row['macro_dev_steps'], 3)} | {_fmt(row['macro_test_f1'])} | {_fmt(row['macro_test_steps'], 3)} |"
        )
    lines.append("")
    lines.append("Interpretation: this is a robustness audit, not a replacement for the main per-dataset operating point.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 shared-default / config-transfer summary")
    p.add_argument("--root-dir", type=Path, default=Path("."))
    p.add_argument("--datasets", type=str, default=",".join(DATASETS))
    p.add_argument(
        "--candidate-suffixes",
        type=str,
        default="",
        help="Comma-separated suffixes to consider. Empty means discover complete non-pdopt Stage2 configs.",
    )
    p.add_argument("--baseline-suffix", type=str, default="pdopt_best")
    p.add_argument("--out-prefix", type=str, default="stage2_config_transfer")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    root = args.root_dir
    datasets = tuple(d.strip().lower() for d in str(args.datasets).split(",") if d.strip())
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}; expected subset of {DATASETS}")

    if str(args.candidate_suffixes).strip():
        suffixes = [s.strip() for s in str(args.candidate_suffixes).split(",")]
        suffixes = [s for s in suffixes if _complete_for_all(root, s, datasets)]
    else:
        suffixes = _default_candidate_filter(_discover_suffixes(root, datasets))
    suffixes = _shared_signature_suffixes(root, suffixes, datasets)
    if not suffixes:
        raise ValueError("No complete candidate suffixes found.")

    baseline = _baseline_rows(root, datasets, str(args.baseline_suffix))
    shared_suffix, shared_raw_rows, candidate_summary = _pick_shared_default(root, suffixes, datasets)
    source_configs = _pick_source_configs(root, suffixes, datasets)

    shared_rows = [_with_delta(row, baseline[str(row["dataset"])]) for row in shared_raw_rows]

    transfer_rows: List[Dict[str, Any]] = []
    for source, selected_suffix in source_configs.items():
        for target in datasets:
            if target == source:
                continue
            row = _metric_bundle(root, target, selected_suffix)
            row = _with_delta(row, baseline[target])
            row["source_dataset"] = source
            row["source_selected_suffix"] = selected_suffix
            transfer_rows.append(row)

    results_dir = root / "results"
    docs_dir = root / "docs"
    candidate_csv = results_dir / f"{args.out_prefix}_candidate_summary.csv"
    shared_csv = results_dir / f"{args.out_prefix}_shared_default.csv"
    transfer_csv = results_dir / f"{args.out_prefix}_source_transfer.csv"
    report_md = docs_dir / f"{args.out_prefix}.md"

    _write_csv(candidate_csv, candidate_summary)
    _write_csv(shared_csv, shared_rows)
    _write_csv(transfer_csv, transfer_rows)
    _write_report(
        report_md,
        candidate_summary,
        shared_suffix,
        shared_rows,
        transfer_rows,
        str(args.baseline_suffix),
        candidate_csv,
        shared_csv,
        transfer_csv,
    )

    LOGGER.info("Shared default suffix: %s", _display_suffix(shared_suffix))
    LOGGER.info("Wrote %s", candidate_csv)
    LOGGER.info("Wrote %s", shared_csv)
    LOGGER.info("Wrote %s", transfer_csv)
    LOGGER.info("Wrote %s", report_md)


if __name__ == "__main__":
    main()
