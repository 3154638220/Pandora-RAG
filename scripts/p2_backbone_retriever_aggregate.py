#!/usr/bin/env python3
"""
Aggregate Stage-1 oracle / fixed-K tables from isolated --root-dir runs (e.g. bm25 vs contriever_bge).

Reads results/stage1_oracle_table_{dataset}.csv produced by stage1.run_stage1 and writes
results/p2_backbone_retriever_sanity.csv for NeurIPS P2 (small-scale backbone / retriever sanity).

Example:
  python scripts/p2_backbone_retriever_aggregate.py \\
    --bm25-root runs/p2_sanity/bm25 \\
    --dense-root runs/p2_sanity/contriever_bge \\
    --datasets hotpotqa,2wiki
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_table(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing oracle table: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _parse_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _rel_under(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _summarize_backend(rows: List[Dict[str, Any]]) -> Tuple[float, float, float, float, int]:
    """Returns oracle_f1, oracle_steps, best_fixed_f1, best_fixed_k, oracle_gain."""
    oracle_f1 = 0.0
    oracle_steps = 0.0
    best_fixed_f1 = 0.0
    best_fixed_k = 0
    for r in rows:
        strat = str(r.get("strategy", ""))
        if strat == "Oracle":
            oracle_f1 = _parse_float(str(r.get("avg_f1", "0")))
            oracle_steps = _parse_float(str(r.get("avg_steps", "0")))
        if strat.startswith("Fixed-K="):
            f1 = _parse_float(str(r.get("avg_f1", "0")))
            if f1 > best_fixed_f1:
                best_fixed_f1 = f1
                try:
                    best_fixed_k = int(strat.split("=", 1)[1])
                except (IndexError, ValueError):
                    best_fixed_k = 0
    oracle_gain = oracle_f1 - best_fixed_f1
    return oracle_f1, oracle_steps, best_fixed_f1, float(best_fixed_k), oracle_gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Stage1 oracle tables for P2 retriever sanity.")
    parser.add_argument("--bm25-root", type=str, required=True, help="Stage1 --root-dir for bm25 run")
    parser.add_argument("--dense-root", type=str, required=True, help="Stage1 --root-dir for contriever_bge run")
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,2wiki",
        help="Comma-separated: hotpotqa,musique,2wiki",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/p2_backbone_retriever_sanity.csv",
        help="Output CSV path (under repo root or cwd)",
    )
    args = parser.parse_args()

    bm25 = Path(args.bm25_root).resolve()
    dense = Path(args.dense_root).resolve()
    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "retriever_backend",
        "oracle_f1",
        "oracle_avg_steps",
        "best_fixed_f1",
        "best_fixed_k",
        "oracle_gain_over_best_fixed",
        "table_csv",
    ]
    out_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        p_bm = bm25 / "results" / f"stage1_oracle_table_{ds}.csv"
        p_dn = dense / "results" / f"stage1_oracle_table_{ds}.csv"
        for backend, root, p in (
            ("bm25", bm25, p_bm),
            ("contriever_bge", dense, p_dn),
        ):
            trows = _read_table(p)
            o_f1, o_st, bf_f1, bf_k, gain = _summarize_backend(trows)
            out_rows.append(
                {
                    "dataset": ds,
                    "retriever_backend": backend,
                    "oracle_f1": f"{o_f1:.6f}",
                    "oracle_avg_steps": f"{o_st:.6f}",
                    "best_fixed_f1": f"{bf_f1:.6f}",
                    "best_fixed_k": f"{int(bf_k)}",
                    "oracle_gain_over_best_fixed": f"{gain:.6f}",
                    "table_csv": _rel_under(root, p),
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # One-line interpretation helpers (same split / seed; absolute F1 not comparable across retrievers)
    print(f"Wrote {out_path}")
    for ds in datasets:
        row_b = next(r for r in out_rows if r["dataset"] == ds and r["retriever_backend"] == "bm25")
        row_d = next(r for r in out_rows if r["dataset"] == ds and r["retriever_backend"] == "contriever_bge")
        g_b = float(row_b["oracle_gain_over_best_fixed"])
        g_d = float(row_d["oracle_gain_over_best_fixed"])
        print(
            f"  {ds}: oracle_gain bm25={g_b:.4f} contriever_bge={g_d:.4f} "
            f"(sign of adaptive headroom; not a cross-retriever F1 claim)"
        )


if __name__ == "__main__":
    main()
