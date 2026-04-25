"""按 Oracle |margin| 分桶的 Probe vs Global-Weitzman 轨迹级 F1 表（仅用缓存轨迹，不重训、不调 LLM）。

用于区分：Probe−GW 的整体差距，是更多落在「Oracle 自身决策边界很窄（label 模糊）」的条上，
还是落在「高 |margin|（决策清晰）」的条上——前者暗示标签噪声/模糊主导，后者暗示可学信号仍不足。

分桶键（--margin-key）：
  min_path  —  路径上 min_k |margin_k|，反映整条回答路径上**最模棱两可**的一步（默认）；
  at_stop  —  最优停步 s 处的 |margin_s|；若 s==max_k 且该步无 target，则退化为 k=max_k-1。

用法:
  python scripts/oracle_margin_bucket_table.py --root-dir .  # 默认 --binning fixed
  python scripts/oracle_margin_bucket_table.py --binning equal_count --n-buckets 4
  python scripts/oracle_margin_bucket_table.py --datasets hotpotqa --margin-key at_stop
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretest.utils.weitzman import compute_trajectory_oracle
from stage3.config import Stage3Config


def _load_t2b():
    t2_path = REPO_ROOT / "scripts" / "table2_paired_bootstrap.py"
    spec = importlib.util.spec_from_file_location("t2b", t2_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 table2_paired_bootstrap")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _equal_count_bins(
    x: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """按 x 值升序分位，尽量等频分桶。返回 (bucket_id 0..B-1, 各桶 [low, high] 闭包展示用)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(x.size)
    if n == 0:
        return np.array([], dtype=np.int64), []
    n_bins = int(min(n_bins, n))
    if n_bins < 1:
        n_bins = 1
    order = np.argsort(x, kind="mergesort")
    bucket = np.empty(n, dtype=np.int64)
    base = n // n_bins
    rem = n % n_bins
    start = 0
    span_edges: List[Tuple[float, float]] = []
    for b in range(n_bins):
        w = base + (1 if b < rem else 0)
        end = start + w
        sl = order[start:end]
        bucket[sl] = b
        lo = float(x[sl[0]])
        hi = float(x[sl[-1]])
        span_edges.append((lo, hi))
        start = end
    return bucket, span_edges


def _fixed_threshold_bins(
    x: np.ndarray, edges: Sequence[float]
) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
    """
    按固定阈值分桶。edges 单调递增，如 (0, 0.01, 0.05, 0.1, 1.0) 表示
    [e0,e1), [e1,e2), …, [e_{B-2}, e_{B-1}]（最后一桶闭右端）.
    """
    e = [float(t) for t in edges]
    if len(e) < 2 or e != sorted(e):
        raise ValueError("edges 需至少 2 个且严格升序")
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(x.size)
    B = len(e) - 1
    bucket = np.full(n, -1, dtype=np.int64)
    labels: List[Tuple[float, str]] = []
    for b in range(B):
        lo, hi = e[b], e[b + 1]
        if b < B - 1:
            mask = (x >= lo) & (x < hi)
            lab = f"[{lo:.4g}, {hi:.4g})"
        else:
            mask = (x >= lo) & (x <= hi)
            lab = f"[{lo:.4g}, {hi:.4g}]"
        bucket[mask] = b
        labels.append((lo, lab))
    if n > 0 and np.any((bucket < 0) & np.isfinite(x)):
        oob = (bucket < 0) & np.isfinite(x)
        raise RuntimeError(
            f"存在 {int(np.sum(oob))} 个 x 落在 edges 外；min={float(np.min(x[np.isfinite(x)]))}, "
            f"max={float(np.max(x[np.isfinite(x)]))}"
        )
    return bucket, labels


def _margin_scalars(
    ordered_test: Sequence[Mapping[str, Any]],
    max_k: int,
    base_cost: float,
    cost_metric: str,
    key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """与 ordered_test 同序；返回 (values, n_invalid)。invalid 为 step_targets 为空的条数。"""
    orows = compute_trajectory_oracle(
        list(ordered_test), base_cost, max_k, cost_metric=cost_metric
    )
    vals: List[float] = []
    bad = 0
    for row in orows:
        st = row.get("step_targets") or {}
        if not st:
            bad += 1
            vals.append(float("nan"))
            continue
        margins: Dict[int, float] = {}
        for raw_k, t in st.items():
            kk = int(raw_k) if not isinstance(raw_k, int) else raw_k
            margins[kk] = float((t or {}).get("margin", 0.0))
        if key == "min_path":
            mabs = [abs(margins[kk]) for kk in margins if kk in margins]
            vals.append(float(min(mabs)) if mabs else float("nan"))
        elif key == "at_stop":
            s = int(row.get("steps_used", 0) or 0)
            if s in margins:
                vals.append(abs(margins[s]))
            elif s >= max_k and (max_k - 1) in margins:
                vals.append(abs(margins[max_k - 1]))
            else:
                # 退而取路径上可得的最近一步
                cand = [abs(margins[kk]) for kk in sorted(margins.keys()) if kk <= s]
                vals.append(float(min(cand)) if cand else float("nan"))
        else:
            raise ValueError(f"unknown margin-key: {key!r}")
    return np.asarray(vals, dtype=np.float64), bad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle margin bucket table: Probe vs Global-Weitzman per trajectory F1")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    p.add_argument("--artifact-suffix", type=str, default="pdopt_best")
    p.add_argument("--probe-checkpoint", type=str, default="")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument(
        "--margin-key",
        type=str,
        default="min_path",
        choices=("min_path", "at_stop"),
        help="分桶用标量：路径最小 |margin| 或 最优停步处 |margin|",
    )
    p.add_argument("--n-buckets", type=int, default=4)
    p.add_argument(
        "--binning",
        type=str,
        default="fixed",
        choices=("equal_count", "fixed"),
        help="分桶：fixed=按 |margin| 绝对阈值（适合离散 F1 导致的 margin 聚集）；equal_count=等频分位。",
    )
    p.add_argument(
        "--fixed-edges",
        type=str,
        default="0,0.01,0.05,0.1,1.0",
        help="--binning fixed 时的单调递增 edges，例如 0,0.01,0.05,0.1,1.0。",
    )
    p.add_argument("--no-outcome-aware", action="store_true", help="与 table2 一致，传给 Stage3 仿真 E-value 分支时关闭（此处主要用 probe F1 与 GW）。")
    p.add_argument("--shift-type", type=str, default="none", choices=("none", "sudden", "gradual", "periodic"))
    p.add_argument("--shuffle-test-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    root = Path(args.root_dir)
    t2b = _load_t2b()

    s3 = Stage3Config(
        root_dir=root,
        results_dir=root / str(args.results_dir),
        artifact_suffix=str(args.artifact_suffix or ""),
        gamma=float(args.gamma),
        gammas=(float(args.gamma),),
        alphas=(float(args.alpha),),
        betting_strategy="predictive",
        betting_lambda=0.5,
        outcome_aware=not bool(args.no_outcome_aware),
        shift_type=str(args.shift_type),  # type: ignore[arg-type]
        shuffle_test_seed=int(args.shuffle_test_seed),
        quality_use_probe_prob=True,
        calib_method="quantile",
    )
    probe_ck = Path(args.probe_checkpoint) if args.probe_checkpoint else None
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]

    out_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        meta, f1_probe, _f1_e, _f1_fix, f1_gw, _bk, ordered_test = t2b._simulate_extended(  # noqa: SLF001
            s3, dataset, float(args.gamma), float(args.alpha), probe_checkpoint=probe_ck
        )
        mk = str(args.margin_key)
        mvals, n_bad = _margin_scalars(ordered_test, s3.max_k, s3.cost_per_step, s3.oracle_cost_metric, mk)
        if n_bad:
            logging.warning("%s: %d 条空 step_targets，对应标量为 NaN，分桶时丢弃", dataset, n_bad)
        valid = np.isfinite(mvals) & np.isfinite(f1_probe) & np.isfinite(f1_gw)
        m2 = mvals[valid]
        p2 = f1_probe[valid]
        g2 = f1_gw[valid]
        n_keep = int(m2.size)
        if n_keep < 2:
            logging.error("%s: 有效样本不足，跳过", dataset)
            continue

        binning = str(args.binning)
        margin_labels: List[Tuple[float, str]] = []
        if binning == "fixed":
            fe = [float(x) for x in str(args.fixed_edges).split(",") if x.strip()]
            bucket_id, margin_labels = _fixed_threshold_bins(m2, fe)
            bmax = len(fe) - 1
        else:
            bucket_id, _span = _equal_count_bins(m2, int(args.n_buckets))
            bmax = int(bucket_id.max()) + 1 if bucket_id.size else 0
        for b in range(bmax):
            sel = bucket_id == b
            n_b = int(np.sum(sel))
            if n_b == 0:
                continue
            lo = float(m2[sel].min())
            hi = float(m2[sel].max())
            mp = float(p2[sel].mean())
            mg = float(g2[sel].mean())
            mlab = margin_labels[b][1] if binning == "fixed" and b < len(margin_labels) else ""
            out_rows.append(
                {
                    "dataset": dataset,
                    "margin_key": mk,
                    "binning": binning,
                    "bucket": b,
                    "n": n_b,
                    "margin_interval": mlab,
                    "margin_low_in_bucket": lo,
                    "margin_high_in_bucket": hi,
                    "mean_f1_probe": mp,
                    "mean_f1_gw": mg,
                    "mean_delta_probe_minus_gw": float(mp - mg),
                    "probe_better_frac": float(np.mean(p2[sel] > g2[sel])),
                    "meta_n_test": int(meta.get("n", n_keep)),
                }
            )
        out_rows.append(
            {
                "dataset": dataset,
                "margin_key": mk,
                "binning": binning,
                "bucket": -1,
                "n": n_keep,
                "margin_interval": "ALL",
                "margin_low_in_bucket": float(m2.min()),
                "margin_high_in_bucket": float(m2.max()),
                "mean_f1_probe": float(p2.mean()),
                "mean_f1_gw": float(g2.mean()),
                "mean_delta_probe_minus_gw": float((p2 - g2).mean()),
                "probe_better_frac": float(np.mean(p2 > g2)),
                "meta_n_test": int(meta.get("n", n_keep)),
            }
        )
        logging.info(
            "%s: overall mean Δ(Probe-GW)=%.4f (N=%d, checkpoint used by table2)",
            dataset,
            float((p2 - g2).mean()),
            n_keep,
        )

    out_dir = root / str(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "oracle_margin_probe_gw_buckets.csv"
    fields = [
        "dataset",
        "margin_key",
        "binning",
        "bucket",
        "n",
        "margin_interval",
        "margin_low_in_bucket",
        "margin_high_in_bucket",
        "mean_f1_probe",
        "mean_f1_gw",
        "mean_delta_probe_minus_gw",
        "probe_better_frac",
        "meta_n_test",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"已写入: {out_csv}\n")
    # 控制台摘要：每个数据集 full + 分桶
    cur = None
    for row in out_rows:
        d = row["dataset"]
        if d != cur:
            print(f"\n=== {d} (margin_key={row['margin_key']}) ===")
            cur = d
        b = row["bucket"]
        tag = "ALL(valid)" if b == -1 else f"bucket {b}"
        ival = row.get("margin_interval") or ""
        ival_s = f"  {ival}" if ival and ival != "ALL" else ""
        print(
            f"  {tag:14} n={int(row['n']):4d}{ival_s}  "
            f"minmax=[{row['margin_low_in_bucket']:.4g},{row['margin_high_in_bucket']:.4g}]  "
            f"meanF1 P={row['mean_f1_probe']:.4f} GW={row['mean_f1_gw']:.4f}  "
            f"Δ={row['mean_delta_probe_minus_gw']:+.4f}  P>GW%={row['probe_better_frac']*100:.1f}"
        )


if __name__ == "__main__":
    main()
