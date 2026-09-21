"""
Phase D5：仅评估 Deployable-GW（train 上用 self_consistency 步间增益估 r_k*，test 上按代理停止；汇报真实 F1/EM）。

无需重训 Probe。产出：results/stage2_deployable_gw.json（可选与现有 stage2_probe_table_*.csv 中的 Probe / Global-Weitzman 对照）。

用法：
  python -m stage2.run_deployable_gw_eval --datasets hotpotqa,musique,2wiki
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pretest.utils.weitzman import (
    compute_all_reservation_values_from_proxy,
    deployable_weitzman_stopping_simulation,
    trajectory_cumulative_cost,
)
from stage2.run_stage2 import Stage2Config, _load_trajectories, _summarize_results

LOGGER = logging.getLogger(__name__)


def _eval_deployable_gw(
    train_traj: List[Dict[str, Any]],
    test_traj: List[Dict[str, Any]],
    cfg: Stage2Config,
    quality_key: str,
) -> Dict[str, Any]:
    reservation = compute_all_reservation_values_from_proxy(
        train_traj, cfg.max_k, cfg.cost_per_step, quality_key
    )
    raw = deployable_weitzman_stopping_simulation(
        test_traj, reservation, cfg.max_k, quality_key
    )
    rows: List[Dict[str, Any]] = []
    for traj, result in zip(test_traj, raw):
        used = int(result.get("steps_used", 0))
        cum_cost = trajectory_cumulative_cost(
            traj, used, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        rows.append(
            {
                "f1": float(result.get("f1", 0.0)),
                "em": int(bool(result.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    summary = _summarize_results(rows, "Deployable-GW")
    return {
        "summary": summary,
        "reservation_values": {str(k): float(v) for k, v in sorted(reservation.items())},
    }


def _read_probe_gw_from_table(results_dir: Path, dataset: str, suffix: str) -> Dict[str, Any]:
    tag = f"_{suffix}" if suffix else ""
    path = results_dir / f"stage2_probe_table_{dataset}{tag}.csv"
    out: Dict[str, Any] = {"probe_table_path": str(path), "probe": None, "global_weitzman": None}
    if not path.exists():
        return out
    df = pd.read_csv(path)
    for name in ("Probe", "Global-Weitzman"):
        sub = df[df["strategy"] == name]
        if sub.empty:
            continue
        row = sub.iloc[0]
        key = "probe" if name == "Probe" else "global_weitzman"
        out[key] = {
            "avg_f1": float(row["avg_f1"]),
            "avg_steps": float(row["avg_steps"]),
            "avg_cost": float(row["avg_cost"]),
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Deployable-GW 快速评估（不重训 Probe）")
    parser.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--cost-per-step", type=float, default=0.05)
    parser.add_argument(
        "--oracle-cost-metric",
        type=str,
        default="fixed",
        choices=("fixed", "token", "latency"),
    )
    parser.add_argument(
        "--quality-key",
        type=str,
        default="self_consistency",
        help="部署可见的代理质量字段（默认 self_consistency）",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="与 run_stage2 --artifact-suffix 一致，用于读取对照表文件名",
    )
    parser.add_argument("--out-json", type=str, default="results/stage2_deployable_gw.json")
    args = parser.parse_args()

    cfg = Stage2Config(
        root_dir=Path(args.root_dir),
        max_k=int(args.max_k),
        cost_per_step=float(args.cost_per_step),
        oracle_cost_metric=str(args.oracle_cost_metric),
    )
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    results_dir = cfg.results_dir
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    suffix = str(args.artifact_suffix or "").strip()

    bundle: Dict[str, Any] = {
        "quality_key": args.quality_key,
        "cost_per_step": cfg.cost_per_step,
        "max_k": cfg.max_k,
        "datasets": {},
    }

    for ds in datasets:
        train_traj = _load_trajectories(cfg, ds, "train")
        test_traj = _load_trajectories(cfg, ds, "test")
        ev = _eval_deployable_gw(train_traj, test_traj, cfg, args.quality_key)
        cmp_tbl = _read_probe_gw_from_table(results_dir, ds, suffix)
        bundle["datasets"][ds] = {
            "deployable_gw": ev["summary"],
            "reservation_values": ev["reservation_values"],
            "comparison_from_probe_table": cmp_tbl,
        }
        s = ev["summary"]
        LOGGER.info(
            "%s Deployable-GW: F1=%.4f steps=%.3f cost=%.4f",
            ds,
            float(s["avg_f1"]),
            float(s["avg_steps"]),
            float(s["avg_cost"]),
        )

    out_path = Path(args.out_json)
    if not out_path.is_absolute():
        out_path = cfg.root_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    LOGGER.info("已写入 %s", out_path)


if __name__ == "__main__":
    main()
