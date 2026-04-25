#!/usr/bin/env python3
"""
将 Stage2 探针行的 avg_steps / F1 与可解释的归一化成本分解合并为一张表（NeurIPS P0：cost accounting）。

默认权重为 **相对单位**（非秒），便于在论文中固定「检索一步 = 1」后比较 Full vs Lite 的特征开销假设：

- 检索：每步 w_ret（通常 1.0）
- 生成：每步 w_gen（可与 token 预算挂钩；也可用 --empirical-token-weight 从轨迹估计）
- 特征：每步 w_feat_full 或 w_feat_lite（Full Probe 侧写 NLI/熵/ROUGE 等额外算子；Lite 仅廉价子集）
- 探针前向：每步 w_probe（相对主干可忽略，但单独列出以满足审稿「probe overhead」口径）

示例::

  python scripts/probe_cost_accounting.py \\
    --root-dir . --results-dir results \\
    --artifact-suffix _pdopt_binary_last_token \\
    --probe-variant full \\
    --out-csv results/probe_cost_accounting_full.csv

  python scripts/probe_cost_accounting.py \\
    --root-dir . --results-dir results \\
    --artifact-suffix _probe_lite \\
    --probe-variant lite \\
    --out-csv results/probe_cost_accounting_lite.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _read_probe_row(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    df = pd.read_csv(csv_path)
    if "strategy" not in df.columns:
        raise ValueError(f"{csv_path} 缺少 strategy 列")
    sub = df[df["strategy"].astype(str).str.strip() == "Probe"]
    if sub.empty:
        raise ValueError(f"{csv_path} 中未找到 strategy==Probe 行")
    row = sub.iloc[0]
    out: Dict[str, Any] = {
        "avg_steps": float(row["avg_steps"]),
        "avg_f1": float(row["avg_f1"]),
    }
    if "avg_em" in row:
        out["avg_em"] = float(row["avg_em"])
    if "avg_cost" in row:
        out["avg_cost_oracle_metric"] = float(row["avg_cost"])
    return out


def _mean_tokens_per_step(
    trajectories_jsonl: Path,
    *,
    max_lines: int = 10_000,
) -> float:
    """对 jsonl 中每条轨迹的所有步，取 token_count 的样本均值。"""
    if not trajectories_jsonl.is_file():
        return float("nan")
    vals: List[float] = []
    n_lines = 0
    with trajectories_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if n_lines >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for st in obj.get("steps") or []:
                c = st.get("cost") or {}
                vals.append(float(c.get("token_count", 0) or 0.0))
    return float(np.mean(vals)) if vals else float("nan")


def _mean_retrieval_calls_per_step(
    trajectories_jsonl: Path,
    *,
    max_lines: int = 10_000,
) -> float:
    if not trajectories_jsonl.is_file():
        return float("nan")
    vals: List[float] = []
    n_lines = 0
    with trajectories_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if n_lines >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for st in obj.get("steps") or []:
                c = st.get("cost") or {}
                vals.append(float(c.get("retrieval_calls", 1) or 1.0))
    return float(np.mean(vals)) if vals else float("nan")


def build_rows_for_datasets(
    *,
    root: Path,
    results_dir: Path,
    datasets: List[str],
    artifact_suffix: str,
    probe_variant: str,
    w_retrieval: float,
    w_generation: float,
    w_feat_full: float,
    w_feat_lite: float,
    w_probe: float,
    split_for_trajectories: str,
    empirical_token_mult: bool,
    empirical_retrieval_mult: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    w_feat = w_feat_lite if probe_variant == "lite" else w_feat_full

    for ds in datasets:
        table = results_dir / f"stage2_probe_table_{ds}{artifact_suffix}.csv"
        pr = _read_probe_row(table)
        avg_steps = float(pr["avg_steps"])

        traj_p = root / "cache" / "trajectories" / ds / split_for_trajectories / "trajectories.jsonl"
        mtok = _mean_tokens_per_step(traj_p) if empirical_token_mult else float("nan")
        mret = _mean_retrieval_calls_per_step(traj_p) if empirical_retrieval_mult else float("nan")

        gen_mult = 1.0
        if empirical_token_mult and np.isfinite(mtok) and mtok > 0:
            # 将「每步 1 次生成」按相对 token 量缩放（仅作论文中可调的展示口径）
            gen_mult = float(mtok)

        ret_mult = 1.0
        if empirical_retrieval_mult and np.isfinite(mret) and mret > 0:
            ret_mult = float(mret)

        c_ret = avg_steps * w_retrieval * ret_mult
        c_gen = avg_steps * w_generation * gen_mult
        c_feat = avg_steps * w_feat
        c_probe = avg_steps * w_probe
        total = c_ret + c_gen + c_feat + c_probe

        rows.append(
            {
                "dataset": ds,
                "probe_variant": probe_variant,
                "artifact_suffix": artifact_suffix or "(default)",
                "avg_steps": avg_steps,
                "avg_f1": float(pr["avg_f1"]),
                "avg_em": float(pr.get("avg_em", float("nan"))),
                "retrieval_component": c_ret,
                "generation_component": c_gen,
                "feature_overhead_component": c_feat,
                "probe_forward_component": c_probe,
                "total_normalized_cost": total,
                "weight_retrieval": w_retrieval,
                "weight_generation": w_generation,
                "weight_feature_per_step": w_feat,
                "weight_probe_forward_per_step": w_probe,
                "empirical_mean_tokens_per_step": mtok,
                "empirical_mean_retrieval_calls_per_step": mret,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe 成本分解表（Stage2 CSV + 可调权重）")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="与 stage2 输出一致，例如 _pdopt_binary_last_token 或 _probe_lite",
    )
    p.add_argument(
        "--probe-variant",
        type=str,
        choices=("full", "lite"),
        required=True,
        help="决定使用 w_feat_full 还是 w_feat_lite 作为每步特征开销权重",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,musique,2wiki",
    )
    p.add_argument("--weight-retrieval", type=float, default=1.0)
    p.add_argument("--weight-generation", type=float, default=1.0)
    p.add_argument(
        "--weight-feat-full",
        type=float,
        default=0.08,
        help="Full Probe：每步额外特征（NLI/熵/ROUGE 等）的相对成本系数",
    )
    p.add_argument(
        "--weight-feat-lite",
        type=float,
        default=0.02,
        help="Lite Probe：仅廉价浅层时的相对成本系数",
    )
    p.add_argument(
        "--weight-probe-forward",
        type=float,
        default=0.01,
        help="每步探针 MLP 前向的相对成本（相对检索步）",
    )
    p.add_argument(
        "--split-for-trajectories",
        type=str,
        default="test",
        help="读取 cache/trajectories/<ds>/<split>/trajectories.jsonl 估计 token/retrieval 均值",
    )
    p.add_argument(
        "--empirical-token-mult",
        action="store_true",
        help="用轨迹中 token_count 均值缩放 generation 分量（gen *= mean_tokens_per_step）",
    )
    p.add_argument(
        "--empirical-retrieval-mult",
        action="store_true",
        help="用轨迹中 retrieval_calls 均值缩放 retrieval 分量",
    )
    p.add_argument("--out-csv", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    res = Path(args.results_dir)
    if not res.is_absolute():
        res = (root / res).resolve()
    ds_list = [x.strip().lower() for x in str(args.datasets).split(",") if x.strip()]
    df = build_rows_for_datasets(
        root=root,
        results_dir=res,
        datasets=ds_list,
        artifact_suffix=str(args.artifact_suffix),
        probe_variant=str(args.probe_variant),
        w_retrieval=float(args.weight_retrieval),
        w_generation=float(args.weight_generation),
        w_feat_full=float(args.weight_feat_full),
        w_feat_lite=float(args.weight_feat_lite),
        w_probe=float(args.weight_probe_forward),
        split_for_trajectories=str(args.split_for_trajectories),
        empirical_token_mult=bool(args.empirical_token_mult),
        empirical_retrieval_mult=bool(args.empirical_retrieval_mult),
    )
    out = Path(args.out_csv)
    if not out.is_absolute():
        out = (root / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
