#!/usr/bin/env python3
"""
从 Stage3 结果 JSON 导出论文用 Markdown 表格（stdout）。

  python scripts/stage3_export_tables.py

依赖：results/e2_predictive/stage3_evalue_*.json 与 results/e4_fixed/stage3_evalue_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def alpha_sensitivity_rows(data: Dict[str, Any]) -> List[Tuple[str, float, float, float, float, float, float]]:
    out: List[Tuple[str, float, float, float, float, float, float]] = []
    for alpha in ["0.05", "0.1", "0.2", "0.3"]:
        step_d, f1_d = [], []
        for gk, gv in data["per_gamma"].items():
            pa = gv.get("per_alpha") or {}
            if alpha not in pa:
                continue
            s = pa[alpha]["summaries"]
            pb, pe = s["probe"], s["probe_evalue"]
            if float(pb["avg_steps"]) > 0:
                step_d.append(100.0 * (float(pe["avg_steps"]) / float(pb["avg_steps"]) - 1.0))
            if float(pb["avg_f1"]) > 0:
                f1_d.append(100.0 * (float(pe["avg_f1"]) / float(pb["avg_f1"]) - 1.0))
        if not step_d:
            continue
        out.append(
            (
                alpha,
                min(step_d),
                max(step_d),
                mean(step_d),
                min(f1_d),
                max(f1_d),
                mean(f1_d),
            )
        )
    return out


def main() -> None:
    root = _ROOT
    e2 = root / "results" / "e2_predictive"
    e4 = root / "results" / "e4_fixed"
    datasets = ["hotpotqa", "musique", "2wiki"]

    print("### Betting 消融（γ=0.5，predictive λ=1−p̂ vs fixed λ=0.5）\n")
    print("| 数据集 | α | 策略 | 终态 E-wealth | Probe+E-value 步数 | F1 |")
    print("|--------|---|------|---------------|-------------------|-----|")
    for ds in datasets:
        pred = _load(e2 / f"stage3_evalue_{ds}.json")
        fix = _load(e4 / f"stage3_evalue_{ds}.json")
        for a in ["0.1", "0.2"]:
            pw = pred["per_gamma"]["0.5"]["per_alpha"][a]["final_e_wealth"]
            fw = fix["per_gamma"]["0.5"]["per_alpha"][a]["final_e_wealth"]
            ps = pred["per_gamma"]["0.5"]["per_alpha"][a]["summaries"]["probe_evalue"]["avg_steps"]
            fs = fix["per_gamma"]["0.5"]["per_alpha"][a]["summaries"]["probe_evalue"]["avg_steps"]
            pf = pred["per_gamma"]["0.5"]["per_alpha"][a]["summaries"]["probe_evalue"]["avg_f1"]
            ff = fix["per_gamma"]["0.5"]["per_alpha"][a]["summaries"]["probe_evalue"]["avg_f1"]
            print(
                f"| {ds} | {float(a):.2f} | predictive | {float(pw):.3f} | {float(ps):.3f} | {float(pf):.4f} |"
            )
            print(
                f"| {ds} | {float(a):.2f} | fixed | {float(fw):.3f} | {float(fs):.3f} | {float(ff):.4f} |"
            )
    print()

    print("### α 敏感性汇总（E2：predictive，相对 Probe 的 Probe+E-value，跨 γ∈{0.3,0.4,0.5,0.6} 的最小–最大区间与均值）\n")
    print("| 数据集 | α | Δ步数% [min,max] | mean Δ步数% | ΔF1% [min,max] | mean ΔF1% |")
    print("|--------|---|------------------|-------------|----------------|----------|")
    for ds in datasets:
        data = _load(e2 / f"stage3_evalue_{ds}.json")
        for r in alpha_sensitivity_rows(data):
            alpha, smin, smax, smean, fmin, fmax, fmean = r
            print(
                f"| {ds} | {float(alpha):.2f} | [{smin:.2f}, {smax:.2f}] | {smean:.2f} | "
                f"[{fmin:.2f}, {fmax:.2f}] | {fmean:.2f} |"
            )
    print()


if __name__ == "__main__":
    main()
