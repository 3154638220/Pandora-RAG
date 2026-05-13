#!/usr/bin/env python3
"""Low-margin / low-confidence 保守 Continue threshold 实验（离线重放，不重训 LLM）。

规则（与论文 Eq. probe 一致）：Continue 当且仅当 p_theta(k) >= eta_k。
若 |Oracle m_k| < margin_abs_lt（默认 0.1）或 |p_theta-0.5| < probe_conf_half_width（低置信），
则 eta_k <- min(max_eta, eta_k + threshold_boost)。

报告：
- 轨迹级：mean F1、EM、error rate (F1<γ)、avg steps
- 步级（Oracle Continue 为正类）：FP = 预测的 Continue 且 Oracle Stop；便于对照 MuSiQue 「误继续」是否下降

用法示例::
  python scripts/experiment_low_margin_conservative_threshold.py --dataset musique --gamma 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretest.utils.weitzman import compute_trajectory_oracle
from stage2.run_stage2 import (
    Stage2Config,
    ThresholdSpec,
    _simulate_probe_policy_from_probs,
    _simulate_probe_policy_low_margin_conservative,
)
from stage3.adapters.stage2_probe import (
    find_best_probe_checkpoint,
    load_stage2_probe_bundle,
    load_trajectories_and_step_features,
    precompute_continue_probabilities,
)
from stage3.config import Stage3Config
from stage3.stopping import attach_error_labels, summarize


def _stage2_cfg(s3: Stage3Config) -> Stage2Config:
    return Stage2Config(
        root_dir=s3.root_dir,
        max_k=s3.max_k,
        cost_per_step=s3.cost_per_step,
        oracle_cost_metric=s3.oracle_cost_metric,
        hidden_state_key=s3.hidden_state_key,
        artifact_suffix=s3.artifact_suffix,
        batch_size=256,
    )


def _threshold_spec_from_ckpt(
    ckpt: Mapping[str, Any], max_k: int, fallback: float
) -> ThresholdSpec:
    pst = ckpt.get("per_step_thresholds")
    if pst is not None and isinstance(pst, (list, tuple)) and len(pst) >= max_k:
        return {k: float(pst[k - 1]) for k in range(1, max_k + 1)}
    return float(ckpt.get("threshold", fallback))


def _effective_eta_at_step(
    *,
    base_t: float,
    p: float,
    margin: Optional[float],
    margin_abs_lt: float,
    threshold_boost: float,
    probe_conf_half_width: float,
    use_oracle_margin: bool,
    use_probe_low_conf: bool,
    max_effective_threshold: float,
) -> float:
    low_m = bool(
        use_oracle_margin
        and margin is not None
        and abs(float(margin)) < float(margin_abs_lt) - 1e-15
    )
    low_c = bool(
        use_probe_low_conf
        and abs(float(p) - 0.5) < float(probe_conf_half_width) - 1e-15
    )
    eff = float(base_t)
    if low_m or low_c:
        eff = min(float(max_effective_threshold), eff + float(threshold_boost))
    return eff


def _step_level_continue_confusion(
    trajectories: List[Dict[str, Any]],
    probs: Dict[Tuple[str, int], float],
    oracle_rows: List[Dict[str, Any]],
    *,
    max_k: int,
    threshold: ThresholdSpec,
    conservative: bool,
    margin_abs_lt: float,
    threshold_boost: float,
    probe_conf_half_width: float,
    use_oracle_margin: bool,
    use_probe_low_conf: bool,
    max_effective_threshold: float,
) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []

    for traj, orow in zip(trajectories, oracle_rows):
        sid = str(traj.get("id", ""))
        st = orow.get("step_targets") or {}
        for k in range(1, max_k):
            if k not in st:
                continue
            tinfo = st[k] or {}
            margin_v = float(tinfo.get("margin", 0.0))
            yt = int(tinfo.get("action_label", 0))
            p = probs.get((sid, k))
            if p is None:
                continue
            if isinstance(threshold, dict):
                base_t = float(threshold.get(k, 0.5))
            else:
                base_t = float(threshold)
            if conservative:
                eta = _effective_eta_at_step(
                    base_t=base_t,
                    p=float(p),
                    margin=margin_v,
                    margin_abs_lt=margin_abs_lt,
                    threshold_boost=threshold_boost,
                    probe_conf_half_width=probe_conf_half_width,
                    use_oracle_margin=use_oracle_margin,
                    use_probe_low_conf=use_probe_low_conf,
                    max_effective_threshold=max_effective_threshold,
                )
            else:
                eta = base_t
            yp = int(float(p) >= eta)
            y_true.append(yt)
            y_pred.append(yp)

    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    if y_t.size == 0:
        return {
            "n_steps": 0,
            "continue_fp": 0,
            "continue_precision": float("nan"),
            "continue_recall": float("nan"),
            "continue_f1": float("nan"),
        }
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = float(2 * prec * rec / (prec + rec)) if prec == prec and rec == rec and (prec + rec) > 0 else float("nan")
    return {
        "n_steps": int(y_t.size),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "continue_fp": int(fp),
        "continue_precision": prec,
        "continue_recall": rec,
        "continue_f1": f1,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Low-margin conservative Continue threshold experiment")
    p.add_argument("--dataset", type=str, default="musique")
    p.add_argument("--root-dir", type=Path, default=REPO_ROOT)
    p.add_argument("--probe-checkpoint", type=Path, default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--margin-abs-lt", type=float, default=0.1)
    p.add_argument("--threshold-boost", type=float, default=0.1)
    p.add_argument("--probe-conf-half-width", type=float, default=0.1)
    p.add_argument("--max-effective-threshold", type=float, default=0.99)
    p.add_argument("--no-oracle-margin", action="store_true")
    p.add_argument("--no-probe-low-conf", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="写入 JSON 的路径（默认 results/low_margin_conservative_{dataset}_g{gamma}.json）",
    )
    args = p.parse_args()

    s3 = Stage3Config(root_dir=args.root_dir)
    cfg2 = _stage2_cfg(s3)
    ds = str(args.dataset).lower().strip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.probe_checkpoint
    if ckpt_path is None:
        cand = find_best_probe_checkpoint(args.root_dir / "artifacts" / "probe", ds)
        if cand is None:
            raise SystemExit(f"未找到 {ds} 的 probe checkpoint")
        ckpt_path = cand
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    bundle = load_stage2_probe_bundle(ckpt_path, device)
    thr_spec = _threshold_spec_from_ckpt(ckpt, s3.max_k, bundle.probe_threshold)
    thr_scalar_stage3 = float(bundle.probe_threshold)

    test_traj, feat_map, hmap, hd = load_trajectories_and_step_features(
        cfg2, ds, "test", bundle
    )
    test_probs = precompute_continue_probabilities(
        test_traj, feat_map, hmap, hd, bundle, cfg2
    )

    oracle_rows = compute_trajectory_oracle(
        test_traj,
        s3.cost_per_step,
        s3.max_k,
        cost_metric=s3.oracle_cost_metric,
    )

    base_rows = _simulate_probe_policy_from_probs(
        test_traj, thr_spec, cfg2, test_probs
    )
    cons_rows = _simulate_probe_policy_low_margin_conservative(
        test_traj,
        thr_spec,
        cfg2,
        test_probs,
        oracle_rows,
        margin_abs_lt=float(args.margin_abs_lt),
        threshold_boost=float(args.threshold_boost),
        probe_conf_half_width=float(args.probe_conf_half_width),
        use_oracle_margin=not args.no_oracle_margin,
        use_probe_low_conf=not args.no_probe_low_conf,
        max_effective_threshold=float(args.max_effective_threshold),
    )

    stage3_style_rows = _simulate_probe_policy_from_probs(
        test_traj, thr_scalar_stage3, cfg2, test_probs
    )

    g = float(args.gamma)
    summ_base = summarize(attach_error_labels(base_rows, g), "Probe(per-step-threshold)")
    summ_cons = summarize(attach_error_labels(cons_rows, g), "Probe+low-margin-conservative")
    summ_s3 = summarize(attach_error_labels(stage3_style_rows, g), "Probe(stage3 scalar ckpt.threshold)")

    step_base = _step_level_continue_confusion(
        test_traj,
        test_probs,
        oracle_rows,
        max_k=s3.max_k,
        threshold=thr_spec,
        conservative=False,
        margin_abs_lt=float(args.margin_abs_lt),
        threshold_boost=float(args.threshold_boost),
        probe_conf_half_width=float(args.probe_conf_half_width),
        use_oracle_margin=not args.no_oracle_margin,
        use_probe_low_conf=not args.no_probe_low_conf,
        max_effective_threshold=float(args.max_effective_threshold),
    )
    step_cons = _step_level_continue_confusion(
        test_traj,
        test_probs,
        oracle_rows,
        max_k=s3.max_k,
        threshold=thr_spec,
        conservative=True,
        margin_abs_lt=float(args.margin_abs_lt),
        threshold_boost=float(args.threshold_boost),
        probe_conf_half_width=float(args.probe_conf_half_width),
        use_oracle_margin=not args.no_oracle_margin,
        use_probe_low_conf=not args.no_probe_low_conf,
        max_effective_threshold=float(args.max_effective_threshold),
    )

    out = {
        "dataset": ds,
        "checkpoint": str(ckpt_path),
        "gamma": g,
        "threshold_spec": thr_spec if isinstance(thr_spec, float) else {str(k): v for k, v in thr_spec.items()},
        "stage3_scalar_threshold": thr_scalar_stage3,
        "conservative_params": {
            "margin_abs_lt": float(args.margin_abs_lt),
            "threshold_boost": float(args.threshold_boost),
            "probe_conf_half_width": float(args.probe_conf_half_width),
            "max_effective_threshold": float(args.max_effective_threshold),
            "use_oracle_margin": not args.no_oracle_margin,
            "use_probe_low_conf": not args.no_probe_low_conf,
        },
        "trajectory_summaries": {
            "per_step_baseline": summ_base,
            "conservative": summ_cons,
            "stage3_scalar_baseline": summ_s3,
        },
        "step_level_continue": {
            "per_step_baseline": step_base,
            "conservative": step_cons,
        },
    }

    res_path = args.output
    if res_path is None:
        res_path = (
            args.root_dir
            / "results"
            / f"low_margin_conservative_{ds}_g{str(g).replace('.','_')}.json"
        )
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {res_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
