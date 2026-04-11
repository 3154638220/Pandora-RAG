"""
Stage3 主入口：加载 Stage2 探针 → Calib 上训练质量头并校准阈值 → Test 上对比
Probe / E-value 门控（结果感知型 + 原版 indicator）/ 分位型 CP 门控，
并输出 JSON + 累积错误率曲线 + E-wealth trace 图 + Markdown 报告。

Stage2 相关细节均经 ``stage3.adapters.stage2_probe``；本脚本只做编排。

Usage:
  python -m stage3.run_stage3 --datasets hotpotqa,musique,2wiki --gammas 0.5
  python -m stage3.run_stage3 --datasets hotpotqa --gammas 0.3,0.4,0.5,0.6 --alphas 0.05,0.1,0.2
  python -m stage3.run_stage3 --datasets hotpotqa --shift-type sudden --shift-fraction 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from stage2.run_stage2 import Stage2Config
from stage3.adapters.stage2_probe import (
    default_probe_checkpoint_path,
    find_best_probe_checkpoint,
    load_stage2_probe_bundle,
    load_trajectories_and_step_features,
    precompute_continue_probabilities,
    shallow_features_from_map,
    simulate_probe_baseline,
)
from stage3.config import Stage3Config
from stage3.evalue import cumulative_error_rate_curve
from stage3.quality_model import (
    build_step_quality_dataset,
    evaluate_quality_model,
    train_quality_logreg,
    tune_quality_bar_on_calib,
)
from stage3.stopping import (
    attach_error_labels,
    build_shift_ordering,
    conformal_min_phat_threshold,
    probe_stop_shallow_and_phat,
    simulate_conformal_phat_gate,
    simulate_evalue_gated_stops,
    simulate_evalue_outcome_aware,
    summarize,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

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


def _collect_calib_stop_stats(
    calib_traj: List[Dict[str, Any]],
    calib_probs: Dict[Tuple[str, int], float],
    shallow_calib: Dict[Tuple[str, int], np.ndarray],
    quality_model: Any,
    cfg2: Stage2Config,
    probe_threshold: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    phats: List[float] = []
    errs: List[float] = []
    for traj in calib_traj:
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        if not steps:
            continue
        _z, ph, chosen = probe_stop_shallow_and_phat(
            traj,
            calib_probs,
            shallow_calib,
            quality_model,
            cfg=cfg2,
            probe_threshold=probe_threshold,
        )
        f1 = float(chosen.get("f1", 0.0))
        phats.append(float(ph))
        errs.append(1.0 if f1 < float(gamma) else 0.0)
    if not phats:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(phats, dtype=np.float64), np.asarray(errs, dtype=np.float64)


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def _plot_cumulative_errors(
    curves: Dict[str, List[float]],
    title: str,
    out_path: Path,
    alpha_line: Optional[float] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    for name, ys in curves.items():
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, label=name, linewidth=1.6)
    ax.set_xlabel("Test samples (ordered)")
    ax.set_ylabel("Cumulative error rate (F1 < γ)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if alpha_line is not None:
        ax.axhline(float(alpha_line), color="#888888", linestyle="--", linewidth=1.0, label=f"α={alpha_line}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_wealth_trace(
    wealth_traces: Dict[str, List[float]],
    title: str,
    out_path: Path,
    alpha: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    for name, ws in wealth_traces.items():
        xs = np.arange(len(ws))
        ax.plot(xs, ws, label=name, linewidth=1.4)
    ax.axhline(1.0 / alpha, color="#d62728", linestyle="--", linewidth=1.0, label=f"1/α={1.0/alpha:.1f}")
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Test samples (ordered)")
    ax.set_ylabel("E-wealth")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 单数据集单 gamma
# ---------------------------------------------------------------------------

def _run_one_gamma(
    s3: Stage3Config,
    dataset: str,
    gamma: float,
    cfg2: Stage2Config,
    bundle: Any,
    calib_traj: List[Dict[str, Any]],
    calib_probs: Dict[Tuple[str, int], float],
    shallow_calib: Dict[Tuple[str, int], np.ndarray],
    test_traj: List[Dict[str, Any]],
    test_probs: Dict[Tuple[str, int], float],
    shallow_test: Dict[Tuple[str, int], np.ndarray],
    test_map: Dict[Tuple[str, int], np.ndarray],
    test_hmap: Dict[Tuple[str, int], np.ndarray],
    test_hd: int,
) -> Dict[str, Any]:
    """对一个 (dataset, gamma) 组合，跑所有 alpha × 策略。"""

    x_q, y_q = build_step_quality_dataset(
        calib_traj,
        shallow_calib,
        gamma=gamma,
        max_k=s3.max_k,
        probe_probs=calib_probs if s3.quality_use_probe_prob else None,
    )
    qmodel = train_quality_logreg(
        x_q, y_q,
        max_iter=s3.quality_max_iter,
        random_state=s3.quality_random_state,
    )
    qm_eval = evaluate_quality_model(qmodel, x_q, y_q)
    LOGGER.info(
        "%s γ=%.2f quality_model: brier=%.4f ece=%.4f pos_rate=%.2f",
        dataset, gamma, qm_eval["brier_score"], qm_eval["ece"], qm_eval["positive_rate"],
    )

    if s3.quality_use_probe_prob:
        shallow_calib_q = {
            k: np.append(v, np.float32(calib_probs.get(k, 0.5)))
            for k, v in shallow_calib.items()
        }
        shallow_test_q = {
            k: np.append(v, np.float32(test_probs.get(k, 0.5)))
            for k, v in shallow_test.items()
        }
    else:
        shallow_calib_q = shallow_calib
        shallow_test_q = shallow_test

    phat_stop, err_stop = _collect_calib_stop_stats(
        calib_traj, calib_probs, shallow_calib_q, qmodel,
        cfg2, bundle.probe_threshold, gamma,
    )

    baseline = simulate_probe_baseline(
        test_traj, test_probs, bundle.probe_threshold, cfg2
    )
    baseline_tagged = attach_error_labels(baseline, gamma)

    shift_order = build_shift_ordering(
        test_traj,
        shift_type=s3.shift_type,
        shift_fraction=s3.shift_fraction,
        rng_seed=s3.shuffle_test_seed,
    )
    ordered_test = [test_traj[int(i)] for i in shift_order]
    baseline_ordered = [baseline_tagged[int(i)] for i in shift_order]

    per_alpha: Dict[str, Any] = {}
    plot_paths: Dict[str, str] = {}

    for alpha in s3.alphas:
        a = float(alpha)
        qbar = tune_quality_bar_on_calib(phat_stop, err_stop, target_error=a)
        cp_tau = conformal_min_phat_threshold(phat_stop, a)

        if s3.outcome_aware:
            ev_rows, wealth_tr = simulate_evalue_outcome_aware(
                ordered_test, test_probs, shallow_test_q, qmodel,
                cfg=cfg2,
                probe_threshold=bundle.probe_threshold,
                gamma=gamma, alpha=a, quality_bar=qbar,
                betting_strategy=s3.betting_strategy,
                betting_lambda=s3.betting_lambda,
            )
        else:
            ev_rows, wealth_tr = simulate_evalue_gated_stops(
                ordered_test, test_probs, shallow_test_q, qmodel,
                cfg=cfg2,
                probe_threshold=bundle.probe_threshold,
                gamma=gamma, alpha=a, quality_bar=qbar,
            )

        cf_rows = simulate_conformal_phat_gate(
            ordered_test, test_probs, shallow_test_q, qmodel,
            cfg=cfg2,
            probe_threshold=bundle.probe_threshold,
            gamma=gamma, min_phat=cp_tau,
        )

        curves = {
            "Probe": cumulative_error_rate_curve([r["error"] for r in baseline_ordered]),
            "Probe+E-value": cumulative_error_rate_curve([r["error"] for r in ev_rows]),
            "Probe+CP-quantile": cumulative_error_rate_curve([r["error"] for r in cf_rows]),
        }

        tag = f"g{str(gamma).replace('.','_')}_a{str(a).replace('.','_')}"
        shift_tag = f"_{s3.shift_type}" if s3.shift_type != "none" else ""

        png_err = s3.results_dir / f"stage3_cum_error_{dataset}_{tag}{shift_tag}.png"
        _plot_cumulative_errors(
            curves,
            title=f"{dataset} cumulative error (α={a}, γ={gamma}){shift_tag}",
            out_path=png_err,
            alpha_line=a,
        )

        wealth_traces = {"Probe+E-value": wealth_tr}
        png_wealth = s3.results_dir / f"stage3_wealth_{dataset}_{tag}{shift_tag}.png"
        _plot_wealth_trace(
            wealth_traces,
            title=f"{dataset} E-wealth trace (α={a}, γ={gamma}){shift_tag}",
            out_path=png_wealth,
            alpha=a,
        )

        plot_paths[str(a)] = str(png_err)

        per_alpha[str(a)] = {
            "evalue_quality_bar": float(qbar),
            "conformal_min_phat": float(cp_tau),
            "summaries": {
                "probe": summarize(baseline_ordered, "Probe"),
                "probe_evalue": summarize(ev_rows, "Probe+E-value"),
                "probe_conformal_quantile": summarize(cf_rows, "Probe+CP-quantile"),
            },
            "final_e_wealth": float(wealth_tr[-1]) if wealth_tr else 1.0,
            "wealth_cap": float(1.0 / a),
            "plot_error_rate": str(png_err),
            "plot_wealth_trace": str(png_wealth),
        }

    return {
        "gamma": float(gamma),
        "quality_model_eval": qm_eval,
        "per_alpha": per_alpha,
        "plot_paths": plot_paths,
    }


# ---------------------------------------------------------------------------
# 单数据集入口
# ---------------------------------------------------------------------------

def run_dataset_stage3(
    s3: Stage3Config,
    dataset: str,
    *,
    probe_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg2 = _stage2_cfg(s3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if probe_checkpoint and probe_checkpoint.exists():
        ckpt = probe_checkpoint
    else:
        ckpt_found = find_best_probe_checkpoint(
            s3.artifacts_probe_dir, dataset,
            preferred_suffix=s3.artifact_suffix or "pdopt_best",
        )
        if ckpt_found is None:
            ckpt = default_probe_checkpoint_path(cfg2, dataset)
        else:
            ckpt = ckpt_found

    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 Stage2 探针 checkpoint：{ckpt}（请先跑 stage2 或指定 --probe-checkpoint）")

    bundle = load_stage2_probe_bundle(ckpt, device)
    arch_tag = (
        "mlp_v2_gru"
        if bundle.sequence_gru
        else ("mlp_v2" if bundle.dual_input else "mlp")
    )
    LOGGER.info(
        "%s 加载探针 arch=%s shallow_only=%s threshold=%.4f ckpt=%s",
        dataset, arch_tag, bundle.shallow_only, bundle.probe_threshold, ckpt,
    )

    calib_traj, calib_map, calib_hmap, calib_hd = load_trajectories_and_step_features(
        cfg2, dataset, "calib", bundle
    )
    test_traj, test_map, test_hmap, test_hd = load_trajectories_and_step_features(
        cfg2, dataset, "test", bundle
    )

    calib_probs = precompute_continue_probabilities(
        calib_traj, calib_map, calib_hmap, calib_hd, bundle, cfg2
    )
    test_probs = precompute_continue_probabilities(
        test_traj, test_map, test_hmap, test_hd, bundle, cfg2
    )

    shallow_calib = shallow_features_from_map(calib_map, bundle.shallow_feature_dim)
    shallow_test = shallow_features_from_map(test_map, bundle.shallow_feature_dim)

    per_gamma: Dict[str, Any] = {}
    for gamma in s3.gammas:
        LOGGER.info("%s γ=%.2f 开始...", dataset, gamma)
        per_gamma[str(gamma)] = _run_one_gamma(
            s3, dataset, float(gamma), cfg2, bundle,
            calib_traj, calib_probs, shallow_calib,
            test_traj, test_probs, shallow_test,
            test_map, test_hmap, test_hd,
        )

    out_bundle: Dict[str, Any] = {
        "dataset": dataset,
        "gammas": [float(g) for g in s3.gammas],
        "alphas": [float(x) for x in s3.alphas],
        "probe_checkpoint": str(ckpt),
        "betting_strategy": s3.betting_strategy,
        "outcome_aware": s3.outcome_aware,
        "shift_type": s3.shift_type,
        "shift_fraction": s3.shift_fraction,
        "quality_use_probe_prob": s3.quality_use_probe_prob,
        "shuffle_test_seed": int(s3.shuffle_test_seed),
        "per_gamma": per_gamma,
    }

    shift_tag = f"_{s3.shift_type}" if s3.shift_type != "none" else ""
    json_path = s3.results_dir / f"stage3_evalue_{dataset}{shift_tag}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_bundle, f, indent=2, ensure_ascii=False)
    LOGGER.info("%s Stage3 结果已写入 %s", dataset, json_path)

    _generate_markdown_report(s3, dataset, out_bundle)

    return {
        "dataset": dataset,
        "json_path": str(json_path),
        "per_gamma": per_gamma,
    }


# ---------------------------------------------------------------------------
# Markdown 报告
# ---------------------------------------------------------------------------

def _generate_markdown_report(
    s3: Stage3Config,
    dataset: str,
    bundle: Dict[str, Any],
) -> None:
    shift_tag = f"_{s3.shift_type}" if s3.shift_type != "none" else ""
    report_path = s3.root_dir / "docs" / f"stage3_report_{dataset}{shift_tag}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Stage3 E-value 报告：{dataset}")
    lines.append("")
    lines.append(f"- Probe checkpoint: `{bundle.get('probe_checkpoint', 'N/A')}`")
    lines.append(f"- Betting strategy: {bundle.get('betting_strategy', 'N/A')}")
    lines.append(f"- Outcome aware: {bundle.get('outcome_aware', True)}")
    lines.append(f"- Quality use probe prob: {bundle.get('quality_use_probe_prob', True)}")
    lines.append(f"- Shift type: {bundle.get('shift_type', 'none')}")
    lines.append("")

    for g_str, g_data in bundle.get("per_gamma", {}).items():
        lines.append(f"## γ = {g_str}")
        lines.append("")

        qm = g_data.get("quality_model_eval", {})
        lines.append(f"Quality Model — Brier: {qm.get('brier_score', 'N/A'):.4f}, "
                      f"ECE: {qm.get('ece', 'N/A'):.4f}, "
                      f"Pos rate: {qm.get('positive_rate', 'N/A'):.2f}")
        lines.append("")

        lines.append("| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |")
        lines.append("|---|------|--------|------------|-----------|-------------------|")

        for a_str, a_data in g_data.get("per_alpha", {}).items():
            qbar = a_data.get("evalue_quality_bar", 0.0)
            cp_tau = a_data.get("conformal_min_phat", 0.0)
            for key, label in [
                ("probe", "Probe"),
                ("probe_evalue", "Probe+E-value"),
                ("probe_conformal_quantile", "Probe+CP"),
            ]:
                sm = a_data.get("summaries", {}).get(key, {})
                bar_val = f"{qbar:.3f}" if "evalue" in key else (f"{cp_tau:.3f}" if "conformal" in key else "—")
                lines.append(
                    f"| {a_str} | {label} | "
                    f"{sm.get('avg_f1', 0.0):.4f} | "
                    f"{sm.get('error_rate', 0.0):.4f} | "
                    f"{sm.get('avg_steps', 0.0):.2f} | "
                    f"{bar_val} |"
                )

            final_w = a_data.get("final_e_wealth", 1.0)
            cap = a_data.get("wealth_cap", 10.0)
            lines.append(f"")
            lines.append(f"  E-wealth final={final_w:.4f}, cap=1/α={cap:.1f}")
            lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    LOGGER.info("Markdown 报告写入 %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage3 E-value 风险控制实验")
    p.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--cost-per-step", type=float, default=0.05)
    p.add_argument(
        "--oracle-cost-metric", type=str, default="fixed",
        choices=("fixed", "token", "latency"),
    )
    p.add_argument(
        "--hidden-state-key", type=str, default="last_token",
        choices=("last_token", "mean_pool", "last_mean_blend"),
    )
    p.add_argument("--artifact-suffix", type=str, default="",
                    help="与 stage2 --artifact-suffix 一致")
    p.add_argument(
        "--probe-checkpoint", type=str, default="",
        help="显式指定 probe_mlp*.pt；默认自动搜索",
    )

    p.add_argument("--gammas", type=str, default="0.5",
                    help="逗号分隔 γ 值（F1 < γ 视为错误）")
    p.add_argument("--alphas", type=str, default="0.1,0.2",
                    help="逗号分隔名义水平")

    p.add_argument("--betting-strategy", type=str, default="predictive",
                    choices=("fixed", "predictive"))
    p.add_argument("--betting-lambda", type=float, default=0.5,
                    help="fixed betting 时的 λ 值")
    p.add_argument("--no-outcome-aware", action="store_true",
                    help="使用原版 indicator betting（不推荐；做对照用）")

    p.add_argument("--quality-use-probe-prob", action="store_true", default=True,
                    help="质量模型加入 Probe p_continue 特征")
    p.add_argument("--no-quality-probe-prob", dest="quality_use_probe_prob",
                    action="store_false")
    p.add_argument("--quality-max-iter", type=int, default=300)
    p.add_argument("--quality-random-state", type=int, default=42)

    p.add_argument("--shift-type", type=str, default="none",
                    choices=("none", "sudden", "gradual", "periodic"),
                    help="分布漂移类型")
    p.add_argument("--shift-fraction", type=float, default=0.5)

    p.add_argument("--shuffle-test-seed", type=int, default=42)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    gammas = tuple(
        float(x) for x in str(args.gammas).split(",") if x.strip()
    ) or (0.5,)
    alphas = tuple(
        float(x) for x in str(args.alphas).split(",") if x.strip()
    ) or (0.1, 0.2)

    root = Path(args.root_dir)
    s3 = Stage3Config(
        root_dir=root,
        max_k=int(args.max_k),
        cost_per_step=float(args.cost_per_step),
        oracle_cost_metric=str(args.oracle_cost_metric),
        hidden_state_key=str(args.hidden_state_key),
        artifact_suffix=str(args.artifact_suffix or ""),
        gamma=gammas[0],
        gammas=gammas,
        alphas=alphas,
        betting_strategy=args.betting_strategy,
        betting_lambda=float(args.betting_lambda),
        outcome_aware=not args.no_outcome_aware,
        quality_use_probe_prob=bool(args.quality_use_probe_prob),
        quality_max_iter=int(args.quality_max_iter),
        quality_random_state=int(args.quality_random_state),
        shift_type=args.shift_type,
        shift_fraction=float(args.shift_fraction),
        shuffle_test_seed=int(args.shuffle_test_seed),
        results_dir=root / str(args.results_dir),
    )

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    allowed = {"hotpotqa", "musique", "2wiki"}
    bad = [d for d in datasets if d not in allowed]
    if bad:
        raise ValueError(f"不支持的数据集：{bad}")

    ckpt_opt = Path(args.probe_checkpoint) if str(args.probe_checkpoint).strip() else None
    if ckpt_opt is not None and not ckpt_opt.is_absolute():
        ckpt_opt = root / ckpt_opt

    all_out: Dict[str, Any] = {}
    for ds in datasets:
        LOGGER.info("===== Stage3 dataset: %s =====", ds)
        all_out[ds] = run_dataset_stage3(s3, ds, probe_checkpoint=ckpt_opt)

    LOGGER.info("Stage3 全部完成。")


if __name__ == "__main__":
    main()
