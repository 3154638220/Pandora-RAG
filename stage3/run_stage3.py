"""
Stage3 主入口：加载 Stage2 探针 → Calib 上训练质量头并校准阈值 → Test 上对比
Probe / E-value 门控 / 分位型 CP 门控，并输出 JSON + 累积错误率曲线。

Stage2 相关细节均经 ``stage3.adapters.stage2_probe``；本脚本只做编排。

Usage:
  python -m stage3.run_stage3 --datasets hotpotqa,musique,2wiki --gamma 0.5
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
    train_quality_logreg,
    tune_quality_bar_on_calib,
)
from stage3.stopping import (
    attach_error_labels,
    conformal_min_phat_threshold,
    probe_stop_shallow_and_phat,
    simulate_conformal_phat_gate,
    simulate_evalue_gated_stops,
    summarize,
)

LOGGER = logging.getLogger(__name__)


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


def run_dataset_stage3(
    s3: Stage3Config,
    dataset: str,
    *,
    probe_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg2 = _stage2_cfg(s3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = probe_checkpoint or default_probe_checkpoint_path(cfg2, dataset)
    if not ckpt.exists():
        raise FileNotFoundError(f"未找到 Stage2 探针 checkpoint：{ckpt}（请先跑 stage2 或指定 --probe-checkpoint）")

    bundle = load_stage2_probe_bundle(ckpt, device)
    LOGGER.info(
        "%s 加载探针 arch=%s shallow_only=%s threshold=%.4f",
        dataset,
        "mlp_v2" if bundle.dual_input else "mlp",
        bundle.shallow_only,
        bundle.probe_threshold,
    )

    calib_traj, calib_map = load_trajectories_and_step_features(cfg2, dataset, "calib", bundle)
    test_traj, test_map = load_trajectories_and_step_features(cfg2, dataset, "test", bundle)

    calib_probs = precompute_continue_probabilities(calib_map, bundle, cfg2)
    test_probs = precompute_continue_probabilities(test_map, bundle, cfg2)

    shallow_calib = shallow_features_from_map(calib_map, bundle.shallow_feature_dim)
    shallow_test = shallow_features_from_map(test_map, bundle.shallow_feature_dim)

    x_q, y_q = build_step_quality_dataset(
        calib_traj,
        shallow_calib,
        gamma=s3.gamma,
        max_k=s3.max_k,
    )
    qmodel = train_quality_logreg(
        x_q,
        y_q,
        max_iter=s3.quality_max_iter,
        random_state=s3.quality_random_state,
    )

    phat_stop, err_stop = _collect_calib_stop_stats(
        calib_traj,
        calib_probs,
        shallow_calib,
        qmodel,
        cfg2,
        bundle.probe_threshold,
        s3.gamma,
    )

    baseline = simulate_probe_baseline(
        test_traj, test_probs, bundle.probe_threshold, cfg2
    )
    baseline_tagged = attach_error_labels(baseline, s3.gamma)

    rng = np.random.RandomState(int(s3.shuffle_test_seed))
    order = rng.permutation(len(test_traj))
    ordered_test = [test_traj[int(i)] for i in order]
    baseline_ordered = [baseline_tagged[int(i)] for i in order]

    per_alpha: Dict[str, Any] = {}
    plot_paths: Dict[str, str] = {}

    for alpha in s3.alphas:
        a = float(alpha)
        qbar = tune_quality_bar_on_calib(phat_stop, err_stop, target_error=a)
        cp_tau = conformal_min_phat_threshold(phat_stop, a)

        ev_rows, wealth_tr = simulate_evalue_gated_stops(
            ordered_test,
            test_probs,
            shallow_test,
            qmodel,
            cfg=cfg2,
            probe_threshold=bundle.probe_threshold,
            gamma=s3.gamma,
            alpha=a,
            quality_bar=qbar,
        )
        cf_rows = simulate_conformal_phat_gate(
            ordered_test,
            test_probs,
            shallow_test,
            qmodel,
            cfg=cfg2,
            probe_threshold=bundle.probe_threshold,
            gamma=s3.gamma,
            min_phat=cp_tau,
        )

        curves = {
            "Probe": cumulative_error_rate_curve([r["error"] for r in baseline_ordered]),
            "Probe+E-value": cumulative_error_rate_curve([r["error"] for r in ev_rows]),
            "Probe+CP-quantile": cumulative_error_rate_curve([r["error"] for r in cf_rows]),
        }
        tag = str(a).replace(".", "_")
        png_name = f"stage3_cumulative_error_{dataset}_alpha{tag}.png"
        png_path = s3.results_dir / png_name
        _plot_cumulative_errors(
            curves,
            title=f"{dataset} cumulative error (α={a}, γ={s3.gamma})",
            out_path=png_path,
            alpha_line=a,
        )
        plot_paths[str(a)] = str(png_path)

        per_alpha[str(a)] = {
            "evalue_quality_bar": float(qbar),
            "conformal_min_phat": float(cp_tau),
            "summaries": {
                "probe": summarize(baseline_ordered, "Probe"),
                "probe_evalue": summarize(ev_rows, "Probe+E-value"),
                "probe_conformal_quantile": summarize(cf_rows, "Probe+CP-quantile"),
            },
            "cumulative_error_curves": {k: v for k, v in curves.items()},
            "final_e_wealth": float(wealth_tr[-1]) if wealth_tr else 1.0,
            "wealth_cap": float(1.0 / a),
            "plot_path": str(png_path),
        }

    out_bundle: Dict[str, Any] = {
        "dataset": dataset,
        "gamma": float(s3.gamma),
        "probe_checkpoint": str(ckpt),
        "alphas": [float(x) for x in s3.alphas],
        "shuffle_test_seed": int(s3.shuffle_test_seed),
        "per_alpha": per_alpha,
    }

    json_path = s3.results_dir / f"stage3_evalue_{dataset}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_bundle, f, indent=2, ensure_ascii=False)
    LOGGER.info("%s Stage3 结果已写入 %s", dataset, json_path)

    return {
        "dataset": dataset,
        "json_path": str(json_path),
        "plots": plot_paths,
        "per_alpha": per_alpha,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage3 E-value 风险控制实验")
    p.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--cost-per-step", type=float, default=0.05)
    p.add_argument(
        "--oracle-cost-metric",
        type=str,
        default="fixed",
        choices=("fixed", "token", "latency"),
    )
    p.add_argument("--hidden-state-key", type=str, default="last_token", choices=("last_token", "mean_pool"))
    p.add_argument("--artifact-suffix", type=str, default="", help="与 stage2 --artifact-suffix 一致")
    p.add_argument(
        "--probe-checkpoint",
        type=str,
        default="",
        help="可选：显式指定 probe_mlp*.pt；默认 artifacts/probe/{ds}/probe_mlp{tag}.pt",
    )
    p.add_argument("--gamma", type=float, default=0.5, help="F1 < γ 视为错误停止")
    p.add_argument("--alphas", type=str, default="0.1,0.2", help="逗号分隔名义水平")
    p.add_argument("--shuffle-test-seed", type=int, default=42)
    p.add_argument("--quality-max-iter", type=int, default=300)
    p.add_argument("--quality-random-state", type=int, default=42)
    p.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="JSON/PNG 输出目录（相对 root-dir）",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    alpha_parts = [x.strip() for x in str(args.alphas).split(",") if x.strip()]
    alphas = tuple(float(x) for x in alpha_parts) if alpha_parts else (0.1, 0.2)

    root = Path(args.root_dir)
    s3 = Stage3Config(
        root_dir=root,
        max_k=int(args.max_k),
        cost_per_step=float(args.cost_per_step),
        oracle_cost_metric=str(args.oracle_cost_metric),
        hidden_state_key=str(args.hidden_state_key),
        artifact_suffix=str(args.artifact_suffix or ""),
        gamma=float(args.gamma),
        alphas=alphas,
        shuffle_test_seed=int(args.shuffle_test_seed),
        quality_max_iter=int(args.quality_max_iter),
        quality_random_state=int(args.quality_random_state),
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
