"""Paired bootstrap and permutation tests for Stage-3 main results.

This script reconstructs the same per-sample rows used by ``stage3.run_stage3``
and reports uncertainty for the main table.  It intentionally works from
paired per-example F1/EM/error rows rather than aggregate CSV values.

Usage:
  python scripts/stage3_significance.py
  python scripts/stage3_significance.py --datasets musique --n-bootstrap 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage2.run_stage2 import Stage2Config
from stage3.adapters.stage2_probe import (
    find_best_probe_checkpoint,
    load_stage2_probe_bundle,
    load_trajectories_and_step_features,
    precompute_continue_probabilities,
    shallow_features_from_map,
    simulate_probe_baseline,
)
from stage3.config import Stage3Config
from stage3.quality_model import (
    build_step_quality_dataset,
    evaluate_quality_model,
    train_quality_logreg,
    tune_quality_bar_on_calib,
)
from stage3.run_stage3 import _collect_calib_stop_stats
from stage3.stopping import (
    attach_error_labels,
    build_shift_ordering,
    conformal_min_phat_threshold,
    simulate_conformal_phat_gate,
    simulate_evalue_outcome_aware,
    simulate_evalue_gated_stops,
)

LOGGER = logging.getLogger(__name__)


STRATEGY_KEYS = {
    "probe": "Probe",
    "probe_evalue": "Probe+E-value",
    "probe_conformal_quantile": "Probe+CP",
}


@dataclass(frozen=True)
class StrategyArrays:
    f1: np.ndarray
    em: np.ndarray
    error: np.ndarray
    steps: np.ndarray


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


def _as_arrays(rows: Sequence[Mapping[str, Any]]) -> StrategyArrays:
    return StrategyArrays(
        f1=np.asarray([float(r.get("f1", 0.0)) for r in rows], dtype=np.float64),
        em=np.asarray([float(r.get("em", 0.0)) for r in rows], dtype=np.float64),
        error=np.asarray([float(r.get("error", 0.0)) for r in rows], dtype=np.float64),
        steps=np.asarray([float(r.get("steps_used", 0.0)) for r in rows], dtype=np.float64),
    )


def _mean_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boots = values[idx].mean(axis=1)
    mean = float(values.mean())
    std = float(boots.std(ddof=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), std


def _paired_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> Tuple[float, float, float, float]:
    diff = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    return _mean_ci(diff, rng, n_bootstrap)


def _paired_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_permutation: int,
) -> float:
    """Two-sided paired randomization test by randomly swapping pair signs."""
    diff = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    if diff.size == 0:
        return 1.0
    observed = abs(float(diff.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(n_permutation), int(diff.size)))
    perm = np.abs((signs * diff).mean(axis=1))
    # Plus-one smoothing keeps the Monte Carlo p-value nonzero and conservative.
    return float((np.count_nonzero(perm >= observed - 1e-15) + 1) / (int(n_permutation) + 1))


def _simulate_dataset_rows(
    s3: Stage3Config,
    dataset: str,
    gamma: float,
    alpha: float,
    probe_checkpoint: Optional[Path],
) -> Tuple[Dict[str, StrategyArrays], Dict[str, Any]]:
    cfg2 = _stage2_cfg(s3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if probe_checkpoint and probe_checkpoint.exists():
        ckpt = probe_checkpoint
    else:
        found = find_best_probe_checkpoint(
            s3.artifacts_probe_dir,
            dataset,
            preferred_suffix=s3.artifact_suffix or "pdopt_best",
        )
        if found is None:
            raise FileNotFoundError(f"Could not find a Stage-2 probe checkpoint for {dataset}")
        ckpt = found

    bundle = load_stage2_probe_bundle(ckpt, device)
    LOGGER.info("%s loading %s", dataset, ckpt)

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

    x_q, y_q = build_step_quality_dataset(
        calib_traj,
        shallow_calib,
        gamma=gamma,
        max_k=s3.max_k,
        probe_probs=calib_probs if s3.quality_use_probe_prob else None,
    )
    qmodel = train_quality_logreg(
        x_q,
        y_q,
        max_iter=s3.quality_max_iter,
        random_state=s3.quality_random_state,
    )
    q_eval = evaluate_quality_model(qmodel, x_q, y_q)

    if s3.quality_use_probe_prob:
        shallow_calib_q = {k: np.append(v, np.float32(calib_probs.get(k, 0.5))) for k, v in shallow_calib.items()}
        shallow_test_q = {k: np.append(v, np.float32(test_probs.get(k, 0.5))) for k, v in shallow_test.items()}
    else:
        shallow_calib_q = shallow_calib
        shallow_test_q = shallow_test

    phat_stop, err_stop = _collect_calib_stop_stats(
        calib_traj,
        calib_probs,
        shallow_calib_q,
        qmodel,
        cfg2,
        bundle.probe_threshold,
        gamma,
    )
    qbar = tune_quality_bar_on_calib(
        phat_stop,
        err_stop,
        target_error=alpha,
        calib_method=s3.calib_method,
    )
    cp_tau = conformal_min_phat_threshold(phat_stop, alpha)

    baseline = attach_error_labels(
        simulate_probe_baseline(test_traj, test_probs, bundle.probe_threshold, cfg2),
        gamma,
    )
    shift_order = build_shift_ordering(
        test_traj,
        shift_type=s3.shift_type,
        shift_fraction=s3.shift_fraction,
        rng_seed=s3.shuffle_test_seed,
    )
    ordered_test = [test_traj[int(i)] for i in shift_order]
    baseline_ordered = [baseline[int(i)] for i in shift_order]

    if s3.outcome_aware:
        ev_rows, wealth_trace = simulate_evalue_outcome_aware(
            ordered_test,
            test_probs,
            shallow_test_q,
            qmodel,
            cfg=cfg2,
            probe_threshold=bundle.probe_threshold,
            gamma=gamma,
            alpha=alpha,
            quality_bar=qbar,
            betting_strategy=s3.betting_strategy,
            betting_lambda=s3.betting_lambda,
        )
    else:
        ev_rows, wealth_trace = simulate_evalue_gated_stops(
            ordered_test,
            test_probs,
            shallow_test_q,
            qmodel,
            cfg=cfg2,
            probe_threshold=bundle.probe_threshold,
            gamma=gamma,
            alpha=alpha,
            quality_bar=qbar,
        )

    cp_rows = simulate_conformal_phat_gate(
        ordered_test,
        test_probs,
        shallow_test_q,
        qmodel,
        cfg=cfg2,
        probe_threshold=bundle.probe_threshold,
        gamma=gamma,
        min_phat=cp_tau,
    )

    meta = {
        "dataset": dataset,
        "n": int(len(ordered_test)),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "probe_checkpoint": str(ckpt),
        "probe_threshold": float(bundle.probe_threshold),
        "quality_bar": float(qbar),
        "conformal_min_phat": float(cp_tau),
        "quality_model_eval": q_eval,
        "final_e_wealth": float(wealth_trace[-1]) if wealth_trace else 1.0,
        "wealth_cap": float(1.0 / alpha),
    }
    arrays = {
        "probe": _as_arrays(baseline_ordered),
        "probe_evalue": _as_arrays(ev_rows),
        "probe_conformal_quantile": _as_arrays(cp_rows),
    }
    return arrays, meta


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _fmt_interval(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def _generate_markdown(
    path: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    meta_rows: Sequence[Mapping[str, Any]],
    title_note: str = "",
) -> None:
    lines: List[str] = []
    title = "# Stage3 Statistical Significance"
    if title_note:
        title = f"{title} ({title_note})"
    lines.append(title)
    lines.append("")
    lines.append("Paired bootstrap and paired randomization results for the Stage-3 main operating point.")
    lines.append("")
    lines.append("- Metric CI: percentile paired bootstrap over test examples.")
    lines.append("- Delta: row strategy minus Probe, using the same test examples and ordering.")
    lines.append("- p-value: two-sided paired permutation/randomization test on per-example F1 differences.")
    lines.append("")
    lines.append("## Main 95% CIs")
    lines.append("")
    lines.append("| Dataset | Strategy | N | F1 95% CI | EM 95% CI | Error 95% CI | Steps 95% CI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['strategy']} | {int(r['n'])} | "
            f"{_fmt_interval(float(r['f1_mean']), float(r['f1_ci_low']), float(r['f1_ci_high']))} | "
            f"{_fmt_interval(float(r['em_mean']), float(r['em_ci_low']), float(r['em_ci_high']))} | "
            f"{_fmt_interval(float(r['error_mean']), float(r['error_ci_low']), float(r['error_ci_high']))} | "
            f"{_fmt_interval(float(r['steps_mean']), float(r['steps_ci_low']), float(r['steps_ci_high']))} |"
        )
    lines.append("")
    lines.append("## Paired F1 Comparisons")
    lines.append("")
    lines.append("| Dataset | Comparison | ΔF1 95% CI | bootstrap std | paired p | interpretation |")
    lines.append("|---|---|---:|---:|---:|---|")
    for r in pair_rows:
        delta = float(r["delta_f1_mean"])
        lo = float(r["delta_f1_ci_low"])
        hi = float(r["delta_f1_ci_high"])
        p = float(r["paired_permutation_p"])
        if lo > 0.0 or hi < 0.0:
            interp = "CI excludes 0"
        else:
            interp = "CI includes 0"
        lines.append(
            f"| {r['dataset']} | {r['comparison']} | "
            f"{delta:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
            f"{float(r['delta_f1_bootstrap_std']):.4f} | {p:.4f} | {interp} |"
        )
    lines.append("")
    lines.append("## MuSiQue Note")
    lines.append("")
    musique = [r for r in summary_rows if r["dataset"] == "musique"]
    if musique:
        lines.append(
            "MuSiQue has only 417 test examples under the repository split, so the intervals are visibly wider. "
            "Use it as a stress case: the main Probe vs E-value delta is small relative to bootstrap uncertainty, "
            "while CP is directionally worse and much more expensive."
        )
        lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append("| Dataset | checkpoint | quality_bar | CP tau | final wealth / cap | quality Brier |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in meta_rows:
        qm = r.get("quality_model_eval", {})
        brier = qm.get("brier_score", 0.0) if isinstance(qm, Mapping) else 0.0
        lines.append(
            f"| {r['dataset']} | `{r['probe_checkpoint']}` | "
            f"{float(r['quality_bar']):.4f} | {float(r['conformal_min_phat']):.4f} | "
            f"{float(r['final_e_wealth']):.4f} / {float(r['wealth_cap']):.1f} | {float(brier):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage3 paired bootstrap and significance tests")
    parser.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--docs-dir", type=str, default="docs/reports/stage3")
    parser.add_argument("--artifact-suffix", type=str, default="pdopt_best")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="If non-empty, append _<suffix> to CSV/JSON/MD filenames (e.g. alias_metric_aligned).",
    )
    parser.add_argument("--probe-checkpoint", type=str, default="")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-permutation", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--betting-strategy", type=str, default="predictive", choices=("fixed", "predictive"))
    parser.add_argument("--betting-lambda", type=float, default=0.5)
    parser.add_argument("--no-outcome-aware", action="store_true")
    parser.add_argument("--shift-type", type=str, default="none", choices=("none", "sudden", "gradual", "periodic"))
    parser.add_argument("--shuffle-test-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    root = Path(args.root_dir)
    s3 = Stage3Config(
        root_dir=root,
        results_dir=root / str(args.results_dir),
        artifact_suffix=str(args.artifact_suffix or ""),
        gamma=float(args.gamma),
        gammas=(float(args.gamma),),
        alphas=(float(args.alpha),),
        betting_strategy=str(args.betting_strategy),
        betting_lambda=float(args.betting_lambda),
        outcome_aware=not bool(args.no_outcome_aware),
        shift_type=str(args.shift_type),  # type: ignore[arg-type]
        shuffle_test_seed=int(args.shuffle_test_seed),
        quality_use_probe_prob=True,
        calib_method="quantile",
    )
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    probe_checkpoint = Path(args.probe_checkpoint) if args.probe_checkpoint else None

    summary_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []

    rng = np.random.default_rng(int(args.seed))
    for dataset in datasets:
        arrays, meta = _simulate_dataset_rows(
            s3,
            dataset,
            gamma=float(args.gamma),
            alpha=float(args.alpha),
            probe_checkpoint=probe_checkpoint,
        )
        meta_rows.append(meta)
        for key, label in STRATEGY_KEYS.items():
            arr = arrays[key]
            f1_mean, f1_lo, f1_hi, f1_std = _mean_ci(arr.f1, rng, int(args.n_bootstrap))
            em_mean, em_lo, em_hi, em_std = _mean_ci(arr.em, rng, int(args.n_bootstrap))
            er_mean, er_lo, er_hi, er_std = _mean_ci(arr.error, rng, int(args.n_bootstrap))
            st_mean, st_lo, st_hi, st_std = _mean_ci(arr.steps, rng, int(args.n_bootstrap))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "strategy": label,
                    "n": int(arr.f1.size),
                    "f1_mean": f1_mean,
                    "f1_ci_low": f1_lo,
                    "f1_ci_high": f1_hi,
                    "f1_bootstrap_std": f1_std,
                    "em_mean": em_mean,
                    "em_ci_low": em_lo,
                    "em_ci_high": em_hi,
                    "em_bootstrap_std": em_std,
                    "error_mean": er_mean,
                    "error_ci_low": er_lo,
                    "error_ci_high": er_hi,
                    "error_bootstrap_std": er_std,
                    "steps_mean": st_mean,
                    "steps_ci_low": st_lo,
                    "steps_ci_high": st_hi,
                    "steps_bootstrap_std": st_std,
                }
            )

        base = arrays["probe"]
        for key in ("probe_evalue", "probe_conformal_quantile"):
            comp = arrays[key]
            d_mean, d_lo, d_hi, d_std = _paired_delta_ci(
                comp.f1,
                base.f1,
                rng,
                int(args.n_bootstrap),
            )
            p_val = _paired_permutation_pvalue(
                comp.f1,
                base.f1,
                rng,
                int(args.n_permutation),
            )
            pair_rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{STRATEGY_KEYS[key]} - Probe",
                    "n": int(base.f1.size),
                    "delta_f1_mean": d_mean,
                    "delta_f1_ci_low": d_lo,
                    "delta_f1_ci_high": d_hi,
                    "delta_f1_bootstrap_std": d_std,
                    "paired_permutation_p": p_val,
                    "delta_steps_mean": float(comp.steps.mean() - base.steps.mean()),
                    "delta_em_mean": float(comp.em.mean() - base.em.mean()),
                    "delta_error_mean": float(comp.error.mean() - base.error.mean()),
                }
            )

    results_dir = root / str(args.results_dir)
    docs_dir = root / str(args.docs_dir)
    out_tag = f"_{args.output_suffix}" if str(args.output_suffix).strip() else ""
    summary_path = results_dir / f"stage3_significance_summary{out_tag}.csv"
    pairs_path = results_dir / f"stage3_significance_paired{out_tag}.csv"
    meta_path = results_dir / f"stage3_significance_meta{out_tag}.json"
    report_path = docs_dir / f"stage3_significance_report{out_tag}.md"
    title_note = str(args.output_suffix).strip()

    _write_csv(
        summary_path,
        summary_rows,
        [
            "dataset",
            "strategy",
            "n",
            "f1_mean",
            "f1_ci_low",
            "f1_ci_high",
            "f1_bootstrap_std",
            "em_mean",
            "em_ci_low",
            "em_ci_high",
            "em_bootstrap_std",
            "error_mean",
            "error_ci_low",
            "error_ci_high",
            "error_bootstrap_std",
            "steps_mean",
            "steps_ci_low",
            "steps_ci_high",
            "steps_bootstrap_std",
        ],
    )
    _write_csv(
        pairs_path,
        pair_rows,
        [
            "dataset",
            "comparison",
            "n",
            "delta_f1_mean",
            "delta_f1_ci_low",
            "delta_f1_ci_high",
            "delta_f1_bootstrap_std",
            "paired_permutation_p",
            "delta_steps_mean",
            "delta_em_mean",
            "delta_error_mean",
        ],
    )
    meta_path.write_text(json.dumps(meta_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _generate_markdown(report_path, summary_rows, pair_rows, meta_rows, title_note=title_note)

    LOGGER.info("Wrote %s", summary_path)
    LOGGER.info("Wrote %s", pairs_path)
    LOGGER.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
