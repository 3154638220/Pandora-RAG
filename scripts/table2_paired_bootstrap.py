"""Paired bootstrap for paper Table-2 style contrasts using cached trajectories only.

Rebuilds the same per-example rows as ``stage3_significance`` (Probe / Probe+E-value),
plus **best Fixed-K** and **Global-Weitzman** on the *same* Stage-3 test permutation,
and optionally **Stop-RAG** per-example F1 from an aligned online ``*.jsonl``.

Does not call the LLM or retrain probes; loads trajectories from ``cache/trajectories``
and runs probe / quality-model forward passes like ``stage3_significance``.

Example:
  python scripts/table2_paired_bootstrap.py --root-dir . --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretest.utils.weitzman import (
    compute_all_reservation_values,
    oracle_stopping_simulation,
    trajectory_cumulative_cost,
)
from stage2.run_stage2 import Stage2Config, _eval_fixed_k, _load_trajectories
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
    simulate_evalue_outcome_aware,
    simulate_evalue_gated_stops,
)
from stage3_significance import (
    _as_arrays,
    _paired_delta_ci,
    _paired_permutation_pvalue,
    _stage2_cfg,
)

LOGGER = logging.getLogger(__name__)

# Default aligned Stop-RAG online jsonl (highest Pareto F1 point per dataset in
# ``results/stop_rag_online_pareto_frontier.csv`` for this checkout).
DEFAULT_STOP_RAG_JSONL: Dict[str, str] = {
    "hotpotqa": (
        "results/stop_rag_sweep_backup/raw_online_test/"
        "hotpotqa_ours_contriever/online_test/hotpotqa_test_ckpt1000_thr-0.03.jsonl"
    ),
    "musique": (
        "results/stop_rag_sweep_backup/raw_online_test/"
        "musique_ours_contriever/online_test/musique_test_ckpt1200_thr-0.03.jsonl"
    ),
    "2wiki": (
        "results/stop_rag_sweep_backup/raw_online_test/"
        "2wikimultihopqa_ours_contriever/online_test/"
        "2wikimultihopqa_test_ckpt2400_thr-0.06.jsonl"
    ),
}


def _global_weitzman_rows(
    ordered_test: Sequence[Mapping[str, Any]],
    train_traj: Sequence[Mapping[str, Any]],
    cfg: Stage2Config,
) -> List[Dict[str, Any]]:
    reservation_values = compute_all_reservation_values(
        list(train_traj), cfg.max_k, cfg.cost_per_step
    )
    raw = oracle_stopping_simulation(list(ordered_test), reservation_values, cfg.max_k)
    out: List[Dict[str, Any]] = []
    for traj, result in zip(ordered_test, raw):
        used = int(result.get("steps_used", 0))
        cum_cost = trajectory_cumulative_cost(
            traj, used, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        out.append(
            {
                "f1": float(result.get("f1", 0.0)),
                "em": int(bool(result.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    return out


def _pick_best_fixed_k_rows(
    ordered_test: Sequence[Mapping[str, Any]],
    cfg: Stage2Config,
) -> Tuple[int, List[Dict[str, Any]]]:
    best_k = 1
    best_mean = -1.0
    best_rows: List[Dict[str, Any]] = []
    for k in range(1, cfg.max_k + 1):
        rows_k = _eval_fixed_k(list(ordered_test), k, cfg)
        m = float(np.mean([float(r.get("f1", 0.0)) for r in rows_k]))
        if m > best_mean:
            best_mean = m
            best_k = k
            best_rows = rows_k
    return best_k, best_rows


def _load_stop_rag_f1_map(path: Path) -> Dict[str, float]:
    m: Dict[str, float] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id") or "")
            if qid:
                m[qid] = float(row.get("f1", 0.0))
    return m


def _align_stop_rag_f1(
    ordered_test: Sequence[Mapping[str, Any]],
    id_to_f1: Mapping[str, float],
) -> np.ndarray:
    vals: List[float] = []
    for traj in ordered_test:
        sid = str(traj.get("id", ""))
        if sid not in id_to_f1:
            raise KeyError(f"Stop-RAG jsonl missing question_id={sid!r}")
        vals.append(id_to_f1[sid])
    return np.asarray(vals, dtype=np.float64)


def _simulate_extended(
    s3: Stage3Config,
    dataset: str,
    gamma: float,
    alpha: float,
    probe_checkpoint: Optional[Path],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, List[Dict[str, Any]]]:
    """Returns meta, f1_probe, f1_evalue, f1_best_fixed_k, f1_gw, best_k, ordered_test."""
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
    train_traj = _load_trajectories(cfg2, dataset, "train")

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

    best_k, fixed_rows = _pick_best_fixed_k_rows(ordered_test, cfg2)
    gw_rows = _global_weitzman_rows(ordered_test, train_traj, cfg2)

    f1_probe = _as_arrays(baseline_ordered).f1
    f1_evalue = _as_arrays(ev_rows).f1
    f1_fixed = _as_arrays(fixed_rows).f1
    f1_gw = _as_arrays(gw_rows).f1

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
        "best_fixed_k": int(best_k),
        "best_fixed_mean_f1": float(f1_fixed.mean()),
        "global_weitzman_mean_f1": float(f1_gw.mean()),
    }
    return meta, f1_probe, f1_evalue, f1_fixed, f1_gw, best_k, list(ordered_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Table-2 style paired bootstrap from cached trajectories")
    p.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--artifact-suffix", type=str, default="pdopt_best")
    p.add_argument("--probe-checkpoint", type=str, default="")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n-bootstrap", type=int, default=10000)
    p.add_argument("--n-permutation", type=int, default=10000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--betting-strategy", type=str, default="predictive", choices=("fixed", "predictive"))
    p.add_argument("--betting-lambda", type=float, default=0.5)
    p.add_argument("--no-outcome-aware", action="store_true")
    p.add_argument("--shift-type", type=str, default="none", choices=("none", "sudden", "gradual", "periodic"))
    p.add_argument("--shuffle-test-seed", type=int, default=42)
    p.add_argument(
        "--stop-rag-jsonl",
        type=str,
        default="",
        help="Optional override: path to Stop-RAG online test jsonl (per-dataset if single run).",
    )
    p.add_argument(
        "--skip-stop-rag",
        action="store_true",
        help="Do not load Stop-RAG jsonl (only Probe / Fixed-K / GW / E-value contrasts).",
    )
    return p.parse_args()


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
    rng = np.random.default_rng(int(args.seed))

    pair_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        meta, f1_probe, f1_evalue, f1_fixed, f1_gw, best_k, ordered_test = _simulate_extended(
            s3,
            dataset,
            gamma=float(args.gamma),
            alpha=float(args.alpha),
            probe_checkpoint=probe_checkpoint,
        )
        LOGGER.info(
            "%s best_fixed_k=%d fixed_mean_f1=%.4f gw_mean_f1=%.4f probe_mean_f1=%.4f evalue_mean_f1=%.4f",
            dataset,
            best_k,
            float(f1_fixed.mean()),
            float(f1_gw.mean()),
            float(f1_probe.mean()),
            float(f1_evalue.mean()),
        )

        comparisons: List[Tuple[str, np.ndarray, np.ndarray]] = [
            ("Probe - Best Fixed-K", f1_probe, f1_fixed),
            ("Probe - Global-Weitzman", f1_probe, f1_gw),
            ("Probe+E-value - Probe", f1_evalue, f1_probe),
        ]

        if not args.skip_stop_rag:
            if args.stop_rag_jsonl:
                sr_path = Path(args.stop_rag_jsonl)
            else:
                rel = DEFAULT_STOP_RAG_JSONL.get(dataset)
                if not rel:
                    raise ValueError(f"No default Stop-RAG jsonl for dataset {dataset!r}")
                sr_path = root / rel
            id_to_f1 = _load_stop_rag_f1_map(sr_path)
            f1_sr = _align_stop_rag_f1(ordered_test, id_to_f1)
            meta["stop_rag_jsonl"] = str(sr_path)
            meta["stop_rag_mean_f1"] = float(f1_sr.mean())
            comparisons.append(("Probe+E-value - Stop-RAG", f1_evalue, f1_sr))

        for name, a, b in comparisons:
            d_mean, d_lo, d_hi, d_std = _paired_delta_ci(
                a, b, rng, int(args.n_bootstrap)
            )
            p_val = _paired_permutation_pvalue(a, b, rng, int(args.n_permutation))
            lo_excl = d_lo > 0.0 or d_hi < 0.0
            pair_rows.append(
                {
                    "dataset": dataset,
                    "comparison": name,
                    "n": int(f1_probe.size),
                    "delta_f1_mean": d_mean,
                    "delta_f1_ci_low": d_lo,
                    "delta_f1_ci_high": d_hi,
                    "delta_f1_bootstrap_std": d_std,
                    "paired_permutation_p": p_val,
                    "ci_excludes_zero": lo_excl,
                }
            )

    results_dir = root / str(args.results_dir)
    csv_path = results_dir / "table2_paired_bootstrap.csv"

    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "comparison",
                "n",
                "delta_f1_mean",
                "delta_f1_ci_low",
                "delta_f1_ci_high",
                "delta_f1_bootstrap_std",
                "paired_permutation_p",
                "ci_excludes_zero",
            ],
        )
        w.writeheader()
        for row in pair_rows:
            w.writerow(row)

    LOGGER.info("Wrote %s", csv_path)
    print("\nTable 2 配对 Bootstrap（ΔF1 = 前者 − 后者；与 stage3_significance 同排列）\n")
    print(f"{'Dataset':<10} {'Comparison':<28} {'N':>5} {'ΔF1':>10} {'95% CI':^26} {'p':>8} {'≠0':>4}")
    for r in pair_rows:
        dm = float(r["delta_f1_mean"])
        lo = float(r["delta_f1_ci_low"])
        hi = float(r["delta_f1_ci_high"])
        flag = "yes" if r["ci_excludes_zero"] else "no"
        print(
            f"{r['dataset']:<10} {r['comparison']:<28} {int(r['n']):>5} "
            f"{dm:+.4f}   [{lo:+.4f}, {hi:+.4f}] "
            f"{float(r['paired_permutation_p']):>8.4f} {flag:>4}"
        )


if __name__ == "__main__":
    main()
