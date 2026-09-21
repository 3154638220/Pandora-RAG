"""
Phase D2：在 Stage2 与 MLP 相同的 (浅层+hidden, y, margin 权重) 上训练 XGBoost，网格搜索超参，
dev 上用与 Neural Probe 相同的 Phase C 阈值逻辑，test 上汇报 F1 / 步数，并与 Global-Weitzman 对照。

产出：docs/reports/stage2/stage2_xgboost_baseline.md、results/stage2_xgboost_baseline.json

用法：
  python -m stage2.run_xgboost_baseline --datasets hotpotqa,musique,2wiki
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from pretest.utils.weitzman import (
    compute_all_reservation_values,
    oracle_stopping_simulation,
    trajectory_cumulative_cost,
)
from stage2.run_stage2 import (
    SHALLOW_FEATURE_DIM,
    Stage2Config,
    _build_step_feature_map,
    _build_xyw,
    _ensure_dirs,
    _eval_fixed_k,
    _eval_oracle_rows,
    _global_weitzman_avg_steps,
    _infer_hidden_dim,
    _load_hidden_map,
    _load_trajectories,
    _make_oracle_maps,
    _pick_best_threshold,
    _set_seed,
    _simulate_probe_policy_from_probs,
    _summarize_results,
    _write_json,
)

LOGGER = logging.getLogger(__name__)

GRID_MAX_DEPTH = (4, 6, 8)
GRID_N_ESTIMATORS = (100, 300, 500)
GRID_LEARNING_RATE = (0.05, 0.1)
PROBA_BATCH = 16384


def _resolve_xgb_device(mode: str) -> str:
    m = str(mode or "auto").strip().lower()
    if m in ("cpu", "cuda"):
        return m
    if m != "auto":
        raise ValueError(f"不支持的 --xgb-device={mode}，可选 auto/cpu/cuda")
    try:
        probe = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        probe.get_params()
        return "cuda"
    except Exception:
        return "cpu"


def _precompute_xgb_probs(
    step_feature_map: Dict[Tuple[str, int], np.ndarray],
    scaler: StandardScaler,
    clf: xgb.XGBClassifier,
) -> Dict[Tuple[str, int], float]:
    if not step_feature_map:
        return {}
    keys = list(step_feature_map.keys())
    out: Dict[Tuple[str, int], float] = {}
    for i in range(0, len(keys), PROBA_BATCH):
        batch_keys = keys[i : i + PROBA_BATCH]
        feats = np.stack([step_feature_map[k] for k in batch_keys], axis=0)
        xs = feats[:, :SHALLOW_FEATURE_DIM].astype(np.float32, copy=False)
        xh = feats[:, SHALLOW_FEATURE_DIM:].astype(np.float32, copy=False)
        xs_s = scaler.transform(xs).astype(np.float32)
        if xh.shape[1] > 0:
            x_all = np.hstack([xs_s, xh])
        else:
            x_all = xs_s
        p1 = clf.predict_proba(x_all)[:, 1]
        for kk, pv in zip(batch_keys, p1):
            out[kk] = float(pv)
    return out


def _run_one_dataset(cfg: Stage2Config, dataset: str) -> Dict[str, Any]:
    train_traj = _load_trajectories(cfg, dataset, "train")
    dev_traj = _load_trajectories(cfg, dataset, "dev")
    test_traj = _load_trajectories(cfg, dataset, "test")

    if cfg.shallow_only:
        hidden_dim = 0
        train_h: Dict[Tuple[str, int], np.ndarray] = {}
        dev_h = {}
        test_h = {}
    else:
        hidden_dim = _infer_hidden_dim(cfg, dataset, "train", cfg.hidden_state_key)
        train_h = _load_hidden_map(cfg, dataset, "train", hidden_dim, cfg.hidden_state_key)
        dev_h = _load_hidden_map(cfg, dataset, "dev", hidden_dim, cfg.hidden_state_key)
        test_h = _load_hidden_map(cfg, dataset, "test", hidden_dim, cfg.hidden_state_key)

    train_oracle_map, _ = _make_oracle_maps(train_traj, cfg)
    dev_oracle_map, _ = _make_oracle_maps(dev_traj, cfg)
    test_oracle_map, test_oracle_rows = _make_oracle_maps(test_traj, cfg)

    x_tr_s, x_tr_h, y_tr, w_tr, _ = _build_xyw(train_traj, train_oracle_map, train_h, hidden_dim, cfg)
    x_dv_s, x_dv_h, y_dv, w_dv, _ = _build_xyw(dev_traj, dev_oracle_map, dev_h, hidden_dim, cfg)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr_s).astype(np.float32)
    x_dv_s = scaler.transform(x_dv_s).astype(np.float32)
    if hidden_dim > 0:
        x_tr = np.hstack([x_tr_s, x_tr_h.astype(np.float32, copy=False)])
        x_dv_fit = np.hstack([x_dv_s, x_dv_h.astype(np.float32, copy=False)])
    else:
        x_tr = x_tr_s
        x_dv_fit = x_dv_s

    dev_step_map = _build_step_feature_map(dev_traj, dev_h, hidden_dim, cfg)
    test_step_map = _build_step_feature_map(test_traj, test_h, hidden_dim, cfg)

    gw_dev_steps = _global_weitzman_avg_steps(train_traj, dev_traj, cfg)
    xgb_device = _resolve_xgb_device(getattr(cfg, "xgb_device", "auto"))
    cpu_threads = max(1, (os.cpu_count() or 1))
    LOGGER.info("%s: XGBoost device=%s", dataset, xgb_device)

    best_dev_f1 = -1.0
    best: Dict[str, Any] = {}
    grid_logs: List[Dict[str, Any]] = []

    for md, ne, lr in product(GRID_MAX_DEPTH, GRID_N_ESTIMATORS, GRID_LEARNING_RATE):
        clf = xgb.XGBClassifier(
            max_depth=int(md),
            n_estimators=int(ne),
            learning_rate=float(lr),
            objective="binary:logistic",
            random_state=int(cfg.seed),
            tree_method="hist",
            device=xgb_device,
            n_jobs=cpu_threads if xgb_device == "cpu" else 1,
            eval_metric="logloss",
            early_stopping_rounds=40,
        )
        clf.fit(
            x_tr,
            y_tr,
            sample_weight=w_tr.astype(np.float32, copy=False),
            eval_set=[(x_dv_fit, y_dv)],
            verbose=False,
        )
        dv_probs = _precompute_xgb_probs(dev_step_map, scaler, clf)
        thr, best_dev_row, thr_diag = _pick_best_threshold(
            train_traj, dev_traj, dv_probs, cfg, gw_dev_avg_steps_precomputed=gw_dev_steps
        )
        dev_f1 = float(best_dev_row["avg_f1"])
        grid_logs.append(
            {
                "max_depth": int(md),
                "n_estimators": int(ne),
                "learning_rate": float(lr),
                "dev_f1_after_phase_c": dev_f1,
                "dev_avg_steps": float(best_dev_row["avg_steps"]),
                "threshold": float(thr),
            }
        )
        if dev_f1 > best_dev_f1 + 1e-12:
            best_dev_f1 = dev_f1
            te_probs = _precompute_xgb_probs(test_step_map, scaler, clf)
            best = {
                "max_depth": int(md),
                "n_estimators": int(ne),
                "learning_rate": float(lr),
                "classifier": clf,
                "scaler": scaler,
                "threshold": float(thr),
                "best_dev_row": dict(best_dev_row),
                "threshold_diag": thr_diag,
                "test_probs": te_probs,
                "dev_probs": dv_probs,
            }

    if not best:
        raise RuntimeError(f"{dataset}: XGBoost 网格搜索未产生有效模型。")

    clf = best["classifier"]
    thr = float(best["threshold"])
    te_probs = best["test_probs"]

    probe_rows = _simulate_probe_policy_from_probs(test_traj, thr, cfg, te_probs)
    probe_summary = _summarize_results(probe_rows, "XGBoost-Probe")

    reservation_values = compute_all_reservation_values(train_traj, cfg.max_k, cfg.cost_per_step)
    gw_raw = oracle_stopping_simulation(test_traj, reservation_values, cfg.max_k)
    gw_rows: List[Dict[str, Any]] = []
    for traj, result in zip(test_traj, gw_raw):
        used = int(result.get("steps_used", 0))
        cum_cost = trajectory_cumulative_cost(
            traj, used, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        gw_rows.append(
            {
                "f1": float(result.get("f1", 0.0)),
                "em": int(bool(result.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    gw_summary = _summarize_results(gw_rows, "Global-Weitzman")

    oracle_eval = _eval_oracle_rows(test_traj, test_oracle_rows, cfg)
    oracle_summary = _summarize_results(oracle_eval, "Oracle")

    fixed_rows: List[Dict[str, Any]] = []
    for k in range(1, cfg.max_k + 1):
        fixed_rows.append(_summarize_results(_eval_fixed_k(test_traj, k, cfg), f"Fixed-K={k}"))
    best_fixed = max((r["avg_f1"] for r in fixed_rows), default=0.0)

    mlp_table = cfg.results_dir / f"stage2_probe_table_{dataset}.csv"
    mlp_probe_f1: Any = None
    mlp_steps: Any = None
    if mlp_table.exists():
        try:
            df_m = pd.read_csv(mlp_table)
            pr = df_m[df_m["strategy"] == "Probe"]
            if len(pr) == 1:
                mlp_probe_f1 = float(pr.iloc[0]["avg_f1"])
                mlp_steps = float(pr.iloc[0]["avg_steps"])
        except Exception:
            pass

    return {
        "dataset": dataset,
        "hidden_dim": int(hidden_dim),
        "shallow_only": bool(cfg.shallow_only),
        "best_hparams": {
            "max_depth": best["max_depth"],
            "n_estimators": best["n_estimators"],
            "learning_rate": best["learning_rate"],
        },
        "xgb_device": xgb_device,
        "threshold": thr,
        "threshold_selection_phase_c": best["threshold_diag"],
        "dev_after_selection": best["best_dev_row"],
        "test_probe": probe_summary,
        "test_global_weitzman": gw_summary,
        "test_oracle": oracle_summary,
        "best_fixed_f1": float(best_fixed),
        "probe_gain_over_best_fixed": float(probe_summary["avg_f1"] - best_fixed),
        "grid_search_log": grid_logs,
        "mlp_default_table_probe_f1": mlp_probe_f1,
        "mlp_default_table_avg_steps": mlp_steps,
    }


def _markdown_report(results: Dict[str, Dict[str, Any]], cfg: Stage2Config) -> str:
    lines: List[str] = [
        "# Stage 2 Phase D2：XGBoost 基线（与 MLP 同特征）",
        "",
        f"- 根目录：`{cfg.root_dir.resolve()}`",
        f"- 网格：`max_depth ∈ {GRID_MAX_DEPTH}`，`n_estimators ∈ {GRID_N_ESTIMATORS}`，`learning_rate ∈ {GRID_LEARNING_RATE}`",
        f"- 阈值：与 Neural Probe 相同 Phase C（GW dev 步数上界 + Pareto λ 网格）",
        f"- 浅层：`StandardScaler` fit on train；hidden 原样拼接（与 MLP 输入一致）",
        "",
        "## 决策对照（plan D2）",
        "",
        "| 数据集 | XGB test F1 | MLP test F1（默认表） | XGB 步数 | MLP 步数 | GW F1 | 解读 |",
        "|--------|------------|----------------------|----------|----------|-------|------|",
    ]
    for ds in sorted(results.keys()):
        r = results[ds]
        xp = r["test_probe"]["avg_f1"]
        xs = r["test_probe"]["avg_steps"]
        gw = r["test_global_weitzman"]["avg_f1"]
        mf = r["mlp_default_table_probe_f1"]
        ms = r["mlp_default_table_avg_steps"]
        mf_s = f"{mf:.4f}" if mf is not None else "—（无 `stage2_probe_table_{ds}.csv`）"
        ms_s = f"{ms:.3f}" if ms is not None else "—"

        if mf is None:
            verdict = "缺 MLP 默认表，无法对比"
        elif abs(xp - mf) < 0.01:
            verdict = "≈ MLP → **特征可能是瓶颈**（→ D4）"
        elif xp > mf + 0.02:
            verdict = "XGB 明显优于 MLP → **MLP 训练/容量可能是瓶颈**（→ D5）"
        else:
            verdict = "略优于或略差于 MLP，需结合方差判断"

        lines.append(
            f"| {ds} | {xp:.4f} | {mf_s} | {xs:.3f} | {ms_s} | {gw:.4f} | {verdict} |"
        )

    lines.extend(
        [
            "",
            "## 各数据集最优超参与 dev/test",
            "",
        ]
    )
    for ds in sorted(results.keys()):
        r = results[ds]
        hp = r["best_hparams"]
        lines.append(f"### {ds}")
        lines.append("")
        lines.append(
            f"- 最优：`max_depth={hp['max_depth']}`, `n_estimators={hp['n_estimators']}`, `lr={hp['learning_rate']}`"
        )
        lines.append(f"- XGBoost device：`{r.get('xgb_device', '—')}`")
        lines.append(f"- 选中阈值：`{r['threshold']:.4f}`（Phase C）")
        lines.append(
            f"- Dev（选完阈值后）：F1={r['dev_after_selection']['avg_f1']:.4f}, avg_steps={r['dev_after_selection']['avg_steps']:.3f}"
        )
        tp = r["test_probe"]
        lines.append(
            f"- Test：F1={tp['avg_f1']:.4f}, avg_steps={tp['avg_steps']:.3f}, avg_cost={tp['avg_cost']:.4f}"
        )
        lines.append("")

    lines.append("## 网格搜索全记录（dev F1 after Phase C）")
    lines.append("")
    for ds in sorted(results.keys()):
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| depth | n_est | lr | dev F1 | dev steps | thr |")
        lines.append("|------|-------|-----|--------|-----------|-----|")
        for row in results[ds]["grid_search_log"]:
            lines.append(
                f"| {row['max_depth']} | {row['n_estimators']} | {row['learning_rate']} | "
                f"{row['dev_f1_after_phase_c']:.4f} | {row['dev_avg_steps']:.3f} | {row['threshold']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 D2 XGBoost baseline")
    p.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--cost-per-step", type=float, default=0.05)
    p.add_argument("--oracle-cost-metric", type=str, default="fixed")
    p.add_argument("--hidden-state-key", type=str, default="last_token")
    p.add_argument(
        "--gw-steps-cap-mult",
        type=float,
        default=1.05,
        help="与 run_stage2 Phase C 一致。",
    )
    p.add_argument(
        "--pareto-lambdas",
        type=str,
        default="0.1,0.3,0.5,1.0",
    )
    p.add_argument("--shallow-only", action="store_true")
    p.add_argument(
        "--xgb-device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="XGBoost 设备：auto 优先 CUDA（不可用时回退 CPU）。",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    allowed = {"hotpotqa", "musique", "2wiki"}
    unknown = [d for d in datasets if d not in allowed]
    if unknown:
        raise ValueError(f"不支持的数据集：{unknown}")

    pl_parts = [p.strip() for p in str(args.pareto_lambdas).split(",") if p.strip()]
    pareto_lambdas: Tuple[float, ...] = tuple(float(x) for x in pl_parts) if pl_parts else (0.1, 0.3, 0.5, 1.0)

    cfg = Stage2Config(
        seed=int(args.seed),
        max_k=int(args.max_k),
        cost_per_step=float(args.cost_per_step),
        oracle_cost_metric=str(args.oracle_cost_metric),
        root_dir=Path(args.root_dir),
        hidden_state_key=str(args.hidden_state_key),
        shallow_only=bool(args.shallow_only),
        threshold_gw_steps_cap_mult=float(args.gw_steps_cap_mult),
        threshold_pareto_lambdas=pareto_lambdas,
    )
    setattr(cfg, "xgb_device", str(args.xgb_device))
    _set_seed(cfg.seed)
    _ensure_dirs(cfg, datasets)

    results: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        LOGGER.info("===== XGBoost baseline: %s =====", ds)
        results[ds] = _run_one_dataset(cfg, ds)

    out_md = cfg.docs_dir / "stage2_xgboost_baseline.md"
    out_json = cfg.results_dir / "stage2_xgboost_baseline.json"

    serializable = {
        ds: {k: v for k, v in r.items() if k not in ("classifier", "scaler", "test_probs", "dev_probs")}
        for ds, r in results.items()
    }

    def _json_cfg(c: Stage2Config) -> Dict[str, Any]:
        d = asdict(c)
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, Path):
                out[k] = str(v)
            elif isinstance(v, tuple):
                out[k] = list(v)
            else:
                out[k] = v
        return out

    _write_json(out_json, {"config": _json_cfg(cfg), "results": serializable})
    out_md.write_text(_markdown_report(results, cfg), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", out_md, out_json)


if __name__ == "__main__":
    main()
