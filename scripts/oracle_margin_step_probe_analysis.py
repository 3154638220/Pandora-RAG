#!/usr/bin/env python3
"""Oracle 逐步 margin m_k 分布 + near-tie 比例 + 分桶 Probe 步级 F1/错误率（不重训 LLM）。

依赖缓存轨迹 ``cache/trajectories`` 与 Probe checkpoint（与 Stage3/table2 一致）。

训练侧样本权重 w_k（实现见 ``stage2/run_stage2.py::_build_xyw``）::
    w_k = max(margin_weight_floor, |m_k|)，默认 margin_weight_floor=0.1。
    即：**clip 型下限**，等价于对 |m_k|<0.1 的步仍用 0.1 加权，而不是 bucket 离散权重。
    Huber（probe_target=f1）与 Focal BCE（probe_target=binary）均在逐步损失上乘以 w_k 后做加权平均。

Near-tie 分桶（针对 |m_k|）::
    ``lt_005``: |m_k| < 0.05
    ``05_01``:  0.05 <= |m_k| < 0.1
    ``ge_01``:  |m_k| >= 0.1

步级 Probe 指标::
    将 checkpoint 中 ``threshold`` / ``per_step_thresholds`` 与 Stage3 部署一致；
    预测 Continue 当且仅当 sigmoid(logit) >= threshold_k；
    与 oracle ``action_label``（margin>0 → Continue）比较，报告 accuracy、error_rate、binary F1；
    另报 **oracle_continue_rate**（正类先验）、**pred_continue_rate**、**TN/FP/FN/TP**、**precision/recall**
    （正类=Continue），便于诊断「F1 低但 accuracy 高」是 **recall 塌** 还是 **precision 塌**。

示例::
    python scripts/oracle_margin_step_probe_analysis.py --root-dir . --datasets hotpotqa,musique,2wiki
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretest.utils.weitzman import compute_trajectory_oracle
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from stage2.run_stage2 import Stage2Config
from stage3.adapters.stage2_probe import (
    find_best_probe_checkpoint,
    load_stage2_probe_bundle,
    load_trajectories_and_step_features,
    precompute_continue_probabilities,
)
from stage3.config import Stage3Config


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


def _threshold_at_step(ckpt: Mapping[str, Any], k: int, fallback: float) -> float:
    """与 ``_simulate_probe_policy_from_probs`` 一致：逐步阈值列表下标 k-1。"""
    pst = ckpt.get("per_step_thresholds")
    if pst is not None and isinstance(pst, (list, tuple)) and len(pst) >= k >= 1:
        return float(pst[k - 1])
    return float(ckpt.get("threshold", fallback))


def _collect_step_rows(
    ordered_test: Sequence[Mapping[str, Any]],
    max_k: int,
    base_cost: float,
    cost_metric: str,
    probs: Dict[Tuple[str, int], float],
    ckpt: Mapping[str, Any],
    bundle_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """返回与逐步样本对齐：abs_margin, signed_margin m_k, y_true, y_pred, sample_ids。"""
    orows = compute_trajectory_oracle(
        list(ordered_test), base_cost, max_k, cost_metric=cost_metric
    )
    abs_list: List[float] = []
    signed_list: List[float] = []
    yt: List[int] = []
    yp: List[int] = []
    ids: List[str] = []
    for traj, orow in zip(ordered_test, orows):
        sid = str(traj.get("id", ""))
        st = orow.get("step_targets") or {}
        for k in range(1, max_k):
            if k not in st:
                continue
            tinfo = st[k] or {}
            margin = float(tinfo.get("margin", 0.0))
            y_true = int(float(tinfo.get("action_label", 0)))
            p = probs.get((sid, k))
            if p is None:
                continue
            tk = _threshold_at_step(ckpt, k, bundle_thr)
            y_pred = int(p >= tk)
            abs_list.append(abs(margin))
            signed_list.append(margin)
            yt.append(y_true)
            yp.append(y_pred)
            ids.append(sid)
    return (
        np.asarray(abs_list, dtype=np.float64),
        np.asarray(signed_list, dtype=np.float64),
        np.asarray(yt, dtype=np.int64),
        np.asarray(yp, dtype=np.int64),
        ids,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    n = int(y_true.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    acc = float(np.mean(y_true == y_pred))
    err = 1.0 - acc
    f1 = float(
        f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    )
    return f1, acc, err


def _diag_continue_stop(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """正类=Continue(1)。返回先验、预测正率、混淆矩阵与 precision/recall。"""
    n = int(y_true.size)
    out: Dict[str, float] = {
        "oracle_continue_rate": float("nan"),
        "pred_continue_rate": float("nan"),
        "cm_tn": float("nan"),
        "cm_fp": float("nan"),
        "cm_fn": float("nan"),
        "cm_tp": float("nan"),
        "precision_continue": float("nan"),
        "recall_continue": float("nan"),
    }
    if n == 0:
        return out
    yt = y_true.astype(np.int64).ravel()
    yp = y_pred.astype(np.int64).ravel()
    out["oracle_continue_rate"] = float(np.mean(yt == 1))
    out["pred_continue_rate"] = float(np.mean(yp == 1))
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    out["cm_tn"], out["cm_fp"], out["cm_fn"], out["cm_tp"] = float(tn), float(fp), float(fn), float(tp)
    out["precision_continue"] = float(
        precision_score(yt, yp, average="binary", pos_label=1, zero_division=0)
    )
    out["recall_continue"] = float(recall_score(yt, yp, average="binary", pos_label=1, zero_division=0))
    return out


def _maybe_hist_png_signed(raw_m: np.ndarray, out_png: Path, title: str) -> None:
    """带符号 margin m_k 直方图（Bellman：margin = continue_val − Q_k）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib 未安装，跳过直方图: %s", out_png)
        return
    x = raw_m[np.isfinite(raw_m)]
    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    span = max(abs(lo), abs(hi), 0.05)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=50, range=(-span, span), color="#6699cc", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("m_k (signed oracle margin)")
    ax.set_ylabel("count (steps)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _maybe_hist_png_abs(abs_m: np.ndarray, out_png: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib 未安装，跳过直方图: %s", out_png)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    mx = float(np.nanmax(abs_m)) + 1e-6
    hi = min(1.0, mx)
    ax.hist(abs_m[np.isfinite(abs_m)], bins=40, range=(0, hi), color="#4477aa", edgecolor="white")
    ax.axvline(0.05, color="crimson", linestyle="--", label="|m|=0.05")
    ax.axvline(0.1, color="darkorange", linestyle="--", label="|m|=0.10")
    ax.set_xlabel("|m_k| (oracle)")
    ax.set_ylabel("count (steps)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle margin step distribution + probe F1/error by |m| bucket")
    p.add_argument("--root-dir", type=str, default=".")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--datasets", type=str, default="hotpotqa,musique,2wiki")
    p.add_argument("--artifact-suffix", type=str, default="pdopt_best")
    p.add_argument("--probe-checkpoint", type=str, default="", help="默认自动探测 artifacts/probe/<ds>/")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    root = Path(args.root_dir)
    res_dir = root / str(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    s3 = Stage3Config(root_dir=root, results_dir=res_dir, artifact_suffix=str(args.artifact_suffix or ""))
    s2 = _stage2_cfg(s3)
    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows: List[Dict[str, Any]] = []
    margin_floor_note = {
        "implementation": "stage2/run_stage2.py::_build_xyw / _build_gru_trajectory_entries",
        "formula": "w_k = max(margin_weight_floor, abs(m_k))",
        "default_margin_weight_floor": 0.1,
        "train_margin_min_abs": "optional filter: skip steps with abs(m_k) < train_margin_min_abs (default 0 = no skip)",
        "loss": "sum(loss_per_step * w_k * mask) / sum(w_k * mask) for GRU; MLP uses w in denominator similarly",
    }
    (res_dir / "oracle_margin_weight_wk_note.json").write_text(
        json.dumps(margin_floor_note, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for dataset in datasets:
        ck_path = Path(args.probe_checkpoint) if args.probe_checkpoint else find_best_probe_checkpoint(
            s3.artifacts_probe_dir, dataset, preferred_suffix=str(args.artifact_suffix or "pdopt_best")
        )
        if ck_path is None or not ck_path.is_file():
            logging.error("%s: 未找到 probe checkpoint，跳过", dataset)
            continue

        ckpt = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        bundle = load_stage2_probe_bundle(ck_path, device)
        test_traj, feat_map, hmap, hd = load_trajectories_and_step_features(s2, dataset, "test", bundle)
        probs = precompute_continue_probabilities(test_traj, feat_map, hmap, hd, bundle, s2)

        abs_m, raw_m, yt, yp, _ids = _collect_step_rows(
            test_traj, s3.max_k, s3.cost_per_step, s3.oracle_cost_metric, probs, ckpt, bundle.probe_threshold
        )
        if abs_m.size == 0:
            logging.error("%s: 无逐步样本", dataset)
            continue

        # --- 全集分布 & near-tie 比例 ---
        total = int(abs_m.size)
        p_lt005 = float(np.mean(abs_m < 0.05))
        p_mid = float(np.mean((abs_m >= 0.05) & (abs_m < 0.1)))
        p_ge01 = float(np.mean(abs_m >= 0.1))

        hist_png = res_dir / f"oracle_margin_abs_histogram_{dataset}.png"
        hist_signed = res_dir / f"oracle_margin_signed_histogram_{dataset}.png"
        _maybe_hist_png_abs(abs_m, hist_png, title=f"{dataset}: |m_k| on supervised steps (test)")
        _maybe_hist_png_signed(raw_m, hist_signed, title=f"{dataset}: signed m_k on supervised steps (test)")

        # MuSiQue id：匹配 cache 内 dataset 字段或路径（兼容）
        musique_mask = np.array([("musique" in str(i).lower()) or i.startswith("musique") for i in _ids], dtype=bool)
        # 若 id 不含别名，整数据集即为 musique
        if dataset.lower() == "musique":
            musique_mask[:] = True

        def emit_slice(tag: str, mask: np.ndarray) -> None:
            am = abs_m[mask]
            ytv = yt[mask]
            ypv = yp[mask]
            n_sub = int(am.size)
            if n_sub == 0:
                return
            for bkey, lo, hi, inclusive_hi in [
                ("lt_005", 0.0, 0.05, False),
                ("05_01", 0.05, 0.1, False),
                ("ge_01", 0.1, 1e9, True),
            ]:
                if inclusive_hi:
                    bm = (am >= lo) & (am <= hi)
                else:
                    bm = (am >= lo) & (am < hi)
                f1, acc, err = _metrics(ytv[bm], ypv[bm])
                nb = int(np.sum(bm))
                dg = _diag_continue_stop(ytv[bm], ypv[bm])
                row_out: Dict[str, Any] = {
                    "dataset": dataset,
                    "slice": tag,
                    "bucket": bkey,
                    "n_steps": nb,
                    "frac_of_slice": float(nb / n_sub) if n_sub else 0.0,
                    "oracle_continue_rate": dg["oracle_continue_rate"],
                    "pred_continue_rate": dg["pred_continue_rate"],
                    "cm_tn": dg["cm_tn"],
                    "cm_fp": dg["cm_fp"],
                    "cm_fn": dg["cm_fn"],
                    "cm_tp": dg["cm_tp"],
                    "precision_continue": dg["precision_continue"],
                    "recall_continue": dg["recall_continue"],
                    "probe_step_f1": f1,
                    "probe_accuracy": acc,
                    "probe_error_rate": err,
                    "mean_abs_margin_in_bucket": float(np.mean(am[bm])) if nb else float("nan"),
                }
                summary_rows.append(row_out)
            # slice ALL
            f1a, acca, erra = _metrics(ytv, ypv)
            dg_all = _diag_continue_stop(ytv, ypv)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "slice": tag,
                    "bucket": "ALL",
                    "n_steps": n_sub,
                    "frac_of_slice": 1.0,
                    "oracle_continue_rate": dg_all["oracle_continue_rate"],
                    "pred_continue_rate": dg_all["pred_continue_rate"],
                    "cm_tn": dg_all["cm_tn"],
                    "cm_fp": dg_all["cm_fp"],
                    "cm_fn": dg_all["cm_fn"],
                    "cm_tp": dg_all["cm_tp"],
                    "precision_continue": dg_all["precision_continue"],
                    "recall_continue": dg_all["recall_continue"],
                    "probe_step_f1": f1a,
                    "probe_accuracy": acca,
                    "probe_error_rate": erra,
                    "mean_abs_margin_in_bucket": float(np.mean(am)),
                }
            )

        emit_slice("full", np.ones_like(abs_m, dtype=bool))
        # 若将来在同一条 jsonl 内混有多数据源，可用 id 切片单独统计 MuSiQue：
        if dataset.lower() != "musique" and np.any(musique_mask):
            emit_slice("musique_id_slice", musique_mask)

        logging.info(
            "%s |m_k| near-tie: <0.05=%.2f%% [0.05,0.1)=%.2f%% >=0.1=%.2f%% (N=%d)",
            dataset,
            100 * p_lt005,
            100 * p_mid,
            100 * p_ge01,
            total,
        )

        dist_path = res_dir / f"oracle_margin_distribution_{dataset}.json"
        dist_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "n_supervised_steps": total,
                    "near_tie_frac_abs_m_lt_0.05": p_lt005,
                    "mid_frac_0.05_le_abs_m_lt_0.1": p_mid,
                    "clear_frac_abs_m_ge_0.1": p_ge01,
                    "abs_m_mean": float(np.mean(abs_m)),
                    "abs_m_std": float(np.std(abs_m)),
                    "abs_m_percentiles": {str(p): float(np.percentile(abs_m, p)) for p in (5, 25, 50, 75, 95)},
                    "signed_m_mean": float(np.mean(raw_m)),
                    "signed_m_std": float(np.std(raw_m)),
                    "frac_margin_positive": float(np.mean(raw_m > 0)),
                    "histogram_png_abs": str(hist_png),
                    "histogram_png_signed": str(hist_signed),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    out_csv = res_dir / "oracle_margin_step_probe_buckets.csv"
    fields = [
        "dataset",
        "slice",
        "bucket",
        "n_steps",
        "frac_of_slice",
        "oracle_continue_rate",
        "pred_continue_rate",
        "cm_tn",
        "cm_fp",
        "cm_fn",
        "cm_tp",
        "precision_continue",
        "recall_continue",
        "probe_step_f1",
        "probe_accuracy",
        "probe_error_rate",
        "mean_abs_margin_in_bucket",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"已写入: {out_csv}")
    print(f"w_k 说明 JSON: {res_dir / 'oracle_margin_weight_wk_note.json'}")
    for row in summary_rows:
        if row["bucket"] == "ALL":
            print(
                f"  [{row['dataset']} {row['slice']}] N={row['n_steps']} "
                f"P(+oracle)={row['oracle_continue_rate']:.4f} P(+pred)={row['pred_continue_rate']:.4f} "
                f"F1={row['probe_step_f1']:.4f} err={row['probe_error_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
