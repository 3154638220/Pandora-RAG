"""
Phase D1：训练集特征 vs action_label 诊断（point-biserial、AUROC、hidden PCA、Oracle margin 分布）。

产出：results/stage2_feature_diagnostic.md 与 results/stage2_feature_diagnostic_pca_<dataset>.png

用法：
  python -m stage2.run_feature_diagnostic --datasets hotpotqa,musique,2wiki
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from stage2.run_stage2 import (
    SHALLOW_FEATURE_NAMES,
    Stage2Config,
    _build_xyw,
    _infer_hidden_dim,
    _load_hidden_map,
    _load_trajectories,
    _make_oracle_maps,
    _set_seed,
)

LOGGER = logging.getLogger(__name__)

MARGIN_FUZZY_EPS = 0.01
PCA_MAX_SAMPLES = 50_000
RANDOM_STATE = 42


def _safe_point_biserial(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    yb = (y >= 0.5).astype(np.int8)
    if np.unique(yb).size < 2:
        return float("nan"), float("nan")
    if float(np.std(x)) < 1e-12:
        return float("nan"), float("nan")
    r, p = stats.pointbiserialr(yb, x)
    return float(r), float(p)


def _safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    yb = (y >= 0.5).astype(np.int8)
    if np.unique(yb).size < 2:
        return float("nan")
    if float(np.std(score)) < 1e-12:
        return float("nan")
    try:
        return float(roc_auc_score(yb, score))
    except ValueError:
        return float("nan")


def _diagnose_shallow(
    x_shallow: np.ndarray, y: np.ndarray
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    assert x_shallow.shape[1] == len(SHALLOW_FEATURE_NAMES)
    for j, name in enumerate(SHALLOW_FEATURE_NAMES):
        col = x_shallow[:, j].astype(np.float64, copy=False)
        r, p = _safe_point_biserial(y, col)
        auc = _safe_auroc(y, col)
        rows.append(
            {
                "feature": name,
                "point_biserial_r": r,
                "point_biserial_p": p,
                "auroc": auc,
            }
        )
    return rows


def _plot_hidden_pca(
    x_hidden: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    dataset: str,
    hidden_dim: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"n_total": int(x_hidden.shape[0]), "n_used": 0, "n_zero_hidden": 0}
    if hidden_dim <= 0 or x_hidden.shape[1] == 0:
        meta["skipped"] = "无 hidden 或未加载"
        return meta

    zero_mask = np.all(np.abs(x_hidden) < 1e-8, axis=1)
    meta["n_zero_hidden"] = int(zero_mask.sum())
    xh = x_hidden[~zero_mask]
    yh = y[~zero_mask]
    if xh.shape[0] < 10:
        meta["skipped"] = "非零 hidden 样本过少"
        return meta

    n = xh.shape[0]
    if n > PCA_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=PCA_MAX_SAMPLES, replace=False)
        xh_fit = xh[idx]
        y_plot = yh[idx]
        xh_all = xh[idx]
        meta["subsampled"] = PCA_MAX_SAMPLES
    else:
        xh_fit = xh
        xh_all = xh
        y_plot = yh

    pca = PCA(n_components=2, svd_solver="randomized", random_state=RANDOM_STATE)
    z = pca.fit_transform(xh_fit)
    if n > PCA_MAX_SAMPLES:
        z = pca.transform(xh_all)
    meta["n_used"] = int(z.shape[0])
    meta["explained_variance_ratio"] = [float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])]

    cont = y_plot >= 0.5
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(z[~cont, 0], z[~cont, 1], c="#1f77b4", s=6, alpha=0.35, label="Stop (y=0)")
    ax.scatter(z[cont, 0], z[cont, 1], c="#d62728", s=6, alpha=0.35, label="Continue (y=1)")
    ax.set_xlabel(f"PC1 ({meta['explained_variance_ratio'][0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({meta['explained_variance_ratio'][1]*100:.1f}% var)")
    ax.set_title(
        f"Hidden states PCA (train) — {dataset}\n"
        f"(excl. all-zero hidden rows: {meta['n_zero_hidden']})"
    )
    ax.legend(markerscale=2.0, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return meta


def _margin_fuzzy_stats(margins: np.ndarray) -> Dict[str, float]:
    m = np.abs(margins.astype(np.float64))
    n = int(m.size)
    if n == 0:
        return {"n": 0.0, "fuzzy_frac": float("nan")}
    fuzzy = int((m < MARGIN_FUZZY_EPS).sum())
    return {
        "n": float(n),
        "fuzzy_count": float(fuzzy),
        "fuzzy_frac": float(fuzzy / n),
        "margin_abs_mean": float(m.mean()),
        "margin_abs_median": float(np.median(m)),
    }


def run_diagnostic(cfg: Stage2Config, dataset: str) -> Dict[str, Any]:
    train_traj = _load_trajectories(cfg, dataset, "train")
    train_oracle_map, _ = _make_oracle_maps(train_traj, cfg)
    hidden_dim = _infer_hidden_dim(cfg, dataset, "train", cfg.hidden_state_key)
    train_hidden = _load_hidden_map(cfg, dataset, "train", hidden_dim, cfg.hidden_state_key)

    x_s, x_h, y, _w, st = _build_xyw(
        train_traj,
        train_oracle_map,
        train_hidden,
        hidden_dim,
        cfg,
        collect_margins=True,
    )
    margins = st.get("oracle_margins")
    if not isinstance(margins, np.ndarray) or margins.shape[0] != y.shape[0]:
        raise RuntimeError("oracle_margins 与标签行数不一致")

    shallow_rows = _diagnose_shallow(x_s, y)
    margin_stats = _margin_fuzzy_stats(margins)

    pca_path = cfg.results_dir / f"stage2_feature_diagnostic_pca_{dataset}.png"
    pca_meta = _plot_hidden_pca(x_h, y, pca_path, dataset, hidden_dim)

    return {
        "dataset": dataset,
        "train_stats": {k: v for k, v in st.items() if k != "oracle_margins"},
        "shallow_feature_diag": shallow_rows,
        "margin_stats": margin_stats,
        "pca": {"path": str(pca_path.relative_to(cfg.root_dir)), **pca_meta},
    }


def _format_float(x: float, nd: int = 4) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.{nd}f}"


def _write_report(cfg: Stage2Config, sections: List[Dict[str, Any]]) -> Path:
    lines: List[str] = []
    lines.append("# Stage 2 Phase D1：特征–标签诊断报告\n")
    lines.append("> 训练集逐步样本：浅层特征与 Oracle `action_label`（Continue=1 / Stop=0）的单变量关联；")
    lines.append("> Hidden states（4096 维）经 PCA 至 2D 着色；Oracle `margin` 小值占比（模糊标签）。\n")
    lines.append(
        f"- **模糊样本定义**：|margin| < {MARGIN_FUZZY_EPS}（Oracle 决策边界附近，标签噪声大）\n"
    )

    for sec in sections:
        ds = sec["dataset"]
        lines.append(f"## 数据集：`{ds}`\n")
        ts = sec["train_stats"]
        lines.append("### 样本量\n")
        lines.append(f"- 可训练步级样本：`{ts.get('trainable_examples', '—')}`\n")

        lines.append("### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）\n")
        lines.append("| feature | point-biserial r | p-value | AUROC |")
        lines.append("|---------|------------------|---------|-------|")
        for row in sec["shallow_feature_diag"]:
            lines.append(
                "| "
                + row["feature"]
                + " | "
                + _format_float(row["point_biserial_r"])
                + " | "
                + _format_float(row["point_biserial_p"])
                + " | "
                + _format_float(row["auroc"])
                + " |"
            )
        lines.append("")
        ms = sec["margin_stats"]
        lines.append("### Oracle margin 分布\n")
        lines.append(f"- 样本数：{int(ms['n'])}")
        lines.append(
            f"- |margin| < {MARGIN_FUZZY_EPS} 占比：**{ms['fuzzy_frac']*100:.2f}%** "
            f"（{int(ms['fuzzy_count'])} / {int(ms['n'])}）"
        )
        lines.append(
            f"- |margin| 绝对值均值：{_format_float(ms['margin_abs_mean'])}；"
            f"中位数：{_format_float(ms['margin_abs_median'])}\n"
        )

        pca = sec["pca"]
        lines.append("### Hidden states PCA（2D）\n")
        if pca.get("skipped"):
            lines.append(f"- 跳过：{pca['skipped']}\n")
        else:
            lines.append(f"- 图：`{pca['path']}`")
            lines.append(
                f"- 用于 PCA 的点数：{pca.get('n_used', '—')}；全零 hidden 行数：{pca.get('n_zero_hidden', '—')}"
            )
            ev = pca.get("explained_variance_ratio")
            if ev:
                lines.append(f"- 前两主成分方差比：{ev[0]*100:.2f}%，{ev[1]*100:.2f}%\n")

    lines.append("## 解读要点（对照 plan D1）\n")
    lines.append("- 若多数浅层特征 **AUROC < 0.60**，单变量上难以区分 Stop/Continue，提示 **特征信号不足**，宜推进 D4 补特征。")
    lines.append("- **模糊样本占比高** 时，硬标签噪声大，会拖累 Probe 与 Focal 等损失；可与 D3 回退损失联动。")
    lines.append("- PCA 平面上两类 **严重重叠** 时，仅靠线性可分 hidden 信息不足，与 ProbeMLP_v2 难提取信号一致。\n")

    out = cfg.results_dir / "stage2_feature_diagnostic.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Stage2 Phase D1 特征诊断")
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,musique,2wiki",
        help="逗号分隔数据集名",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="仓库根目录")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Stage2Config(root_dir=args.root.resolve(), max_k=args.max_k, seed=args.seed)
    _set_seed(cfg.seed)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    sections: List[Dict[str, Any]] = []
    for ds in datasets:
        LOGGER.info("D1 诊断：%s", ds)
        sections.append(run_diagnostic(cfg, ds))

    report_path = _write_report(cfg, sections)
    LOGGER.info("已写入 %s", report_path)


if __name__ == "__main__":
    main()
