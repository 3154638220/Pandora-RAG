from __future__ import annotations

"""
在 Calib 上训练轻量质量预测器（逻辑回归），估计 P(F1≥γ|浅层特征)。

支持两种特征集：
  1. 仅浅层特征（原版）
  2. 浅层特征 + Probe p_continue（增强版，零成本额外维）

附带校准评估工具（Brier Score / ECE）。
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 数据构建
# ---------------------------------------------------------------------------

def build_step_quality_dataset(
    trajectories: List[Dict[str, Any]],
    shallow_by_step: Dict[Tuple[str, int], np.ndarray],
    *,
    gamma: float,
    max_k: int,
    probe_probs: Optional[Dict[Tuple[str, int], float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    每一步一行：特征为浅层向量（可选拼 probe p_continue），标签为 1[该步 F1 >= γ]。

    若 probe_probs 非 None，则在浅层向量末尾拼一维 p_continue，可提升质量模型。
    """
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for traj in trajectories:
        sid = str(traj.get("id", ""))
        if not sid:
            continue
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        for step in steps:
            k = int(step.get("step", 0))
            if k <= 0 or k > max_k:
                continue
            vec = shallow_by_step.get((sid, k))
            if vec is None:
                continue
            base = np.asarray(vec, dtype=np.float32).reshape(-1)
            if probe_probs is not None:
                p_cont = float(probe_probs.get((sid, k), 0.5))
                base = np.append(base, np.float32(p_cont))
            f1 = float(step.get("f1", 0.0) or 0.0)
            xs.append(base)
            ys.append(1.0 if f1 >= float(gamma) else 0.0)
    if not xs:
        raise RuntimeError("质量模型：Calib 上无有效 (浅层特征, 标签) 样本。")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_quality_logreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """标准化 + L2 逻辑回归（class_weight 平衡）。"""
    clf = LogisticRegression(
        max_iter=int(max_iter),
        random_state=int(random_state),
        class_weight="balanced",
        solver="lbfgs",
    )
    pipe: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    pipe.fit(x, y)
    return pipe


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

def predict_success_prob(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """返回正类（F1≥γ）概率。"""
    proba = model.predict_proba(x.astype(np.float64, copy=False))
    if proba.shape[1] < 2:
        return proba[:, 0]
    return proba[:, 1]


# ---------------------------------------------------------------------------
# 阈值调优
# ---------------------------------------------------------------------------

def tune_quality_bar_error_based(
    p_hat_stop: np.ndarray,
    error_stop: np.ndarray,
    *,
    target_error: float,
) -> float:
    """
    在 Calib 上，仅使用「探针自然停止步」上的 (p_hat, error)。
    在 [0,1] 上网格搜索 quality_bar，使得 p_hat >= bar 的子集中
    经验错误率 <= target_error，并取满足条件的最大 bar。
    """
    ph = np.asarray(p_hat_stop, dtype=np.float64).reshape(-1)
    er = np.asarray(error_stop, dtype=np.float64).reshape(-1)
    if ph.size != er.size or ph.size == 0:
        return 0.0
    grid = np.linspace(0.0, 1.0, 101)
    best = 0.0
    for bar in grid:
        mask = ph >= bar - 1e-12
        if not np.any(mask):
            continue
        rate = float(np.mean(er[mask]))
        if rate <= float(target_error) + 1e-9:
            best = float(bar)
    return best


def tune_quality_bar_on_calib(
    p_hat_stop: np.ndarray,
    error_stop: np.ndarray,
    *,
    target_error: float,
    calib_method: Literal["quantile", "error_rate"] = "quantile",
) -> float:
    """
    在 Calib 上校准 quality_bar。

    - quantile（默认，推荐）：bar = p_hat 的 α 分位数，阻断最低质量的 α 比例停止。
    - error_rate（保留对照）：最大化 bar 且满足子集经验错误率 <= α。
    """
    ph = np.asarray(p_hat_stop, dtype=np.float64).reshape(-1)
    er = np.asarray(error_stop, dtype=np.float64).reshape(-1)
    if ph.size == 0:
        return 0.0

    method = str(calib_method).strip().lower()
    if method == "quantile":
        q = float(np.clip(float(target_error), 0.0, 1.0))
        return float(np.quantile(ph, q))
    if method == "error_rate":
        return tune_quality_bar_error_based(ph, er, target_error=target_error)
    raise ValueError(f"未知 calib_method: {calib_method}")


# ---------------------------------------------------------------------------
# 校准评估
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score：越小越好，完美校准 = 0。"""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    return float(np.mean((yp - yt) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE（Expected Calibration Error）：分 bin 后加权平均 |accuracy - confidence|。"""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (yp >= lo) & (yp < hi + 1e-12)
        if not np.any(mask):
            continue
        acc = float(np.mean(yt[mask]))
        conf = float(np.mean(yp[mask]))
        ece += float(np.sum(mask)) / float(len(yt)) * abs(acc - conf)
    return ece


def evaluate_quality_model(
    model: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """返回 Brier Score 和 ECE 的字典。"""
    probs = predict_success_prob(model, x)
    return {
        "brier_score": brier_score(y, probs),
        "ece": expected_calibration_error(y, probs),
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }
