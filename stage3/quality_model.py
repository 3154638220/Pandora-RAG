from __future__ import annotations

"""
在 Calib 上训练轻量质量预测器（逻辑回归），估计 P(F1≥γ|浅层特征)。

仅依赖 numpy / sklearn；特征维度与 Stage2 的 SHALLOW_FEATURE_DIM 由调用方保证一致。
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_step_quality_dataset(
    trajectories: List[Dict[str, Any]],
    shallow_by_step: Dict[Tuple[str, int], np.ndarray],
    *,
    gamma: float,
    max_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    每一步一行：特征为浅层向量，标签为 1[该步 F1 >= γ]。
    与 plan 中「Calib 上训练质量预测器」一致；使用全步可增广样本量。
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
            f1 = float(step.get("f1", 0.0) or 0.0)
            xs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
            ys.append(1.0 if f1 >= float(gamma) else 0.0)
    if not xs:
        raise RuntimeError("质量模型：Calib 上无有效 (浅层特征, 标签) 样本。")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32)


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


def predict_success_prob(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """返回正类（F1≥γ）概率。"""
    proba = model.predict_proba(x.astype(np.float64, copy=False))
    # 正类列索引：sklearn 按标签排序，0/1 二分类时列为 [p0, p1]
    if proba.shape[1] < 2:
        return proba[:, 0]
    return proba[:, 1]


def tune_quality_bar_on_calib(
    p_hat_stop: np.ndarray,
    error_stop: np.ndarray,
    *,
    target_error: float,
) -> float:
    """
    在 Calib 上，仅使用「探针自然停止步」上的 (p_hat, error)。
    在 [0,1] 上网格搜索 quality_bar，使得 p_hat >= bar 的子集中经验错误率 <= target_error，
    并取满足条件的最大 bar（更保守的下注门槛）。
    """
    ph = np.asarray(p_hat_stop, dtype=np.float64).reshape(-1)
    er = np.asarray(error_stop, dtype=np.float64).reshape(-1)
    if ph.size != er.size or ph.size == 0:
        return 0.5
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
