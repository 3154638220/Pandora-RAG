"""
Weitzman Pandora's Box 最优停止规则。

核心方程（每步 k 的保留值 r_k*）：
    c = E[max(G_k - r*, 0)] = (1/n) Σ max(g_i - r*, 0)

其中 G_k = Q(s_k) - Q(s_{k-1}) 是第 k 步的信息增益，c 为每步检索成本。
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def compute_reservation_value(
    gains: List[float],
    cost: float,
    bounds: tuple = (-1.0, 1.0),
) -> Optional[float]:
    """
    给定信息增益的经验样本 {g_1,...,g_n}，求解 Weitzman 保留值 r*，使得：
        c = (1/n) Σ max(g_i - r*, 0)

    若所有增益都 ≤ cost，说明该步检索不值得，返回 None（永远停止）。
    """
    gains_arr = np.array(gains, dtype=float)

    def equation(r):
        return np.mean(np.maximum(gains_arr - r, 0.0)) - cost

    # 检查边界条件
    val_at_lb = equation(bounds[0])
    val_at_ub = equation(bounds[1])

    if val_at_lb < 0:
        # 即使 r=lb，期望增益也不够覆盖成本 -> 步骤不值得，r* 很大 -> 总是停止
        return float(bounds[1])
    if val_at_ub > 0:
        # r=ub 时仍有剩余期望增益 -> 调宽上界
        bounds = (bounds[0], bounds[1] * 2)
        val_at_ub = equation(bounds[1])
        if val_at_ub > 0:
            return float(bounds[0])  # 几乎不会停止

    try:
        r_star = brentq(equation, bounds[0], bounds[1], xtol=1e-6)
        return float(r_star)
    except ValueError as e:
        logger.warning("保留值求解失败（%s），增益样本: mean=%.3f", e, gains_arr.mean())
        return None


def compute_all_reservation_values(
    trajectories: List[dict],
    max_k: int,
    cost: float,
) -> Dict[int, float]:
    """
    从训练轨迹中统计每步 k 的信息增益分布，求解所有 r_k*。

    返回字典 {step_k: r_k*}，k 从 1 开始。
    """
    gains_by_step: Dict[int, List[float]] = {k: [] for k in range(1, max_k + 1)}

    for traj in trajectories:
        steps = traj["steps"]
        prev_f1 = 0.0
        for step_data in steps:
            k = step_data["step"]
            curr_f1 = step_data["f1"]
            gain = curr_f1 - prev_f1
            gains_by_step[k].append(gain)
            prev_f1 = curr_f1

    reservation_values: Dict[int, float] = {}
    for k in range(1, max_k + 1):
        gains = gains_by_step[k]
        if not gains:
            reservation_values[k] = 0.0
            continue
        r_star = compute_reservation_value(gains, cost)
        reservation_values[k] = r_star if r_star is not None else 0.0
        logger.info(
            "Step %d: n=%d, mean_gain=%.4f, r*=%.4f",
            k, len(gains), float(np.mean(gains)), reservation_values[k],
        )

    return reservation_values


def oracle_stopping_simulation(
    trajectories: List[dict],
    reservation_values: Dict[int, float],
    max_k: int,
) -> List[dict]:
    """
    在测试轨迹上模拟 Oracle 停止策略（使用真实 F1 + 真实 r*）。

    停止规则：执行第 k 步检索后，若 Q(s_k) >= r_{k+1}* 则停止。
    最后一步（k=max_k）强制停止。

    返回每条轨迹的评估结果：{f1, em, steps_used}。
    """
    results = []
    for traj in trajectories:
        steps = traj["steps"]
        final_f1, final_em, steps_used = 0.0, False, 0

        for step_data in steps:
            k = step_data["step"]
            curr_f1 = step_data["f1"]
            curr_em = step_data["em"]
            steps_used = k

            # 是否停止？比较当前质量与下一步保留值
            next_r = reservation_values.get(k + 1, -999.0) if k < max_k else -999.0
            if curr_f1 >= next_r or k == max_k:
                final_f1, final_em = curr_f1, curr_em
                break

        results.append({"f1": final_f1, "em": final_em, "steps_used": steps_used})
    return results
