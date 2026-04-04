"""
Weitzman Pandora's Box 最优停止规则。

1) 全局 Weitzman（训练集经验分布）：每步 k 的保留值 r_k* 满足
    c = E[max(G_k - r*, 0)] = (1/n) Σ max(g_i - r*, 0)
   其中 G_k = Q(s_k) - Q(s_{k-1}) 为信息增益。用于 Static / Global-Weitzman 基线。

2) 单条轨迹 DP Oracle（上帝视角）：已知该轨迹上每一步真实 Q(s_k)=F1 时，
    V_K = Q(s_K)，V_k = max(Q(s_k), V_{k+1} - c_{k+1})，
   在最小 k 满足 Q(s_k) >= V_{k+1} - c_{k+1} 处停止。
   支持每步成本 c_k（固定 / 按 token / 按延迟归一化），与 NeurIPS 形式
   payoff Q(s_τ) - Σ_{j=1}^τ c_j 对齐。
   step_targets[k] 提供 Phase 2 用的期望继续价值、margin 与二分类标签。
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def _unit_step_cost(step: dict, base_cost_per_step: float, cost_metric: str) -> float:
    """从单步轨迹记录解析该步检索–生成的成本 c_k（与 cost_metric 一致）。"""
    cost = step.get("cost") or {}
    if cost_metric == "token":
        tokens = int(cost.get("token_count", 0) or 0)
        return base_cost_per_step * (tokens / 1000.0 if tokens > 0 else 1.0)
    if cost_metric == "latency":
        ms = float(cost.get("latency_ms", 0) or 0)
        return base_cost_per_step * (ms / 1000.0 if ms > 0 else 1.0)
    if cost_metric == "fixed":
        return float(base_cost_per_step)
    raise ValueError(f"未知 cost_metric: {cost_metric!r}，应为 fixed|token|latency")


def q_and_c_values_for_trajectory(
    steps_raw: List[dict],
    base_cost_per_step: float,
    max_k: int,
    cost_metric: str = "fixed",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    将轨迹补全到 1..max_k：缺失步上 Q 沿用上一观测 F1，成本对缺失步使用 base_cost_per_step。
    返回 q_values[k]、c_values[k]（c_k 表示从 k-1 走到 k 这一步的成本）。
    """
    q_partial: Dict[int, float] = {}
    c_partial: Dict[int, float] = {}
    for s in sorted(steps_raw, key=lambda x: int(x["step"])):
        k = int(s["step"])
        q_partial[k] = float(s["f1"])
        c_partial[k] = _unit_step_cost(s, base_cost_per_step, cost_metric)

    q_values: Dict[int, float] = {}
    c_values: Dict[int, float] = {}
    prev_q = 0.0
    for k in range(1, max_k + 1):
        if k in q_partial:
            prev_q = q_partial[k]
            c_values[k] = c_partial[k]
        else:
            c_values[k] = float(base_cost_per_step)
        q_values[k] = prev_q
    return q_values, c_values


def trajectory_cumulative_cost(
    traj: dict,
    upto_step: int,
    base_cost_per_step: float,
    max_k: int,
    cost_metric: str = "fixed",
) -> float:
    """从轨迹缓存计算走到第 upto_step 步（含）的累计成本 Σ_{j=1}^{upto_step} c_j。"""
    steps_raw = traj.get("steps") or []
    if upto_step <= 0:
        return 0.0
    _, c_values = q_and_c_values_for_trajectory(
        steps_raw, base_cost_per_step, max_k, cost_metric
    )
    hi = min(int(upto_step), max_k)
    return float(sum(c_values[j] for j in range(1, hi + 1)))


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


def compute_trajectory_oracle(
    trajectories: List[dict],
    base_cost_per_step: float,
    max_k: int,
    cost_metric: str = "fixed",
) -> List[dict]:
    """
    对每条轨迹用后向归纳（DP）计算实例级 Oracle 停止步与逐步训练标签。

    V_max_k = Q(s_max_k)；V_k = max(Q(s_k), V_{k+1} - c_{k+1})。
    停止：最小的 k ∈ {1,…,max_k-1} 使得 Q(s_k) >= V_{k+1} - c_{k+1}，否则在 max_k 停止。

    返回每条轨迹：
        id, steps_used, f1, em, step_targets
    其中 step_targets[k]（k=1..max_k-1）为字典，含：
        absolute_v_next, cost_next, expected_continue_val（=V_{k+1}-c_{k+1}）,
        margin（= expected_continue_val - Q_k；>0 表示应继续）,
        action_label（1=Continue, 0=Stop，用于二分类预案）。
    """
    if cost_metric not in ("fixed", "token", "latency"):
        raise ValueError("cost_metric 必须是 fixed、token 或 latency")

    oracle_results: List[dict] = []

    for traj in trajectories:
        steps_raw = traj.get("steps") or []
        if not steps_raw:
            oracle_results.append(
                {
                    "id": traj.get("id", ""),
                    "steps_used": 0,
                    "f1": 0.0,
                    "em": False,
                    "step_targets": {},
                }
            )
            continue

        steps = sorted(steps_raw, key=lambda x: int(x["step"]))
        q_values, c_values = q_and_c_values_for_trajectory(
            steps, base_cost_per_step, max_k, cost_metric
        )

        v_values: Dict[int, float] = {}
        v_values[max_k] = q_values[max_k]
        for k in range(max_k - 1, 0, -1):
            continue_val = v_values[k + 1] - c_values[k + 1]
            v_values[k] = max(q_values[k], continue_val)

        stop_step = max_k
        for k in range(1, max_k):
            continue_val = v_values[k + 1] - c_values[k + 1]
            if q_values[k] >= continue_val:
                stop_step = k
                break

        by_step = {int(s["step"]): s for s in steps}
        if stop_step in by_step:
            target_step_info = by_step[stop_step]
        else:
            target_step_info = max(
                (s for s in steps if int(s["step"]) <= stop_step),
                key=lambda s: int(s["step"]),
                default=steps[-1],
            )

        step_targets: Dict[int, dict] = {}
        for k in range(1, max_k):
            continue_val = v_values[k + 1] - c_values[k + 1]
            margin = continue_val - q_values[k]
            step_targets[k] = {
                "absolute_v_next": float(v_values[k + 1]),
                "cost_next": float(c_values[k + 1]),
                "expected_continue_val": float(continue_val),
                "margin": float(margin),
                "action_label": 1 if margin > 0 else 0,
            }

        oracle_results.append(
            {
                "id": traj.get("id", ""),
                "steps_used": int(stop_step),
                "f1": float(target_step_info["f1"]),
                "em": bool(target_step_info["em"]),
                "step_targets": step_targets,
            }
        )

    return oracle_results
