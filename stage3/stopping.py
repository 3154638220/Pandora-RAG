from __future__ import annotations

"""
停止策略仿真：Probe / E-value 门控 / Conformal 门控 / 分布漂移。

依赖 ``pretest.utils.weitzman.trajectory_cumulative_cost`` 仅用于汇总成本。
所有仿真函数返回统一的 ``List[Dict]`` 行格式 + 可选的 wealth trace。
"""

import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from pretest.utils.weitzman import trajectory_cumulative_cost

from stage3.evalue import (
    BettingStrategy,
    EWealthTracker,
    betting_multiplier_indicator,
    compute_betting_lambda,
    outcome_aware_multiplier,
)
from stage3.quality_model import predict_success_prob


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _sorted_steps(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))


def _empty_outcome(cfg: Any, gamma: float) -> Dict[str, Any]:
    return {
        "f1": 0.0,
        "em": 0,
        "steps_used": 0,
        "avg_cost": 0.0,
        "error": int(0.0 < float(gamma)),
    }


def _finalize_row(
    traj: Dict[str, Any],
    chosen: Dict[str, Any],
    cfg: Any,
    gamma: float,
) -> Dict[str, Any]:
    used = int(chosen.get("step", 0))
    cum_cost = trajectory_cumulative_cost(
        traj,
        used,
        cfg.cost_per_step,
        cfg.max_k,
        cfg.oracle_cost_metric,
    )
    f1 = float(chosen.get("f1", 0.0))
    return {
        "f1": f1,
        "em": int(bool(chosen.get("em", False))),
        "steps_used": used,
        "avg_cost": float(cum_cost),
        "error": int(f1 < float(gamma)),
    }


# ---------------------------------------------------------------------------
# Probe-only baseline（无门控）
# ---------------------------------------------------------------------------

def probe_stop_shallow_and_phat(
    traj: Dict[str, Any],
    continue_probs: Dict[Tuple[str, int], float],
    shallow_by_step: Dict[Tuple[str, int], np.ndarray],
    quality_model: Pipeline,
    *,
    cfg: Any,
    probe_threshold: float,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """探针在 dev/calib 上的「自然停止」步，用于阈值校准。"""
    sample_id = str(traj.get("id", ""))
    steps = _sorted_steps(traj)
    if not steps:
        z0 = np.zeros((1,), dtype=np.float32)
        return z0, 0.0, {}

    chosen = steps[-1]
    for step in steps:
        k = int(step.get("step", 0))
        if k >= cfg.max_k:
            chosen = step
            break
        p_cont = continue_probs.get((sample_id, k))
        if p_cont is None:
            continue
        if float(p_cont) >= float(probe_threshold):
            continue
        chosen = step
        break

    kf = int(chosen.get("step", 0))
    z = shallow_by_step.get((sample_id, kf))
    if z is None:
        return np.zeros((1,), dtype=np.float32), 0.0, chosen
    phat = float(predict_success_prob(quality_model, np.asarray(z).reshape(1, -1))[0])
    return np.asarray(z, dtype=np.float32), phat, chosen


# ---------------------------------------------------------------------------
# E-value 门控 — 结果感知型（推荐）
# ---------------------------------------------------------------------------

def simulate_evalue_outcome_aware(
    trajectories: Sequence[Dict[str, Any]],
    continue_probs: Dict[Tuple[str, int], float],
    shallow_by_step: Dict[Tuple[str, int], np.ndarray],
    quality_model: Pipeline,
    *,
    cfg: Any,
    probe_threshold: float,
    gamma: float,
    alpha: float,
    quality_bar: float,
    betting_strategy: BettingStrategy = "predictive",
    betting_lambda: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    结果感知型 E-value 门控：

    1. Probe 建议停止时，计算 p_hat
    2. 若 p_hat < quality_bar → 强制继续（质量预测不达标）
    3. 若允许停止 → 观测真实 F1 → 用 outcome_aware_multiplier 更新 wealth
    4. 当 wealth 接近 1/α → 下一次停止前会更保守（wealth 越高说明累积错误越多）

    每个样本处理完（无论是否提前停止）都会更新 wealth。
    """
    tracker = EWealthTracker(alpha=float(alpha))
    rows: List[Dict[str, Any]] = []

    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        steps = _sorted_steps(traj)
        if not steps:
            rows.append(_empty_outcome(cfg, gamma))
            tracker.trace.append(tracker.wealth)
            continue

        chosen = steps[-1]
        stopped_early = False

        for step in steps:
            k = int(step.get("step", 0))
            if k >= cfg.max_k:
                chosen = step
                break

            p_cont = continue_probs.get((sample_id, k))
            if p_cont is None:
                continue
            if float(p_cont) >= float(probe_threshold):
                continue

            z = shallow_by_step.get((sample_id, k))
            if z is None:
                continue
            phat = float(predict_success_prob(quality_model, z.reshape(1, -1))[0])

            should_gate = (phat < float(quality_bar)) or tracker.exceeded_cap
            if should_gate:
                continue

            chosen = step
            stopped_early = True
            break

        f1 = float(chosen.get("f1", 0.0))
        error = int(f1 < float(gamma))

        z_final = shallow_by_step.get((sample_id, int(chosen.get("step", 0))))
        if z_final is not None:
            phat_final = float(predict_success_prob(quality_model, z_final.reshape(1, -1))[0])
        else:
            phat_final = 0.5

        lam = compute_betting_lambda(
            betting_strategy,
            p_hat=phat_final,
            fixed_lambda=betting_lambda,
        )
        tracker.apply_outcome(error, lam)

        rows.append(_finalize_row(traj, chosen, cfg, gamma))

    return rows, tracker.trace


# ---------------------------------------------------------------------------
# E-value 门控 — 原版 indicator（对照基线）
# ---------------------------------------------------------------------------

def simulate_evalue_gated_stops(
    trajectories: Sequence[Dict[str, Any]],
    continue_probs: Dict[Tuple[str, int], float],
    shallow_by_step: Dict[Tuple[str, int], np.ndarray],
    quality_model: Pipeline,
    *,
    cfg: Any,
    probe_threshold: float,
    gamma: float,
    alpha: float,
    quality_bar: float,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    原版 indicator betting（保留做对照）：
    Probe 要求停止时，若 E-wealth * multiplier 会突破 1/α 则强制继续。
    """
    tracker = EWealthTracker(alpha=float(alpha))
    rows: List[Dict[str, Any]] = []

    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        steps = _sorted_steps(traj)
        if not steps:
            rows.append(_empty_outcome(cfg, gamma))
            tracker.trace.append(tracker.wealth)
            continue

        chosen = steps[-1]
        stopped_early = False

        for step in steps:
            k = int(step.get("step", 0))
            if k >= cfg.max_k:
                chosen = step
                break

            p_cont = continue_probs.get((sample_id, k))
            if p_cont is None:
                continue

            if float(p_cont) >= float(probe_threshold):
                continue

            z = shallow_by_step.get((sample_id, k))
            if z is None:
                continue
            phat = float(predict_success_prob(quality_model, z.reshape(1, -1))[0])
            mult_raw = betting_multiplier_indicator(
                phat, alpha=float(alpha), quality_bar=float(quality_bar)
            )
            next_w = tracker.wealth * mult_raw
            if next_w <= tracker.cap + 1e-12:
                tracker.apply(mult_raw)
                chosen = step
                stopped_early = True
                break

        if not stopped_early:
            z = shallow_by_step.get((sample_id, int(chosen.get("step", 0))))
            if z is not None:
                phat = float(predict_success_prob(quality_model, z.reshape(1, -1))[0])
            else:
                phat = 0.0
            mult_raw = betting_multiplier_indicator(
                phat, alpha=float(alpha), quality_bar=float(quality_bar)
            )
            m = tracker.try_apply_multiplier(mult_raw)
            tracker.apply(m)

        rows.append(_finalize_row(traj, chosen, cfg, gamma))

    return rows, tracker.trace


# ---------------------------------------------------------------------------
# Conformal (split) 门控
# ---------------------------------------------------------------------------

def simulate_conformal_phat_gate(
    trajectories: Sequence[Dict[str, Any]],
    continue_probs: Dict[Tuple[str, int], float],
    shallow_by_step: Dict[Tuple[str, int], np.ndarray],
    quality_model: Pipeline,
    *,
    cfg: Any,
    probe_threshold: float,
    gamma: float,
    min_phat: float,
) -> List[Dict[str, Any]]:
    """
    Split 型门控：Probe 想停时，额外要求 p_hat(F1≥γ) ≥ min_phat。
    min_phat 在 Calib 上按 (1-α) 分位校准。无跨样本 wealth。
    """
    rows: List[Dict[str, Any]] = []
    mp = float(min_phat)

    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        steps = _sorted_steps(traj)
        if not steps:
            rows.append(_empty_outcome(cfg, gamma))
            continue

        chosen = steps[-1]
        for step in steps:
            k = int(step.get("step", 0))
            if k >= cfg.max_k:
                chosen = step
                break
            p_cont = continue_probs.get((sample_id, k))
            if p_cont is None:
                continue
            if float(p_cont) >= float(probe_threshold):
                continue
            z = shallow_by_step.get((sample_id, k))
            if z is None:
                continue
            phat = float(predict_success_prob(quality_model, z.reshape(1, -1))[0])
            if phat >= mp - 1e-12:
                chosen = step
                break

        rows.append(_finalize_row(traj, chosen, cfg, gamma))

    return rows


def conformal_min_phat_threshold(phat_stop: np.ndarray, alpha: float) -> float:
    """Calib 上 probe 停止处的 p_hat 的保守下分位。"""
    p = np.sort(np.asarray(phat_stop, dtype=np.float64).reshape(-1))
    n = int(p.size)
    if n == 0:
        return 0.5
    idx = int(math.ceil((1.0 - float(alpha)) * (n + 1))) - 1
    idx = min(max(idx, 0), n - 1)
    return float(p[idx])


# ---------------------------------------------------------------------------
# 分布漂移序列构造
# ---------------------------------------------------------------------------

def build_shift_ordering(
    trajectories: List[Dict[str, Any]],
    *,
    shift_type: Literal["none", "sudden", "gradual", "periodic"],
    shift_fraction: float = 0.5,
    shift_sort_key: str = "f1",
    rng_seed: int = 42,
) -> List[int]:
    """
    构造漂移测试序列的样本排列。

    - none:    随机排列（基准）
    - sudden:  前 fraction 按原序随机；后 (1-fraction) 全换成最难样本
    - gradual: 按 shift_sort_key 升序排列（从易到难，模拟逐渐恶化）
    - periodic: 交替排列 easy/hard batch（周期=10）
    """
    n = len(trajectories)
    rng = np.random.RandomState(rng_seed)

    if shift_type == "none":
        return list(rng.permutation(n))

    max_f1_per_traj = []
    for i, traj in enumerate(trajectories):
        steps = traj.get("steps") or []
        if not steps:
            max_f1_per_traj.append(0.0)
            continue
        max_f1 = max(float(s.get(shift_sort_key, 0.0) or 0.0) for s in steps)
        max_f1_per_traj.append(max_f1)

    sorted_by_difficulty = np.argsort(max_f1_per_traj)
    easy_half = sorted_by_difficulty[n // 2:]
    hard_half = sorted_by_difficulty[: n // 2]

    if shift_type == "sudden":
        cut = int(n * float(shift_fraction))
        normal_part = list(rng.permutation(n)[:cut])
        shift_part = list(hard_half[: n - cut])
        if len(shift_part) < n - cut:
            shift_part = list(rng.choice(hard_half, size=n - cut, replace=True))
        rng.shuffle(shift_part)
        return normal_part + shift_part

    if shift_type == "gradual":
        return list(reversed(sorted_by_difficulty))

    if shift_type == "periodic":
        period = 10
        order: List[int] = []
        easy_list = list(rng.permutation(easy_half))
        hard_list = list(rng.permutation(hard_half))
        ei, hi = 0, 0
        for batch_idx in range(0, n, period):
            if batch_idx // period % 2 == 0:
                for _ in range(period):
                    if ei < len(easy_list):
                        order.append(int(easy_list[ei]))
                        ei += 1
            else:
                for _ in range(period):
                    if hi < len(hard_list):
                        order.append(int(hard_list[hi]))
                        hi += 1
        while len(order) < n:
            remaining = [i for i in range(n) if i not in set(order)]
            order.extend(remaining)
        return order[:n]

    return list(rng.permutation(n))


# ---------------------------------------------------------------------------
# 错误标签 & 汇总
# ---------------------------------------------------------------------------

def attach_error_labels(rows: Sequence[Dict[str, Any]], gamma: float) -> List[Dict[str, Any]]:
    """为仅含 f1 的行补充 error = 1[f1 < γ]。"""
    g = float(gamma)
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["error"] = int(float(d.get("f1", 0.0)) < g)
        out.append(d)
    return out


def summarize(rows: Sequence[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    f1s = [float(r.get("f1", 0.0)) for r in rows]
    ems = [int(r.get("em", 0)) for r in rows]
    steps = [int(r.get("steps_used", 0)) for r in rows]
    costs = [float(r.get("avg_cost", 0.0)) for r in rows]
    errs = [int(r.get("error", 0)) for r in rows]
    return {
        "strategy": strategy,
        "avg_steps": float(np.mean(steps) if steps else 0.0),
        "avg_cost": float(np.mean(costs) if costs else 0.0),
        "avg_f1": float(np.mean(f1s) if f1s else 0.0),
        "avg_em": float(np.mean(ems) if ems else 0.0),
        "error_rate": float(np.mean(errs) if errs else 0.0),
        "n": int(len(rows)),
    }
