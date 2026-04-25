from __future__ import annotations

"""
E-wealth 更新与多种 betting multiplier 实现。

支持两类 betting 策略：
  1. **Indicator**（原版）：p_hat ≥ bar 时乘 1/(1-α)，否则乘 1（仅增不减，不观测结果）。
  2. **Outcome-aware**（推荐）：观测真实错误指标 e_n 后更新，构成严格超鞅：
     M_n = 1 - λ_n + λ_n * e_n / α
     其中 λ_n 为 betting fraction。λ 越大越激进。

不含 PyTorch / 轨迹 IO；Stage2 变更不应需要修改本文件。
"""

import math
from dataclasses import dataclass, field
from typing import List, Literal, Sequence


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def wealth_cap(alpha: float) -> float:
    """Ville 型边界：关注 E_n 是否超过 1/α。"""
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha}")
    return 1.0 / a


# ---------------------------------------------------------------------------
# Indicator betting（原版，保留做基线对照）
# ---------------------------------------------------------------------------

def betting_multiplier_indicator(
    quality_prob: float,
    *,
    alpha: float,
    quality_bar: float,
) -> float:
    """
    原版 indicator 策略：p_hat ≥ bar 时乘 1/(1-α)，否则不下注（乘 1）。
    不观测真实结果，wealth 只增不减。保留做对照基线。
    """
    if quality_prob >= float(quality_bar):
        return 1.0 / (1.0 - float(alpha))
    return 1.0


# ---------------------------------------------------------------------------
# Outcome-aware betting（推荐）
# ---------------------------------------------------------------------------

BettingStrategy = Literal["fixed", "predictive"]


def compute_betting_lambda(
    strategy: BettingStrategy,
    *,
    p_hat: float = 0.5,
    fixed_lambda: float = 0.5,
    eps: float = 0.01,
) -> float:
    """
    计算 betting fraction λ_n ∈ (0, 1)。

    - fixed:       λ = fixed_lambda（常数）
    - predictive:  λ = clip(1 - p_hat, eps, 1-eps)，质量差 → 下注更大
    """
    if strategy == "fixed":
        return max(eps, min(1.0 - eps, float(fixed_lambda)))
    if strategy == "predictive":
        return max(eps, min(1.0 - eps, 1.0 - float(p_hat)))
    raise ValueError(f"未知 betting strategy: {strategy}")


def outcome_aware_multiplier(
    lambda_n: float,
    error: int,
    alpha: float,
) -> float:
    """
    结果感知型 betting multiplier：

        M_n = 1 - λ_n + λ_n * e_n / α

    性质：
    - 若 e_n=0（正确停止）：M = 1 - λ < 1 → wealth 下降
    - 若 e_n=1（错误停止）：M = 1 - λ + λ/α > 1 → wealth 上升
    - E_H₀[M] = 1（当真实错误率 = α 时为鞅）
    - E_H₀[M] < 1（当真实错误率 < α 时为超鞅）
    """
    lam = float(lambda_n)
    e = int(error)
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha 需在 (0,1)，收到 {a}")
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lambda 需在 (0,1)，收到 {lam}")
    return 1.0 - lam + lam * float(e) / a


# ---------------------------------------------------------------------------
# E-Wealth Tracker
# ---------------------------------------------------------------------------

@dataclass
class EWealthTracker:
    """跨样本累积 E-wealth（乘积过程）。"""

    alpha: float
    wealth: float = 1.0
    trace: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.trace:
            self.trace = [self.wealth]

    @property
    def cap(self) -> float:
        return wealth_cap(self.alpha)

    @property
    def exceeded_cap(self) -> bool:
        return self.wealth >= self.cap - 1e-12

    def try_apply_multiplier(self, mult: float) -> float:
        """
        若 wealth * mult 超过 1/α，返回裁剪后乘子；否则返回原 mult。
        """
        m = float(mult)
        if m <= 0.0 or math.isnan(m) or math.isinf(m):
            raise ValueError(f"非法 betting 乘子: {m}")
        next_w = self.wealth * m
        cap = self.cap
        if next_w <= cap + 1e-12:
            return m
        return cap / self.wealth * (1.0 - 1e-9)

    def apply(self, mult: float) -> None:
        """应用乘子并记录 trace。"""
        m = self.try_apply_multiplier(mult)
        self.wealth *= m
        self.trace.append(self.wealth)

    def apply_outcome(
        self,
        error: int,
        lambda_n: float,
    ) -> float:
        """
        结果感知更新：观测 e_n 后计算 M_n 并应用。返回实际使用的 multiplier。
        """
        mult = outcome_aware_multiplier(lambda_n, error, self.alpha)
        self.apply(mult)
        return mult

    def would_exceed_cap_with_indicator(self, quality_bar: float, p_hat: float) -> bool:
        """原版 indicator 策略下，是否会超过 cap（用于向后兼容的门控判断）。"""
        mult = betting_multiplier_indicator(p_hat, alpha=self.alpha, quality_bar=quality_bar)
        return self.wealth * mult > self.cap + 1e-12


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def cumulative_error_rate_curve(errors: Sequence[int]) -> List[float]:
    """errors[i] ∈ {0,1} 表示第 i 个样本是否错误；返回前缀平均错误率。"""
    out: List[float] = []
    s = 0
    for i, e in enumerate(errors, start=1):
        s += int(e)
        out.append(float(s) / float(i))
    return out
