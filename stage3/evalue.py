from __future__ import annotations

"""
纯函数：E-wealth 更新与 README 中基于示性函数的 betting multiplier。

不含 PyTorch / 轨迹 IO；Stage2 变更不应需要修改本文件。
"""

import math
from dataclasses import dataclass
from typing import List, Sequence


def wealth_cap(alpha: float) -> float:
    """Ville 型边界：关注 E_n 是否超过 1/α。"""
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha 必须在 (0,1)，收到 {alpha}")
    return 1.0 / a


def betting_multiplier_indicator(
    quality_prob: float,
    *,
    alpha: float,
    quality_bar: float,
) -> float:
    """
    README 风格：当质量指标超过阈值时乘以 1/(1-α)，否则视为不增加 wealth（乘子 1）。

    quality_prob：校准集上估计的 P(F1≥γ|特征)（或其它 [0,1] 质量分数）。
    quality_bar：在 calib 上调出的阈值；只有 quality_prob >= quality_bar 才「下注」。
    """
    if quality_prob >= float(quality_bar):
        return 1.0 / (1.0 - float(alpha))
    return 1.0


@dataclass
class EWealthTracker:
    """跨样本累积 E-wealth（乘积过程）。"""

    alpha: float
    wealth: float = 1.0

    @property
    def cap(self) -> float:
        return wealth_cap(self.alpha)

    def try_apply_multiplier(self, mult: float) -> float:
        """
        若 wealth * mult 超过 1/α，返回应使用的**裁剪后**乘子（使更新后恰落在 cap 内）。
        若无需裁剪，返回原 mult。
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
        m = self.try_apply_multiplier(mult)
        self.wealth *= m


def cumulative_error_rate_curve(errors: Sequence[int]) -> List[float]:
    """errors[i] ∈ {0,1} 表示第 i 个样本是否错误；返回前缀平均错误率。"""
    out: List[float] = []
    s = 0
    for i, e in enumerate(errors, start=1):
        s += int(e)
        out.append(float(s) / float(i))
    return out
