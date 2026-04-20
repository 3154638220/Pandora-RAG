"""
E-wealth 触及 cap 的时刻统计（用于漂移后「检测延迟」与 JSON 报告）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def evalue_cap_timing(
    wealth_trace: List[float],
    *,
    cap: float,
    n_samples: int,
    shift_type: str,
    shift_fraction: float,
) -> Dict[str, Any]:
    """
    wealth_trace[k]：处理完前 k 个（按当前测试排列顺序）样本后的 E-wealth；k=0 为初值 1。

    sudden：前 ``cut=int(n*shift_fraction)`` 个样本为「正常段」，其后为难例段（与 ``build_shift_ordering`` 一致）。
    gradual / periodic：同样用 ``cut`` 表示流前半/后半分界，用于报告「后半段首次触 cap」。
    """
    cap_tol = max(1e-6 * cap, 1e-9)

    def at_cap(w: float) -> bool:
        return w >= cap - cap_tol

    first_global: Optional[int] = None
    for i, w in enumerate(wealth_trace):
        if at_cap(w):
            first_global = int(i)
            break

    cut: Optional[int] = None
    first_after_shift: Optional[int] = None
    samples_after_shift: Optional[int] = None
    if shift_type != "none":
        cut = int(float(n_samples) * float(shift_fraction))
        # 首个后段样本处理完毕后 wealth 下标为 cut+1
        for i in range(cut + 1, len(wealth_trace)):
            if at_cap(wealth_trace[i]):
                first_after_shift = int(i)
                samples_after_shift = int(i - cut)
                break

    return {
        "shift_stream_cut_processed": cut,
        "first_trace_index_reaching_cap": first_global,
        "samples_processed_at_first_cap": first_global,
        "post_shift_first_trace_index_reaching_cap": first_after_shift,
        "samples_after_shift_to_first_cap": samples_after_shift,
        "ever_reaches_cap": bool(first_global is not None),
    }
