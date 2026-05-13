"""标准 HotpotQA 评估指标：F1 Score 与 Exact Match。"""

from qa_shared.metrics import (
    compute_exact_match,
    compute_f1,
    compute_metrics,
    compute_metrics_multi,
    normalize_answer,
)
