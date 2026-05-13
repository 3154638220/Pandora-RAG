"""Shared HotpotQA-style answer normalization and metrics."""

import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(prediction: str, ground_truth: str) -> Tuple[float, bool]:
    return compute_f1(prediction, ground_truth), compute_exact_match(prediction, ground_truth)


def _answer_piece_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_answer_piece_str(v) for v in value if _answer_piece_str(v))
    return str(value).strip()


def build_gold_answers(example: Dict[str, Any]) -> List[str]:
    """
    主答案 + answer_aliases，去空、去重，主答案始终在下标 0。
    与 Stop-RAG MuSiQue 口径一致：[answer] + answer_aliases。
    """
    main = _answer_piece_str(
        example.get("answer") or example.get("answers") or example.get("output")
    )
    raw_aliases = example.get("answer_aliases")
    extras: List[str] = []
    if isinstance(raw_aliases, (list, tuple)):
        for a in raw_aliases:
            t = _answer_piece_str(a)
            if t:
                extras.append(t)
    elif raw_aliases is not None:
        t = _answer_piece_str(raw_aliases)
        if t:
            extras.append(t)
    out: List[str] = []
    seen: set = set()
    if main:
        out.append(main)
        seen.add(main)
    for a in extras:
        if a not in seen:
            out.append(a)
            seen.add(a)
    if not out:
        return [""]
    return out


def compute_metrics_multi(prediction: str, ground_truths: Iterable[str]) -> Tuple[float, bool]:
    golds = [str(g) for g in ground_truths if g is not None and str(g).strip()]
    if not golds:
        return 0.0, False
    return (
        max(compute_f1(prediction, g) for g in golds),
        any(compute_exact_match(prediction, g) for g in golds),
    )
