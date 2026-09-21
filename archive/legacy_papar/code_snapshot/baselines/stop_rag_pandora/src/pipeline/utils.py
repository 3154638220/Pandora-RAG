from typing import List, Dict, Any, Tuple

from ..pandora_repo_defaults import ensure_pandora_repo_root_on_path

ensure_pandora_repo_root_on_path()

from qa_shared.metrics import compute_exact_match, compute_f1, normalize_answer


def _answer_aliases(question: Dict[str, Any], fields: Dict[str, str]) -> List[str]:
    raw = [question[fields["answer"]]] + question.get(fields["answer_aliases"], [])
    return [str(g) for g in raw if g is not None and str(g).strip() != ""]


def print_metrics(metrics_list, metric_name="Metrics"):
    if not metrics_list or not any(metrics_list):
        print(f"No {metric_name} available.")
        return

    metrics_flat = [item for sublist in metrics_list for item in sublist]

    print(f"===== {metric_name} =====")
    print(f"Count: {[len(m) for m in metrics_list]}")

    averages = [sum(m) / len(m) if m else 0 for m in metrics_list]
    print(f"{metric_name} by hop: {[f'{avg:.4f}' for avg in averages]}")

    total_avg = sum(metrics_flat) / len(metrics_flat) if metrics_flat else 0
    print(f"Total {metric_name}: {total_avg:.4f}")


def compute_retrieval_metrics(
    questions: List[Dict[str, Any]],
    batch_history: List[List[Dict[str, Any]]],
    stop_logs: List[Dict[str, Any]],
    fields: Dict[str, str],
) -> Tuple[List[List[float]], ...]:
    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(questions, batch_history):
        qid = question[fields["id"]]
        gold_hop = len(question.get(fields["supporting_facts"], []))
        correct = sum(int(qid + "-sf" in doc["id"]) for doc in history)
        retrieved = len(history)

        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        for log in stop_logs:
            if log["question_id"] == qid:
                log.update({"em": em, "precision": precision, "recall": recall, "f1": f1})
                break

        idx = min(max(gold_hop - 2, 0), 2)
        em_list[idx].append(em)
        precision_list[idx].append(precision)
        recall_list[idx].append(recall)
        f1_list[idx].append(f1)

    return em_list, precision_list, recall_list, f1_list


def compute_answer_metrics(
    questions: List[Dict[str, Any]], predictions: List[str], fields: Dict[str, str]
) -> Tuple[List[List[float]], List[List[float]]]:
    em_list = [[], [], []]
    f1_list = [[], [], []]
    acc_list = [[], [], []]

    for question, prediction in zip(questions, predictions):
        hop = len(question.get(fields["supporting_facts"], []))
        idx = min(max(hop - 2, 0), 2)

        gold_answers = _answer_aliases(question, fields)
        pred = prediction if prediction is not None else ""

        if not gold_answers:
            em, f1, acc = 0.0, 0.0, 0
        else:
            em = int(any(compute_exact_match(pred, g) for g in gold_answers))
            f1 = max(compute_f1(pred, g) for g in gold_answers)
            npred = normalize_answer(pred)
            acc = int(any(npred in normalize_answer(g) for g in gold_answers))

        em_list[idx].append(em)
        f1_list[idx].append(f1)
        acc_list[idx].append(acc)

    return em_list, f1_list, acc_list


def compute_all_answer_metrics(
    questions: List[Dict[str, Any]], predictions: List[str], fields: Dict[str, str]
) -> Tuple[List[float], List[float]]:
    em_list = []
    f1_list = []
    acc_list = []

    for question, prediction in zip(questions, predictions):
        gold_answers = _answer_aliases(question, fields)
        pred = prediction if prediction is not None else ""

        if not gold_answers:
            em, f1, acc = 0.0, 0.0, 0
        else:
            em = int(any(compute_exact_match(pred, g) for g in gold_answers))
            f1 = max(compute_f1(pred, g) for g in gold_answers)
            npred = normalize_answer(pred)
            acc = int(any(npred in normalize_answer(g) for g in gold_answers))

        em_list.append(em)
        f1_list.append(f1)
        acc_list.append(acc)

    return em_list, f1_list, acc_list
