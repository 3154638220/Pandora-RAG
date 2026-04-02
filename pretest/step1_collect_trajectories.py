"""
Step 1：构建离线 RAG 轨迹缓存
==============================
对 HotpotQA 的每一个 query，强制执行 K=5 轮检索，记录每步的：
  - 检索到的文档
  - 生成的答案
  - 真实 F1 / EM 得分
  - LLM 置信度特征（logprob 均值、熵、token 数、BM25 分数）

输出：data/trajectories.json
运行：python -m pretest.step1_collect_trajectories
"""
import json
import logging
import os
import random
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from pretest.config import cfg
from pretest.utils.llm_client import LLMClient
from pretest.utils.metrics import compute_metrics
from pretest.utils.retriever import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
def build_retrieval_query(question: str, current_answer: str, step: int) -> str:
    """迭代检索的查询拼接策略：首步用原问题，后续步带入当前答案引导检索。"""
    if step == 1 or not current_answer:
        return question
    return f"{question} [当前答案提示: {current_answer}]"


def parse_context(raw_context) -> tuple:
    """
    HotpotQA context 格式：{'title': [...], 'sentences': [[...], ...]}
    返回 (List[str] paragraphs, List[str] titles)
    """
    paragraphs, titles = [], []
    for title, sentences in zip(raw_context["title"], raw_context["sentences"]):
        text = " ".join(sentences).strip()
        paragraphs.append(f"[{title}] {text}")
        titles.append(title)
    return paragraphs, titles


def run_single_trajectory(
    example: dict,
    llm: LLMClient,
    split: str,
) -> dict:
    """对单条 HotpotQA 样例运行完整 K 步 RAG 轨迹并返回结构化记录。"""
    question = example["question"]
    gold_answer = example["answer"]
    paragraphs, titles = parse_context(example["context"])

    retriever = BM25Retriever(paragraphs)
    retrieved_indices: List[int] = []
    accumulated_context = ""
    current_answer = ""
    steps_data = []

    for k in range(1, cfg.max_k + 1):
        # ── 1. 检索 ───────────────────────────────────────────
        query = build_retrieval_query(question, current_answer, k)
        docs, scores, indices = retriever.retrieve(
            query, k=1, exclude_indices=retrieved_indices
        )
        if not docs:
            break

        retrieved_indices.extend(indices)
        retrieved_doc = docs[0]
        bm25_score = scores[0]
        accumulated_context = (
            (accumulated_context + "\n\n" + retrieved_doc).strip()
        )

        # ── 2. 生成答案 ────────────────────────────────────────
        result = llm.generate(question, accumulated_context)
        current_answer = result["answer"]

        # ── 3. 评估质量 ────────────────────────────────────────
        f1, em = compute_metrics(current_answer, gold_answer)

        steps_data.append({
            "step": k,
            "retrieval_query": query,
            "retrieved_idx": indices[0],
            "retrieved_title": titles[indices[0]],
            "retrieved_doc": retrieved_doc,
            "bm25_score": round(bm25_score, 4),
            "answer": current_answer,
            "f1": round(f1, 4),
            "em": bool(em),
            "features": {
                "mean_logprob": round(result["mean_logprob"], 4),
                "entropy": round(result["entropy"], 4),
                "token_count": result["token_count"],
                "bm25_score": round(bm25_score, 4),
                "step": k,
            },
        })

    return {
        "id": example.get("id", ""),
        "question": question,
        "gold_answer": gold_answer,
        "split": split,
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "steps": steps_data,
    }


# ──────────────────────────────────────────────────────────────
def main():
    if os.path.exists(cfg.trajectory_file):
        logger.info("轨迹文件已存在：%s，跳过收集。如需重新收集，请删除该文件。", cfg.trajectory_file)
        return

    logger.info("加载 HotpotQA（%s）数据集...", cfg.dataset_config)
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)

    rng = random.Random(cfg.random_seed)

    # HotpotQA 官方 train split 有 ~90k 条，validation 有 ~7k 条
    train_pool = list(dataset["train"])
    val_pool = list(dataset["validation"])

    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    train_samples = train_pool[: cfg.train_size]
    # 测试集从 validation 抽取，避免数据泄漏
    test_samples = val_pool[: cfg.test_size]

    logger.info(
        "采样完成：train=%d，test=%d，每题最多 %d 步检索。",
        len(train_samples), len(test_samples), cfg.max_k,
    )

    llm = LLMClient(cfg)
    trajectories = []
    total = len(train_samples) + len(test_samples)

    with tqdm(total=total, desc="收集轨迹") as pbar:
        for sample in train_samples:
            traj = run_single_trajectory(sample, llm, split="train")
            trajectories.append(traj)
            pbar.update(1)

        for sample in test_samples:
            traj = run_single_trajectory(sample, llm, split="test")
            trajectories.append(traj)
            pbar.update(1)

    with open(cfg.trajectory_file, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)

    logger.info("轨迹已保存至 %s（共 %d 条）。", cfg.trajectory_file, len(trajectories))

    # 打印简单统计
    train_f1s = [
        t["steps"][-1]["f1"]
        for t in trajectories
        if t["split"] == "train" and t["steps"]
    ]
    test_f1s = [
        t["steps"][-1]["f1"]
        for t in trajectories
        if t["split"] == "test" and t["steps"]
    ]
    import numpy as np
    logger.info(
        "K=%d 时 F1：train=%.4f，test=%.4f",
        cfg.max_k,
        float(np.mean(train_f1s)) if train_f1s else 0,
        float(np.mean(test_f1s)) if test_f1s else 0,
    )


if __name__ == "__main__":
    main()
