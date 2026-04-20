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
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from pretest.hf_env import init_pandora_hf_home

init_pandora_hf_home()

from datasets import load_dataset
from tqdm import tqdm

from pretest.config import cfg
from pretest.utils.llm_client import LLMClient
from pretest.utils.metrics import compute_metrics
from pretest.utils.retriever import BM25Retriever, ContrieverBgeRetriever
from qa_shared.prompts import append_trace_step

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
    dense_rerank: Optional[ContrieverBgeRetriever] = None,
) -> dict:
    """对单条 HotpotQA 样例运行完整 K 步 RAG 轨迹并返回结构化记录。"""
    question = example["question"]
    gold_answer = example["answer"]
    paragraphs, titles = parse_context(example["context"])

    use_dense = dense_rerank is not None
    if use_dense:
        dense_rerank.prepare_docs(paragraphs)
    else:
        bm25_retriever = BM25Retriever(paragraphs)
    retrieved_indices: List[int] = []
    accumulated_context = ""
    current_answer = ""
    trace = ""
    steps_data = []
    backend = (os.getenv("PANDORA_RETRIEVER_BACKEND", "bm25") or "bm25").strip().lower()

    for k in range(1, cfg.max_k + 1):
        # ── 1. 检索 ───────────────────────────────────────────
        query = llm.generate_follow_up_query(question, trace, backend)
        if use_dense:
            retrieved_doc, retr_score, doc_idx = dense_rerank.retrieve_top1(
                query, retrieved_indices
            )
            if doc_idx < 0:
                break
            indices = [doc_idx]
        else:
            docs, scores, indices = bm25_retriever.retrieve(
                query, k=1, exclude_indices=retrieved_indices
            )
            if not docs:
                break
            retrieved_doc = docs[0]
            retr_score = scores[0]

        retrieved_indices.extend(indices)
        bm25_score = retr_score
        accumulated_context = (
            (accumulated_context + "\n\n" + retrieved_doc).strip()
        )
        intermediate_answer = llm.generate_intermediate_answer(query, retrieved_doc)
        trace = append_trace_step(trace, query, retrieved_doc, intermediate_answer)

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
            "intermediate_answer": intermediate_answer,
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
    if cfg.dataset_name == "hotpot_qa" and (cfg.dataset_config or "").strip() == "distractor":
        dataset = load_dataset(
            cfg.dataset_name,
            revision="refs/convert/parquet",
            data_dir="distractor",
        )
    else:
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
    backend = (os.getenv("PANDORA_RETRIEVER_BACKEND", "bm25") or "bm25").strip().lower()
    dense: Optional[ContrieverBgeRetriever] = None
    if backend == "contriever_bge":
        dense = ContrieverBgeRetriever(
            contriever_model=os.getenv("PANDORA_CONTRIEVER_MODEL", "facebook/contriever-msmarco"),
            reranker_model=os.getenv("PANDORA_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            device=(os.getenv("RETRIEVER_DEVICE") or "").strip() or None,
            shortlist_k=int(os.getenv("PANDORA_CONTRIEVER_SHORTLIST_K", "32")),
            rerank_batch_size=int(os.getenv("PANDORA_RERANK_BATCH_SIZE", "8")),
        )
        logger.info("pretest step1：使用 Contriever+BGE 检索（见 PANDORA_CONTRIEVER_MODEL / PANDORA_RERANKER_MODEL）")
    elif backend != "bm25":
        raise ValueError(f"未知 PANDORA_RETRIEVER_BACKEND={backend!r}，可选 bm25 / contriever_bge")

    trajectories = []
    total = len(train_samples) + len(test_samples)

    with tqdm(total=total, desc="收集轨迹") as pbar:
        for sample in train_samples:
            traj = run_single_trajectory(sample, llm, split="train", dense_rerank=dense)
            trajectories.append(traj)
            pbar.update(1)

        for sample in test_samples:
            traj = run_single_trajectory(sample, llm, split="test", dense_rerank=dense)
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
