"""基于 BM25 的文档检索器，用于在 HotpotQA 候选段落中进行迭代检索。"""
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """在给定文档集合上做 BM25 检索，支持逐步排除已检索文档。"""

    def __init__(self, documents: List[str]):
        self.documents = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        k: int = 1,
        exclude_indices: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        返回 (文档列表, BM25分数列表, 文档索引列表)。
        exclude_indices: 已检索过的文档下标，本次不再重复选取。
        """
        exclude = set(exclude_indices or [])
        scores = self.bm25.get_scores(query.lower().split())

        ranked = sorted(
            [(i, s) for i, s in enumerate(scores) if i not in exclude],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        indices = [i for i, _ in ranked]
        retrieved_docs = [self.documents[i] for i in indices]
        retrieved_scores = [s for _, s in ranked]
        return retrieved_docs, retrieved_scores, indices
