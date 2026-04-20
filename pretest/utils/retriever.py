"""文档检索：BM25；以及 Contriever-MS MARCO 召回 + BGE cross-encoder 重排序。"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


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


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class ContrieverBgeRetriever:
    """
两阶段检索：Contriever 双塔向量打分短名单，再用 bge-reranker cross-encoder 精排取1 条。
    需在每条样本开始时调用 ``prepare_docs`` 缓存段落向量；查询随步变化（含 current_answer 拼接）。
    """

    def __init__(
        self,
        contriever_model: str = "facebook/contriever-msmarco",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        shortlist_k: int = 32,
        rerank_batch_size: int = 8,
        contriever_max_length: int = 512,
        rerank_max_length: int = 512,
    ) -> None:
        dev = (device or "").strip()
        if not dev:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.shortlist_k = max(1, int(shortlist_k))
        self.rerank_batch_size = max(1, int(rerank_batch_size))
        self.contriever_max_length = int(contriever_max_length)
        self.rerank_max_length = int(rerank_max_length)

        self._ctok = AutoTokenizer.from_pretrained(contriever_model)
        self._cmodel = AutoModel.from_pretrained(contriever_model)
        self._cmodel.to(self.device)
        self._cmodel.eval()

        self._rtok = AutoTokenizer.from_pretrained(reranker_model)
        self._rmodel = AutoModelForSequenceClassification.from_pretrained(reranker_model)
        self._rmodel.to(self.device)
        self._rmodel.eval()

        self._docs_pool: List[str] = []
        self._doc_emb: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def _encode_contriever_batch(self, texts: Sequence[str]) -> torch.Tensor:
        out_chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), self.rerank_batch_size):
            batch = list(texts[i : i + self.rerank_batch_size])
            enc = self._ctok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.contriever_max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            mo = self._cmodel(**enc)
            pooled = _mean_pool(mo.last_hidden_state, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            out_chunks.append(pooled)
        return torch.cat(out_chunks, dim=0)

    def prepare_docs(self, docs_pool: List[str]) -> None:
        """同一题内文档集合固定时调用一次，预计算 Contriever 段落向量。"""
        self._docs_pool = list(docs_pool)
        if not self._docs_pool:
            self._doc_emb = None
            return
        self._doc_emb = self._encode_contriever_batch(self._docs_pool)

    def encode_docs(
        self, docs_pool: List[str]
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """编码文档并返回 (docs_copy, doc_emb)，不修改实例状态（线程安全）。
        配合 retrieve_top1 的 docs_pool/doc_emb 参数使用，无需持有锁。"""
        if not docs_pool:
            return list(docs_pool), None
        emb = self._encode_contriever_batch(docs_pool)
        return list(docs_pool), emb

    @torch.inference_mode()
    def retrieve_top1(
        self,
        query: str,
        used: List[int],
        docs_pool: Optional[List[str]] = None,
        doc_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[str, float, int]:
        """在未使用过的文档中返回精排最高分的一条及 reranker 分数。

        docs_pool / doc_emb 可由 encode_docs() 预计算后传入（线程安全无状态模式）。
        不传则退化为使用 prepare_docs() 缓存的共享状态（单线程模式）。
        """
        _pool = docs_pool if docs_pool is not None else self._docs_pool
        _emb = doc_emb if doc_emb is not None else self._doc_emb
        if not _pool or _emb is None:
            return "", 0.0, -1
        exclude = set(used)
        cand_idx = [i for i in range(len(_pool)) if i not in exclude]
        if not cand_idx:
            return "", 0.0, -1

        q_emb = self._encode_contriever_batch([query])
        sub = _emb[torch.tensor(cand_idx, device=self.device)]
        dense_scores = (q_emb @ sub.T).squeeze(0)

        m = len(cand_idx)
        cap = min(self.shortlist_k, m)
        if cap < m:
            top_rel = torch.topk(dense_scores, k=cap, largest=True).indices.tolist()
            shortlist = [cand_idx[j] for j in top_rel]
        else:
            shortlist = cand_idx

        pairs = [[query, _pool[i]] for i in shortlist]
        all_scores: List[float] = []
        for s in range(0, len(pairs), self.rerank_batch_size):
            chunk = pairs[s : s + self.rerank_batch_size]
            enc = self._rtok(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.rerank_max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self._rmodel(**enc).logits.view(-1).float()
            all_scores.extend(logits.detach().cpu().tolist())

        j_best = int(max(range(len(all_scores)), key=lambda j: all_scores[j]))
        doc_i = shortlist[j_best]
        return _pool[doc_i], float(all_scores[j_best]), doc_i
