"""
Stage-1 pipeline for Pandora-RAG.

Covers:
  A. Data normalization and deterministic split manifests
  B. Trajectory caching with deep-ish features
  C. Oracle labels + Pareto frontier
  D. Stage report generation and quality gates

Usage:
  python -m stage1.run_stage1 --datasets hotpotqa,musique,2wiki --max-k 5

动态特征（B1）：每步检索与生成后，将 hidden state 存为 ``cache/features/.../hidden_states/{id}_step{k}.npz``。
若已有轨迹但缺 .npz，可用 ``--skip-prepare --reextract-hidden-only`` 仅从 trajectories.jsonl 重提（不重复调用 LLM）。
若某 split 的 trajectories.jsonl 条数少于 data/processed 下对应 jsonl，说明该 split 轨迹未跑满；可用 ``--skip-prepare --collect-splits train`` 只补跑指定 split 的检索+LLM+每步 hidden（其余 split 沿用已有缓存）。

``--skip-prepare`` 表示「跳过已具备完整 ``data/processed`` 与 manifest 的数据集的 prepare」；若多数据集串联时某一数据集尚缺这些文件，则**仅对该数据集自动执行** ``prepare_data``（从 HF 拉取并写出），不会重算已有产物的数据集。

数据划分默认 Train=4000 / Calib=1000 / Dev=1000 / Test=1000；无独立 test split 时从 validation 划 test，
若 validation 总条数不足 Calib+Dev+Test，则自动收窄 test（保证 Calib/Dev 满额），详见 docs/experiments.md A2。
NLI 默认 CPU（NLI_DEVICE）。权重可放任意盘：设 NLI_MODEL_DIR 指向本地下载目录
（如 models/cross-encoder-nli-deberta-v3-small）即离线加载。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pretest.hf_env import init_pandora_hf_home

init_pandora_hf_home()

# 未设置 HF_ENDPOINT 时默认走 hf-mirror，减轻国内直连 huggingface.co 的延迟。
if os.environ.get("PANDORA_NO_CN_HF_MIRROR", "").lower() not in ("1", "true", "yes"):
    if not (os.environ.get("HF_ENDPOINT") or "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import Dataset, load_dataset
from tqdm import tqdm

from pretest.utils.llm_client import LLMClient
from qa_shared.metrics import build_gold_answers, compute_metrics_multi
from pretest.utils.retriever import BM25Retriever, ContrieverBgeRetriever
from pretest.utils.weitzman import (
    compute_all_reservation_values,
    compute_trajectory_oracle,
    oracle_stopping_simulation,
    trajectory_cumulative_cost,
)
from qa_shared.prompts import append_trace_step, format_answer_prompt

# Hugging Face 上可用的 Parquet 镜像（旧名 musique / 2wikimultihopqa 已不可用）
# hotpotqa：datasets>=3 下官方脚本含已废弃的 List feature；用 Hub ``refs/convert/parquet`` 下的
# ``distractor/`` 目录（load_dataset(..., data_dir="distractor")），勿再传 config 名 ``distractor``。
HF_DATASET_IDS = {
    "hotpotqa": ("hotpot_qa", None, "refs/convert/parquet", "distractor"),
    "musique": ("dgslibisey/MuSiQue", None, None, None),
    "2wiki": ("framolfese/2WikiMultihopQA", None, None, None),
}

# 勿请求 HF 的 test split，由 prepare_data 从 validation 尾部切出 test：
# - hotpotqa / musique：镜像无 test；
# - 2wiki：framolfese 镜像的 test 分割 gold 为空（answer、supporting_facts 全空），无法算 F1/Oracle。
HF_DATASETS_WITHOUT_TEST_SPLIT = frozenset({"hotpotqa", "musique", "2wiki"})

DEFAULT_LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_LLM_API_BASE = "http://127.0.0.1:8000/v1"

LOGGER = logging.getLogger(__name__)


@dataclass
class Stage1Config:
    seed: int = 42
    max_k: int = 5
    train_quota: int = 4000
    calib_quota: int = 1000
    dev_quota: int = 1000
    test_quota: int = 1000
    temperature: float = 0.0
    n_samples: int = 1
    cost_per_step: float = 0.05
    oracle_cost_metric: str = "fixed"
    embed_dim: int = 256
    hidden_state_model: Optional[str] = None
    hidden_state_max_length: int = 2048
    root_dir: Path = Path(".")
    # 迭代检索后端：bm25（默认）或 contriever_bge（facebook/contriever-msmarco + BAAI/bge-reranker-v2-m3）
    retriever_backend: str = "bm25"
    contriever_model_id: str = "facebook/contriever-msmarco"
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    contriever_shortlist_k: int = 32
    rerank_batch_size: int = 8
    retriever_device: Optional[str] = None
    # 并行 worker 数（多线程并发 item 处理，LLM 调用并发，本地模型推理加锁串行）
    num_workers: int = 1
    # 跳过每步 self_eval HTTP 调用（节省 50% LLM 请求，self_eval_score 记为 0）
    skip_self_eval: bool = False

    @property
    def data_processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def split_manifest_dir(self) -> Path:
        return self.root_dir / "data" / "splits"

    @property
    def trajectories_dir(self) -> Path:
        return self.root_dir / "cache" / "trajectories"

    @property
    def features_dir(self) -> Path:
        return self.root_dir / "cache" / "features"

    @property
    def oracle_dir(self) -> Path:
        return self.root_dir / "artifacts" / "oracle"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    @property
    def docs_dir(self) -> Path:
        return self.root_dir / "docs"


def _ensure_dirs(cfg: Stage1Config) -> None:
    for p in [
        cfg.data_processed_dir,
        cfg.split_manifest_dir,
        cfg.trajectories_dir,
        cfg.features_dir,
        cfg.oracle_dir,
        cfg.results_dir,
        cfg.docs_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_normalize_text(v) for v in value if _normalize_text(v))
    return str(value).strip()


def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def _heuristic_nli(question: str, context: str) -> Tuple[float, float]:
    q_tokens = set(question.lower().split())
    c_tokens = set(context.lower().split())
    if not q_tokens or not c_tokens:
        return 0.0, 0.0
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    entail = min(1.0, overlap * 1.2)
    contra = max(0.0, 0.3 - overlap * 0.5)
    return float(entail), float(contra)


class NLICrossEncoderScorer:
    """cross-encoder/nli-deberta-v3-small：文档 vs 问题+历史上下文（docs/experiments.md B2）。"""

    DEFAULT_HUB_ID = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, device: str = "cpu", model_id: Optional[str] = None) -> None:
        self._device = device
        env_id = (os.getenv("NLI_MODEL_DIR") or "").strip()
        self.model_id = (model_id or env_id or self.DEFAULT_HUB_ID).strip()
        self._tokenizer = None
        self._model = None
        self._ent_idx: Optional[int] = None
        self._con_idx: Optional[int] = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.to(self._device)
        self._model.eval()
        id2l = getattr(self._model.config, "id2label", None) or {}
        for idx, lab in id2l.items():
            low = str(lab).lower()
            if "entail" in low:
                self._ent_idx = int(idx)
            if "contrad" in low:
                self._con_idx = int(idx)
        if self._ent_idx is None:
            self._ent_idx = 1
        if self._con_idx is None:
            self._con_idx = 0
        LOGGER.info("NLI CrossEncoder 已加载: %s (%s)", self.model_id, self._device)

    def entail_contra(self, premise: str, hypothesis: str) -> Tuple[float, float]:
        if not _normalize_text(premise) or not _normalize_text(hypothesis):
            return 0.0, 0.0
        try:
            self._lazy_init()
            import torch
            import torch.nn.functional as F

            batch = self._tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                logits = self._model(**batch).logits
                probs = F.softmax(logits, dim=-1)[0]
            ent = float(probs[self._ent_idx].item())
            con = float(probs[self._con_idx].item()) if self._con_idx is not None else 0.0
            return ent, con
        except Exception as exc:
            LOGGER.warning("NLI 推理失败，回退启发式：%s", exc)
            return _heuristic_nli(hypothesis, premise)


def _resolve_hidden_state_device(torch_module: Any) -> str:
    """
    为 transformers 侧选择运行设备。
    - 未设置 HIDDEN_STATE_DEVICE 或设为 auto：在可见 GPU 中选剩余显存（free）最大的一张。
    - 否则：cpu / cuda:N / cuda:0 等形式按字面使用（便于调试或与 vLLM 错卡）。
    """
    raw = (os.getenv("HIDDEN_STATE_DEVICE") or "").strip()
    if raw:
        low = raw.lower()
        if low == "auto":
            pass
        elif low == "cpu":
            return "cpu"
        elif low.startswith("cuda:"):
            return raw
        elif raw.isdigit():
            return f"cuda:{raw}"
        return raw

    if not torch_module.cuda.is_available():
        return "cpu"

    best_idx = 0
    best_free = -1
    for idx in range(torch_module.cuda.device_count()):
        free_b, _total_b = torch_module.cuda.mem_get_info(idx)
        if free_b > best_free:
            best_free = free_b
            best_idx = idx
    dev = f"cuda:{best_idx}"
    LOGGER.info(
        "HiddenStateExtractor 选用剩余显存最多的 GPU: %s (约 %.2f GiB 空闲)",
        dev,
        best_free / (1024**3),
    )
    return dev


def _resolve_local_llama_weights_dir(cfg: Stage1Config) -> Optional[Path]:
    """
    解析 Meta-Llama-3.1-8B-Instruct 本地权重目录（见 docs/STORAGE_LAYOUT.md：优先 --root-dir、再仓库 models/、再 PANDORA_MODELS_ROOT）。
    顺序：--root-dir 下 models/ → 本仓库根目录 models/ → 环境变量 PANDORA_MODELS_ROOT。
    """
    name = "Meta-Llama-3.1-8B-Instruct"
    repo_root = Path(__file__).resolve().parent.parent
    candidates: List[Path] = [
        cfg.root_dir / "models" / name,
        repo_root / "models" / name,
    ]
    env_root = (os.getenv("PANDORA_MODELS_ROOT") or "").strip()
    if env_root:
        candidates.append(Path(env_root) / name)
    for p in candidates:
        if p.is_dir() and (p / "config.json").exists():
            return p
    return None


class HiddenStateExtractor:
    """Extracts real final-layer hidden states from a causal LM."""

    def __init__(self, cfg: Stage1Config):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "提取真实 hidden states 需要安装 transformers 与 torch。"
                "请先执行: pip install -U transformers torch"
            ) from exc

        model_hint = (cfg.hidden_state_model or "").strip()
        env_model = (os.getenv("HIDDEN_STATE_MODEL") or "").strip()
        model_name = model_hint or env_model or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        local_llama_path = _resolve_local_llama_weights_dir(cfg)
        if not model_hint and not env_model and local_llama_path is not None:
            model_name = str(local_llama_path)

        self._torch = torch
        self.max_length = max(64, int(cfg.hidden_state_max_length))
        self.model_name = model_name
        self.device = _resolve_hidden_state_device(torch)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        LOGGER.info(
            "HiddenStateExtractor ready: model=%s, device=%s, max_length=%d",
            self.model_name,
            self.device,
            self.max_length,
        )

    def extract(self, question: str, context: str, answer: str) -> Tuple[np.ndarray, np.ndarray]:
        prompt = format_answer_prompt(question=question, context=context, answer=answer)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self._torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)

        last_layer = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        attn_mask = encoded["attention_mask"][0].bool()
        valid_states = last_layer[attn_mask]
        if valid_states.shape[0] == 0:
            valid_states = last_layer

        last_token = valid_states[-1]
        mean_pool = valid_states.mean(dim=0)
        return (
            last_token.detach().cpu().to(self._torch.float16).numpy(),
            mean_pool.detach().cpu().to(self._torch.float16).numpy(),
        )


def _semantic_entropy_and_consistency(samples: Sequence[str]) -> Tuple[float, float]:
    cleaned = [_normalize_text(s).lower() for s in samples if _normalize_text(s)]
    if not cleaned:
        return 0.0, 0.0
    counts = Counter(cleaned)
    probs = np.array([c / len(cleaned) for c in counts.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    majority_ratio = float(max(counts.values()) / len(cleaned))
    return entropy, majority_ratio


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_docs_from_raw(example: Dict[str, Any]) -> List[str]:
    docs: List[str] = []
    if "context" in example and isinstance(example["context"], dict):
        ctx = example["context"]
        if "title" in ctx and "sentences" in ctx:
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                text = " ".join(sents) if isinstance(sents, list) else _normalize_text(sents)
                docs.append(f"[{_normalize_text(title)}] {_normalize_text(text)}".strip())
    if not docs and "paragraphs" in example and isinstance(example["paragraphs"], list):
        for p in example["paragraphs"]:
            if isinstance(p, dict):
                title = _normalize_text(p.get("title"))
                text = _normalize_text(
                    p.get("paragraph_text") or p.get("text") or p.get("paragraph")
                )
                chunk = f"[{title}] {text}".strip() if title else text
                if chunk:
                    docs.append(chunk)
            else:
                t = _normalize_text(p)
                if t:
                    docs.append(t)
    if not docs and "documents" in example and isinstance(example["documents"], list):
        docs.extend(_normalize_text(p) for p in example["documents"] if _normalize_text(p))
    if not docs and "supporting_facts" in example:
        docs.append(_normalize_text(example["supporting_facts"]))
    if not docs and "question" in example:
        docs.append(_normalize_text(example["question"]))
    return [d for d in docs if d]


def _extract_hop_count(example: Dict[str, Any], dataset_name: str, record_id: str) -> int:
    """Heuristically extract ground-truth required hop count."""

    def _from_supporting_facts(obj: Any) -> int:
        if isinstance(obj, dict):
            titles = obj.get("title")
            if isinstance(titles, list):
                uniq = {_normalize_text(t) for t in titles if _normalize_text(t)}
                if uniq:
                    return len(uniq)
        if isinstance(obj, list):
            uniq_titles = set()
            for item in obj:
                if isinstance(item, (list, tuple)) and item:
                    title = _normalize_text(item[0])
                    if title:
                        uniq_titles.add(title)
                elif isinstance(item, dict):
                    title = _normalize_text(item.get("title"))
                    if title:
                        uniq_titles.add(title)
            if uniq_titles:
                return len(uniq_titles)
        return -1

    ds = dataset_name.lower()
    sf_hop = _from_supporting_facts(example.get("supporting_facts"))
    if sf_hop > 0:
        return sf_hop

    if "musique" in ds:
        decomposition = example.get("question_decomposition") or example.get("decomposition")
        if isinstance(decomposition, list) and decomposition:
            return len(decomposition)
        m = re.match(r"^\s*(\d+)hop__", record_id)
        if m:
            return int(m.group(1))
        return -1

    if "hotpotqa" in ds:
        return 2

    return -1


def _normalize_record(example: Dict[str, Any], dataset_name: str, split: str, idx: int) -> Dict[str, Any]:
    q = _normalize_text(example.get("question") or example.get("query") or example.get("input"))
    gold_answers = build_gold_answers(example)
    record_id = _normalize_text(example.get("id") or example.get("_id")) or f"{dataset_name}_{split}_{idx:07d}"
    gt_hop_count = _extract_hop_count(example, dataset_name, record_id)
    return {
        "id": record_id,
        "dataset": dataset_name,
        "split": split,
        "question": q,
        "answer": gold_answers[0],
        "answer_aliases": gold_answers[1:],
        "gold_answers": gold_answers,
        "gt_hop_count": int(gt_hop_count),
        "supporting_facts": example.get("supporting_facts", None),
        "documents": _to_docs_from_raw(example),
    }


def _sample_indices(n_total: int, wanted: int, rng: random.Random) -> List[int]:
    if n_total <= wanted:
        return list(range(n_total))
    all_idx = list(range(n_total))
    rng.shuffle(all_idx)
    selected = sorted(all_idx[:wanted])
    return selected


def _prepared_manifest_path(cfg: Stage1Config, dataset_name: str) -> Path:
    return cfg.split_manifest_dir / f"{dataset_name}_seed{cfg.seed}_manifest.json"


def _missing_prepared_artifacts(cfg: Stage1Config, dataset_name: str) -> List[Path]:
    required: List[Path] = [
        cfg.data_processed_dir / dataset_name / "train.jsonl",
        cfg.data_processed_dir / dataset_name / "calib.jsonl",
        cfg.data_processed_dir / dataset_name / "dev.jsonl",
        cfg.data_processed_dir / dataset_name / "test.jsonl",
        _prepared_manifest_path(cfg, dataset_name),
    ]
    return [p for p in required if not p.exists()]


def _load_hf_split(dataset_name: str, split_name: str) -> Optional[Dataset]:
    try:
        spec = HF_DATASET_IDS.get(dataset_name)
        if not spec:
            return None
        repo, config_name, revision, data_dir = spec
        kw: Dict[str, Any] = {}
        if revision:
            kw["revision"] = revision
        if data_dir:
            kw["data_dir"] = data_dir
        if config_name:
            ds = load_dataset(repo, config_name, split=split_name, **kw)
        else:
            ds = load_dataset(repo, split=split_name, **kw)
        return ds
    except Exception as exc:
        LOGGER.warning("加载数据集 %s split=%s 失败：%s", dataset_name, split_name, exc)
        return None


def prepare_data(cfg: Stage1Config, dataset_name: str) -> Dict[str, int]:
    """
    按 docs/experiments.md：Train/Calib/Dev/Test 四切分，id 互不重叠；Calib+Dev 均从同一条 validation
    池中用 seed 打乱后顺序切出，专用于后续 E-value / CP 校准（严禁与 Test 重叠）。
    """
    rng = random.Random(cfg.seed)
    train_raw = _load_hf_split(dataset_name, "train")
    val_raw = _load_hf_split(dataset_name, "validation")
    if dataset_name in HF_DATASETS_WITHOUT_TEST_SPLIT:
        test_raw = None
    else:
        test_raw = _load_hf_split(dataset_name, "test")

    if train_raw is None and val_raw is None and test_raw is None:
        raise RuntimeError(
            f"无法加载数据集 {dataset_name}。请检查 Hugging Face 可访问性（或 HF_ENDPOINT 设置），"
            "或先手动准备 data/processed 与 data/splits。"
        )

    train_list = list(train_raw) if train_raw is not None else []
    val_list = list(val_raw) if val_raw is not None else []
    test_list = list(test_raw) if test_raw is not None else []

    need_val = cfg.calib_quota + cfg.dev_quota
    if not test_list and val_list:
        V = len(val_list)
        if V < need_val:
            raise RuntimeError(
                f"{dataset_name}: validation 共 {V} 条，不足以划分 Calib+Dev（需 {need_val} 条互不重复样本）。"
            )
        # 无独立 test split 时从 validation 尾部划 test，但必须为 Calib+Dev 留出空间。
        # MuSiQue 等数据集的 validation 较小（≈2.4k），若固定 test=1000 会导致剩余 <2000。
        max_test = V - need_val
        cut = min(cfg.test_quota, max_test)
        if cut < 1:
            raise RuntimeError(
                f"{dataset_name}: validation={V} 在预留 Calib+Dev={need_val} 后无法划出 test。"
            )
        if cut < cfg.test_quota:
            LOGGER.warning(
                "%s: validation=%d 条，test 由请求的 %d 收窄为 %d，以保证 Calib=%d + Dev=%d。",
                dataset_name,
                V,
                cfg.test_quota,
                cut,
                cfg.calib_quota,
                cfg.dev_quota,
            )
        test_list = val_list[-cut:]
        val_list = val_list[:-cut]

    if len(val_list) < need_val:
        raise RuntimeError(
            f"{dataset_name}: validation 池不足以划分 Calib+Dev（需 {need_val} 条互不重复样本，"
            f"当前 validation 剩余 {len(val_list)} 条）。"
        )

    val_order = list(range(len(val_list)))
    rng.shuffle(val_order)
    calib_pick = sorted(val_order[: cfg.calib_quota])
    dev_pick = sorted(val_order[cfg.calib_quota : need_val])

    calib_rows = [
        _normalize_record(val_list[i], dataset_name, "calib", j) for j, i in enumerate(calib_pick)
    ]
    dev_rows = [
        _normalize_record(val_list[i], dataset_name, "dev", j) for j, i in enumerate(dev_pick)
    ]

    train_idx = _sample_indices(len(train_list), cfg.train_quota, rng)
    train_rows = [
        _normalize_record(train_list[i], dataset_name, "train", j) for j, i in enumerate(train_idx)
    ]

    test_idx = _sample_indices(len(test_list), cfg.test_quota, rng)
    test_rows = [
        _normalize_record(test_list[i], dataset_name, "test", j) for j, i in enumerate(test_idx)
    ]

    splits_out: Dict[str, List[Dict[str, Any]]] = {
        "train": train_rows,
        "calib": calib_rows,
        "dev": dev_rows,
        "test": test_rows,
    }
    quota_requested = {
        "train": cfg.train_quota,
        "calib": cfg.calib_quota,
        "dev": cfg.dev_quota,
        "test": cfg.test_quota,
    }
    quotas_effective = {
        "train": len(train_rows),
        "calib": len(calib_rows),
        "dev": len(dev_rows),
        "test": len(test_rows),
    }

    counts: Dict[str, int] = {}
    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "seed": cfg.seed,
        "quota_requested": quota_requested,
        "quota": quotas_effective,
        "selected_ids": {},
        "timestamp": int(time.time()),
    }

    for split, normalized in splits_out.items():
        out_path = cfg.data_processed_dir / dataset_name / f"{split}.jsonl"
        _write_jsonl(out_path, normalized)
        manifest["selected_ids"][split] = [r["id"] for r in normalized]
        counts[split] = len(normalized)

    manifest_path = _prepared_manifest_path(cfg, dataset_name)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return counts


def _context_upto_step(steps_sorted: List[Dict[str, Any]], k_target: int) -> str:
    """
    与 collect_trajectories 中一致：按步序累积非空 retrieved_doc，得到第 k_target 步 extract 时用的上下文。
    """
    acc = ""
    for s in steps_sorted:
        kk = int(s.get("step", 0))
        if kk > k_target:
            break
        doc = _normalize_text(s.get("retrieved_doc", ""))
        if doc:
            acc = (acc + "\n\n" + doc).strip()
    return acc


def reextract_hidden_states_from_trajectories(
    cfg: Stage1Config,
    dataset_name: str,
    split: str,
    hidden_state_extractor: HiddenStateExtractor,
) -> int:
    """
    不调用检索/LLM，仅根据已有 trajectories.jsonl 重建每步 (question, context, answer) 并写入 ``{id}_step{k}.npz``。
    """
    traj_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
    if not traj_path.exists():
        raise FileNotFoundError(f"缺少轨迹缓存，无法重提 hidden states：{traj_path}")
    rows = _read_jsonl(traj_path)
    feature_dir = cfg.features_dir / dataset_name / split / "hidden_states"
    feature_dir.mkdir(parents=True, exist_ok=True)
    for old_fp in feature_dir.glob("*.npz"):
        old_fp.unlink()

    for row in tqdm(rows, desc=f"reextract-hidden::{dataset_name}/{split}"):
        q = _normalize_text(row.get("question", ""))
        sample_id = row.get("id")
        if not sample_id or not q:
            continue
        steps = sorted(row.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        for s in steps:
            k = int(s.get("step", 0))
            if k < 1:
                continue
            acc_context = _context_upto_step(steps, k)
            current_answer = _normalize_text(s.get("current_answer", ""))
            embedding_last, embedding_mean = hidden_state_extractor.extract(
                q, acc_context, current_answer
            )
            feat_path = feature_dir / f"{sample_id}_step{k}.npz"
            np.savez_compressed(
                feat_path,
                last_token=embedding_last.astype(np.float16),
                mean_pool=embedding_mean.astype(np.float16),
            )
    return len(rows)


def _retrieve_step_docs(
    cfg: Stage1Config,
    dense_rerank: Optional[ContrieverBgeRetriever],
    query: str,
    docs_pool: List[str],
    used: List[int],
) -> Tuple[str, float, int]:
    if not docs_pool:
        return "", 0.0, -1
    backend = (cfg.retriever_backend or "bm25").strip().lower()
    if backend == "bm25":
        retriever = BM25Retriever(docs_pool)
        docs, scores, idxs = retriever.retrieve(query, k=1, exclude_indices=used)
        if not docs:
            return "", 0.0, -1
        return docs[0], float(scores[0]), int(idxs[0])
    if backend == "contriever_bge":
        if dense_rerank is None:
            raise ValueError("retriever_backend=contriever_bge 但未提供 ContrieverBgeRetriever 实例")
        return dense_rerank.retrieve_top1(query, used)
    raise ValueError(f"未知 retriever_backend={cfg.retriever_backend!r}，可选: bm25, contriever_bge")


def collect_trajectories(
    cfg: Stage1Config,
    dataset_name: str,
    split: str,
    hidden_state_extractor: HiddenStateExtractor,
    nli_scorer: NLICrossEncoderScorer,
    dense_rerank: Optional[ContrieverBgeRetriever] = None,
) -> int:
    in_path = cfg.data_processed_dir / dataset_name / f"{split}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"缺少处理后数据：{in_path}")
    rows = _read_jsonl(in_path)

    llm_cfg_obj = type("TmpCfg", (), {})()
    llm_cfg_obj.api_base = os.getenv("OPENAI_API_BASE", DEFAULT_LLM_API_BASE)
    llm_cfg_obj.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    llm_cfg_obj.model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm_cfg_obj.max_tokens = 150
    llm_cfg_obj.temperature = float(cfg.temperature)
    llm = LLMClient(llm_cfg_obj)

    out_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dir = cfg.features_dir / dataset_name / split / "hidden_states"
    feature_dir.mkdir(parents=True, exist_ok=True)
    # 仅在全量新跑（非续跑）时清空旧 .npz；续跑时保留已完成 item 的特征文件。
    _is_fresh_run = not out_path.exists() or out_path.stat().st_size == 0
    if _is_fresh_run:
        for old_fp in feature_dir.glob("*.npz"):
            old_fp.unlink()

    skip_self_eval: bool = cfg.skip_self_eval or os.getenv(
        "PANDORA_STAGE1_SKIP_SELF_EVAL", ""
    ).lower() in ("1", "true", "yes")
    backend = (cfg.retriever_backend or "bm25").strip().lower()
    num_workers = max(1, int(cfg.num_workers))

    if skip_self_eval:
        LOGGER.info(
            "trajectory::%s/%s skip_self_eval=True（self_eval_score 记为 0）",
            dataset_name, split,
        )
    if num_workers > 1:
        LOGGER.info(
            "trajectory::%s/%s 并行模式 num_workers=%d",
            dataset_name, split, num_workers,
        )

    # 每个本地 GPU 模型用独立锁串行调用，防止多线程并发的 CUDA stream 竞争。
    # vLLM HTTP 调用无需锁（httpx 客户端线程安全）。
    _hidden_lock = threading.Lock()
    _retriever_lock = threading.Lock()
    _nli_lock = threading.Lock()

    def _process_item(
        row: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], List[Tuple[Path, np.ndarray, np.ndarray]]]]:
        """处理单条轨迹，返回 (traj_dict, [(feat_path, emb_last, emb_mean)])；失败返回 None。"""
        try:
            q = row["question"]
            gold_answers = row.get("gold_answers")
            if not gold_answers:
                gold_answers = [row.get("answer") or ""]
            gold = gold_answers[0]
            docs_pool = row.get("documents", []) or [q]

            # 预编码段落向量（无状态，每个 item 独立，加锁防止 CUDA 并发）
            _enc_pool: Optional[List[str]] = None
            _enc_emb = None
            if backend == "contriever_bge":
                if dense_rerank is None:
                    raise ValueError("contriever_bge 检索需传入 dense_rerank")
                with _retriever_lock:
                    _enc_pool, _enc_emb = dense_rerank.encode_docs(docs_pool)

            used: List[int] = []
            acc_context = ""
            current_answer = ""
            trace = ""
            hist_docs: List[str] = []
            steps: List[Dict[str, Any]] = []
            npz_writes: List[Tuple[Path, np.ndarray, np.ndarray]] = []

            for k in range(1, cfg.max_k + 1):
                query = llm.generate_follow_up_query(q, trace, backend)

                # ── 检索 ──────────────────────────────────────────
                if backend == "bm25":
                    _bm25 = BM25Retriever(docs_pool)
                    _docs, _scores, _idxs = _bm25.retrieve(query, k=1, exclude_indices=used)
                    if not _docs:
                        break
                    doc, score, doc_idx = _docs[0], float(_scores[0]), int(_idxs[0])
                elif backend == "contriever_bge":
                    with _retriever_lock:
                        doc, score, doc_idx = dense_rerank.retrieve_top1(
                            query, used, _enc_pool, _enc_emb
                        )
                    if doc_idx < 0:
                        break
                else:
                    raise ValueError(f"未知 retriever_backend={cfg.retriever_backend!r}")

                acc_before_doc = acc_context
                if doc_idx >= 0:
                    used.append(doc_idx)
                    hist_docs.append(doc)
                if doc:
                    acc_context = (acc_context + "\n\n" + doc).strip()

                intermediate_answer = (
                    _normalize_text(llm.generate_intermediate_answer(query, doc)) if doc else ""
                )
                trace = append_trace_step(trace, query, doc, intermediate_answer)

                # ── LLM 生成（HTTP，无需锁）─────────────────────
                samples, gen_meta = llm.generate_n(q, acc_context, cfg.n_samples, cfg.temperature)
                current_answer = _normalize_text(samples[0]) if samples else ""
                f1, em = compute_metrics_multi(current_answer, gold_answers)

                semantic_entropy, self_consistency = _semantic_entropy_and_consistency(samples)
                overlap = _jaccard(doc, " ".join(hist_docs[:-1])) if len(hist_docs) > 1 else 0.0
                hyp_piece = f"{q} {acc_before_doc}".strip()
                if len(hyp_piece) > 512:
                    hyp_piece = hyp_piece[:512]

                with _nli_lock:
                    nli_entail, nli_contra = (
                        nli_scorer.entail_contra(doc, hyp_piece) if doc else (0.0, 0.0)
                    )

                tok = int(round(float(gen_meta.get("token_count", 0))))
                lat_ms = float(gen_meta.get("latency_ms", 0.0))
                answer_logprob = float(gen_meta.get("answer_logprob", 0.0) or 0.0)

                if skip_self_eval:
                    self_eval_score = 0.0
                else:
                    self_eval_score = float(llm.self_evaluate_score(q, acc_context, current_answer))

                steps.append(
                    {
                        "step": k,
                        "retrieval_query": query,
                        "retrieved_doc": doc,
                        "retrieval_score": round(score, 4),
                        "intermediate_answer": intermediate_answer,
                        "current_answer": current_answer,
                        "f1": round(float(f1), 4),
                        "em": bool(em),
                        "cost": {
                            "token_count": max(1, tok),
                            "retrieval_calls": 1,
                            "latency_ms": round(lat_ms, 3),
                        },
                        "semantic_entropy": round(semantic_entropy, 6),
                        "self_consistency": round(self_consistency, 6),
                        "answer_logprob": round(answer_logprob, 6),
                        "self_eval_score": round(self_eval_score, 6),
                        "ctx_overlap": round(float(overlap), 6),
                        "nli_entail": round(float(nli_entail), 6),
                        "nli_contra": round(float(nli_contra), 6),
                    }
                )

                # ── 提取 hidden states（GPU，加锁）────────────────
                with _hidden_lock:
                    embedding_last, embedding_mean = hidden_state_extractor.extract(
                        q, acc_context, current_answer
                    )
                feat_path = feature_dir / f"{row['id']}_step{k}.npz"
                npz_writes.append((feat_path, embedding_last, embedding_mean))

            traj = {
                "id": row["id"],
                "dataset": dataset_name,
                "split": split,
                "question": q,
                "gold_answer": gold,
                "answer_aliases": gold_answers[1:],
                "gold_answers": list(gold_answers),
                "gt_hop_count": int(row.get("gt_hop_count", -1)),
                "steps": steps,
            }
            return traj, npz_writes

        except Exception as exc:
            LOGGER.error(
                "轨迹处理失败 id=%s: %s", row.get("id", "?"), exc, exc_info=True
            )
            return None

    def _write_result(
        writer,
        result: Optional[Tuple[Dict[str, Any], List[Tuple[Path, np.ndarray, np.ndarray]]]],
    ) -> bool:
        if result is None:
            return False
        traj, npz_writes = result
        for fp, emb_last, emb_mean in npz_writes:
            np.savez_compressed(
                fp,
                last_token=emb_last.astype(np.float16),
                mean_pool=emb_mean.astype(np.float16),
            )
        writer.write(json.dumps(traj, ensure_ascii=False) + "\n")
        writer.flush()
        return True

    desc = f"trajectory::{dataset_name}/{split}"
    produced = 0

    # 断点续跑：读取已有 JSONL 中的 id，跳过已完成的 rows
    done_ids: set = set()
    if out_path.exists():
        for rec in _read_jsonl(out_path):
            _rid = rec.get("id")
            if _rid:
                done_ids.add(_rid)
    if done_ids:
        LOGGER.info(
            "trajectory::%s/%s 续跑模式：已完成 %d 条，跳过重复。",
            dataset_name, split, len(done_ids),
        )
        produced = len(done_ids)

    pending_rows = [r for r in rows if r.get("id") not in done_ids]
    file_mode = "a" if done_ids else "w"

    with out_path.open(file_mode, encoding="utf-8") as writer:
        if not pending_rows:
            LOGGER.info("trajectory::%s/%s 已全部完成，无需重跑。", dataset_name, split)
        elif num_workers <= 1:
            for row in tqdm(pending_rows, desc=desc):
                if _write_result(writer, _process_item(row)):
                    produced += 1
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for result in tqdm(
                    executor.map(_process_item, pending_rows),
                    total=len(pending_rows),
                    desc=desc,
                ):
                    if _write_result(writer, result):
                        produced += 1

    return produced


def _load_trajectory_jsonl(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


def _pareto_nondominated_min_cost_max_f1(
    points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """横轴为 cost（越小越好）、纵轴为 F1（越大越好）时的非支配点集，用于包络折线。"""
    pts = [(float(a), float(b)) for a, b in points]
    nd: List[Tuple[float, float]] = []
    for i, (cx, fy) in enumerate(pts):
        dominated = False
        for j, (ox, oy) in enumerate(pts):
            if i == j:
                continue
            if (ox <= cx and oy >= fy) and (ox < cx or oy > fy):
                dominated = True
                break
        if not dominated:
            nd.append((cx, fy))
    return sorted(nd, key=lambda t: (t[0], -t[1]))


def compute_oracle_and_pareto(cfg: Stage1Config, dataset_name: str) -> Dict[str, Any]:
    train_path = cfg.trajectories_dir / dataset_name / "train" / "trajectories.jsonl"
    dev_path = cfg.trajectories_dir / dataset_name / "dev" / "trajectories.jsonl"
    test_path = cfg.trajectories_dir / dataset_name / "test" / "trajectories.jsonl"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"缺少轨迹缓存：{train_path} 或 {test_path}")

    train_traj = _load_trajectory_jsonl(train_path)
    dev_traj = _load_trajectory_jsonl(dev_path) if dev_path.exists() else []
    test_traj = _load_trajectory_jsonl(test_path)

    global_reservation_values = compute_all_reservation_values(
        train_traj, cfg.max_k, cfg.cost_per_step
    )
    oracle_dev = (
        compute_trajectory_oracle(
            dev_traj, cfg.cost_per_step, cfg.max_k, cost_metric=cfg.oracle_cost_metric
        )
        if dev_traj
        else []
    )
    oracle_test = compute_trajectory_oracle(
        test_traj, cfg.cost_per_step, cfg.max_k, cost_metric=cfg.oracle_cost_metric
    )
    static_weitzman_test = oracle_stopping_simulation(
        test_traj, global_reservation_values, cfg.max_k
    )

    rows: List[Dict[str, Any]] = []
    for k in range(1, cfg.max_k + 1):
        f1s: List[float] = []
        ems: List[int] = []
        costs: List[float] = []
        for traj in test_traj:
            target = next((s for s in traj["steps"] if s["step"] == k), traj["steps"][-1])
            f1s.append(float(target["f1"]))
            ems.append(int(bool(target["em"])))
            costs.append(
                trajectory_cumulative_cost(
                    traj, k, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
                )
            )
        rows.append(
            {
                "strategy": f"Fixed-K={k}",
                "avg_steps": float(k),
                "avg_cost": float(np.mean(costs) if costs else 0.0),
                "avg_f1": float(np.mean(f1s) if f1s else 0.0),
                "avg_em": float(np.mean(ems) if ems else 0.0),
            }
        )

    static_costs = [
        trajectory_cumulative_cost(
            traj, int(r["steps_used"]), cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        for traj, r in zip(test_traj, static_weitzman_test)
    ]
    static_row = {
        "strategy": "Global-Weitzman",
        "avg_steps": float(
            np.mean([r["steps_used"] for r in static_weitzman_test]) if static_weitzman_test else 0.0
        ),
        "avg_cost": float(np.mean(static_costs) if static_costs else 0.0),
        "avg_f1": float(
            np.mean([r["f1"] for r in static_weitzman_test]) if static_weitzman_test else 0.0
        ),
        "avg_em": float(
            np.mean([int(r["em"]) for r in static_weitzman_test]) if static_weitzman_test else 0.0
        ),
    }
    rows.append(static_row)

    oracle_costs = [
        trajectory_cumulative_cost(
            traj, int(r["steps_used"]), cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        for traj, r in zip(test_traj, oracle_test)
    ]
    oracle_row = {
        "strategy": "Oracle",
        "avg_steps": float(np.mean([r["steps_used"] for r in oracle_test]) if oracle_test else 0.0),
        "avg_cost": float(np.mean(oracle_costs) if oracle_costs else 0.0),
        "avg_f1": float(np.mean([r["f1"] for r in oracle_test]) if oracle_test else 0.0),
        "avg_em": float(np.mean([int(r["em"]) for r in oracle_test]) if oracle_test else 0.0),
    }
    rows.append(oracle_row)

    df = pd.DataFrame(rows)
    table_csv = cfg.results_dir / f"stage1_oracle_table_{dataset_name}.csv"
    table_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_csv, index=False)

    use_cost_x = cfg.oracle_cost_metric != "fixed"
    x_key = "avg_cost" if use_cost_x else "avg_steps"
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    scatter_xy: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        strat = row["strategy"]
        if strat == "Oracle":
            color, marker, size = "#d62728", "*", 180
        elif strat == "Global-Weitzman":
            color, marker, size = "#ff7f0e", "^", 130
        else:
            color, marker, size = "#1f77b4", "o", 90
        x_val = float(row[x_key])
        scatter_xy.append((x_val, float(row["avg_f1"])))
        ax.scatter(
            x_val,
            row["avg_f1"],
            color=color,
            marker=marker,
            s=size,
            label=row["strategy"],
        )
        ax.annotate(row["strategy"], (x_val, row["avg_f1"]), fontsize=8)
    nd = _pareto_nondominated_min_cost_max_f1(scatter_xy)
    if len(nd) >= 2:
        ax.plot(
            [p[0] for p in nd],
            [p[1] for p in nd],
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label="Pareto envelope",
        )
    ax.set_xlabel(
        "Avg cumulative cost (normalized)"
        if use_cost_x
        else "Avg cost (retrieval steps)"
    )
    ax.set_ylabel("F1")
    ax.set_title(f"Stage1 Oracle Pareto - {dataset_name}")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    pareto_path = cfg.results_dir / f"stage1_oracle_pareto_{dataset_name}.png"
    fig.savefig(pareto_path, dpi=160)
    plt.close(fig)

    label_dir = cfg.oracle_dir / dataset_name
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / "test_oracle_labels.jsonl"
    label_rows = []
    hop_alignment_counter: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "exact_match": 0})
    known_hop_count = 0
    for traj, result in zip(test_traj, oracle_test):
        raw_gt_hop = traj.get("gt_hop_count", -1)
        try:
            gt_hop = int(raw_gt_hop)
        except (TypeError, ValueError):
            gt_hop = -1
        if gt_hop > 0:
            known_hop_count += 1
            hop_alignment_counter[gt_hop]["total"] += 1
            if int(result["steps_used"]) == gt_hop:
                hop_alignment_counter[gt_hop]["exact_match"] += 1
        tau = int(result["steps_used"])
        targets = result["step_targets"] or {}
        if tau >= cfg.max_k:
            td = targets.get(cfg.max_k - 1, {})
        else:
            td = targets.get(tau, {})
        label_rows.append(
            {
                "id": traj["id"],
                "gt_hop_count": gt_hop,
                "oracle_stop_step": tau,
                "oracle_steps_used": result["steps_used"],
                "oracle_f1": result["f1"],
                "oracle_em": result["em"],
                "margin": float(td.get("margin", 0.0)),
                "action_label": int(td.get("action_label", 0)),
                "expected_continue_val": float(td.get("expected_continue_val", 0.0)),
                "step_targets": result["step_targets"],
            }
        )
    _write_jsonl(label_path, label_rows)

    per_hop_alignment = []
    total_exact_match = 0
    for hop in sorted(hop_alignment_counter.keys()):
        total = hop_alignment_counter[hop]["total"]
        exact = hop_alignment_counter[hop]["exact_match"]
        total_exact_match += exact
        per_hop_alignment.append(
            {
                "gt_hop_count": int(hop),
                "total": int(total),
                "exact_match": int(exact),
                "exact_match_rate": float(exact / total) if total > 0 else 0.0,
            }
        )
    hop_alignment = {
        "known_gt_count": int(known_hop_count),
        "unknown_gt_count": int(max(0, len(test_traj) - known_hop_count)),
        "exact_match_rate_over_known": float(total_exact_match / known_hop_count) if known_hop_count > 0 else 0.0,
        "per_hop": per_hop_alignment,
    }

    return {
        "reservation_values": global_reservation_values,
        "oracle_test": oracle_test,
        "oracle_dev": oracle_dev,
        "pareto_path": str(pareto_path),
        "label_path": str(label_path),
        "hop_alignment": hop_alignment,
        "table": rows,
    }


def _count_missing_key(items: Sequence[Dict[str, Any]], keys: Sequence[str]) -> float:
    if not items:
        return 1.0
    miss = 0
    total = 0
    for item in items:
        for k in keys:
            total += 1
            if item.get(k) is None:
                miss += 1
    return miss / max(1, total)


def build_stage1_report(cfg: Stage1Config, datasets: Sequence[str], metrics: Dict[str, Dict[str, Any]]) -> Path:
    report_lines: List[str] = []
    report_lines.append("# Stage1 Report")
    report_lines.append("")
    report_lines.append("## Data Statistics")
    report_lines.append("")
    for ds in datasets:
        m = metrics[ds]
        report_lines.append(
            f"- `{ds}`: train={m['counts'].get('train', 0)}, calib={m['counts'].get('calib', 0)}, "
            f"dev={m['counts'].get('dev', 0)}, test={m['counts'].get('test', 0)}"
        )
    report_lines.append("")
    report_lines.append("## Cache Integrity")
    report_lines.append("")
    for ds in datasets:
        m = metrics[ds]
        hb = m.get("hidden_state_npz_by_split") or {}
        if hb:
            order = ("train", "calib", "dev", "test")
            parts = ", ".join(f"{k}={hb.get(k, 0)}" for k in order)
            htot = int(m.get("hidden_state_npz_total", sum(hb.values())))
            btot = int(m.get("bad_feature_npz_total", m["bad_feature_files"]))
            report_lines.append(
                f"- `{ds}`: trajectory_cached={m['trajectory_cached']}, "
                f"hidden_npz_total={htot} ({parts}), bad_npz_total={btot}"
            )
        else:
            report_lines.append(
                f"- `{ds}`: trajectory_cached={m['trajectory_cached']}, "
                f"hidden_state_files={m['hidden_state_files']}, bad_feature_files={m['bad_feature_files']}"
            )
    report_lines.append("")
    report_lines.append("## Feature Distribution")
    report_lines.append("")
    for ds in datasets:
        m = metrics[ds]
        report_lines.append(
            f"- `{ds}`: entropy_mean={m['entropy_mean']:.4f}, self_consistency_mean={m['self_consistency_mean']:.4f}, overlap_mean={m['overlap_mean']:.4f}"
        )
    report_lines.append("")
    report_lines.append("## Oracle Frontier")
    report_lines.append("")
    report_out = cfg.docs_dir / "stage1_report.md"
    root = cfg.root_dir.resolve()
    for ds in datasets:
        pp = Path(str(metrics[ds]["pareto_path"]))
        full = pp.resolve() if pp.is_absolute() else (root / pp).resolve()
        rel = os.path.relpath(str(full), start=str(report_out.parent.resolve()))
        report_lines.append(f"- `{ds}` pareto: `{rel}`")
    report_lines.append("")
    report_lines.append("## Hop Alignment (Oracle Step vs GT Hop)")
    report_lines.append("")
    for ds in datasets:
        m = metrics[ds]
        hop_info = m.get("hop_alignment", {})
        known_gt = int(hop_info.get("known_gt_count", 0))
        unknown_gt = int(hop_info.get("unknown_gt_count", 0))
        exact_rate = float(hop_info.get("exact_match_rate_over_known", 0.0))
        if known_gt <= 0:
            report_lines.append(f"- `{ds}`: known_gt=0, unknown_gt={unknown_gt} (缺少可用 hop 标签)")
            continue
        per_hop = hop_info.get("per_hop", [])
        details = []
        for item in per_hop:
            hop = int(item.get("gt_hop_count", -1))
            exact = int(item.get("exact_match", 0))
            total = int(item.get("total", 0))
            rate = float(item.get("exact_match_rate", 0.0))
            details.append(f"{hop}-hop {exact}/{total} ({rate:.2%})")
        details_text = "; ".join(details) if details else "no per-hop breakdown"
        report_lines.append(
            f"- `{ds}`: known_gt={known_gt}, unknown_gt={unknown_gt}, exact_match_over_known={exact_rate:.2%}; {details_text}"
        )
    report_lines.append("")
    report_lines.append("## Go/No-Go Checks")
    report_lines.append("")
    gate_msgs = []
    for ds in datasets:
        m = metrics[ds]
        cache_complete = (
            m["counts"].get("train", 0) > 0
            and m["counts"].get("calib", 0) > 0
            and m["counts"].get("dev", 0) > 0
            and m["counts"].get("test", 0) > 0
        )
        key_missing = m["key_feature_missing_rate"]
        oracle_gain = m["oracle_gain_over_best_fixed"]
        gate_msgs.append(
            (
                ds,
                cache_complete,
                key_missing <= 0.01,
                oracle_gain > 0.0,
                key_missing,
                oracle_gain,
            )
        )
    for ds, g1, g2, g3, miss_rate, gain in gate_msgs:
        report_lines.append(
            f"- `{ds}`: cache_ok={g1}, feature_missing_rate={miss_rate:.4f}, oracle_gain={gain:.4f}, pass={g1 and g2 and g3}"
        )

    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_out
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    return report_path


def _scan_feature_files(path: Path) -> Tuple[int, int]:
    files = list(path.glob("*.npz"))
    bad = 0
    for fp in files:
        try:
            with np.load(fp) as obj:
                _ = obj["last_token"]
                _ = obj["mean_pool"]
        except Exception:
            bad += 1
    return len(files), bad


def run_dataset_stage1(
    cfg: Stage1Config,
    dataset_name: str,
    skip_prepare: bool,
    skip_trajectories: bool,
    reextract_hidden_only: bool,
    reextract_splits: Sequence[str],
    collect_splits: Sequence[str],
    hidden_state_extractor: HiddenStateExtractor,
    nli_scorer: NLICrossEncoderScorer,
    dense_rerank: Optional[ContrieverBgeRetriever] = None,
) -> Dict[str, Any]:
    if not skip_prepare:
        counts = prepare_data(cfg, dataset_name)
    else:
        missing_prep = _missing_prepared_artifacts(cfg, dataset_name)
        if missing_prep:
            LOGGER.warning(
                "--skip-prepare 已开启，但数据集 %s 仍缺 %d 个预处理文件，将自动执行 prepare_data（"
                "仅从 HF 写入本数据集的 data/processed 与 manifest，不会重跑已存在完整产物的数据集）。",
                dataset_name,
                len(missing_prep),
            )
            counts = prepare_data(cfg, dataset_name)
        else:
            counts = {}
            for split in ["train", "calib", "dev", "test"]:
                p = cfg.data_processed_dir / dataset_name / f"{split}.jsonl"
                counts[split] = len(_read_jsonl(p)) if p.exists() else 0

    cached_counts: Dict[str, int] = {}
    all_splits = ("train", "calib", "dev", "test")
    if skip_trajectories and reextract_hidden_only:
        raise ValueError("--skip-trajectories 与 --reextract-hidden-only 互斥。")
    if skip_trajectories and set(collect_splits) != set(all_splits):
        raise ValueError(
            "--skip-trajectories 已开启时不能缩小 --collect-splits（未列出的 split 不会从磁盘补算轨迹）。"
        )
    if reextract_hidden_only and set(collect_splits) != set(all_splits):
        raise ValueError("--reextract-hidden-only 与缩小 --collect-splits 互斥。")
    if skip_trajectories:
        for split in all_splits:
            traj_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
            if not traj_path.exists():
                raise FileNotFoundError(
                    f"--skip-trajectories 已开启但缺少轨迹文件：{traj_path}。"
                    "请先对该 split 跑过轨迹收集，或去掉该标志。"
                )
            cached_counts[split] = len(_read_jsonl(traj_path))
    elif reextract_hidden_only:
        allowed = set(all_splits)
        bad = [s for s in reextract_splits if s not in allowed]
        if bad:
            raise ValueError(f"--reextract-splits 含非法项 {bad}，仅允许 {sorted(allowed)}")
        for split in all_splits:
            traj_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
            if split in reextract_splits:
                cached_counts[split] = reextract_hidden_states_from_trajectories(
                    cfg, dataset_name, split, hidden_state_extractor
                )
            else:
                if not traj_path.exists():
                    raise FileNotFoundError(
                        f"--reextract-hidden-only 要求各 split 均存在轨迹；缺失：{traj_path}"
                    )
                cached_counts[split] = len(_read_jsonl(traj_path))
    else:
        allowed_c = set(all_splits)
        bad_c = [s for s in collect_splits if s not in allowed_c]
        if bad_c:
            raise ValueError(f"--collect-splits 含非法项 {bad_c}，仅允许 {sorted(allowed_c)}")
        if not collect_splits:
            raise ValueError("--collect-splits 不能为空")
        for split in all_splits:
            if split in collect_splits:
                cached_counts[split] = collect_trajectories(
                    cfg, dataset_name, split, hidden_state_extractor, nli_scorer, dense_rerank
                )
            else:
                traj_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
                if not traj_path.exists():
                    raise FileNotFoundError(
                        f"未对 split={split} 执行轨迹收集（不在 --collect-splits 中），但缺少文件：{traj_path}"
                    )
                cached_counts[split] = len(_read_jsonl(traj_path))

    oracle_info = compute_oracle_and_pareto(cfg, dataset_name)

    test_traj_path = cfg.trajectories_dir / dataset_name / "test" / "trajectories.jsonl"
    all_steps = []
    for row in _read_jsonl(test_traj_path):
        all_steps.extend(row.get("steps", []))
    key_missing_rate = _count_missing_key(
        all_steps,
        [
            "semantic_entropy",
            "self_consistency",
            "answer_logprob",
            "self_eval_score",
            "ctx_overlap",
            "nli_entail",
            "nli_contra",
        ],
    )
    entropy_vals = [float(s.get("semantic_entropy", 0.0)) for s in all_steps]
    consistency_vals = [float(s.get("self_consistency", 0.0)) for s in all_steps]
    overlap_vals = [float(s.get("ctx_overlap", 0.0)) for s in all_steps]

    hidden_npz_by_split: Dict[str, int] = {}
    bad_npz_by_split: Dict[str, int] = {}
    for split in all_splits:
        fd = cfg.features_dir / dataset_name / split / "hidden_states"
        n, b = _scan_feature_files(fd)
        hidden_npz_by_split[split] = n
        bad_npz_by_split[split] = b
    # 历史字段名保留：仅 test split 的 .npz 数量（MuSiQue test 仅 417 条 → 2085 个文件，易与「全量轨迹」混淆）
    total_feats = int(hidden_npz_by_split.get("test", 0))
    bad_feats = int(bad_npz_by_split.get("test", 0))
    hidden_npz_total = int(sum(hidden_npz_by_split.values()))
    bad_npz_total = int(sum(bad_npz_by_split.values()))

    fixed_rows = [r for r in oracle_info["table"] if r["strategy"].startswith("Fixed-K=")]
    best_fixed_f1 = max((r["avg_f1"] for r in fixed_rows), default=0.0)
    oracle_f1 = next((r["avg_f1"] for r in oracle_info["table"] if r["strategy"] == "Oracle"), 0.0)
    oracle_gain = float(oracle_f1 - best_fixed_f1)

    return {
        "counts": counts if counts else cached_counts,
        "trajectory_cached": sum(cached_counts.values()),
        "hidden_state_files": total_feats,
        "bad_feature_files": bad_feats,
        "hidden_state_npz_by_split": hidden_npz_by_split,
        "hidden_state_npz_total": hidden_npz_total,
        "bad_feature_npz_by_split": bad_npz_by_split,
        "bad_feature_npz_total": bad_npz_total,
        "entropy_mean": float(np.mean(entropy_vals) if entropy_vals else 0.0),
        "self_consistency_mean": float(np.mean(consistency_vals) if consistency_vals else 0.0),
        "overlap_mean": float(np.mean(overlap_vals) if overlap_vals else 0.0),
        "key_feature_missing_rate": key_missing_rate,
        "oracle_gain_over_best_fixed": oracle_gain,
        "pareto_path": oracle_info["pareto_path"],
        "hop_alignment": oracle_info["hop_alignment"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pandora-RAG Stage 1 pipeline")
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,musique,2wiki",
        help="Comma-separated dataset names in {hotpotqa,musique,2wiki}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--train-quota", type=int, default=4000)
    parser.add_argument("--calib-quota", type=int, default=1000)
    parser.add_argument("--dev-quota", type=int, default=1000)
    parser.add_argument("--test-quota", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--cost-per-step", type=float, default=0.05)
    parser.add_argument(
        "--oracle-cost-metric",
        type=str,
        choices=("fixed", "token", "latency"),
        default="fixed",
        help="Oracle DP 与 Pareto 横轴用的步级成本：fixed=每步常数；token/latency=按轨迹缓存归一化",
    )
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument(
        "--hidden-state-model",
        type=str,
        default="",
        help="用于提取真实 hidden states 的本地目录或 Hugging Face 模型名（默认优先 models/Meta-Llama-3.1-8B-Instruct）",
    )
    parser.add_argument(
        "--hidden-state-max-length",
        type=int,
        default=2048,
        help="提取 hidden states 时的最大 token 长度（超长会截断）",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument(
        "--skip-trajectories",
        action="store_true",
        help="不调用 LLM 重算轨迹与 hidden states，直接读 cache/trajectories 与 cache/features（用于只重跑部分数据集后合并报告）",
    )
    parser.add_argument(
        "--reextract-hidden-only",
        action="store_true",
        help="保留已有 trajectories.jsonl，仅按每步重建上下文并重提 hidden states 写入 {id}_step{k}.npz（不调用检索/LLM）",
    )
    parser.add_argument(
        "--reextract-splits",
        type=str,
        default="train,calib,dev,test",
        help="与 --reextract-hidden-only 联用：逗号分隔，指定要重写 .npz 的 split；未列出的 split 只校验轨迹存在并计入条数",
    )
    parser.add_argument(
        "--collect-splits",
        type=str,
        default="train,calib,dev,test",
        help="默认全量。逗号分隔，仅对这些 split 调用检索+LLM+写 trajectories 与每步 .npz；其余 split 必须已有 trajectories.jsonl（用于补跑缺条数的 split）",
    )
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument(
        "--retriever-backend",
        type=str,
        choices=("bm25", "contriever_bge"),
        default=((os.getenv("PANDORA_RETRIEVER_BACKEND") or "bm25").strip().lower()),
        help="迭代检索：bm25或 contriever_bge（Contriever-MS MARCO 短名单 + bge-reranker-v2-m3）",
    )
    parser.add_argument(
        "--contriever-model",
        type=str,
        default=((os.getenv("PANDORA_CONTRIEVER_MODEL") or "facebook/contriever-msmarco").strip()),
        help="Contriever 模型名或本地目录",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=((os.getenv("PANDORA_RERANKER_MODEL") or "BAAI/bge-reranker-v2-m3").strip()),
        help="BGE cross-encoder 重排序模型名或本地目录",
    )
    parser.add_argument(
        "--contriever-shortlist-k",
        type=int,
        default=int((os.getenv("PANDORA_CONTRIEVER_SHORTLIST_K") or "32").strip()),
        help="Contriever 向量召回进入 cross-encoder 的短名单上限（候选更少时全进重排）",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=int((os.getenv("PANDORA_RERANK_BATCH_SIZE") or "8").strip()),
        help="重排序与 Contriever 编码的批大小",
    )
    parser.add_argument(
        "--retriever-device",
        type=str,
        default=(os.getenv("RETRIEVER_DEVICE") or "").strip(),
        help="检索模型设备，如 cuda / cuda:0 / cpu；默认自动检测",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int((os.getenv("PANDORA_NUM_WORKERS") or "1").strip()),
        help="并行轨迹收集的线程数（>1 时开启并行，LLM HTTP 并发，本地 GPU 模型加锁串行）",
    )
    parser.add_argument(
        "--skip-self-eval",
        action="store_true",
        default=os.getenv("PANDORA_STAGE1_SKIP_SELF_EVAL", "").lower() in ("1", "true", "yes"),
        help="跳过每步 self_evaluate_score HTTP 调用（节省 50%% LLM 请求，self_eval_score 记为 0）",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    allowed = {"hotpotqa", "musique", "2wiki"}
    unknown = [d for d in datasets if d not in allowed]
    if unknown:
        raise ValueError(f"不支持的数据集：{unknown}，只支持 {sorted(allowed)}")

    reextract_splits = tuple(
        x.strip().lower()
        for x in (args.reextract_splits or "").split(",")
        if x.strip()
    )
    if args.reextract_hidden_only and not reextract_splits:
        raise ValueError("--reextract-hidden-only 需要非空的 --reextract-splits")

    collect_splits = tuple(
        x.strip().lower()
        for x in (args.collect_splits or "").split(",")
        if x.strip()
    )
    if not args.skip_trajectories and not args.reextract_hidden_only and not collect_splits:
        raise ValueError("--collect-splits 不能为空")

    cfg = Stage1Config(
        seed=args.seed,
        max_k=args.max_k,
        train_quota=args.train_quota,
        calib_quota=args.calib_quota,
        dev_quota=args.dev_quota,
        test_quota=args.test_quota,
        temperature=args.temperature,
        n_samples=args.n_samples,
        cost_per_step=args.cost_per_step,
        oracle_cost_metric=args.oracle_cost_metric,
        embed_dim=args.embed_dim,
        hidden_state_model=args.hidden_state_model,
        hidden_state_max_length=args.hidden_state_max_length,
        root_dir=Path(args.root_dir),
        retriever_backend=str(args.retriever_backend),
        contriever_model_id=str(args.contriever_model),
        reranker_model_id=str(args.reranker_model),
        contriever_shortlist_k=int(args.contriever_shortlist_k),
        rerank_batch_size=int(args.rerank_batch_size),
        retriever_device=(args.retriever_device or None),
        num_workers=int(args.num_workers),
        skip_self_eval=bool(args.skip_self_eval),
    )
    _ensure_dirs(cfg)
    if args.skip_prepare:
        for ds in datasets:
            miss = _missing_prepared_artifacts(cfg, ds)
            if miss:
                LOGGER.info(
                    "预处理：数据集 %s 在 --skip-prepare 下缺少 %d 个文件，进入该数据集时会自动 prepare。",
                    ds,
                    len(miss),
                )
    hidden_state_extractor = HiddenStateExtractor(cfg)
    nli_device = (os.getenv("NLI_DEVICE") or "cpu").strip()
    local_nli = (cfg.root_dir / "models" / "cross-encoder-nli-deberta-v3-small").resolve()
    nli_model = str(local_nli) if local_nli.is_dir() and (local_nli / "config.json").exists() else None
    nli_scorer = NLICrossEncoderScorer(device=nli_device, model_id=nli_model)

    dense_rerank: Optional[ContrieverBgeRetriever] = None
    if (cfg.retriever_backend or "").strip().lower() == "contriever_bge":
        rdev = (cfg.retriever_device or "").strip() or None
        LOGGER.info(
            "加载 Contriever+BGE 检索：contriever=%s reranker=%s device=%s",
            cfg.contriever_model_id,
            cfg.reranker_model_id,
            rdev or "auto",
        )
        dense_rerank = ContrieverBgeRetriever(
            contriever_model=cfg.contriever_model_id,
            reranker_model=cfg.reranker_model_id,
            device=rdev,
            shortlist_k=cfg.contriever_shortlist_k,
            rerank_batch_size=cfg.rerank_batch_size,
        )

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        LOGGER.info("===== Stage1 dataset: %s =====", ds)
        all_metrics[ds] = run_dataset_stage1(
            cfg,
            ds,
            skip_prepare=args.skip_prepare,
            skip_trajectories=args.skip_trajectories,
            reextract_hidden_only=args.reextract_hidden_only,
            reextract_splits=reextract_splits,
            collect_splits=collect_splits,
            hidden_state_extractor=hidden_state_extractor,
            nli_scorer=nli_scorer,
            dense_rerank=dense_rerank,
        )

    report_path = build_stage1_report(cfg, datasets, all_metrics)
    LOGGER.info("Stage1 complete. Report: %s", report_path)
    for ds in datasets:
        LOGGER.info(
            "%s | cached=%d | feat_missing=%.4f | oracle_gain=%.4f",
            ds,
            all_metrics[ds]["trajectory_cached"],
            all_metrics[ds]["key_feature_missing_rate"],
            all_metrics[ds]["oracle_gain_over_best_fixed"],
        )


if __name__ == "__main__":
    main()
