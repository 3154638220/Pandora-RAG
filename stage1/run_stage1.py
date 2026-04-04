"""
Stage-1 pipeline for Pandora-RAG.

Covers:
  A. Data normalization and deterministic split manifests
  B. Trajectory caching with deep-ish features
  C. Oracle labels + Pareto frontier
  D. Stage report generation and quality gates

Usage:
  python -m stage1.run_stage1 --datasets hotpotqa,musique,2wiki --max-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# 未设置 HF_ENDPOINT 时默认走 hf-mirror，减轻国内直连 huggingface.co 的延迟。
if os.environ.get("PANDORA_NO_CN_HF_MIRROR", "").lower() not in ("1", "true", "yes"):
    if not (os.environ.get("HF_ENDPOINT") or "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import Dataset, load_dataset
from tqdm import tqdm

from pretest.utils.llm_client import LLMClient
from pretest.utils.metrics import compute_metrics
from pretest.utils.retriever import BM25Retriever
from pretest.utils.weitzman import (
    compute_all_reservation_values,
    compute_trajectory_oracle,
    oracle_stopping_simulation,
    trajectory_cumulative_cost,
)

# Hugging Face 上可用的 Parquet 镜像（旧名 musique / 2wikimultihopqa 已不可用）
HF_DATASET_IDS = {
    "hotpotqa": ("hotpot_qa", "distractor"),
    "musique": ("dgslibisey/MuSiQue", None),
    "2wiki": ("framolfese/2WikiMultihopQA", None),
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
    train_quota: int = 5000
    dev_quota: int = 1000
    test_quota: int = 1000
    temperature: float = 0.7
    n_samples: int = 10
    cost_per_step: float = 0.05
    oracle_cost_metric: str = "fixed"
    embed_dim: int = 256
    hidden_state_model: Optional[str] = None
    hidden_state_max_length: int = 2048
    root_dir: Path = Path(".")

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


def _ensure_dirs(cfg: Stage1Config) -> None:
    for p in [
        cfg.data_processed_dir,
        cfg.split_manifest_dir,
        cfg.trajectories_dir,
        cfg.features_dir,
        cfg.oracle_dir,
        cfg.results_dir,
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
    解析 Meta-Llama-3.1-8B-Instruct 本地权重目录（与 STORAGE_LAYOUT.md 一致：仓库内 models/ 常 symlink 到 haoge）。
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
        prompt = (
            "你是一个严谨的问答助手。请根据以下检索到的上下文回答问题。\n"
            "如无法确定，仍需给出最佳猜测，请直接输出答案（一个词组或短语），不要解释。\n\n"
            f"问题：{question}\n\n"
            f"已检索上下文：\n{context}\n\n"
            f"答案：{answer}"
        )
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


def _sample_answers(base_answer: str, n: int, rng: random.Random) -> List[str]:
    if not base_answer:
        return ["" for _ in range(n)]
    parts = base_answer.split()
    sampled = []
    for _ in range(n):
        if len(parts) <= 2:
            sampled.append(base_answer)
            continue
        k = max(1, int(len(parts) * rng.uniform(0.6, 1.0)))
        shuffled = parts[:]
        rng.shuffle(shuffled)
        sampled.append(" ".join(shuffled[:k]))
    return sampled


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
    a = _normalize_text(example.get("answer") or example.get("answers") or example.get("output"))
    record_id = _normalize_text(example.get("id") or example.get("_id")) or f"{dataset_name}_{split}_{idx:07d}"
    gt_hop_count = _extract_hop_count(example, dataset_name, record_id)
    return {
        "id": record_id,
        "dataset": dataset_name,
        "split": split,
        "question": q,
        "answer": a,
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
        cfg.data_processed_dir / dataset_name / "dev.jsonl",
        cfg.data_processed_dir / dataset_name / "test.jsonl",
        _prepared_manifest_path(cfg, dataset_name),
    ]
    return [p for p in required if not p.exists()]


def _validate_skip_prepare_inputs(cfg: Stage1Config, dataset_name: str) -> None:
    missing = _missing_prepared_artifacts(cfg, dataset_name)
    if not missing:
        return
    missing_list = "\n".join(f"- {p}" for p in missing)
    raise FileNotFoundError(
        "检测到 --skip-prepare，但缺少预处理产物。\n"
        f"数据集: {dataset_name}\n"
        f"缺失文件:\n{missing_list}\n"
        "请去掉 --skip-prepare 重新运行（prepare 阶段会从 Hugging Face 拉取并写出这些文件）；"
        "若必须离线运行，请先手动准备上述文件。"
    )


def _load_hf_split(dataset_name: str, split_name: str) -> Optional[Dataset]:
    try:
        spec = HF_DATASET_IDS.get(dataset_name)
        if not spec:
            return None
        repo, config_name = spec
        if config_name:
            ds = load_dataset(repo, config_name, split=split_name)
        else:
            ds = load_dataset(repo, split=split_name)
        return ds
    except Exception as exc:
        LOGGER.warning("加载数据集 %s split=%s 失败：%s", dataset_name, split_name, exc)
        return None


def prepare_data(cfg: Stage1Config, dataset_name: str) -> Dict[str, int]:
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

    if not test_list and val_list:
        cut = min(len(val_list), cfg.test_quota)
        test_list = val_list[-cut:]
        val_list = val_list[:-cut]

    split_map = {
        "train": train_list,
        "dev": val_list,
        "test": test_list,
    }
    quotas = {
        "train": cfg.train_quota,
        "dev": cfg.dev_quota,
        "test": cfg.test_quota,
    }

    counts: Dict[str, int] = {}
    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "seed": cfg.seed,
        "quota": quotas,
        "selected_ids": {},
        "timestamp": int(time.time()),
    }

    for split, rows in split_map.items():
        selected_idx = _sample_indices(len(rows), quotas[split], rng)
        normalized: List[Dict[str, Any]] = []
        for local_idx, src_idx in enumerate(selected_idx):
            normalized.append(_normalize_record(rows[src_idx], dataset_name, split, local_idx))
        out_path = cfg.data_processed_dir / dataset_name / f"{split}.jsonl"
        _write_jsonl(out_path, normalized)
        manifest["selected_ids"][split] = [r["id"] for r in normalized]
        counts[split] = len(normalized)

    manifest_path = _prepared_manifest_path(cfg, dataset_name)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return counts


def _retrieve_step_docs(question: str, current_answer: str, docs_pool: List[str], used: List[int]) -> Tuple[str, float, int]:
    if not docs_pool:
        return "", 0.0, -1
    query = question if not current_answer else f"{question} {current_answer}"
    retriever = BM25Retriever(docs_pool)
    docs, scores, idxs = retriever.retrieve(query, k=1, exclude_indices=used)
    if not docs:
        return "", 0.0, -1
    return docs[0], float(scores[0]), int(idxs[0])


def collect_trajectories(
    cfg: Stage1Config,
    dataset_name: str,
    split: str,
    hidden_state_extractor: HiddenStateExtractor,
) -> int:
    in_path = cfg.data_processed_dir / dataset_name / f"{split}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"缺少处理后数据：{in_path}")
    rows = _read_jsonl(in_path)
    llm_cfg = type("TmpCfg", (), {})()
    llm_cfg.api_base = os.getenv("OPENAI_API_BASE", DEFAULT_LLM_API_BASE)
    llm_cfg.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    llm_cfg.model_name = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    llm_cfg.max_tokens = 150
    llm_cfg.temperature = 0.0
    llm = LLMClient(llm_cfg)
    rng = random.Random(cfg.seed + len(rows))

    out_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_dir = cfg.features_dir / dataset_name / split / "hidden_states"
    feature_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    with out_path.open("w", encoding="utf-8") as writer:
        for row in tqdm(rows, desc=f"trajectory::{dataset_name}/{split}"):
            q = row["question"]
            gold = row["answer"]
            docs_pool = row.get("documents", []) or [q]
            used: List[int] = []
            acc_context = ""
            current_answer = ""
            hist_docs: List[str] = []
            steps: List[Dict[str, Any]] = []

            for k in range(1, cfg.max_k + 1):
                doc, score, doc_idx = _retrieve_step_docs(q, current_answer, docs_pool, used)
                if doc_idx >= 0:
                    used.append(doc_idx)
                    hist_docs.append(doc)
                if doc:
                    acc_context = (acc_context + "\n\n" + doc).strip()

                result = llm.generate(q, acc_context)
                current_answer = _normalize_text(result.get("answer", ""))
                f1, em = compute_metrics(current_answer, gold)

                samples = _sample_answers(current_answer, cfg.n_samples, rng)
                semantic_entropy, self_consistency = _semantic_entropy_and_consistency(samples)
                overlap = _jaccard(doc, " ".join(hist_docs[:-1])) if len(hist_docs) > 1 else 0.0
                nli_entail, nli_contra = _heuristic_nli(q, doc)

                steps.append(
                    {
                        "step": k,
                        "retrieved_doc": doc,
                        "retrieval_score": round(score, 4),
                        "current_answer": current_answer,
                        "f1": round(float(f1), 4),
                        "em": bool(em),
                        "cost": {
                            "token_count": int(result.get("token_count", 0)),
                            "retrieval_calls": k,
                            "latency_ms": 0,
                        },
                        "semantic_entropy": round(semantic_entropy, 6),
                        "self_consistency": round(self_consistency, 6),
                        "ctx_overlap": round(float(overlap), 6),
                        "nli_entail": round(float(nli_entail), 6),
                        "nli_contra": round(float(nli_contra), 6),
                    }
                )

            embedding_last, embedding_mean = hidden_state_extractor.extract(q, acc_context, current_answer)
            feat_path = feature_dir / f"{row['id']}.npz"
            np.savez_compressed(
                feat_path,
                last_token=embedding_last.astype(np.float16),
                mean_pool=embedding_mean.astype(np.float16),
            )

            writer.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "dataset": dataset_name,
                        "split": split,
                        "question": q,
                        "gold_answer": gold,
                        "gt_hop_count": int(row.get("gt_hop_count", -1)),
                        "steps": steps,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            produced += 1
    return produced


def _load_trajectory_jsonl(path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(path)


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
    use_cost_x = cfg.oracle_cost_metric != "fixed"
    x_key = "avg_cost" if use_cost_x else "avg_steps"
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    for _, row in df.iterrows():
        strat = row["strategy"]
        if strat == "Oracle":
            color, marker, size = "#d62728", "*", 180
        elif strat == "Global-Weitzman":
            color, marker, size = "#ff7f0e", "^", 130
        else:
            color, marker, size = "#1f77b4", "o", 90
        x_val = float(row[x_key])
        ax.scatter(
            x_val,
            row["avg_f1"],
            color=color,
            marker=marker,
            s=size,
            label=row["strategy"],
        )
        ax.annotate(row["strategy"], (x_val, row["avg_f1"]), fontsize=8)
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
        label_rows.append(
            {
                "id": traj["id"],
                "gt_hop_count": gt_hop,
                "oracle_steps_used": result["steps_used"],
                "oracle_f1": result["f1"],
                "oracle_em": result["em"],
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
            f"- `{ds}`: train={m['counts'].get('train', 0)}, dev={m['counts'].get('dev', 0)}, test={m['counts'].get('test', 0)}"
        )
    report_lines.append("")
    report_lines.append("## Cache Integrity")
    report_lines.append("")
    for ds in datasets:
        m = metrics[ds]
        report_lines.append(
            f"- `{ds}`: trajectory_cached={m['trajectory_cached']}, hidden_state_files={m['hidden_state_files']}, bad_feature_files={m['bad_feature_files']}"
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
    for ds in datasets:
        report_lines.append(f"- `{ds}` pareto: `{metrics[ds]['pareto_path']}`")
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

    report_path = cfg.results_dir / "stage1_report.md"
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
    hidden_state_extractor: HiddenStateExtractor,
) -> Dict[str, Any]:
    if not skip_prepare:
        counts = prepare_data(cfg, dataset_name)
    else:
        _validate_skip_prepare_inputs(cfg, dataset_name)
        counts = {}
        for split in ["train", "dev", "test"]:
            p = cfg.data_processed_dir / dataset_name / f"{split}.jsonl"
            counts[split] = len(_read_jsonl(p)) if p.exists() else 0

    cached_counts: Dict[str, int] = {}
    if skip_trajectories:
        for split in ["train", "dev", "test"]:
            traj_path = cfg.trajectories_dir / dataset_name / split / "trajectories.jsonl"
            if not traj_path.exists():
                raise FileNotFoundError(
                    f"--skip-trajectories 已开启但缺少轨迹文件：{traj_path}。"
                    "请先对该 split 跑过轨迹收集，或去掉该标志。"
                )
            cached_counts[split] = len(_read_jsonl(traj_path))
    else:
        for split in ["train", "dev", "test"]:
            cached_counts[split] = collect_trajectories(cfg, dataset_name, split, hidden_state_extractor)

    oracle_info = compute_oracle_and_pareto(cfg, dataset_name)

    test_traj_path = cfg.trajectories_dir / dataset_name / "test" / "trajectories.jsonl"
    all_steps = []
    for row in _read_jsonl(test_traj_path):
        all_steps.extend(row.get("steps", []))
    key_missing_rate = _count_missing_key(
        all_steps, ["semantic_entropy", "self_consistency", "ctx_overlap", "nli_entail", "nli_contra"]
    )
    entropy_vals = [float(s.get("semantic_entropy", 0.0)) for s in all_steps]
    consistency_vals = [float(s.get("self_consistency", 0.0)) for s in all_steps]
    overlap_vals = [float(s.get("ctx_overlap", 0.0)) for s in all_steps]

    feature_dir = cfg.features_dir / dataset_name / "test" / "hidden_states"
    total_feats, bad_feats = _scan_feature_files(feature_dir)

    fixed_rows = [r for r in oracle_info["table"] if r["strategy"].startswith("Fixed-K=")]
    best_fixed_f1 = max((r["avg_f1"] for r in fixed_rows), default=0.0)
    oracle_f1 = next((r["avg_f1"] for r in oracle_info["table"] if r["strategy"] == "Oracle"), 0.0)
    oracle_gain = float(oracle_f1 - best_fixed_f1)

    return {
        "counts": counts if counts else cached_counts,
        "trajectory_cached": sum(cached_counts.values()),
        "hidden_state_files": total_feats,
        "bad_feature_files": bad_feats,
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
    parser.add_argument("--train-quota", type=int, default=5000)
    parser.add_argument("--dev-quota", type=int, default=1000)
    parser.add_argument("--test-quota", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-samples", type=int, default=10)
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
    parser.add_argument("--root-dir", type=str, default=".")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    datasets = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    allowed = {"hotpotqa", "musique", "2wiki"}
    unknown = [d for d in datasets if d not in allowed]
    if unknown:
        raise ValueError(f"不支持的数据集：{unknown}，只支持 {sorted(allowed)}")

    cfg = Stage1Config(
        seed=args.seed,
        max_k=args.max_k,
        train_quota=args.train_quota,
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
    )
    _ensure_dirs(cfg)
    hidden_state_extractor = HiddenStateExtractor(cfg)

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        LOGGER.info("===== Stage1 dataset: %s =====", ds)
        all_metrics[ds] = run_dataset_stage1(
            cfg,
            ds,
            skip_prepare=args.skip_prepare,
            skip_trajectories=args.skip_trajectories,
            hidden_state_extractor=hidden_state_extractor,
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
