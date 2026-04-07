"""
Stage-2 pipeline for Pandora-RAG.

Covers:
  A. Build Oracle step labels from Stage-1 cached trajectories
  B. Train Neural Probe（完整数据用双分支 `ProbeMLP_v2`，`--shallow-only` 用浅层 `ProbeMLP`）与 margin-weighted Focal BCE + 可选 label smoothing（Phase B）
  C. Tune stop threshold on dev split（Phase C：GW(dev) 步数约束 + Pareto 分数 F1−λ·归一化成本）
  D. Evaluate on test and compare with baselines

Usage:
  python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki --max-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from pretest.utils.weitzman import (
    compute_all_reservation_values,
    compute_all_reservation_values_from_proxy,
    compute_trajectory_oracle,
    deployable_weitzman_stopping_simulation,
    oracle_stopping_simulation,
    trajectory_cumulative_cost,
)

LOGGER = logging.getLogger(__name__)

# 9 维基础 + 5 维步间差分 + cumulative_cost_ratio + Phase D4 答案/检索代理（3）
SHALLOW_FEATURE_DIM = 18
SHALLOW_FEATURE_NAMES: Tuple[str, ...] = (
    "k_norm",
    "retrieval_score",
    "semantic_entropy",
    "self_consistency",
    "ctx_overlap",
    "nli_entail",
    "nli_contra",
    "log1p_token_count",
    "log1p_latency_ms",
    "delta_retrieval_score",
    "delta_semantic_entropy",
    "delta_self_consistency",
    "delta_ctx_overlap",
    "delta_nli_entail",
    "cumulative_cost_ratio",
    "answer_changed",
    "answer_consistency_streak_norm",
    "retrieval_marginal_novelty",
)

# D4：ROUGE-L 用截断词序列，避免超长 retrieved_doc 导致 LCS 过慢
_D4_ROUGE_MAX_TOKENS = 256


@dataclass
class Stage2Config:
    seed: int = 42
    max_k: int = 5
    cost_per_step: float = 0.05
    oracle_cost_metric: str = "fixed"
    root_dir: Path = Path(".")
    hidden_state_key: str = "last_token"
    # C1：Shallow-Only — 不使用 Stage1 的 hidden states，仅浅层特征（含 Delta）训练 Probe（w/o Deep Features）。
    shallow_only: bool = False

    # Training（浅层单塔 ProbeMLP 用 hidden_dim；完整 Probe 用 ProbeMLP_v2 的 compress / fuse）
    hidden_dim: int = 256
    compress_dim: int = 64
    fuse_dim: int = 128
    dropout: float = 0.30
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    batch_size: int = 256
    epochs: int = 60
    patience: int = 12
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    margin_weight_floor: float = 0.1
    # Phase B：Focal BCE + label smoothing（None 表示按训练集 Continue 比例自适应 focal 正类权重）
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    label_smoothing: float = 0.05
    # Phase C：阈值在 dev 上联合「Pareto 分数 F1−λ·归一化成本」与 GW 步数上界（avg_steps ≤ GW_dev×mult）
    threshold_pareto_lambdas: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    threshold_gw_steps_cap_mult: float = 1.05
    # 结果文件名后缀（如 D3 消融 `--artifact-suffix d3_bce`，避免覆盖默认 `stage2_probe_table_*.csv`）
    artifact_suffix: str = ""

    @property
    def trajectories_dir(self) -> Path:
        return self.root_dir / "cache" / "trajectories"

    @property
    def features_dir(self) -> Path:
        return self.root_dir / "cache" / "features"

    @property
    def artifacts_probe_dir(self) -> Path:
        return self.root_dir / "artifacts" / "probe"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"


class ProbeDataset(Dataset):
    """浅层 + 可选 hidden；无 hidden 时（shallow-only）仅返回 x_shallow, y, w。"""

    def __init__(
        self,
        x_shallow: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        x_hidden: Optional[np.ndarray] = None,
    ):
        self.xs = torch.tensor(x_shallow, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.w = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)
        self._dual = x_hidden is not None and x_hidden.shape[1] > 0
        self.xh: Optional[torch.Tensor]
        if self._dual:
            self.xh = torch.tensor(x_hidden, dtype=torch.float32)
        else:
            self.xh = None

    def __len__(self) -> int:
        return int(self.xs.shape[0])

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self.xh is None:
            return self.xs[idx], self.y[idx], self.w[idx]
        return self.xs[idx], self.xh[idx], self.y[idx], self.w[idx]


class ProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        mid = max(64, hidden_dim)
        hi = max(32, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hi),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hi, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProbeMLP_v2(nn.Module):
    """双分支融合：hidden 经 LayerNorm + 压缩，与浅层分支拼接后分类。"""

    def __init__(
        self,
        hidden_state_dim: int,
        shallow_dim: int,
        compress_dim: int = 64,
        fuse_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.hidden_branch = nn.Sequential(
            nn.LayerNorm(hidden_state_dim),
            nn.Linear(hidden_state_dim, compress_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shallow_branch = nn.Sequential(
            nn.Linear(shallow_dim, shallow_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fuse_input = compress_dim + shallow_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fuse_input, fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.GELU(),
            nn.Linear(fuse_dim // 2, 1),
        )

    def forward(self, x_shallow: torch.Tensor, x_hidden: torch.Tensor) -> torch.Tensor:
        h = self.hidden_branch(x_hidden)
        s = self.shallow_branch(x_shallow)
        fused = torch.cat([h, s], dim=-1)
        return self.classifier(fused)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs(cfg: Stage2Config, datasets: Sequence[str]) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_probe_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        (cfg.artifacts_probe_dir / ds).mkdir(parents=True, exist_ok=True)


def _stage2_artifact_tag(cfg: Stage2Config) -> str:
    parts: List[str] = []
    if cfg.shallow_only:
        parts.append("shallow")
    extra = (cfg.artifact_suffix or "").strip()
    if extra:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", extra).strip("_")
        if safe:
            parts.append(safe)
    return ("_" + "_".join(parts)) if parts else ""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_trajectories(cfg: Stage2Config, dataset: str, split: str) -> List[Dict[str, Any]]:
    p = cfg.trajectories_dir / dataset / split / "trajectories.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"缺少 Stage1 轨迹缓存：{p}")
    rows = _read_jsonl(p)
    if not rows:
        raise RuntimeError(f"轨迹文件为空：{p}")
    return rows


def _infer_hidden_dim(cfg: Stage2Config, dataset: str, split: str, key: str) -> int:
    feat_dir = cfg.features_dir / dataset / split / "hidden_states"
    if not feat_dir.exists():
        LOGGER.warning("%s 不存在，回退为仅浅层特征。", feat_dir)
        return 0
    sample_files = sorted(feat_dir.glob("*.npz"))
    if not sample_files:
        LOGGER.warning("%s 为空，回退为仅浅层特征。", feat_dir)
        return 0
    for fp in sample_files[:20]:
        try:
            with np.load(fp) as obj:
                if key in obj:
                    vec = np.asarray(obj[key]).reshape(-1)
                    if vec.size > 0:
                        return int(vec.size)
        except Exception:
            continue
    LOGGER.warning("无法从 %s 推断 hidden 维度（key=%s），回退为仅浅层特征。", feat_dir, key)
    return 0


def _load_hidden_map(
    cfg: Stage2Config,
    dataset: str,
    split: str,
    hidden_dim: int,
    key: str,
) -> Dict[Tuple[str, int], np.ndarray]:
    if hidden_dim <= 0:
        return {}
    feat_dir = cfg.features_dir / dataset / split / "hidden_states"
    if not feat_dir.exists():
        LOGGER.warning("%s 不存在，当前 split 使用零向量 hidden。", feat_dir)
        return {}

    out: Dict[Tuple[str, int], np.ndarray] = {}
    bad = 0
    for fp in feat_dir.glob("*.npz"):
        stem = fp.stem
        # 新格式：{sample_id}_step{k}.npz；旧格式：{sample_id}.npz（视为 step=0 的兜底）。
        match = re.match(r"^(?P<sample_id>.+)_step(?P<step>\d+)$", stem)
        if match:
            sample_id = match.group("sample_id")
            step_idx = int(match.group("step"))
        else:
            sample_id = stem
            step_idx = 0
        try:
            with np.load(fp) as obj:
                if key not in obj:
                    bad += 1
                    continue
                vec = np.asarray(obj[key], dtype=np.float32).reshape(-1)
                if vec.size != hidden_dim:
                    bad += 1
                    continue
                out[(sample_id, step_idx)] = vec
        except Exception:
            bad += 1
    if bad > 0:
        LOGGER.warning("%s/%s hidden states 异常文件数：%d", dataset, split, bad)
    return out


def _delta_source_scalars(step: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """用于差分特征的 5 个标量：与 plan 中 delta_* 一一对应。"""
    return (
        float(step.get("retrieval_score", 0.0) or 0.0),
        float(step.get("semantic_entropy", 0.0) or 0.0),
        float(step.get("self_consistency", 0.0) or 0.0),
        float(step.get("ctx_overlap", 0.0) or 0.0),
        float(step.get("nli_entail", 0.0) or 0.0),
    )


def _canonical_answer_text(value: Any) -> str:
    """与 Stage1 轨迹中 current_answer 可比对的规范字符串（小写、压缩空白）。"""
    if value is None:
        return ""
    if isinstance(value, str):
        t = value.strip().lower()
    else:
        t = str(value).strip().lower()
    return " ".join(t.split())


def _prior_retrieved_context(steps_by_k: Dict[int, Dict[str, Any]], k: int) -> str:
    """第 k 步之前累积的检索正文（不含第 k 步新文档）。"""
    parts: List[str] = []
    for j in range(1, k):
        s = steps_by_k.get(j)
        if s is None:
            continue
        doc = s.get("retrieved_doc")
        if doc is None:
            continue
        t = str(doc).strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def _tokenize_words_truncated(text: str, max_tokens: int) -> List[str]:
    toks = re.findall(r"\S+", text.lower())
    if len(toks) > max_tokens:
        return toks[:max_tokens]
    return toks


def _rouge_l_f1(ref_tokens: List[str], cand_tokens: List[str]) -> float:
    """ROUGE-L F1（基于词级 LCS）。任一为空且另一非空则返回 0；二者皆空返回 1。"""
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    # dp[i][j] = LCS length of ref[:i] and cand[:j]
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ri = ref_tokens[i - 1]
        for j in range(1, n + 1):
            if ri == cand_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs_len = prev[n]
    r = lcs_len / m
    p = lcs_len / n
    if r + p <= 1e-12:
        return 0.0
    return float(2.0 * r * p / (r + p))


def _d4_answer_retrieval_features(
    steps_by_k: Dict[int, Dict[str, Any]],
    k: int,
    max_k: int,
) -> Tuple[float, float, float]:
    """Phase D4：answer_changed、连续同答案步数（归一化）、检索边际新颖度（1−ROUGE-L）。"""
    step_k = steps_by_k.get(k)
    prev_step = steps_by_k.get(k - 1) if k > 1 else None
    ans_k = _canonical_answer_text(step_k.get("current_answer", "") if step_k else "")

    if k <= 1 or prev_step is None:
        changed = 0.0
    else:
        ans_prev = _canonical_answer_text(prev_step.get("current_answer", ""))
        changed = 1.0 if ans_k != ans_prev else 0.0

    streak = 0
    j = k
    while j >= 1:
        sj = steps_by_k.get(j)
        if sj is None:
            break
        if _canonical_answer_text(sj.get("current_answer", "")) != ans_k:
            break
        streak += 1
        j -= 1
    streak_norm = float(streak) / float(max(1, max_k))

    prior = _prior_retrieved_context(steps_by_k, k)
    new_doc = ""
    if step_k is not None:
        new_doc = str(step_k.get("retrieved_doc", "") or "").strip()
    if not new_doc:
        marginal = 0.0
    elif not prior.strip():
        marginal = 1.0
    else:
        tok_prior = _tokenize_words_truncated(prior, _D4_ROUGE_MAX_TOKENS)
        tok_new = _tokenize_words_truncated(new_doc, _D4_ROUGE_MAX_TOKENS)
        sim = _rouge_l_f1(tok_prior, tok_new)
        marginal = float(max(0.0, min(1.0, 1.0 - sim)))

    return changed, streak_norm, marginal


def _step_shallow_features(
    step: Dict[str, Any],
    k: int,
    cfg: Stage2Config,
    *,
    prev_delta_scalars: Optional[Tuple[float, float, float, float, float]] = None,
    cumulative_cost_ratio: float = 0.0,
) -> List[float]:
    cost = step.get("cost") or {}
    token_count = float(cost.get("token_count", 0) or 0.0)
    latency_ms = float(cost.get("latency_ms", 0.0) or 0.0)
    base = [
        float(k) / float(max(1, cfg.max_k)),
        float(step.get("retrieval_score", 0.0) or 0.0),
        float(step.get("semantic_entropy", 0.0) or 0.0),
        float(step.get("self_consistency", 0.0) or 0.0),
        float(step.get("ctx_overlap", 0.0) or 0.0),
        float(step.get("nli_entail", 0.0) or 0.0),
        float(step.get("nli_contra", 0.0) or 0.0),
        math.log1p(max(0.0, token_count)),
        math.log1p(max(0.0, latency_ms)),
    ]
    curr = _delta_source_scalars(step)
    if k <= 1 or prev_delta_scalars is None:
        deltas = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        deltas = [c - p for c, p in zip(curr, prev_delta_scalars)]
    return base + deltas + [float(cumulative_cost_ratio)]


def _trajectory_max_cumulative_cost(traj: Dict[str, Any], cfg: Stage2Config) -> float:
    v = trajectory_cumulative_cost(
        traj, cfg.max_k, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
    )
    return float(max(v, 1e-8))


def _shallow_row_for_step(
    traj: Dict[str, Any],
    step: Dict[str, Any],
    k: int,
    cfg: Stage2Config,
    steps_by_k: Dict[int, Dict[str, Any]],
    max_total_cost: float,
) -> np.ndarray:
    prev = steps_by_k.get(k - 1) if k > 1 else None
    prev_sc = _delta_source_scalars(prev) if prev is not None else None
    cum = trajectory_cumulative_cost(
        traj, k, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
    )
    ratio = float(cum / max_total_cost)
    vec = _step_shallow_features(
        step,
        k,
        cfg,
        prev_delta_scalars=prev_sc,
        cumulative_cost_ratio=ratio,
    )
    d4 = _d4_answer_retrieval_features(steps_by_k, k, cfg.max_k)
    return np.asarray(list(vec) + list(d4), dtype=np.float32)


def _make_oracle_maps(
    trajectories: List[Dict[str, Any]],
    cfg: Stage2Config,
) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], List[Dict[str, Any]]]:
    oracle_rows = compute_trajectory_oracle(
        trajectories,
        cfg.cost_per_step,
        cfg.max_k,
        cost_metric=cfg.oracle_cost_metric,
    )
    by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in oracle_rows:
        targets = row.get("step_targets") or {}
        step_map: Dict[int, Dict[str, Any]] = {}
        for raw_k, info in targets.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError):
                continue
            step_map[k] = dict(info)
        by_id[str(row.get("id", ""))] = step_map
    return by_id, oracle_rows


def _build_xyw(
    trajectories: List[Dict[str, Any]],
    oracle_by_id: Dict[str, Dict[int, Dict[str, Any]]],
    hidden_map: Dict[Tuple[str, int], np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
    *,
    collect_margins: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    xs_rows: List[np.ndarray] = []
    xh_rows: List[np.ndarray] = []
    y_vals: List[float] = []
    w_vals: List[float] = []
    margin_vals: List[float] = []

    n_missing_hidden = 0
    n_missing_label = 0
    total_steps = 0
    supervised_steps = 0

    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)

    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        if not sample_id:
            continue
        step_targets = oracle_by_id.get(sample_id, {})
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        steps_by_k = {int(s.get("step", 0)): s for s in steps}
        max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
        for step in steps:
            total_steps += 1
            k = int(step.get("step", 0))
            if k <= 0 or k >= cfg.max_k:
                # 最后一步固定 Stop，不参与学习。
                continue
            if k not in step_targets:
                n_missing_label += 1
                continue
            target = step_targets[k]
            action_label = float(target.get("action_label", 0))
            margin = float(target.get("margin", 0.0))
            if collect_margins:
                margin_vals.append(margin)
            weight = max(cfg.margin_weight_floor, abs(margin))
            supervised_steps += 1
            hidden = hidden_map.get((sample_id, k), hidden_map.get((sample_id, 0), zero_hidden))
            if hidden_dim > 0 and (sample_id, k) not in hidden_map and (sample_id, 0) not in hidden_map:
                n_missing_hidden += 1

            shallow = _shallow_row_for_step(
                traj, step, k, cfg, steps_by_k, max_total_cost
            )
            xs_rows.append(shallow)
            xh_rows.append(hidden.astype(np.float32, copy=False))
            y_vals.append(action_label)
            w_vals.append(weight)

    if not xs_rows:
        raise RuntimeError("可训练样本为空，请确认 Stage1 轨迹与 Oracle 标签是否完整。")

    n_supervised = max(1, supervised_steps)
    if hidden_dim > 0 and n_missing_hidden / n_supervised > 0.05:
        LOGGER.warning(
            "超过 5%% 的监督步缺失 Hidden States（missing_hidden_steps=%d / supervised_steps=%d），"
            "请检查 Stage 1 的 hidden_states 缓存是否完整。",
            n_missing_hidden,
            supervised_steps,
        )

    stats: Dict[str, Any] = {
        "total_steps_seen": int(total_steps),
        "trainable_examples": int(len(xs_rows)),
        "missing_hidden_ids": int(n_missing_hidden),
        "missing_step_labels": int(n_missing_label),
    }
    if collect_margins:
        stats["oracle_margins"] = np.asarray(margin_vals, dtype=np.float32)
    x_shallow = np.stack(xs_rows).astype(np.float32)
    x_hidden = np.stack(xh_rows).astype(np.float32)
    y = np.asarray(y_vals, dtype=np.float32)
    w = np.asarray(w_vals, dtype=np.float32)
    return x_shallow, x_hidden, y, w, stats


def _log_stop_continue_balance(split_name: str, y: np.ndarray) -> None:
    n = int(y.size)
    if n <= 0:
        return
    n_cont = int((y > 0.5).sum())
    n_stop = n - n_cont
    LOGGER.info(
        "%s 步级标签 stop/continue：stop=%.2f%% (%d), continue=%.2f%% (%d)",
        split_name,
        100.0 * n_stop / n,
        n_stop,
        100.0 * n_cont / n,
        n_cont,
    )


def _adaptive_focal_alpha_pos(y_train: np.ndarray) -> float:
    """Continue=1 为少数类时提高其在 focal 中的 α，使 alpha_t(label=1) > alpha_t(label=0)。"""
    p_cont = float(np.mean(y_train)) if y_train.size else 0.5
    p_cont = min(max(p_cont, 1e-6), 1.0 - 1e-6)
    return float(min(0.95, max(0.05, 1.0 - p_cont)))


def _focal_weighted_bce_loss(
    logits: torch.Tensor,
    y_hard: torch.Tensor,
    y_for_bce: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
    alpha_pos: float,
) -> torch.Tensor:
    """Focal + 样本权重：BCE 目标可为 label smoothing 后的软标签；focal 的 pt/α 仍用硬标签。"""
    bce = F.binary_cross_entropy_with_logits(logits, y_for_bce, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(y_hard >= 0.5, prob, 1.0 - prob)
    pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)
    focal_w = (1.0 - pt) ** gamma
    alpha_t = torch.where(y_hard >= 0.5, alpha_pos, 1.0 - alpha_pos)
    per = alpha_t * focal_w * bce * weights
    denom = torch.clamp(weights.sum(), min=1e-6)
    return per.sum() / denom


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    dual_input: bool,
    grad_clip_norm: float = 0.0,
    *,
    focal_gamma: float = 2.0,
    focal_alpha_pos: float = 0.25,
    label_smoothing: float = 0.0,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_weight = 0.0
    for batch in loader:
        if dual_input:
            xs, xh, y, w = batch
            xs = xs.to(device)
            xh = xh.to(device)
            y = y.to(device)
            w = w.to(device)
            logits = model(xs, xh)
        else:
            xs, y, w = batch
            xs = xs.to(device)
            y = y.to(device)
            w = w.to(device)
            logits = model(xs)
        y_hard = y
        eps = float(label_smoothing) if is_train else 0.0
        if eps > 0.0:
            y_bce = y_hard * (1.0 - 2.0 * eps) + eps
        else:
            y_bce = y_hard
        loss = _focal_weighted_bce_loss(
            logits, y_hard, y_bce, w, focal_gamma, focal_alpha_pos
        )

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        batch_weight = float(w.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight

    if total_weight <= 0:
        return 0.0
    return total_loss / total_weight


def _predict_probs(
    model: nn.Module,
    x_shallow: np.ndarray,
    x_hidden: np.ndarray,
    batch_size: int,
    device: torch.device,
    dual_input: bool,
) -> np.ndarray:
    model.eval()
    n = int(x_shallow.shape[0])
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xsb = torch.tensor(x_shallow[i : i + batch_size], dtype=torch.float32, device=device)
            if dual_input:
                xhb = torch.tensor(x_hidden[i : i + batch_size], dtype=torch.float32, device=device)
                logits = model(xsb, xhb)
            else:
                logits = model(xsb)
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            out.append(prob)
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def _train_probe(
    x_train_shallow: np.ndarray,
    x_train_hidden: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_dev_shallow: np.ndarray,
    x_dev_hidden: np.ndarray,
    y_dev: np.ndarray,
    w_dev: np.ndarray,
    cfg: Stage2Config,
) -> Tuple[nn.Module, Dict[str, Any], StandardScaler, str]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_shallow).astype(np.float32)
    x_dev_s = scaler.transform(x_dev_shallow).astype(np.float32)

    dual = x_train_hidden.shape[1] > 0
    train_ds = ProbeDataset(x_train_s, y_train, w_train, x_train_hidden if dual else None)
    dev_ds = ProbeDataset(x_dev_s, y_dev, w_dev, x_dev_hidden if dual else None)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dual:
        hd = int(x_train_hidden.shape[1])
        model = ProbeMLP_v2(
            hidden_state_dim=hd,
            shallow_dim=SHALLOW_FEATURE_DIM,
            compress_dim=cfg.compress_dim,
            fuse_dim=cfg.fuse_dim,
            dropout=cfg.dropout,
        ).to(device)
        arch = "mlp_v2"
        LOGGER.info(
            "ProbeMLP_v2：hidden_dim=%d, compress=%d, fuse=%d（浅层 StandardScaler，hidden 用 LayerNorm）。",
            hd,
            cfg.compress_dim,
            cfg.fuse_dim,
        )
    else:
        model = ProbeMLP(
            in_dim=SHALLOW_FEATURE_DIM, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout
        ).to(device)
        arch = "mlp"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    cosine_epochs = cfg.epochs - warmup_epochs
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    if cosine_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )

    clip = float(cfg.grad_clip_norm)
    focal_alpha_pos = (
        float(cfg.focal_alpha)
        if cfg.focal_alpha is not None
        else _adaptive_focal_alpha_pos(y_train)
    )
    focal_gamma = float(cfg.focal_gamma)
    ls_eps = float(cfg.label_smoothing)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_dev_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    hist_rows: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        if epoch <= warmup_epochs:
            denom = max(warmup_epochs - 1, 1)
            frac = (epoch - 1) / denom
            lr = cfg.learning_rate * (0.1 + 0.9 * frac)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            dual,
            grad_clip_norm=clip,
            focal_gamma=focal_gamma,
            focal_alpha_pos=focal_alpha_pos,
            label_smoothing=ls_eps,
        )
        dev_loss = _run_epoch(
            model,
            dev_loader,
            None,
            device,
            dual,
            grad_clip_norm=0.0,
            focal_gamma=focal_gamma,
            focal_alpha_pos=focal_alpha_pos,
            label_smoothing=0.0,
        )
        cur_lr = float(optimizer.param_groups[0]["lr"])
        hist_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "lr": cur_lr,
            }
        )
        LOGGER.info(
            "Probe epoch %02d | lr=%.2e | train_loss=%.6f | dev_loss=%.6f",
            epoch,
            cur_lr,
            train_loss,
            dev_loss,
        )

        if scheduler is not None and warmup_epochs <= epoch < cfg.epochs:
            scheduler.step()

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                LOGGER.info("Early stop at epoch %d (best epoch=%d)", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Probe 训练失败：未得到有效模型状态。")

    model.load_state_dict(best_state)

    dev_probs = _predict_probs(
        model, x_dev_s, x_dev_hidden, cfg.batch_size, device, dual
    )
    dev_pred = (dev_probs >= 0.5).astype(np.int32)
    dev_acc = float((dev_pred == y_dev.astype(np.int32)).mean()) if len(y_dev) > 0 else 0.0

    train_info = {
        "best_epoch": int(best_epoch),
        "best_dev_loss": float(best_dev_loss),
        "dev_acc_at_0.5": dev_acc,
        "history": hist_rows,
        "device": str(device),
        "probe_arch": arch,
        "warmup_epochs": int(warmup_epochs),
        "grad_clip_norm": float(clip),
        "cosine_T_max": int(cosine_epochs) if cosine_epochs > 0 else 0,
        "focal_gamma": focal_gamma,
        "focal_alpha_pos": focal_alpha_pos,
        "focal_alpha_fixed": cfg.focal_alpha is not None,
        "label_smoothing": ls_eps,
        "train_continue_ratio": float(np.mean(y_train)) if y_train.size else 0.0,
    }
    return model, train_info, scaler, arch


def _build_step_feature_map(
    trajectories: List[Dict[str, Any]],
    hidden_map: Dict[Tuple[str, int], np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
) -> Dict[Tuple[str, int], np.ndarray]:
    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        if not sample_id:
            continue
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        steps_by_k = {int(s.get("step", 0)): s for s in steps}
        max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
        for step in steps:
            k = int(step.get("step", 0))
            if k <= 0:
                continue
            hidden = hidden_map.get((sample_id, k), hidden_map.get((sample_id, 0), zero_hidden))
            shallow = _shallow_row_for_step(
                traj, step, k, cfg, steps_by_k, max_total_cost
            )
            out[(sample_id, k)] = np.concatenate([shallow, hidden], axis=0)
    return out


def _precompute_probe_probs(
    step_feature_map: Dict[Tuple[str, int], np.ndarray],
    model: nn.Module,
    scaler: StandardScaler,
    cfg: Stage2Config,
) -> Dict[Tuple[str, int], float]:
    """对 step_feature_map 中所有步特征批量标准化并推理，避免逐步 CPU/GPU 往返。"""
    if not step_feature_map:
        return {}
    device = next(model.parameters()).device
    model.eval()
    sdim = SHALLOW_FEATURE_DIM
    keys = list(step_feature_map.keys())
    feats_arr = np.stack([step_feature_map[k] for k in keys], axis=0)
    x_s = feats_arr[:, :sdim].astype(np.float32, copy=False)
    x_h = feats_arr[:, sdim:].astype(np.float32, copy=False)
    x_s_scaled = scaler.transform(x_s).astype(np.float32)
    dual = isinstance(model, ProbeMLP_v2)
    probs = _predict_probs(model, x_s_scaled, x_h, cfg.batch_size, device, dual)
    return {k: float(p) for k, p in zip(keys, probs)}


def _simulate_probe_policy_from_probs(
    trajectories: List[Dict[str, Any]],
    threshold: float,
    cfg: Stage2Config,
    precomputed_probs: Dict[Tuple[str, int], float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        if not steps:
            rows.append({"f1": 0.0, "em": 0, "steps_used": 0, "avg_cost": 0.0})
            continue

        chosen = steps[-1]
        for step in steps:
            k = int(step.get("step", 0))
            if k >= cfg.max_k:
                chosen = step
                break

            p_continue = precomputed_probs.get((sample_id, k))
            if p_continue is None:
                continue

            if p_continue < threshold:
                chosen = step
                break

        used = int(chosen.get("step", len(steps)))
        cum_cost = trajectory_cumulative_cost(
            traj,
            used,
            cfg.cost_per_step,
            cfg.max_k,
            cfg.oracle_cost_metric,
        )
        rows.append(
            {
                "f1": float(chosen.get("f1", 0.0)),
                "em": int(bool(chosen.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    return rows


def _simulate_probe_policy(
    trajectories: List[Dict[str, Any]],
    step_feature_map: Dict[Tuple[str, int], np.ndarray],
    model: nn.Module,
    scaler: StandardScaler,
    threshold: float,
    cfg: Stage2Config,
    precomputed_probs: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[Dict[str, Any]]:
    if precomputed_probs is None:
        precomputed_probs = _precompute_probe_probs(step_feature_map, model, scaler, cfg)
    return _simulate_probe_policy_from_probs(trajectories, threshold, cfg, precomputed_probs)


def _summarize_results(rows: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    f1s = [float(r.get("f1", 0.0)) for r in rows]
    ems = [int(r.get("em", 0)) for r in rows]
    steps = [int(r.get("steps_used", 0)) for r in rows]
    costs = [float(r.get("avg_cost", 0.0)) for r in rows]
    return {
        "strategy": strategy,
        "avg_steps": float(np.mean(steps) if steps else 0.0),
        "avg_cost": float(np.mean(costs) if costs else 0.0),
        "avg_f1": float(np.mean(f1s) if f1s else 0.0),
        "avg_em": float(np.mean(ems) if ems else 0.0),
    }


def _eval_fixed_k(
    trajectories: List[Dict[str, Any]],
    k: int,
    cfg: Stage2Config,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for traj in trajectories:
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        if not steps:
            out.append({"f1": 0.0, "em": 0, "steps_used": 0, "avg_cost": 0.0})
            continue
        target = next((s for s in steps if int(s.get("step", 0)) == k), steps[-1])
        used = int(target.get("step", len(steps)))
        cum_cost = trajectory_cumulative_cost(
            traj,
            used,
            cfg.cost_per_step,
            cfg.max_k,
            cfg.oracle_cost_metric,
        )
        out.append(
            {
                "f1": float(target.get("f1", 0.0)),
                "em": int(bool(target.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    return out


def _eval_oracle_rows(
    trajectories: List[Dict[str, Any]],
    oracle_rows: List[Dict[str, Any]],
    cfg: Stage2Config,
) -> List[Dict[str, Any]]:
    by_id = {str(t.get("id", "")): t for t in trajectories}
    out: List[Dict[str, Any]] = []
    for row in oracle_rows:
        sample_id = str(row.get("id", ""))
        traj = by_id.get(sample_id)
        if traj is None:
            continue
        used = int(row.get("steps_used", 0))
        cum_cost = trajectory_cumulative_cost(
            traj,
            used,
            cfg.cost_per_step,
            cfg.max_k,
            cfg.oracle_cost_metric,
        )
        out.append(
            {
                "f1": float(row.get("f1", 0.0)),
                "em": int(bool(row.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    return out


def _global_weitzman_avg_steps(
    train_trajectories: List[Dict[str, Any]],
    eval_trajectories: List[Dict[str, Any]],
    cfg: Stage2Config,
) -> float:
    """用 train 上估计的保留值，在 eval 上跑 Global-Weitzman，返回平均停止步数。"""
    if not eval_trajectories:
        return float("nan")
    reservation_values = compute_all_reservation_values(
        train_trajectories, cfg.max_k, cfg.cost_per_step
    )
    raw = oracle_stopping_simulation(eval_trajectories, reservation_values, cfg.max_k)
    if not raw:
        return float("nan")
    steps = [int(r.get("steps_used", 0)) for r in raw]
    return float(np.mean(steps))


def _pick_best_threshold(
    train_trajectories: List[Dict[str, Any]],
    dev_trajectories: List[Dict[str, Any]],
    dev_probs: Dict[Tuple[str, int], float],
    cfg: Stage2Config,
    *,
    gw_dev_avg_steps_precomputed: Optional[float] = None,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """
    Phase C：在 dev 上
    1) 以 GW 平均步数为锚：仅考虑 avg_steps ≤ gw_dev_steps × threshold_gw_steps_cap_mult 的阈值（无可行则回退全体候选）；
    2) 对每个 λ ∈ threshold_pareto_lambdas，在可行集内最大化 F1 − λ * normalized_cost（成本在候选阈值间 min-max 归一化）；
    3) 在四个 λ 各自得到的阈值中，选 dev F1 最高者（平手则更低 avg_cost）。
    """
    candidates = [round(x, 3) for x in np.linspace(0.01, 0.99, 50)]

    if gw_dev_avg_steps_precomputed is not None and not math.isnan(float(gw_dev_avg_steps_precomputed)):
        gw_dev_steps = float(gw_dev_avg_steps_precomputed)
    else:
        gw_dev_steps = _global_weitzman_avg_steps(train_trajectories, dev_trajectories, cfg)
    cap_mult = float(cfg.threshold_gw_steps_cap_mult)
    step_cap = gw_dev_steps * cap_mult if not math.isnan(gw_dev_steps) else float("inf")

    per_t: List[Tuple[float, Dict[str, Any]]] = []
    for t in candidates:
        rs = _simulate_probe_policy_from_probs(dev_trajectories, t, cfg, dev_probs)
        row = _summarize_results(rs, f"Probe@{t:.2f}")
        per_t.append((t, row))

    costs = [row["avg_cost"] for _, row in per_t]
    cmin, cmax = (min(costs), max(costs)) if costs else (0.0, 1.0)
    cspan = cmax - cmin
    if cspan < 1e-12:
        cspan = 1.0

    def norm_cost(row: Dict[str, Any]) -> float:
        return float((row["avg_cost"] - cmin) / cspan)

    feasible_idx = [
        i
        for i, (_, r) in enumerate(per_t)
        if math.isnan(gw_dev_steps) or r["avg_steps"] <= step_cap + 1e-9
    ]
    if not feasible_idx:
        LOGGER.warning(
            "Phase C：无阈值满足 Probe dev avg_steps ≤ GW_dev×%.3f（GW_steps=%.4f, cap=%.4f），"
            "回退为不施加步数上界。",
            cap_mult,
            gw_dev_steps if not math.isnan(gw_dev_steps) else -1.0,
            step_cap if not math.isnan(step_cap) else -1.0,
        )
        feasible_idx = list(range(len(per_t)))

    def pick_for_lambda(lam: float) -> Tuple[float, Dict[str, Any], float]:
        best_t_local = per_t[feasible_idx[0]][0]
        best_row_local = per_t[feasible_idx[0]][1]
        best_score = -1e30
        for i in feasible_idx:
            t, row = per_t[i]
            sc = float(row["avg_f1"]) - lam * norm_cost(row)
            if sc > best_score + 1e-12:
                best_score = sc
                best_t_local, best_row_local = t, row
            elif math.isclose(sc, best_score, rel_tol=1e-9, abs_tol=1e-9):
                if row["avg_f1"] > best_row_local["avg_f1"] or (
                    math.isclose(row["avg_f1"], best_row_local["avg_f1"], rel_tol=1e-9, abs_tol=1e-9)
                    and row["avg_cost"] < best_row_local["avg_cost"]
                ):
                    best_t_local, best_row_local = t, row
        return best_t_local, best_row_local, best_score

    lambda_trace: List[Dict[str, Any]] = []
    best_t = per_t[feasible_idx[0]][0]
    best_row = per_t[feasible_idx[0]][1]
    best_f1_for_lambda_pick = -1.0
    chosen_lambda: Optional[float] = None

    for lam in cfg.threshold_pareto_lambdas:
        t_l, row_l, sc_l = pick_for_lambda(float(lam))
        lambda_trace.append(
            {
                "lambda": float(lam),
                "threshold": float(t_l),
                "dev_avg_f1": float(row_l["avg_f1"]),
                "dev_avg_steps": float(row_l["avg_steps"]),
                "dev_avg_cost": float(row_l["avg_cost"]),
                "pareto_score": float(sc_l),
            }
        )
        if row_l["avg_f1"] > best_f1_for_lambda_pick + 1e-12:
            best_f1_for_lambda_pick = float(row_l["avg_f1"])
            best_t, best_row = t_l, row_l
            chosen_lambda = float(lam)
        elif math.isclose(row_l["avg_f1"], best_f1_for_lambda_pick, rel_tol=1e-9, abs_tol=1e-9):
            if row_l["avg_cost"] < best_row["avg_cost"]:
                best_t, best_row = t_l, row_l
                chosen_lambda = float(lam)

    diag: Dict[str, Any] = {
        "gw_dev_avg_steps": float(gw_dev_steps) if not math.isnan(gw_dev_steps) else None,
        "step_cap": float(step_cap) if not math.isnan(step_cap) else None,
        "cap_mult": cap_mult,
        "normalized_cost_span": {"min": cmin, "max": cmax},
        "feasible_threshold_count": len(feasible_idx),
        "lambda_grid_trace": lambda_trace,
        "chosen_lambda": chosen_lambda,
    }
    best_row = dict(best_row)
    best_row["strategy"] = f"Probe@{best_t:.2f}"
    return float(best_t), best_row, diag


def _pareto_nondominated_min_cost_max_f1(
    points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
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


def _plot_dataset_pareto(
    df: pd.DataFrame, dataset: str, out_path: Path, title_extra: str = ""
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
    points: List[Tuple[float, float]] = []
    for _, row in df.iterrows():
        strategy = str(row["strategy"])
        x = float(row["avg_cost"])
        y = float(row["avg_f1"])
        points.append((x, y))

        if strategy == "Oracle":
            color, marker, size = "#d62728", "*", 180
        elif strategy == "Probe":
            color, marker, size = "#2ca02c", "D", 120
        elif strategy == "Global-Weitzman":
            color, marker, size = "#ff7f0e", "^", 120
        elif strategy == "Deployable-GW":
            color, marker, size = "#9467bd", "v", 120
        else:
            color, marker, size = "#1f77b4", "o", 90

        ax.scatter(x, y, color=color, marker=marker, s=size, label=strategy)
        ax.annotate(strategy, (x, y), fontsize=8)

    nd = _pareto_nondominated_min_cost_max_f1(points)
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

    ax.set_xlabel("Avg cumulative cost (normalized)")
    ax.set_ylabel("F1")
    suffix = f" ({title_extra})" if title_extra else ""
    ax.set_title(f"Stage2 Probe Pareto - {dataset}{suffix}")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_dataset_stage2(cfg: Stage2Config, dataset: str) -> Dict[str, Any]:
    train_traj = _load_trajectories(cfg, dataset, "train")
    dev_traj = _load_trajectories(cfg, dataset, "dev")
    test_traj = _load_trajectories(cfg, dataset, "test")

    if cfg.shallow_only:
        LOGGER.info("Shallow-Only 模式：不使用 hidden states，输入维度=%d。", SHALLOW_FEATURE_DIM)
        hidden_dim = 0
        train_hidden: Dict[str, np.ndarray] = {}
        dev_hidden = {}
        test_hidden = {}
    else:
        hidden_dim = _infer_hidden_dim(cfg, dataset, "train", cfg.hidden_state_key)
        train_hidden = _load_hidden_map(cfg, dataset, "train", hidden_dim, cfg.hidden_state_key)
        dev_hidden = _load_hidden_map(cfg, dataset, "dev", hidden_dim, cfg.hidden_state_key)
        test_hidden = _load_hidden_map(cfg, dataset, "test", hidden_dim, cfg.hidden_state_key)

    train_oracle_map, _train_oracle_rows = _make_oracle_maps(train_traj, cfg)
    dev_oracle_map, _dev_oracle_rows = _make_oracle_maps(dev_traj, cfg)
    _test_oracle_map, test_oracle_rows = _make_oracle_maps(test_traj, cfg)

    x_train_s, x_train_h, y_train, w_train, train_stats = _build_xyw(
        train_traj, train_oracle_map, train_hidden, hidden_dim, cfg
    )
    x_dev_s, x_dev_h, y_dev, w_dev, dev_stats = _build_xyw(
        dev_traj, dev_oracle_map, dev_hidden, hidden_dim, cfg
    )
    _log_stop_continue_balance(f"{dataset} train", y_train)
    _log_stop_continue_balance(f"{dataset} dev", y_dev)
    if x_dev_s.shape[0] > 0 and x_dev_s.shape[1] == SHALLOW_FEATURE_DIM:
        ext_std = x_dev_s[:, 9:].std(axis=0)
        LOGGER.info(
            "%s dev 浅层扩展维 std（Delta×5 + cum_ratio + D4×3，StandardScaler 前）: %s",
            dataset,
            np.array2string(ext_std, precision=4, suppress_small=True),
        )

    model, train_info, scaler, probe_arch = _train_probe(
        x_train_s,
        x_train_h,
        y_train,
        w_train,
        x_dev_s,
        x_dev_h,
        y_dev,
        w_dev,
        cfg,
    )

    dev_step_feat_map = _build_step_feature_map(dev_traj, dev_hidden, hidden_dim, cfg)
    test_step_feat_map = _build_step_feature_map(test_traj, test_hidden, hidden_dim, cfg)
    dev_probs_nn = _precompute_probe_probs(dev_step_feat_map, model, scaler, cfg)
    threshold, best_dev_row, threshold_diag = _pick_best_threshold(
        train_traj, dev_traj, dev_probs_nn, cfg
    )
    LOGGER.info(
        "%s Phase C 阈值：GW_dev_avg_steps=%s cap_mult=%.3f chosen_λ=%s threshold=%.3f dev_f1=%.4f dev_steps=%.3f",
        dataset,
        threshold_diag.get("gw_dev_avg_steps"),
        float(cfg.threshold_gw_steps_cap_mult),
        threshold_diag.get("chosen_lambda"),
        threshold,
        float(best_dev_row["avg_f1"]),
        float(best_dev_row["avg_steps"]),
    )

    rows: List[Dict[str, Any]] = []

    test_probs = _precompute_probe_probs(test_step_feat_map, model, scaler, cfg)
    probe_test = _simulate_probe_policy(
        test_traj,
        test_step_feat_map,
        model,
        scaler,
        threshold,
        cfg,
        precomputed_probs=test_probs,
    )
    probe_row = _summarize_results(probe_test, "Probe")
    rows.append(probe_row)

    fixed_rows: List[Dict[str, Any]] = []
    for k in range(1, cfg.max_k + 1):
        rk = _summarize_results(_eval_fixed_k(test_traj, k, cfg), f"Fixed-K={k}")
        rows.append(rk)
        fixed_rows.append(rk)

    reservation_values = compute_all_reservation_values(train_traj, cfg.max_k, cfg.cost_per_step)
    global_weitzman_raw = oracle_stopping_simulation(test_traj, reservation_values, cfg.max_k)
    global_weitzman_rows: List[Dict[str, Any]] = []
    for traj, result in zip(test_traj, global_weitzman_raw):
        used = int(result.get("steps_used", 0))
        cum_cost = trajectory_cumulative_cost(
            traj, used, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        global_weitzman_rows.append(
            {
                "f1": float(result.get("f1", 0.0)),
                "em": int(bool(result.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    rows.append(_summarize_results(global_weitzman_rows, "Global-Weitzman"))

    dgw_quality_key = "self_consistency"
    dgw_reservation = compute_all_reservation_values_from_proxy(
        train_traj, cfg.max_k, cfg.cost_per_step, dgw_quality_key
    )
    dgw_raw = deployable_weitzman_stopping_simulation(
        test_traj, dgw_reservation, cfg.max_k, dgw_quality_key
    )
    deployable_gw_rows: List[Dict[str, Any]] = []
    for traj, result in zip(test_traj, dgw_raw):
        used = int(result.get("steps_used", 0))
        cum_cost = trajectory_cumulative_cost(
            traj, used, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
        )
        deployable_gw_rows.append(
            {
                "f1": float(result.get("f1", 0.0)),
                "em": int(bool(result.get("em", False))),
                "steps_used": used,
                "avg_cost": float(cum_cost),
            }
        )
    rows.append(_summarize_results(deployable_gw_rows, "Deployable-GW"))

    oracle_rows = _eval_oracle_rows(test_traj, test_oracle_rows, cfg)
    oracle_summary = _summarize_results(oracle_rows, "Oracle")
    rows.append(oracle_summary)

    table_df = pd.DataFrame(rows).sort_values(by=["avg_cost", "avg_f1"], ascending=[True, False])
    tag = _stage2_artifact_tag(cfg)
    table_path = cfg.results_dir / f"stage2_probe_table_{dataset}{tag}.csv"
    table_df.to_csv(table_path, index=False)

    pareto_path = cfg.results_dir / f"stage2_probe_pareto_{dataset}{tag}.png"
    _plot_dataset_pareto(
        table_df, dataset, pareto_path, title_extra="Shallow-Only" if cfg.shallow_only else ""
    )

    best_fixed = max((r["avg_f1"] for r in fixed_rows), default=0.0)
    probe_gain = float(probe_row["avg_f1"] - best_fixed)
    oracle_gap = float(oracle_summary["avg_f1"] - probe_row["avg_f1"])

    model_path = cfg.artifacts_probe_dir / dataset / f"probe_mlp{tag}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "probe_arch": probe_arch,
            "input_dim": int(SHALLOW_FEATURE_DIM + hidden_dim),
            # 与历史 checkpoint 兼容：浅层 ProbeMLP 的塔宽；mlp_v2 时见 fuse_dim / compress_dim。
            "hidden_dim": int(cfg.hidden_dim),
            "mlp_hidden_dim": int(cfg.hidden_dim),
            "compress_dim": int(cfg.compress_dim) if probe_arch == "mlp_v2" else None,
            "fuse_dim": int(cfg.fuse_dim) if probe_arch == "mlp_v2" else None,
            "dropout": float(cfg.dropout),
            "threshold": float(threshold),
            "hidden_state_key": cfg.hidden_state_key,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "shallow_feature_dim": SHALLOW_FEATURE_DIM,
            "shallow_only": bool(cfg.shallow_only),
            "stage1_hidden_dim": int(hidden_dim),
        },
        model_path,
    )

    meta_path = cfg.artifacts_probe_dir / dataset / f"stage2_train_meta{tag}.json"
    _write_json(
        meta_path,
        {
            "dataset": dataset,
            "seed": cfg.seed,
            "hidden_dim_from_stage1": hidden_dim,
            "train_stats": train_stats,
            "dev_stats": dev_stats,
            "train_info": train_info,
            "best_dev_threshold": threshold,
            "best_dev_row": best_dev_row,
            "threshold_selection_phase_c": threshold_diag,
            "probe_summary": probe_row,
            "oracle_summary": oracle_summary,
            "best_fixed_f1": best_fixed,
            "probe_gain_over_best_fixed": probe_gain,
            "oracle_gap_to_probe": oracle_gap,
            "shallow_only": bool(cfg.shallow_only),
            "table_path": str(table_path),
            "pareto_path": str(pareto_path),
            "model_path": str(model_path),
        },
    )

    return {
        "dataset": dataset,
        "table_path": str(table_path),
        "pareto_path": str(pareto_path),
        "model_path": str(model_path),
        "threshold": float(threshold),
        "probe_summary": probe_row,
        "oracle_summary": oracle_summary,
        "best_fixed_f1": best_fixed,
        "probe_gain_over_best_fixed": probe_gain,
        "oracle_gap_to_probe": oracle_gap,
        "train_info": train_info,
    }


def build_stage2_report(
    results: Dict[str, Dict[str, Any]], out_path: Path, title_suffix: str = ""
) -> Path:
    lines: List[str] = []
    lines.append("# Stage2 Report" + (f" ({title_suffix})" if title_suffix else ""))
    lines.append("")
    lines.append("## Probe Performance")
    lines.append("")
    for ds, info in results.items():
        p = info["probe_summary"]
        o = info["oracle_summary"]
        lines.append(
            f"- `{ds}`: threshold={info['threshold']:.2f}, "
            f"probe_f1={p['avg_f1']:.4f}, probe_steps={p['avg_steps']:.3f}, "
            f"oracle_f1={o['avg_f1']:.4f}, oracle_gap={info['oracle_gap_to_probe']:.4f}"
        )
    lines.append("")
    lines.append("## Probe vs Best Fixed-K")
    lines.append("")
    for ds, info in results.items():
        lines.append(
            f"- `{ds}`: best_fixed_f1={info['best_fixed_f1']:.4f}, "
            f"probe_gain={info['probe_gain_over_best_fixed']:+.4f}"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for ds, info in results.items():
        lines.append(
            f"- `{ds}`: table=`{info['table_path']}`, pareto=`{info['pareto_path']}`, "
            f"model=`{info['model_path']}`"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pandora-RAG Stage 2 (Neural Probe)")
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,musique,2wiki",
        help="Comma-separated dataset names in {hotpotqa,musique,2wiki}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--cost-per-step", type=float, default=0.05)
    parser.add_argument(
        "--oracle-cost-metric",
        type=str,
        choices=("fixed", "token", "latency"),
        default="fixed",
        help="与 Stage1 一致的成本口径（用于 Oracle 标签和评估成本）。",
    )
    parser.add_argument(
        "--hidden-state-key",
        type=str,
        choices=("last_token", "mean_pool"),
        default="last_token",
        help="从 Stage1 的 .npz 中取哪个 hidden 向量。",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="浅层单塔 ProbeMLP 的隐层宽度。")
    parser.add_argument(
        "--compress-dim",
        type=int,
        default=64,
        help="ProbeMLP_v2：hidden 压缩维度（默认与 plan 一致）。",
    )
    parser.add_argument(
        "--fuse-dim",
        type=int,
        default=128,
        help="ProbeMLP_v2：融合分类头宽度。",
    )
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="线性 warmup 轮数：lr 从 learning_rate/10 线性升至 learning_rate；之后 CosineAnnealingLR（T_max=epochs−warmup）。",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="训练步梯度裁剪阈值（L2）；0 表示关闭。",
    )
    parser.add_argument(
        "--margin-weight-floor",
        type=float,
        default=0.1,
        help="margin 加权 BCE 的最小样本权重。",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal BCE 的 γ；0 等价于退化为加权 BCE（仍含 α 项）。",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Focal 中正类（Continue=1）的 α；省略则按训练集 Continue 比例自适应。",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="训练步 BCE 目标的 label smoothing ε（0 关闭）。",
    )
    parser.add_argument(
        "--gw-steps-cap-mult",
        type=float,
        default=1.05,
        help="Phase C：dev 上调阈值时 Probe avg_steps 相对 GW(dev) 平均步数的上界倍数（默认 1.05）。",
    )
    parser.add_argument(
        "--pareto-lambdas",
        type=str,
        default="0.1,0.3,0.5,1.0",
        help="Phase C：F1−λ·归一化成本的 λ 网格（逗号分隔）。",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="写入 probe 表 / checkpoint / meta 时的文件名后缀（避免并行或多配置覆盖）。",
    )
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument(
        "--shallow-only",
        action="store_true",
        help="仅使用浅层特征（含 Delta）训练 Probe，不使用 Stage1 的 hidden states（w/o Deep Features 对照）。",
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

    pl_parts = [p.strip() for p in str(args.pareto_lambdas).split(",") if p.strip()]
    pareto_lambdas: Tuple[float, ...] = tuple(float(x) for x in pl_parts) if pl_parts else (0.1, 0.3, 0.5, 1.0)

    cfg = Stage2Config(
        seed=args.seed,
        max_k=args.max_k,
        cost_per_step=args.cost_per_step,
        oracle_cost_metric=args.oracle_cost_metric,
        root_dir=Path(args.root_dir),
        hidden_state_key=args.hidden_state_key,
        hidden_dim=args.hidden_dim,
        compress_dim=args.compress_dim,
        fuse_dim=args.fuse_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        grad_clip_norm=args.grad_clip_norm,
        margin_weight_floor=args.margin_weight_floor,
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha) if args.focal_alpha is not None else None,
        label_smoothing=float(args.label_smoothing),
        shallow_only=bool(args.shallow_only),
        threshold_gw_steps_cap_mult=float(args.gw_steps_cap_mult),
        threshold_pareto_lambdas=pareto_lambdas,
        artifact_suffix=str(args.artifact_suffix or ""),
    )

    _set_seed(cfg.seed)
    _ensure_dirs(cfg, datasets)

    all_results: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        LOGGER.info("===== Stage2 dataset: %s =====", ds)
        all_results[ds] = run_dataset_stage2(cfg, ds)

    tag = _stage2_artifact_tag(cfg)
    report_name = f"stage2_report{tag}.md" if tag else "stage2_report.md"
    report_title = "Shallow-Only Probe" if cfg.shallow_only else ""
    if cfg.artifact_suffix.strip() and not cfg.shallow_only:
        report_title = (report_title + f" ({cfg.artifact_suffix.strip()})").strip()
    report_path = build_stage2_report(
        all_results, cfg.results_dir / report_name, title_suffix=report_title
    )
    LOGGER.info("Stage2 complete. Report: %s", report_path)
    for ds in datasets:
        info = all_results[ds]
        LOGGER.info(
            "%s | probe_f1=%.4f | gain_vs_fixed=%+.4f | oracle_gap=%.4f",
            ds,
            info["probe_summary"]["avg_f1"],
            info["probe_gain_over_best_fixed"],
            info["oracle_gap_to_probe"],
        )


if __name__ == "__main__":
    main()
