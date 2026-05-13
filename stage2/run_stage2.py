"""
Stage-2 pipeline for Pandora-RAG.

Covers:
  A. Build Oracle step labels from Stage-1 cached trajectories
  B. Train Neural Probe（完整数据用双分支 `ProbeMLP_v2`，`--shallow-only` 用浅层 `ProbeMLP`）：
     默认使用 F1 回归（sigmoid + weighted Huber），可选回退为二分类 Focal BCE（Phase B）
  C. Tune stop threshold on dev split（Phase C：GW(dev) 步数约束 + Pareto 分数 F1−λ·归一化成本；P1 支持保守逐步阈值 refinement）
  D. Evaluate on test and compare with baselines

Usage:
  python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki --max-k 5
  python -m stage2.run_stage2 --per-dataset-optimal --datasets hotpotqa,musique,2wiki
  python -m stage2.run_stage2 --rethreshold-only --gw-steps-cap-mult 1.2 --artifact-suffix ablate_cap120
  python -m stage2.run_stage2 --rethreshold-only --artifact-suffix pdopt_best
  python -m stage2.run_stage2 --rethreshold-only --artifact-suffix pdopt_best --disable-step-threshold-refine
  python -m stage2.run_stage2 --rethreshold-only --artifact-suffix ablate_cap120 --load-artifact-suffix pdopt_best
  python -m stage2.run_stage2 --train-margin-min-abs 0.05 --artifact-suffix ablate_margin_m05
  python -m stage2.run_stage2 --hidden-state-key mean_pool --artifact-suffix ablate_hidden_mean_pool
  python -m stage2.run_stage2 --hidden-state-key last_mean_blend --artifact-suffix ablate_hidden_last_mean_blend
  python -m stage2.run_stage2 --hidden-branch-residual --artifact-suffix ablate_hidden_residual
  # Hidden 消融：三数据集统一二分类时显式传 --probe-target binary（覆盖表内 per-dataset probe_target）
  python -m stage2.run_stage2 --per-dataset-optimal --probe-target binary --artifact-suffix pdopt_binary_last_token
  python -m stage2.run_stage2 --per-dataset-optimal --probe-target binary --hidden-branch-residual --artifact-suffix pdopt_binary_hidden_residual
  # 序列建模消融（plan §二.3）：混合最优头（省略 --probe-target）见 pdopt_true_*；全 f1 对照见 pdopt_seqhist / pdopt_seq_gru
  python -m stage2.run_stage2 --per-dataset-optimal --artifact-suffix pdopt_true_baseline
  python -m stage2.run_stage2 --per-dataset-optimal --seq-history-features --artifact-suffix pdopt_true_seqhist
  python -m stage2.run_stage2 --per-dataset-optimal --sequence-gru --artifact-suffix pdopt_true_seq_gru
  python -m stage2.run_stage2 --per-dataset-optimal --probe-target f1 --seq-history-features --artifact-suffix pdopt_seqhist
  python -m stage2.run_stage2 --per-dataset-optimal --probe-target f1 --sequence-gru --artifact-suffix pdopt_seq_gru
  # Lite Probe（部署口径浅层：步数/检索分/廉价词面，无 NLI/熵/自洽/ROUGE 边际新颖度等）
  python -m stage2.run_stage2 --per-dataset-optimal --probe-feature-mode lite --artifact-suffix probe_lite
  （汇总报告写入 docs/stage2_report*.md；CSV/PNG 仍在 results/。rethreshold-only：加载已有 checkpoint，不重训。）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import math
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

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

# 11 维基础 + 7 维步间/历史代理 + cumulative_cost_ratio + Phase D4(3) + P2 稳定性/重写/overlap(9)
BASE_SHALLOW_FEATURE_DIM = 31
# Lite：k_norm + retrieval_score + Δretrieval + cum_cost_ratio + log1p(token/latency)
# + D4 中廉价二值/ streak（无 ROUGE 边际新颖度）+ P2 词面/重写（无 history_best_* 需额外打分器的 gap）
LITE_SHALLOW_FEATURE_DIM = 15
# 历史兼容名：不含序列聚合维（与 XGBoost 基线、旧 checkpoint 一致）
SHALLOW_FEATURE_DIM = BASE_SHALLOW_FEATURE_DIM
# 序列聚合（plan §二.3 建议 A）：teacher forcing 训练 / 自回归推理
SEQ_AGG_FEATURE_DIM = 3
LITE_SHALLOW_FEATURE_NAMES: Tuple[str, ...] = (
    "k_norm",
    "retrieval_score",
    "delta_retrieval_score",
    "cumulative_cost_ratio",
    "log1p_token_count",
    "log1p_latency_ms",
    "answer_changed",
    "answer_consistency_streak_norm",
    "answer_change_rate",
    "answer_flip_count_norm",
    "query_rewrite_jaccard_prev",
    "query_is_repeat",
    "retrieved_doc_jaccard_prev",
    "answer_doc_overlap",
    "query_change_rate",
)
SHALLOW_FEATURE_NAMES: Tuple[str, ...] = (
    "k_norm",
    "retrieval_score",
    "semantic_entropy",
    "self_consistency",
    "answer_logprob",
    "self_eval_score",
    "ctx_overlap",
    "nli_entail",
    "nli_contra",
    "log1p_token_count",
    "log1p_latency_ms",
    "delta_answer_logprob",
    "delta_self_eval_score",
    "delta_retrieval_score",
    "delta_semantic_entropy",
    "delta_self_consistency",
    "delta_ctx_overlap",
    "delta_nli_entail",
    "cumulative_cost_ratio",
    "answer_changed",
    "answer_consistency_streak_norm",
    "retrieval_marginal_novelty",
    "answer_change_rate",
    "answer_flip_count_norm",
    "query_rewrite_jaccard_prev",
    "query_is_repeat",
    "retrieved_doc_jaccard_prev",
    "answer_doc_overlap",
    "history_best_self_eval_gap",
    "history_best_logprob_gap",
    "query_change_rate",
)

ThresholdSpec = Union[float, Dict[int, float]]


def effective_shallow_dim(cfg: Stage2Config) -> int:
    """GRU 序列探针不扩维；否则可选拼接 SEQ_AGG（oracle teacher / 推理用自回归 pred）。"""
    base = LITE_SHALLOW_FEATURE_DIM if cfg.probe_feature_mode == "lite" else BASE_SHALLOW_FEATURE_DIM
    if bool(cfg.sequence_gru):
        return base
    if bool(cfg.seq_history_features):
        return base + SEQ_AGG_FEATURE_DIM
    return base


# D4：ROUGE-L 用截断词序列，避免超长 retrieved_doc 导致 LCS 过慢
_D4_ROUGE_MAX_TOKENS = 256

# Per-dataset 最优：来自 Stage2 消融（dev 选参，见 docs/plan-04-10.md §二.2）
# 值：(compress_dim, train_margin_min_abs, probe_target, hidden_branch_residual)
# 2026-04-11 修正：三数据集均以 binary 头为优。
# 旧注"HotpotQA/MuSiQue binary 与 f1 数值相同"及"2Wiki f1 更优"均源于将旧 binary 运行结果
# 误贴 f1 标签，实际 pdopt_f1_baseline 补跑后 binary 在三数据集均领先 0.05-0.07 F1。
# 2026-04-11 更新：2Wiki 以 --hidden-branch-residual 为最优（+0.0275 F1，0.3463→0.3738）；
# HotpotQA/MuSiQue last_token 仍为最优，残差未带来净收益。
PER_DATASET_OPTIMAL: Dict[str, Tuple[int, float, Literal["binary", "f1"], bool]] = {
    "hotpotqa": (256, 0.0, "binary", False),
    "musique": (256, 0.0, "binary", False),
    "2wiki": (64, 0.02, "binary", True),
}

# 兼容旧名（仅 compress + margin，不含 probe_target）
PER_DATASET_OPTIMAL_COMPRESS_AND_MARGIN: Dict[str, Tuple[int, float]] = {
    k: (v[0], v[1]) for k, v in PER_DATASET_OPTIMAL.items()
}


@dataclass
class Stage2Config:
    seed: int = 42
    max_k: int = 5
    cost_per_step: float = 0.05
    oracle_cost_metric: str = "fixed"
    root_dir: Path = Path(".")
    hidden_state_key: str = "last_token"
    # full：31 维浅层（与历史一致）；lite：15 维部署口径（无 NLI/熵/自洽/ROUGE 边际新颖度等昂贵特征）
    probe_feature_mode: Literal["full", "lite"] = "full"
    # Hidden 分支：在 LayerNorm→Linear→GELU 后加 z+Linear(z) 残差（plan：Residual Compression）。
    hidden_branch_residual: bool = False
    # C1：Shallow-Only — 不使用 Stage1 的 hidden states，仅浅层特征（含 Delta）训练 Probe（w/o Deep Features）。
    shallow_only: bool = False
    # §二.3：在浅层拼接「上一步 oracle 标量 / 历史均值 / 历史 max」；推理时用自回归 pred 填这三维。
    seq_history_features: bool = False
    # §二.3 建议 B：按轨迹 GRU，不在浅层手工拼历史维（与 seq_history_features 二选一）。
    sequence_gru: bool = False
    gru_hidden_dim: int = 128

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
    # 训练集：丢弃 |Oracle margin| < 此阈值的步（模糊标签）；0 表示不过滤。dev/test 仍用全量步做阈值与评估。
    train_margin_min_abs: float = 0.0
    # Phase B：Focal BCE + label smoothing（None 表示按训练集 Continue 比例自适应 focal 正类权重）
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    label_smoothing: float = 0.05
    # Phase C：阈值在 dev 上联合「Pareto 分数 F1−λ·归一化成本」与 GW 步数上界（avg_steps ≤ GW_dev×mult）
    threshold_pareto_lambdas: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    threshold_gw_steps_cap_mult: float = 1.05
    # P1：在通过 GW cap 的前提下，允许对逐步阈值做保守 refinement；
    # 仅当它真正改进 cost-aware utility / matched-budget F1 / Pareto 前沿时才接管主策略。
    enable_step_threshold_refine: bool = True
    step_threshold_refine_passes: int = 3
    # 结果文件名后缀（如 D3 消融 `--artifact-suffix d3_bce`，避免覆盖默认 `stage2_probe_table_*.csv`）
    artifact_suffix: str = ""
    # rethreshold-only 时 checkpoint 加载后缀：
    # 1) 若 load_artifact_suffix 非空，优先用它；
    # 2) 否则回退到 artifact_suffix；
    # 3) 再回退到默认空后缀（历史 probe_mlp*.pt）。
    load_artifact_suffix: str = ""
    # 仅重跑 Phase C 阈值 + test 评估：从 checkpoint 加载权重（不重新训练）。
    rethreshold_only: bool = False
    # Probe 训练目标：binary=Continue/Stop 二分类；f1=回归当前步 F1（0~1）。
    probe_target: Literal["binary", "f1"] = "binary"

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

    @property
    def docs_dir(self) -> Path:
        return self.root_dir / "docs"


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
        hidden_branch_residual: bool = False,
    ):
        super().__init__()
        self.hidden_branch_residual = bool(hidden_branch_residual)
        self.ln_h = nn.LayerNorm(hidden_state_dim)
        self.lin_h = nn.Linear(hidden_state_dim, compress_dim)
        self.residual_h = (
            nn.Linear(compress_dim, compress_dim) if self.hidden_branch_residual else None
        )
        self.drop_h = nn.Dropout(dropout)
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

    def _encode_hidden(self, x_hidden: torch.Tensor) -> torch.Tensor:
        x = self.ln_h(x_hidden)
        u = self.lin_h(x)
        z = F.gelu(u)
        if self.residual_h is not None:
            z = z + self.residual_h(z)
        return self.drop_h(z)

    def forward(self, x_shallow: torch.Tensor, x_hidden: torch.Tensor) -> torch.Tensor:
        h = self._encode_hidden(x_hidden)
        s = self.shallow_branch(x_shallow)
        fused = torch.cat([h, s], dim=-1)
        return self.classifier(fused)


class ProbeSequenceGRU(nn.Module):
    """轨迹级 GRU：与 ProbeMLP_v2 相同的步级编码，再沿时间维 GRU + 逐步 logits。"""

    def __init__(
        self,
        hidden_state_dim: int,
        shallow_dim: int,
        compress_dim: int = 64,
        fuse_dim: int = 128,
        gru_hidden_dim: int = 128,
        dropout: float = 0.25,
        hidden_branch_residual: bool = False,
    ):
        super().__init__()
        self.hidden_branch_residual = bool(hidden_branch_residual)
        self.ln_h = nn.LayerNorm(hidden_state_dim)
        self.lin_h = nn.Linear(hidden_state_dim, compress_dim)
        self.residual_h = (
            nn.Linear(compress_dim, compress_dim) if self.hidden_branch_residual else None
        )
        self.drop_h = nn.Dropout(dropout)
        self.shallow_branch = nn.Sequential(
            nn.Linear(shallow_dim, shallow_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fuse_input = compress_dim + shallow_dim * 2
        self.gru = nn.GRU(fuse_input, gru_hidden_dim, batch_first=True)
        self.drop_seq = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden_dim, 1)

    def encode_step(self, x_shallow: torch.Tensor, x_hidden: torch.Tensor) -> torch.Tensor:
        x = self.ln_h(x_hidden)
        u = self.lin_h(x)
        z = F.gelu(u)
        if self.residual_h is not None:
            z = z + self.residual_h(z)
        z = self.drop_h(z)
        s = self.shallow_branch(x_shallow)
        return torch.cat([z, s], dim=-1)

    def forward(
        self,
        x_shallow: torch.Tensor,
        x_hidden: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = x_shallow.shape
        xs = x_shallow.reshape(b * t, -1)
        xh = x_hidden.reshape(b * t, -1)
        emb = self.encode_step(xs, xh).view(b, t, -1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)
        out = self.drop_seq(out)
        return self.head(out).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dirs(cfg: Stage2Config, datasets: Sequence[str]) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
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


def _checkpoint_tag_for_load(cfg: Stage2Config) -> str:
    """rethreshold-only 的 checkpoint 选择：load_artifact_suffix > artifact_suffix > 空后缀。"""
    load_suffix = (cfg.load_artifact_suffix or "").strip()
    if not load_suffix:
        load_suffix = (cfg.artifact_suffix or "").strip()
    return _stage2_artifact_tag(replace(cfg, artifact_suffix=load_suffix))


def _load_probe_from_checkpoint(
    path: Path,
    cfg: Stage2Config,
) -> Tuple[nn.Module, StandardScaler, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Probe checkpoint 不存在：{path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    arch = str(ckpt.get("probe_arch", "mlp"))
    dropout = float(ckpt.get("dropout", cfg.dropout))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(scaler.mean_.shape[0])
    scaler.n_samples_seen_ = np.array([1], dtype=np.int64)

    state_dict = cast(Dict[str, torch.Tensor], ckpt["model_state_dict"])
    stage1_hd = int(ckpt.get("stage1_hidden_dim", 0))
    shallow_d = int(ckpt.get("shallow_feature_dim") or SHALLOW_FEATURE_DIM)
    if arch == "mlp_v2":
        compress_d = int(
            state_dict["lin_h.weight"].shape[0]
            if "lin_h.weight" in state_dict
            else (ckpt.get("compress_dim") or cfg.compress_dim)
        )
        fuse_d = int(
            state_dict["classifier.0.weight"].shape[0]
            if "classifier.0.weight" in state_dict
            else (ckpt.get("fuse_dim") or cfg.fuse_dim)
        )
        hid_res = bool(
            ckpt.get("hidden_branch_residual", False) or "residual_h.weight" in state_dict
        )
        model = ProbeMLP_v2(
            hidden_state_dim=stage1_hd,
            shallow_dim=shallow_d,
            compress_dim=compress_d,
            fuse_dim=fuse_d,
            dropout=dropout,
            hidden_branch_residual=hid_res,
        ).to(device)
    elif arch == "mlp_v2_gru":
        compress_d = int(
            state_dict["lin_h.weight"].shape[0]
            if "lin_h.weight" in state_dict
            else (ckpt.get("compress_dim") or cfg.compress_dim)
        )
        fuse_d = int(
            state_dict["shallow_branch.0.weight"].shape[0] // 2
            if "shallow_branch.0.weight" in state_dict
            else (ckpt.get("fuse_dim") or cfg.fuse_dim)
        )
        hid_res = bool(
            ckpt.get("hidden_branch_residual", False) or "residual_h.weight" in state_dict
        )
        gru_h = int(
            state_dict["head.weight"].shape[1]
            if "head.weight" in state_dict
            else (ckpt.get("gru_hidden_dim") or cfg.gru_hidden_dim)
        )
        model = ProbeSequenceGRU(
            hidden_state_dim=stage1_hd,
            shallow_dim=shallow_d,
            compress_dim=compress_d,
            fuse_dim=fuse_d,
            gru_hidden_dim=gru_h,
            dropout=dropout,
            hidden_branch_residual=hid_res,
        ).to(device)
    else:
        mlp_h = int(ckpt.get("mlp_hidden_dim") or ckpt.get("hidden_dim") or cfg.hidden_dim)
        model = ProbeMLP(in_dim=shallow_d, hidden_dim=mlp_h, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, arch


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
                if key == "last_mean_blend":
                    if "last_token" not in obj or "mean_pool" not in obj:
                        continue
                    a = np.asarray(obj["last_token"]).reshape(-1)
                    b = np.asarray(obj["mean_pool"]).reshape(-1)
                    if a.size > 0 and a.size == b.size:
                        return int(a.size)
                elif key in obj:
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
                if key == "last_mean_blend":
                    if "last_token" not in obj or "mean_pool" not in obj:
                        bad += 1
                        continue
                    a = np.asarray(obj["last_token"], dtype=np.float32).reshape(-1)
                    b = np.asarray(obj["mean_pool"], dtype=np.float32).reshape(-1)
                    if a.size != hidden_dim or b.size != hidden_dim:
                        bad += 1
                        continue
                    vec = 0.5 * (a + b)
                elif key not in obj:
                    bad += 1
                    continue
                else:
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


def _delta_source_scalars(step: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
    """用于差分特征的 7 个标量：logprob / self-eval + 原有 5 个连续代理。"""
    return (
        float(step.get("answer_logprob", 0.0) or 0.0),
        float(step.get("self_eval_score", 0.0) or 0.0),
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


def _token_set_truncated(text: Any, max_tokens: int = _D4_ROUGE_MAX_TOKENS) -> set[str]:
    if text is None:
        return set()
    return set(_tokenize_words_truncated(str(text), max_tokens))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return float(len(a & b) / max(1, len(a)))


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


def _p2_history_stability_features(
    steps_by_k: Dict[int, Dict[str, Any]],
    k: int,
    max_k: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """P2：只依赖现有轨迹缓存的稳定性/重写/overlap/history-best 代理。"""
    step_k = steps_by_k.get(k) or {}
    ans_k = _canonical_answer_text(step_k.get("current_answer", ""))
    q_k = str(step_k.get("retrieval_query", "") or "").strip().lower()
    doc_k = str(step_k.get("retrieved_doc", "") or "").strip()

    prev_answers = [
        _canonical_answer_text((steps_by_k.get(j) or {}).get("current_answer", ""))
        for j in range(1, k)
        if steps_by_k.get(j) is not None
    ]
    flips = 0
    if prev_answers:
        chain = prev_answers + [ans_k]
        for prev_ans, cur_ans in zip(chain[:-1], chain[1:]):
            if cur_ans != prev_ans:
                flips += 1
    ans_change_rate = float(flips / max(1, k - 1)) if k > 1 else 0.0
    ans_flip_count_norm = float(flips / max(1, max_k - 1))

    prev_q = str((steps_by_k.get(k - 1) or {}).get("retrieval_query", "") or "").strip().lower()
    q_tokens = _token_set_truncated(q_k)
    prev_q_tokens = _token_set_truncated(prev_q)
    query_rewrite_jaccard_prev = _jaccard_similarity(q_tokens, prev_q_tokens) if k > 1 else 0.0

    prior_queries = [
        str((steps_by_k.get(j) or {}).get("retrieval_query", "") or "").strip().lower()
        for j in range(1, k)
        if steps_by_k.get(j) is not None
    ]
    query_is_repeat = 1.0 if q_k and q_k in prior_queries else 0.0
    unique_queries = len({q for q in prior_queries + ([q_k] if q_k else []) if q})
    query_change_rate = (
        float((unique_queries - 1) / max(1, k - 1)) if k > 1 and unique_queries > 0 else 0.0
    )

    prev_doc = str((steps_by_k.get(k - 1) or {}).get("retrieved_doc", "") or "").strip()
    doc_tokens = _token_set_truncated(doc_k)
    prev_doc_tokens = _token_set_truncated(prev_doc)
    retrieved_doc_jaccard_prev = _jaccard_similarity(doc_tokens, prev_doc_tokens) if k > 1 else 0.0

    answer_doc_overlap = _overlap_ratio(_token_set_truncated(ans_k, 64), doc_tokens)

    prev_self_eval = [
        float((steps_by_k.get(j) or {}).get("self_eval_score", 0.0) or 0.0)
        for j in range(1, k)
        if steps_by_k.get(j) is not None
    ]
    cur_self_eval = float(step_k.get("self_eval_score", 0.0) or 0.0)
    history_best_self_eval_gap = (
        float(max(prev_self_eval) - cur_self_eval) if prev_self_eval else 0.0
    )

    prev_logprob = [
        float((steps_by_k.get(j) or {}).get("answer_logprob", 0.0) or 0.0)
        for j in range(1, k)
        if steps_by_k.get(j) is not None
    ]
    cur_logprob = float(step_k.get("answer_logprob", 0.0) or 0.0)
    history_best_logprob_gap = float(max(prev_logprob) - cur_logprob) if prev_logprob else 0.0

    return (
        ans_change_rate,
        ans_flip_count_norm,
        query_rewrite_jaccard_prev,
        query_is_repeat,
        retrieved_doc_jaccard_prev,
        answer_doc_overlap,
        history_best_self_eval_gap,
        history_best_logprob_gap,
        query_change_rate,
    )


def _lite_shallow_row_for_step(
    traj: Dict[str, Any],
    step: Dict[str, Any],
    k: int,
    cfg: Stage2Config,
    steps_by_k: Dict[int, Dict[str, Any]],
    max_total_cost: float,
) -> np.ndarray:
    """Lite 浅层：步归一化、检索分、廉价日志与词面特征；不含 NLI/熵/ROUGE 边际新颖度等。"""
    prev = steps_by_k.get(k - 1) if k > 1 else None
    prev_rs = float((prev or {}).get("retrieval_score", 0.0) or 0.0)
    cur_rs = float(step.get("retrieval_score", 0.0) or 0.0)
    delta_rs = 0.0 if k <= 1 or prev is None else float(cur_rs - prev_rs)
    cum = trajectory_cumulative_cost(
        traj, k, cfg.cost_per_step, cfg.max_k, cfg.oracle_cost_metric
    )
    ratio = float(cum / max_total_cost)
    cost = step.get("cost") or {}
    token_count = float(cost.get("token_count", 0) or 0.0)
    latency_ms = float(cost.get("latency_ms", 0.0) or 0.0)
    parts = [
        float(k) / float(max(1, cfg.max_k)),
        cur_rs,
        delta_rs,
        ratio,
        math.log1p(max(0.0, token_count)),
        math.log1p(max(0.0, latency_ms)),
    ]
    changed, streak_norm, _marginal = _d4_answer_retrieval_features(steps_by_k, k, cfg.max_k)
    p2 = _p2_history_stability_features(steps_by_k, k, cfg.max_k)
    p2_lite = (p2[0], p2[1], p2[2], p2[3], p2[4], p2[5], p2[8])
    return np.asarray(
        list(parts) + [float(changed), float(streak_norm)] + [float(x) for x in p2_lite],
        dtype=np.float32,
    )


def _step_shallow_features(
    step: Dict[str, Any],
    k: int,
    cfg: Stage2Config,
    *,
    prev_delta_scalars: Optional[Tuple[float, float, float, float, float, float, float]] = None,
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
        float(step.get("answer_logprob", 0.0) or 0.0),
        float(step.get("self_eval_score", 0.0) or 0.0),
        float(step.get("ctx_overlap", 0.0) or 0.0),
        float(step.get("nli_entail", 0.0) or 0.0),
        float(step.get("nli_contra", 0.0) or 0.0),
        math.log1p(max(0.0, token_count)),
        math.log1p(max(0.0, latency_ms)),
    ]
    curr = _delta_source_scalars(step)
    if k <= 1 or prev_delta_scalars is None:
        deltas = [0.0] * len(curr)
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
    if cfg.probe_feature_mode == "lite":
        return _lite_shallow_row_for_step(traj, step, k, cfg, steps_by_k, max_total_cost)
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
    p2 = _p2_history_stability_features(steps_by_k, k, cfg.max_k)
    return np.asarray(list(vec) + list(d4) + list(p2), dtype=np.float32)


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
    margin_min_abs: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    xs_rows: List[np.ndarray] = []
    xh_rows: List[np.ndarray] = []
    y_vals: List[float] = []
    w_vals: List[float] = []
    margin_vals: List[float] = []

    n_missing_hidden = 0
    n_missing_label = 0
    n_skipped_low_margin = 0
    total_steps = 0
    supervised_steps = 0

    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    m_floor = float(margin_min_abs)

    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        if not sample_id:
            continue
        step_targets = oracle_by_id.get(sample_id, {})
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        steps_by_k = {int(s.get("step", 0)): s for s in steps}
        max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
        oracle_scalar_by_k: Dict[int, float] = {}
        for raw_k, tinfo in step_targets.items():
            try:
                kk = int(raw_k)
            except (TypeError, ValueError):
                continue
            if kk <= 0:
                continue
            st_k = steps_by_k.get(kk)
            if cfg.probe_target == "binary":
                oracle_scalar_by_k[kk] = float(tinfo.get("action_label", 0))
            else:
                ev = float(
                    tinfo.get("expected_continue_val", st_k.get("f1", 0.0) if st_k else 0.0)
                    or 0.0
                )
                oracle_scalar_by_k[kk] = float(max(0.0, min(1.0, ev)))
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
            if m_floor > 0.0 and abs(margin) < m_floor:
                n_skipped_low_margin += 1
                continue
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
            if cfg.seq_history_features and not cfg.sequence_gru:
                prevs = [oracle_scalar_by_k[j] for j in range(1, k) if j in oracle_scalar_by_k]
                prev = (
                    float(oracle_scalar_by_k[k - 1])
                    if k > 1 and (k - 1) in oracle_scalar_by_k
                    else 0.0
                )
                run_mean = float(np.mean(prevs)) if prevs else 0.0
                run_max = float(np.max(prevs)) if prevs else 0.0
                shallow = np.concatenate(
                    [shallow, np.asarray([prev, run_mean, run_max], dtype=np.float32)]
                )
            if cfg.probe_target == "binary":
                target_val = action_label
            else:
                # 回归 Oracle 的 expected_continue_val（连续目标）；缺失时回退当前步 F1。
                target_val = float(target.get("expected_continue_val", step.get("f1", 0.0)) or 0.0)
                target_val = float(max(0.0, min(1.0, target_val)))
            xs_rows.append(shallow)
            xh_rows.append(hidden.astype(np.float32, copy=False))
            y_vals.append(target_val)
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
        "skipped_low_margin": int(n_skipped_low_margin),
        "margin_min_abs_filter": float(m_floor),
    }
    if collect_margins:
        stats["oracle_margins"] = np.asarray(margin_vals, dtype=np.float32)
    x_shallow = np.stack(xs_rows).astype(np.float32)
    x_hidden = np.stack(xh_rows).astype(np.float32)
    y = np.asarray(y_vals, dtype=np.float32)
    w = np.asarray(w_vals, dtype=np.float32)
    return x_shallow, x_hidden, y, w, stats


def _build_gru_trajectory_entries(
    trajectories: List[Dict[str, Any]],
    oracle_by_id: Dict[str, Dict[int, Dict[str, Any]]],
    hidden_map: Dict[Tuple[str, int], np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
) -> List[Dict[str, Any]]:
    """每条轨迹一条序列：k=1..max_k−1 且轨迹中存在该步；监督位由 mask 标出（含 margin 过滤）。"""
    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    m_floor = float(cfg.train_margin_min_abs)
    out: List[Dict[str, Any]] = []
    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        if not sample_id:
            continue
        step_targets = oracle_by_id.get(sample_id, {})
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        steps_by_k = {int(s.get("step", 0)): s for s in steps}
        max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
        seq_xs: List[np.ndarray] = []
        seq_xh: List[np.ndarray] = []
        seq_y: List[float] = []
        seq_w: List[float] = []
        seq_m: List[float] = []
        seq_k: List[int] = []
        for k in range(1, cfg.max_k):
            step = steps_by_k.get(k)
            if step is None:
                continue
            hidden = hidden_map.get((sample_id, k), hidden_map.get((sample_id, 0), zero_hidden))
            shallow = _shallow_row_for_step(traj, step, k, cfg, steps_by_k, max_total_cost)
            seq_k.append(k)
            seq_xs.append(np.asarray(shallow, dtype=np.float32, copy=False))
            seq_xh.append(hidden.astype(np.float32, copy=False))
            if k not in step_targets:
                seq_y.append(0.0)
                seq_w.append(0.0)
                seq_m.append(0.0)
                continue
            target = step_targets[k]
            margin = float(target.get("margin", 0.0))
            if m_floor > 0.0 and abs(margin) < m_floor:
                seq_y.append(0.0)
                seq_w.append(0.0)
                seq_m.append(0.0)
                continue
            weight = max(cfg.margin_weight_floor, abs(margin))
            if cfg.probe_target == "binary":
                yv = float(target.get("action_label", 0))
            else:
                yv = float(target.get("expected_continue_val", step.get("f1", 0.0)) or 0.0)
                yv = float(max(0.0, min(1.0, yv)))
            seq_y.append(yv)
            seq_w.append(weight)
            seq_m.append(1.0)
        if not seq_xs:
            continue
        out.append(
            {
                "sample_id": sample_id,
                "ks": list(seq_k),
                "xs": np.stack(seq_xs),
                "xh": np.stack(seq_xh),
                "y": np.asarray(seq_y, dtype=np.float32),
                "w": np.asarray(seq_w, dtype=np.float32),
                "mask": np.asarray(seq_m, dtype=np.float32),
            }
        )
    return out


def _collate_gru_batch(
    batch: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
    ],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs_l, xh_l, y_l, w_l, m_l, lens_t = zip(*batch)
    t_max = int(max(lens_t))
    b = len(batch)
    sdim = int(xs_l[0].shape[1])
    hd = int(xh_l[0].shape[1])
    xs_pad = torch.zeros(b, t_max, sdim, dtype=torch.float32)
    xh_pad = torch.zeros(b, t_max, hd, dtype=torch.float32)
    y_pad = torch.zeros(b, t_max, dtype=torch.float32)
    w_pad = torch.zeros(b, t_max, dtype=torch.float32)
    m_pad = torch.zeros(b, t_max, dtype=torch.float32)
    for i, ell in enumerate(lens_t):
        xs_pad[i, :ell] = xs_l[i]
        xh_pad[i, :ell] = xh_l[i]
        y_pad[i, :ell] = y_l[i]
        w_pad[i, :ell] = w_l[i]
        m_pad[i, :ell] = m_l[i]
    lengths = torch.tensor(lens_t, dtype=torch.long)
    return xs_pad, xh_pad, y_pad, w_pad, m_pad, lengths


class _TrajectoryGRUDataset(Dataset):
    def __init__(self, entries: List[Dict[str, Any]], scaler: StandardScaler):
        self.entries = entries
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        e = self.entries[idx]
        xs = self.scaler.transform(e["xs"]).astype(np.float32)
        return (
            torch.from_numpy(xs),
            torch.from_numpy(e["xh"]),
            torch.from_numpy(e["y"]),
            torch.from_numpy(e["w"]),
            torch.from_numpy(e["mask"]),
            int(xs.shape[0]),
        )


def _run_epoch_gru(
    model: ProbeSequenceGRU,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip_norm: float = 0.0,
    *,
    focal_gamma: float = 2.0,
    focal_alpha_pos: float = 0.25,
    label_smoothing: float = 0.0,
    objective: Literal["binary", "f1"] = "f1",
    huber_delta: float = 0.1,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_weight = 0.0
    for batch in loader:
        xs, xh, y, w, m, lens = batch
        xs = xs.to(device)
        xh = xh.to(device)
        y = y.to(device)
        w = w.to(device)
        m = m.to(device)
        logits = model(xs, xh, lens)
        wm = w * m
        if float(wm.sum().item()) <= 0.0:
            continue
        if objective == "binary":
            y_hard = y
            eps = float(label_smoothing) if is_train else 0.0
            if eps > 0.0:
                y_bce = y_hard * (1.0 - 2.0 * eps) + eps
            else:
                y_bce = y_hard
            bce = F.binary_cross_entropy_with_logits(logits, y_bce, reduction="none")
            prob = torch.sigmoid(logits)
            pt = torch.where(y_hard >= 0.5, prob, 1.0 - prob)
            pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)
            focal_w = (1.0 - pt) ** focal_gamma
            alpha_t = torch.where(y_hard >= 0.5, focal_alpha_pos, 1.0 - focal_alpha_pos)
            per = alpha_t * focal_w * bce * wm
            denom = torch.clamp(wm.sum(), min=1e-6)
            loss = per.sum() / denom
        else:
            pred_f1 = torch.sigmoid(logits)
            per = F.huber_loss(
                pred_f1,
                torch.clamp(y, min=0.0, max=1.0),
                delta=float(huber_delta),
                reduction="none",
            )
            per = per * wm
            denom = torch.clamp(wm.sum(), min=1e-6)
            loss = per.sum() / denom

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        batch_weight = float(wm.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight

    if total_weight <= 0:
        return 0.0
    return total_loss / total_weight


def _train_probe_gru(
    train_entries: List[Dict[str, Any]],
    dev_entries: List[Dict[str, Any]],
    cfg: Stage2Config,
    hidden_dim: int,
) -> Tuple[nn.Module, Dict[str, Any], StandardScaler, str]:
    if not train_entries:
        raise RuntimeError("GRU：train 轨迹序列为空。")
    scaler = StandardScaler()
    scaler.fit(np.vstack([e["xs"] for e in train_entries]).astype(np.float32))

    train_ds = _TrajectoryGRUDataset(train_entries, scaler)
    dev_ds = _TrajectoryGRUDataset(dev_entries, scaler)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch_size, max(1, len(train_ds))),
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_gru_batch,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=min(cfg.batch_size, max(1, len(dev_ds))),
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_gru_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sdim = effective_shallow_dim(cfg)
    model = ProbeSequenceGRU(
        hidden_state_dim=hidden_dim,
        shallow_dim=sdim,
        compress_dim=cfg.compress_dim,
        fuse_dim=cfg.fuse_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        dropout=cfg.dropout,
        hidden_branch_residual=bool(cfg.hidden_branch_residual),
    ).to(device)
    arch = "mlp_v2_gru"
    LOGGER.info(
        "ProbeSequenceGRU：hidden_dim=%d, shallow=%d, compress=%d, fuse=%d, gru_hidden=%d, hidden_residual=%s。",
        hidden_dim,
        sdim,
        cfg.compress_dim,
        cfg.fuse_dim,
        cfg.gru_hidden_dim,
        cfg.hidden_branch_residual,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    cosine_epochs = cfg.epochs - warmup_epochs
    scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    if cosine_epochs > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )

    clip = float(cfg.grad_clip_norm)
    y_flat = np.concatenate([e["y"][e["mask"] > 0.5] for e in train_entries if (e["mask"] > 0.5).any()])
    focal_alpha_pos = (
        float(cfg.focal_alpha)
        if cfg.focal_alpha is not None
        else (_adaptive_focal_alpha_pos(y_flat) if y_flat.size else 0.25)
    )
    objective = str(cfg.probe_target)
    if objective not in {"binary", "f1"}:
        raise ValueError(f"不支持的 probe_target={objective!r}")

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

        train_loss = _run_epoch_gru(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=clip,
            focal_gamma=float(cfg.focal_gamma),
            focal_alpha_pos=focal_alpha_pos,
            label_smoothing=float(cfg.label_smoothing),
            objective=objective,
        )
        dev_loss = _run_epoch_gru(
            model,
            dev_loader,
            None,
            device,
            grad_clip_norm=0.0,
            focal_gamma=float(cfg.focal_gamma),
            focal_alpha_pos=focal_alpha_pos,
            label_smoothing=0.0,
            objective=objective,
        )
        cur_lr = float(optimizer.param_groups[0]["lr"])
        hist_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "lr": cur_lr}
        )
        LOGGER.info(
            "Probe GRU epoch %02d | lr=%.2e | train_loss=%.6f | dev_loss=%.6f",
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
                LOGGER.info("Early stop GRU at epoch %d (best epoch=%d)", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("GRU 训练失败：未得到有效模型状态。")
    model.load_state_dict(best_state)

    train_info: Dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "best_dev_loss": float(best_dev_loss),
        "history": hist_rows,
        "device": str(device),
        "probe_arch": arch,
        "gru_hidden_dim": int(cfg.gru_hidden_dim),
        "warmup_epochs": int(warmup_epochs),
        "grad_clip_norm": float(clip),
        "cosine_T_max": int(cosine_epochs) if cosine_epochs > 0 else 0,
        "focal_gamma": float(cfg.focal_gamma),
        "focal_alpha": focal_alpha_pos,
        "focal_alpha_fixed": cfg.focal_alpha is not None,
        "label_smoothing": float(cfg.label_smoothing),
        "train_continue_ratio": float(np.mean(y_flat)) if (y_flat.size and objective == "binary") else None,
        "probe_target": objective,
    }
    return model, train_info, scaler, arch


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
    objective: Literal["binary", "f1"] = "f1",
    huber_delta: float = 0.1,
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
        if objective == "binary":
            y_hard = y
            eps = float(label_smoothing) if is_train else 0.0
            if eps > 0.0:
                y_bce = y_hard * (1.0 - 2.0 * eps) + eps
            else:
                y_bce = y_hard
            loss = _focal_weighted_bce_loss(
                logits, y_hard, y_bce, w, focal_gamma, focal_alpha_pos
            )
        else:
            pred_f1 = torch.sigmoid(logits)
            per = F.huber_loss(
                pred_f1,
                torch.clamp(y, min=0.0, max=1.0),
                delta=float(huber_delta),
                reduction="none",
            )
            per = per * w
            denom = torch.clamp(w.sum(), min=1e-6)
            loss = per.sum() / denom

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
            pred = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            out.append(pred)
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
    shallow_in = int(x_train_shallow.shape[1])
    if dual:
        hd = int(x_train_hidden.shape[1])
        model = ProbeMLP_v2(
            hidden_state_dim=hd,
            shallow_dim=shallow_in,
            compress_dim=cfg.compress_dim,
            fuse_dim=cfg.fuse_dim,
            dropout=cfg.dropout,
            hidden_branch_residual=bool(cfg.hidden_branch_residual),
        ).to(device)
        arch = "mlp_v2"
        LOGGER.info(
            "ProbeMLP_v2：hidden_dim=%d, shallow_in=%d, compress=%d, fuse=%d, hidden_residual=%s（浅层 StandardScaler，hidden 用 LayerNorm）。",
            hd,
            shallow_in,
            cfg.compress_dim,
            cfg.fuse_dim,
            cfg.hidden_branch_residual,
        )
    else:
        model = ProbeMLP(
            in_dim=shallow_in, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout
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
    objective = str(cfg.probe_target)
    if objective not in {"binary", "f1"}:
        raise ValueError(f"不支持的 probe_target={objective!r}")

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
            objective=objective,
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
            objective=objective,
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
    dev_acc = float((dev_pred == y_dev.astype(np.int32)).mean()) if (len(y_dev) > 0 and objective == "binary") else None
    dev_mae = float(np.mean(np.abs(dev_probs - y_dev))) if len(y_dev) > 0 else 0.0

    train_info = {
        "best_epoch": int(best_epoch),
        "best_dev_loss": float(best_dev_loss),
        "dev_acc_at_0.5": dev_acc,
        "dev_mae": dev_mae,
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
        "train_continue_ratio": float(np.mean(y_train)) if (y_train.size and objective == "binary") else None,
        "probe_target": objective,
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


def _precompute_probe_probs_from_map(
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
    sdim = int(scaler.n_features_in_)
    keys = list(step_feature_map.keys())
    feats_arr = np.stack([step_feature_map[k] for k in keys], axis=0)
    dual = isinstance(model, ProbeMLP_v2)
    if dual:
        hidden_dim = int(model.ln_h.normalized_shape[0])
        x_s = feats_arr[:, :sdim].astype(np.float32, copy=False)
        x_h = feats_arr[:, -hidden_dim:].astype(np.float32, copy=False)
    else:
        x_s = feats_arr[:, :sdim].astype(np.float32, copy=False)
        x_h = feats_arr[:, 0:0].astype(np.float32, copy=False)
    x_s_scaled = scaler.transform(x_s).astype(np.float32)
    probs = _predict_probs(model, x_s_scaled, x_h, cfg.batch_size, device, dual)
    return {k: float(p) for k, p in zip(keys, probs)}


def _precompute_probe_probs_autoreg_mlp(
    trajectories: List[Dict[str, Any]],
    hidden_map: Dict[Tuple[str, int], np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
    model: nn.Module,
    scaler: StandardScaler,
) -> Dict[Tuple[str, int], float]:
    """序列浅层聚合：按轨迹时间序用前几步 pred 填 SEQ_AGG 三维后再推理。"""
    device = next(model.parameters()).device
    model.eval()
    sdim = int(scaler.n_features_in_)
    dual = isinstance(model, ProbeMLP_v2)
    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    out: Dict[Tuple[str, int], float] = {}
    with torch.no_grad():
        for traj in trajectories:
            sample_id = str(traj.get("id", ""))
            if not sample_id:
                continue
            steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
            steps_by_k = {int(s.get("step", 0)): s for s in steps}
            max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
            preds_so_far: List[float] = []
            for step in steps:
                k = int(step.get("step", 0))
                if k <= 0 or k >= cfg.max_k:
                    continue
                shallow_b = _shallow_row_for_step(
                    traj, step, k, cfg, steps_by_k, max_total_cost
                )
                if preds_so_far:
                    pkm1 = preds_so_far[-1]
                    rm = float(np.mean(preds_so_far))
                    mx = float(np.max(preds_so_far))
                else:
                    pkm1 = rm = mx = 0.0
                shallow_f = np.concatenate(
                    [shallow_b, np.asarray([pkm1, rm, mx], dtype=np.float32)]
                )
                assert int(shallow_f.shape[0]) == sdim, (shallow_f.shape[0], sdim)
                hidden = hidden_map.get((sample_id, k), hidden_map.get((sample_id, 0), zero_hidden))
                x_s = scaler.transform(shallow_f.reshape(1, -1)).astype(np.float32)
                x_s_t = torch.from_numpy(x_s).to(device)
                if dual:
                    x_h = hidden.reshape(1, -1).astype(np.float32, copy=False)
                    x_h_t = torch.from_numpy(x_h).to(device)
                    logit = model(x_s_t, x_h_t)
                else:
                    logit = model(x_s_t)
                p = float(torch.sigmoid(logit).detach().cpu().item())
                preds_so_far.append(p)
                out[(sample_id, k)] = p
    return out


def _build_autoreg_step_feature_map(
    trajectories: List[Dict[str, Any]],
    hidden_map: Dict[Tuple[str, int], np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
    model: nn.Module,
    scaler: StandardScaler,
) -> Dict[Tuple[str, int], np.ndarray]:
    """与 autoreg 推理一致：浅层 20+3 维拼接 hidden，供 Stage3 取 shallow 子向量。"""
    sdim = int(scaler.n_features_in_)
    dual = isinstance(model, ProbeMLP_v2)
    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    out: Dict[Tuple[str, int], np.ndarray] = {}
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for traj in trajectories:
            sample_id = str(traj.get("id", ""))
            if not sample_id:
                continue
            steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
            steps_by_k = {int(s.get("step", 0)): s for s in steps}
            max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
            preds_so_far: List[float] = []
            for step in steps:
                k = int(step.get("step", 0))
                if k <= 0 or k >= cfg.max_k:
                    continue
                shallow_b = _shallow_row_for_step(
                    traj, step, k, cfg, steps_by_k, max_total_cost
                )
                if preds_so_far:
                    pkm1 = preds_so_far[-1]
                    rm = float(np.mean(preds_so_far))
                    mx = float(np.max(preds_so_far))
                else:
                    pkm1 = rm = mx = 0.0
                shallow_f = np.concatenate(
                    [shallow_b, np.asarray([pkm1, rm, mx], dtype=np.float32)]
                )
                hidden = hidden_map.get((sample_id, k), hidden_map.get((sample_id, 0), zero_hidden))
                x_s = scaler.transform(shallow_f.reshape(1, -1)).astype(np.float32)
                x_s_t = torch.from_numpy(x_s).to(device)
                if dual:
                    x_h = hidden.reshape(1, -1).astype(np.float32, copy=False)
                    x_h_t = torch.from_numpy(x_h).to(device)
                    logit = model(x_s_t, x_h_t)
                else:
                    logit = model(x_s_t)
                p = float(torch.sigmoid(logit).detach().cpu().item())
                preds_so_far.append(p)
                out[(sample_id, k)] = np.concatenate([shallow_f, hidden.astype(np.float32, copy=False)])
    assert not out or next(iter(out.values())).shape[0] == sdim + hidden_dim
    return out


def _precompute_probe_probs_gru(
    trajectories: List[Dict[str, Any]],
    hidden_map: Dict[Tuple[str, int], np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
    model: ProbeSequenceGRU,
    scaler: StandardScaler,
) -> Dict[Tuple[str, int], float]:
    device = next(model.parameters()).device
    model.eval()
    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    out: Dict[Tuple[str, int], float] = {}
    with torch.no_grad():
        for traj in trajectories:
            sample_id = str(traj.get("id", ""))
            if not sample_id:
                continue
            steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
            steps_by_k = {int(s.get("step", 0)): s for s in steps}
            max_total_cost = _trajectory_max_cumulative_cost(traj, cfg)
            seq_xs: List[np.ndarray] = []
            seq_xh: List[np.ndarray] = []
            ks: List[int] = []
            for k in range(1, cfg.max_k):
                step = steps_by_k.get(k)
                if step is None:
                    continue
                hidden = hidden_map.get((sample_id, k), hidden_map.get((sample_id, 0), zero_hidden))
                shallow = _shallow_row_for_step(traj, step, k, cfg, steps_by_k, max_total_cost)
                seq_xs.append(np.asarray(shallow, dtype=np.float32, copy=False))
                seq_xh.append(hidden.astype(np.float32, copy=False))
                ks.append(k)
            if not ks:
                continue
            xs = scaler.transform(np.stack(seq_xs)).astype(np.float32)
            xh = np.stack(seq_xh).astype(np.float32, copy=False)
            t = xs.shape[0]
            xs_t = torch.from_numpy(xs).unsqueeze(0).to(device)
            xh_t = torch.from_numpy(xh).unsqueeze(0).to(device)
            lens = torch.tensor([t], dtype=torch.long, device=device)
            logits = model(xs_t, xh_t, lens)
            probs = torch.sigmoid(logits[0, :t]).detach().cpu().numpy().reshape(-1)
            for j, kk in enumerate(ks):
                out[(sample_id, kk)] = float(probs[j])
    return out


def _precompute_probe_probs(
    step_feature_map: Dict[Tuple[str, int], np.ndarray],
    model: nn.Module,
    scaler: StandardScaler,
    cfg: Stage2Config,
) -> Dict[Tuple[str, int], float]:
    return _precompute_probe_probs_from_map(step_feature_map, model, scaler, cfg)


def _simulate_probe_policy_from_probs(
    trajectories: List[Dict[str, Any]],
    threshold: ThresholdSpec,
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

            threshold_k = (
                float(threshold.get(k, 0.5))
                if isinstance(threshold, dict)
                else float(threshold)
            )
            if p_continue < threshold_k:
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


def _simulate_probe_policy_low_margin_conservative(
    trajectories: List[Dict[str, Any]],
    threshold: ThresholdSpec,
    cfg: Stage2Config,
    precomputed_probs: Dict[Tuple[str, int], float],
    oracle_rows: List[Dict[str, Any]],
    *,
    margin_abs_lt: float = 0.1,
    threshold_boost: float = 0.1,
    probe_conf_half_width: float = 0.1,
    use_oracle_margin: bool = True,
    use_probe_low_conf: bool = True,
    max_effective_threshold: float = 0.99,
) -> List[Dict[str, Any]]:
    """
    逐步 Continue 决策：在 Oracle |m_k| 小（模糊）或 Probe p≈0.5（低置信）时，将阈值提高 Δ，使继续更保守。

    oracle_rows 须与 trajectories 同序；每行含 compute_trajectory_oracle 的 step_targets[k]["margin"]。
    """
    if len(oracle_rows) != len(trajectories):
        raise ValueError("oracle_rows 与 trajectories 长度不一致")

    boost = float(threshold_boost)
    mcut = float(margin_abs_lt)
    phw = float(probe_conf_half_width)
    cap_t = float(max_effective_threshold)

    rows: List[Dict[str, Any]] = []
    for traj, orow in zip(trajectories, oracle_rows):
        sample_id = str(traj.get("id", ""))
        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
        step_targets: Dict[int, Any] = orow.get("step_targets") or {}

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

            base_t = (
                float(threshold.get(k, 0.5))
                if isinstance(threshold, dict)
                else float(threshold)
            )
            tinfo = step_targets.get(k)
            margin_v: Optional[float] = None
            if isinstance(tinfo, dict):
                margin_v = float(tinfo.get("margin", 0.0))

            low_m = bool(
                use_oracle_margin
                and margin_v is not None
                and abs(margin_v) < mcut - 1e-15
            )
            low_c = bool(
                use_probe_low_conf
                and abs(float(p_continue) - 0.5) < phw - 1e-15
            )
            eff_t = base_t
            if low_m or low_c:
                eff_t = min(cap_t, base_t + boost)

            if float(p_continue) < eff_t:
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
    threshold: ThresholdSpec,
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

    utility_lambda = (
        float(chosen_lambda)
        if chosen_lambda is not None
        else (float(cfg.threshold_pareto_lambdas[0]) if cfg.threshold_pareto_lambdas else 0.0)
    )

    sweep_rows: List[Dict[str, Any]] = []
    for i, (t, row) in enumerate(per_t):
        ncost = norm_cost(row)
        sweep_rows.append(
            {
                "policy_name": f"global@{t:.2f}",
                "policy_type": "global",
                "threshold": float(t),
                "avg_f1": float(row["avg_f1"]),
                "avg_em": float(row["avg_em"]),
                "avg_steps": float(row["avg_steps"]),
                "avg_cost": float(row["avg_cost"]),
                "normalized_cost": float(ncost),
                "utility": float(float(row["avg_f1"]) - utility_lambda * ncost),
                "utility_lambda": float(utility_lambda),
                "is_feasible_under_gw_cap": bool(i in feasible_idx),
                "steps_gap_to_gw_dev": (
                    float(float(row["avg_steps"]) - gw_dev_steps) if not math.isnan(gw_dev_steps) else None
                ),
            }
        )

    matched_budget_candidates: List[Dict[str, Any]] = []
    if not math.isnan(gw_dev_steps):
        feasible_rows = [sweep_rows[i] for i in feasible_idx]
        feasible_rows_sorted = sorted(
            feasible_rows,
            key=lambda r: (abs(float(r["avg_steps"]) - gw_dev_steps), -float(r["avg_f1"])),
        )
        for r in feasible_rows_sorted[:5]:
            matched_budget_candidates.append(
                {
                    "threshold": float(r["threshold"]),
                    "avg_f1": float(r["avg_f1"]),
                    "avg_steps": float(r["avg_steps"]),
                    "steps_gap_to_gw_dev": float(float(r["avg_steps"]) - gw_dev_steps),
                    "utility": float(r["utility"]),
                }
            )

    diag: Dict[str, Any] = {
        "gw_dev_avg_steps": float(gw_dev_steps) if not math.isnan(gw_dev_steps) else None,
        "step_cap": float(step_cap) if not math.isnan(step_cap) else None,
        "cap_mult": cap_mult,
        "normalized_cost_span": {"min": cmin, "max": cmax},
        "feasible_threshold_count": len(feasible_idx),
        "lambda_grid_trace": lambda_trace,
        "chosen_lambda": chosen_lambda,
        "matched_budget_candidates": matched_budget_candidates,
        "threshold_sweep_rows": sweep_rows,
    }
    best_row = dict(best_row)
    best_row["strategy"] = f"Probe@{best_t:.2f}"
    return float(best_t), best_row, diag


def _pick_max_f1_operating_point(
    sweep_rows: Sequence[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Appendix 补充：同一 checkpoint 下，dev 上按纯 F1 选全局阈值。"""
    global_rows = [r for r in sweep_rows if str(r.get("policy_type", "global")) == "global"]
    if not global_rows:
        return None, None

    best_row = dict(global_rows[0])
    for row in global_rows[1:]:
        row_f1 = float(row.get("avg_f1", 0.0))
        best_f1 = float(best_row.get("avg_f1", 0.0))
        if row_f1 > best_f1 + 1e-12:
            best_row = dict(row)
            continue
        if math.isclose(row_f1, best_f1, rel_tol=1e-9, abs_tol=1e-9):
            row_cost = float(row.get("avg_cost", 0.0))
            best_cost = float(best_row.get("avg_cost", 0.0))
            if row_cost < best_cost - 1e-12:
                best_row = dict(row)

    threshold = best_row.get("threshold")
    if threshold is None:
        return None, None
    best_row["selection_rule"] = "dev_max_f1_global_threshold"
    return float(threshold), best_row


def _step_threshold_policy_to_list(policy: Dict[int, float], max_k: int) -> List[float]:
    return [float(policy.get(k, 0.5)) for k in range(1, max_k + 1)]


def _summarize_policy_candidate(
    policy_name: str,
    policy_type: str,
    threshold_spec: ThresholdSpec,
    row: Dict[str, Any],
    utility_lambda: float,
    cost_span: Dict[str, Any],
    gw_dev_steps: Optional[float],
    step_cap: Optional[float],
) -> Dict[str, Any]:
    cmin = float(cost_span.get("min", 0.0) or 0.0)
    cmax = float(cost_span.get("max", 1.0) or 1.0)
    cspan = cmax - cmin
    if abs(cspan) < 1e-12:
        cspan = 1.0
    norm_cost = float((float(row["avg_cost"]) - cmin) / cspan)
    steps_gap = None
    if gw_dev_steps is not None:
        steps_gap = float(row["avg_steps"]) - float(gw_dev_steps)
    feasible = True
    if step_cap is not None:
        feasible = float(row["avg_steps"]) <= float(step_cap) + 1e-9
    payload: Dict[str, Any] = {
        "policy_name": policy_name,
        "policy_type": policy_type,
        "avg_f1": float(row["avg_f1"]),
        "avg_em": float(row["avg_em"]),
        "avg_steps": float(row["avg_steps"]),
        "avg_cost": float(row["avg_cost"]),
        "normalized_cost": norm_cost,
        "utility": float(row["avg_f1"]) - float(utility_lambda) * norm_cost,
        "utility_lambda": float(utility_lambda),
        "is_feasible_under_gw_cap": bool(feasible),
        "steps_gap_to_gw_dev": steps_gap,
    }
    if policy_type == "global":
        payload["threshold"] = float(threshold_spec)
    else:
        payload["per_step_thresholds"] = _step_threshold_policy_to_list(
            cast(Dict[int, float], threshold_spec), max_k=len(cast(Dict[int, float], threshold_spec))
        )
    return payload


def _is_policy_nondominated(
    candidate_row: Dict[str, Any],
    baseline_rows: Sequence[Dict[str, Any]],
) -> bool:
    c_cost = float(candidate_row["avg_cost"])
    c_f1 = float(candidate_row["avg_f1"])
    for row in baseline_rows:
        b_cost = float(row["avg_cost"])
        b_f1 = float(row["avg_f1"])
        if (
            b_cost <= c_cost + 1e-12
            and b_f1 >= c_f1 - 1e-12
            and (b_cost < c_cost - 1e-12 or b_f1 > c_f1 + 1e-12)
        ):
            return False
    return True


def _pick_step_threshold_refinement(
    dev_trajectories: List[Dict[str, Any]],
    dev_probs: Dict[Tuple[str, int], float],
    cfg: Stage2Config,
    *,
    base_threshold: float,
    base_row: Dict[str, Any],
    base_diag: Dict[str, Any],
) -> Tuple[Optional[Dict[int, float]], Optional[Dict[str, Any]], Dict[str, Any]]:
    if cfg.max_k <= 1:
        return None

    candidates = [round(x, 3) for x in np.linspace(0.01, 0.99, 50)]
    gw_dev_steps = base_diag.get("gw_dev_avg_steps")
    step_cap = base_diag.get("step_cap")
    utility_lambda = float(
        base_diag.get("chosen_lambda")
        if base_diag.get("chosen_lambda") is not None
        else (cfg.threshold_pareto_lambdas[0] if cfg.threshold_pareto_lambdas else 0.0)
    )
    cost_span = cast(Dict[str, Any], base_diag.get("normalized_cost_span", {}))
    global_sweep_rows = cast(List[Dict[str, Any]], base_diag.get("threshold_sweep_rows", []))
    feasible_global_rows = [r for r in global_sweep_rows if bool(r.get("is_feasible_under_gw_cap", False))]
    if not feasible_global_rows:
        feasible_global_rows = global_sweep_rows

    cmin = float(cost_span.get("min", 0.0) or 0.0)
    cmax = float(cost_span.get("max", 1.0) or 1.0)
    cspan = cmax - cmin
    if abs(cspan) < 1e-12:
        cspan = 1.0

    def norm_cost(cost: float) -> float:
        return float((float(cost) - cmin) / cspan)

    def score_row(row: Dict[str, Any]) -> float:
        return float(row["avg_f1"]) - utility_lambda * norm_cost(float(row["avg_cost"]))

    def better_row(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
        score_a = score_row(row_a)
        score_b = score_row(row_b)
        if score_a > score_b + 1e-12:
            return True
        if math.isclose(score_a, score_b, rel_tol=1e-9, abs_tol=1e-9):
            if row_a["avg_f1"] > row_b["avg_f1"] + 1e-12:
                return True
            if math.isclose(row_a["avg_f1"], row_b["avg_f1"], rel_tol=1e-9, abs_tol=1e-9):
                return row_a["avg_cost"] < row_b["avg_cost"] - 1e-12
        return False

    policy: Dict[int, float] = {k: float(base_threshold) for k in range(1, cfg.max_k + 1)}
    current_rows = _simulate_probe_policy_from_probs(dev_trajectories, policy, cfg, dev_probs)
    current_row = _summarize_results(current_rows, "Probe-step-threshold")
    if step_cap is not None and float(current_row["avg_steps"]) > float(step_cap) + 1e-9:
        current_row = dict(base_row)

    improved = False
    trace: List[Dict[str, Any]] = []
    for pass_idx in range(max(1, int(cfg.step_threshold_refine_passes))):
        changed = False
        for k in range(1, cfg.max_k):
            best_t = float(policy[k])
            best_row = dict(current_row)
            for cand_t in candidates:
                if math.isclose(float(cand_t), float(policy[k]), rel_tol=1e-9, abs_tol=1e-9):
                    continue
                cand_policy = dict(policy)
                cand_policy[k] = float(cand_t)
                cand_rows = _simulate_probe_policy_from_probs(dev_trajectories, cand_policy, cfg, dev_probs)
                cand_row = _summarize_results(cand_rows, "Probe-step-threshold")
                if step_cap is not None and float(cand_row["avg_steps"]) > float(step_cap) + 1e-9:
                    continue
                if better_row(cand_row, best_row):
                    best_t = float(cand_t)
                    best_row = cand_row
            if not math.isclose(float(policy[k]), best_t, rel_tol=1e-9, abs_tol=1e-9):
                policy[k] = best_t
                current_row = best_row
                changed = True
                improved = True
        trace.append(
            {
                "pass_index": int(pass_idx + 1),
                "avg_f1": float(current_row["avg_f1"]),
                "avg_steps": float(current_row["avg_steps"]),
                "avg_cost": float(current_row["avg_cost"]),
                "utility": score_row(current_row),
                "per_step_thresholds": _step_threshold_policy_to_list(policy, cfg.max_k),
            }
        )
        if not changed:
            break

    if step_cap is not None and float(current_row["avg_steps"]) > float(step_cap) + 1e-9:
        return None

    base_score = score_row(base_row)
    cand_score = score_row(current_row)
    improves_utility = cand_score > base_score + 1e-12
    improves_matched_budget_f1 = (
        float(current_row["avg_f1"]) > float(base_row["avg_f1"]) + 1e-12
        and float(current_row["avg_steps"]) <= float(base_row["avg_steps"]) + 1e-9
    )
    expands_frontier = _is_policy_nondominated(current_row, feasible_global_rows) and (
        float(current_row["avg_f1"]) > float(base_row["avg_f1"]) + 1e-12
        or float(current_row["avg_cost"]) < float(base_row["avg_cost"]) - 1e-12
    )

    should_adopt = bool(improves_utility or improves_matched_budget_f1 or expands_frontier)
    diag = {
        "enabled": True,
        "searched": True,
        "adopted": should_adopt,
        "started_from_threshold": float(base_threshold),
        "base_policy_summary": _summarize_policy_candidate(
            "global_baseline",
            "global",
            float(base_threshold),
            base_row,
            utility_lambda,
            cost_span,
            float(gw_dev_steps) if gw_dev_steps is not None else None,
            float(step_cap) if step_cap is not None else None,
        ),
        "candidate_policy_summary": _summarize_policy_candidate(
            "step_refined_candidate",
            "per_step",
            policy,
            current_row,
            utility_lambda,
            cost_span,
            float(gw_dev_steps) if gw_dev_steps is not None else None,
            float(step_cap) if step_cap is not None else None,
        ),
        "adoption_reasons": {
            "improves_utility": bool(improves_utility),
            "improves_matched_budget_f1": bool(improves_matched_budget_f1),
            "expands_frontier": bool(expands_frontier),
        },
        "coordinate_descent_trace": trace,
    }
    if not should_adopt and not improved:
        diag["searched"] = False
    if should_adopt:
        row = dict(current_row)
        row["strategy"] = "Probe(step-threshold)"
        return policy, row, diag
    return None, None, diag


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
    # rethreshold-only：必须先与 checkpoint 对齐浅层口径与序列开关，再构建 _build_xyw / feature_map
    rethreshold_ck_meta: Optional[Dict[str, Any]] = None
    if cfg.rethreshold_only:
        load_tag_pre = _checkpoint_tag_for_load(cfg)
        ckpt_pre = cfg.artifacts_probe_dir / dataset / f"probe_mlp{load_tag_pre}.pt"
        if ckpt_pre.is_file():
            try:
                rethreshold_ck_meta = torch.load(ckpt_pre, map_location="cpu", weights_only=False)
            except TypeError:
                rethreshold_ck_meta = torch.load(ckpt_pre, map_location="cpu")
            ck_mode = str((rethreshold_ck_meta or {}).get("probe_feature_mode", "full"))
            if ck_mode in ("full", "lite"):
                cfg = replace(cfg, probe_feature_mode=cast(Literal["full", "lite"], ck_mode))
            if bool((rethreshold_ck_meta or {}).get("sequence_gru", False)):
                cfg = replace(cfg, sequence_gru=True, seq_history_features=False)
            elif bool((rethreshold_ck_meta or {}).get("seq_history_features", False)):
                cfg = replace(cfg, seq_history_features=True, sequence_gru=False)
            LOGGER.info(
                "%s rethreshold-only：预加载 checkpoint 对齐 probe_feature_mode=%s seq_hist=%s gru=%s",
                dataset,
                cfg.probe_feature_mode,
                bool(cfg.seq_history_features),
                bool(cfg.sequence_gru),
            )
        else:
            LOGGER.warning(
                "%s rethreshold-only：未找到 %s，浅层/序列配置沿用 CLI。",
                dataset,
                ckpt_pre,
            )

    train_traj = _load_trajectories(cfg, dataset, "train")
    dev_traj = _load_trajectories(cfg, dataset, "dev")
    test_traj = _load_trajectories(cfg, dataset, "test")

    if cfg.shallow_only:
        LOGGER.info(
            "Shallow-Only 模式：不使用 hidden states，浅层维度=%d（probe_feature_mode=%s）。",
            effective_shallow_dim(cfg),
            cfg.probe_feature_mode,
        )
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

    tm = float(cfg.train_margin_min_abs)
    stat_cfg = (
        replace(cfg, seq_history_features=False, sequence_gru=False)
        if cfg.sequence_gru
        else cfg
    )
    x_train_s, x_train_h, y_train, w_train, train_stats = _build_xyw(
        train_traj,
        train_oracle_map,
        train_hidden,
        hidden_dim,
        stat_cfg,
        margin_min_abs=tm,
    )
    x_dev_s, x_dev_h, y_dev, w_dev, dev_stats = _build_xyw(
        dev_traj, dev_oracle_map, dev_hidden, hidden_dim, stat_cfg
    )
    if tm > 0.0:
        n0 = int(train_stats.get("skipped_low_margin", 0))
        n1 = int(train_stats.get("trainable_examples", 0))
        LOGGER.info(
            "%s 硬 margin 过滤：train_margin_min_abs=%.4f，丢弃步数=%d，保留可训练步数=%d",
            dataset,
            tm,
            n0,
            n1,
        )
    if cfg.probe_target == "binary":
        _log_stop_continue_balance(f"{dataset} train", y_train)
        _log_stop_continue_balance(f"{dataset} dev", y_dev)
    else:
        LOGGER.info(
            "%s F1 回归标签分布：train mean=%.4f std=%.4f | dev mean=%.4f std=%.4f",
            dataset,
            float(np.mean(y_train)) if y_train.size else 0.0,
            float(np.std(y_train)) if y_train.size else 0.0,
            float(np.mean(y_dev)) if y_dev.size else 0.0,
            float(np.std(y_dev)) if y_dev.size else 0.0,
        )
    if (
        x_dev_s.shape[0] > 0
        and cfg.probe_feature_mode == "full"
        and x_dev_s.shape[1] == BASE_SHALLOW_FEATURE_DIM
    ):
        ext_std = x_dev_s[:, 9:].std(axis=0)
        LOGGER.info(
            "%s dev 浅层扩展维 std（Delta×7 + cum_ratio + D4×3 + P2 稳定性特征，StandardScaler 前）: %s",
            dataset,
            np.array2string(ext_std, precision=4, suppress_small=True),
        )

    ckpt_seq_hist = False
    ckpt_seq_gru = False
    if cfg.rethreshold_only:
        load_tag = _checkpoint_tag_for_load(cfg)
        ckpt_path = cfg.artifacts_probe_dir / dataset / f"probe_mlp{load_tag}.pt"
        if rethreshold_ck_meta is not None:
            ck_meta = rethreshold_ck_meta
        else:
            try:
                ck_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ck_meta = torch.load(ckpt_path, map_location="cpu")
        ckpt_seq_hist = bool(ck_meta.get("seq_history_features", False))
        ckpt_seq_gru = bool(ck_meta.get("sequence_gru", False))
        model, scaler, probe_arch = _load_probe_from_checkpoint(ckpt_path, cfg)
        meta_load_path = cfg.artifacts_probe_dir / dataset / f"stage2_train_meta{load_tag}.json"
        if meta_load_path.is_file():
            with meta_load_path.open("r", encoding="utf-8") as f:
                prior_meta = json.load(f)
            train_info = prior_meta.get("train_info") or {}
            train_info = dict(train_info)
            train_info["rethreshold_only"] = True
            train_info["loaded_checkpoint"] = str(ckpt_path)
        else:
            LOGGER.warning("未找到 %s，train_info 仅含占位字段。", meta_load_path)
            train_info = {
                "rethreshold_only": True,
                "loaded_checkpoint": str(ckpt_path),
                "note": "missing stage2_train_meta next to checkpoint",
            }
        LOGGER.info(
            "%s rethreshold-only：已加载 %s（arch=%s），将用 cap_mult=%.3f 重选阈值。",
            dataset,
            ckpt_path,
            probe_arch,
            float(cfg.threshold_gw_steps_cap_mult),
        )
    else:
        if cfg.sequence_gru:
            if hidden_dim <= 0:
                raise ValueError("sequence_gru 需要 Stage1 hidden states，请勿与 --shallow-only 同用。")
            train_entries = _build_gru_trajectory_entries(
                train_traj, train_oracle_map, train_hidden, hidden_dim, cfg
            )
            dev_entries = _build_gru_trajectory_entries(
                dev_traj, dev_oracle_map, dev_hidden, hidden_dim, cfg
            )
            model, train_info, scaler, probe_arch = _train_probe_gru(
                train_entries, dev_entries, cfg, hidden_dim
            )
        else:
            if stat_cfg is not cfg:
                x_train_s, x_train_h, y_train, w_train, train_stats = _build_xyw(
                    train_traj,
                    train_oracle_map,
                    train_hidden,
                    hidden_dim,
                    cfg,
                    margin_min_abs=tm,
                )
                x_dev_s, x_dev_h, y_dev, w_dev, dev_stats = _build_xyw(
                    dev_traj, dev_oracle_map, dev_hidden, hidden_dim, cfg
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
        train_info["train_margin_min_abs"] = float(tm)
        train_info["train_skipped_low_margin"] = int(train_stats.get("skipped_low_margin", 0))
        train_info["trainable_examples_after_margin_filter"] = int(
            train_stats.get("trainable_examples", 0)
        )

    use_seq_hist = ckpt_seq_hist if cfg.rethreshold_only else bool(cfg.seq_history_features)
    use_seq_gru = ckpt_seq_gru if cfg.rethreshold_only else bool(cfg.sequence_gru)

    test_step_feat_map = _build_step_feature_map(test_traj, test_hidden, hidden_dim, cfg)
    if use_seq_gru:
        assert isinstance(model, ProbeSequenceGRU)
        dev_probs_nn = _precompute_probe_probs_gru(
            dev_traj, dev_hidden, hidden_dim, cfg, model, scaler
        )
        test_probs = _precompute_probe_probs_gru(
            test_traj, test_hidden, hidden_dim, cfg, model, scaler
        )
    elif use_seq_hist:
        dev_probs_nn = _precompute_probe_probs_autoreg_mlp(
            dev_traj, dev_hidden, hidden_dim, cfg, model, scaler
        )
        test_probs = _precompute_probe_probs_autoreg_mlp(
            test_traj, test_hidden, hidden_dim, cfg, model, scaler
        )
    else:
        dev_step_feat_map = _build_step_feature_map(dev_traj, dev_hidden, hidden_dim, cfg)
        dev_probs_nn = _precompute_probe_probs_from_map(dev_step_feat_map, model, scaler, cfg)
        test_probs = _precompute_probe_probs_from_map(test_step_feat_map, model, scaler, cfg)
    threshold, best_dev_row, threshold_diag = _pick_best_threshold(
        train_traj, dev_traj, dev_probs_nn, cfg
    )
    deployed_threshold: ThresholdSpec = float(threshold)
    deployed_policy_type = "global"
    step_refine_diag: Dict[str, Any] = {
        "enabled": bool(cfg.enable_step_threshold_refine),
        "searched": False,
        "adopted": False,
    }
    if cfg.enable_step_threshold_refine:
        step_policy, step_row, step_diag = _pick_step_threshold_refinement(
            dev_traj,
            dev_probs_nn,
            cfg,
            base_threshold=threshold,
            base_row=best_dev_row,
            base_diag=threshold_diag,
        )
        step_refine_diag = step_diag
        if step_policy is not None and step_row is not None:
            deployed_threshold = step_policy
            deployed_policy_type = "per_step"
            best_dev_row = step_row
    LOGGER.info(
        "%s Phase C 阈值：GW_dev_avg_steps=%s cap_mult=%.3f chosen_λ=%s policy=%s threshold=%s dev_f1=%.4f dev_steps=%.3f",
        dataset,
        threshold_diag.get("gw_dev_avg_steps"),
        float(cfg.threshold_gw_steps_cap_mult),
        threshold_diag.get("chosen_lambda"),
        deployed_policy_type,
        (
            f"{threshold:.3f}"
            if deployed_policy_type == "global"
            else json.dumps(_step_threshold_policy_to_list(cast(Dict[int, float], deployed_threshold), cfg.max_k))
        ),
        float(best_dev_row["avg_f1"]),
        float(best_dev_row["avg_steps"]),
    )

    appendix_max_f1_threshold, appendix_max_f1_dev_row = _pick_max_f1_operating_point(
        cast(List[Dict[str, Any]], threshold_diag.get("threshold_sweep_rows", []))
    )
    appendix_max_f1_test_row: Optional[Dict[str, Any]] = None
    if appendix_max_f1_threshold is not None and appendix_max_f1_dev_row is not None:
        appendix_test = _simulate_probe_policy(
            test_traj,
            test_step_feat_map,
            model,
            scaler,
            float(appendix_max_f1_threshold),
            cfg,
            precomputed_probs=test_probs,
        )
        appendix_max_f1_test_row = _summarize_results(appendix_test, "Probe-Appendix-MaxF1")
        LOGGER.info(
            "%s Appendix max-F1：dev threshold=%.3f dev_f1=%.4f dev_steps=%.3f | test_f1=%.4f test_steps=%.3f",
            dataset,
            float(appendix_max_f1_threshold),
            float(appendix_max_f1_dev_row["avg_f1"]),
            float(appendix_max_f1_dev_row["avg_steps"]),
            float(appendix_max_f1_test_row["avg_f1"]),
            float(appendix_max_f1_test_row["avg_steps"]),
        )

    rows: List[Dict[str, Any]] = []

    probe_test = _simulate_probe_policy(
        test_traj,
        test_step_feat_map,
        model,
        scaler,
        deployed_threshold,
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

    sweep_rows_for_export = list(threshold_diag.get("threshold_sweep_rows", []))
    step_refine_candidate = cast(Dict[str, Any], step_refine_diag.get("candidate_policy_summary", {}))
    if step_refine_candidate:
        sweep_rows_for_export.append(step_refine_candidate)
    sweep_df = pd.DataFrame(sweep_rows_for_export)
    sweep_path = cfg.results_dir / f"stage2_threshold_sweep_{dataset}{tag}.csv"
    if not sweep_df.empty:
        desired_cols = [
            "policy_name",
            "policy_type",
            "threshold",
            "per_step_thresholds",
            "avg_f1",
            "avg_em",
            "avg_steps",
            "avg_cost",
            "normalized_cost",
            "utility",
            "utility_lambda",
            "is_feasible_under_gw_cap",
            "steps_gap_to_gw_dev",
        ]
        existing_cols = [c for c in desired_cols if c in sweep_df.columns]
        remaining_cols = [c for c in sweep_df.columns if c not in existing_cols]
        sweep_df = sweep_df[existing_cols + remaining_cols]
    sweep_df.to_csv(sweep_path, index=False)

    pareto_path = cfg.results_dir / f"stage2_probe_pareto_{dataset}{tag}.png"
    _plot_dataset_pareto(
        table_df, dataset, pareto_path, title_extra="Shallow-Only" if cfg.shallow_only else ""
    )

    best_fixed = max((r["avg_f1"] for r in fixed_rows), default=0.0)
    probe_gain = float(probe_row["avg_f1"] - best_fixed)
    oracle_gap = float(oracle_summary["avg_f1"] - probe_row["avg_f1"])

    model_path = cfg.artifacts_probe_dir / dataset / f"probe_mlp{tag}.pt"
    sdim_ckpt = int(scaler.n_features_in_)
    compress_ckpt = probe_arch in ("mlp_v2", "mlp_v2_gru")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "probe_arch": probe_arch,
            "input_dim": int(sdim_ckpt + hidden_dim),
            # 与历史 checkpoint 兼容：浅层 ProbeMLP 的塔宽；mlp_v2 时见 fuse_dim / compress_dim。
            "hidden_dim": int(cfg.hidden_dim),
            "mlp_hidden_dim": int(cfg.hidden_dim),
            "compress_dim": int(cfg.compress_dim) if compress_ckpt else None,
            "fuse_dim": int(cfg.fuse_dim) if compress_ckpt else None,
            "gru_hidden_dim": int(cfg.gru_hidden_dim) if probe_arch == "mlp_v2_gru" else None,
            "dropout": float(cfg.dropout),
            "threshold": float(threshold),
            "threshold_policy_type": deployed_policy_type,
            "per_step_thresholds": (
                _step_threshold_policy_to_list(cast(Dict[int, float], deployed_threshold), cfg.max_k)
                if deployed_policy_type == "per_step"
                else None
            ),
            "hidden_state_key": cfg.hidden_state_key,
            "hidden_branch_residual": bool(cfg.hidden_branch_residual),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "shallow_feature_dim": sdim_ckpt,
            "seq_history_features": bool(cfg.seq_history_features),
            "sequence_gru": bool(cfg.sequence_gru),
            "shallow_only": bool(cfg.shallow_only),
            "stage1_hidden_dim": int(hidden_dim),
            "probe_target": str(cfg.probe_target),
            "probe_feature_mode": str(cfg.probe_feature_mode),
        },
        model_path,
    )

    meta_path = cfg.artifacts_probe_dir / dataset / f"stage2_train_meta{tag}.json"
    _write_json(
        meta_path,
        {
            "dataset": dataset,
            "seed": cfg.seed,
            "probe_feature_mode": str(cfg.probe_feature_mode),
            "hidden_dim_from_stage1": hidden_dim,
            "train_stats": train_stats,
            "dev_stats": dev_stats,
            "train_info": train_info,
            "best_dev_threshold": threshold,
            "best_dev_threshold_policy_type": deployed_policy_type,
            "best_dev_per_step_thresholds": (
                _step_threshold_policy_to_list(cast(Dict[int, float], deployed_threshold), cfg.max_k)
                if deployed_policy_type == "per_step"
                else None
            ),
            "best_dev_row": best_dev_row,
            "threshold_selection_phase_c": threshold_diag,
            "step_threshold_refine": step_refine_diag,
            "appendix_max_f1_operating_point": {
                "selection_rule": "dev_max_f1_global_threshold",
                "dev_threshold": appendix_max_f1_threshold,
                "dev_summary": appendix_max_f1_dev_row,
                "test_summary": appendix_max_f1_test_row,
            },
            "probe_summary": probe_row,
            "oracle_summary": oracle_summary,
            "best_fixed_f1": best_fixed,
            "probe_gain_over_best_fixed": probe_gain,
            "oracle_gap_to_probe": oracle_gap,
            "shallow_only": bool(cfg.shallow_only),
            "hidden_state_key": str(cfg.hidden_state_key),
            "hidden_branch_residual": bool(cfg.hidden_branch_residual),
            "seq_history_features": bool(cfg.seq_history_features),
            "sequence_gru": bool(cfg.sequence_gru),
            "probe_target": str(cfg.probe_target),
            "table_path": str(table_path),
            "threshold_sweep_path": str(sweep_path),
            "pareto_path": str(pareto_path),
            "model_path": str(model_path),
        },
    )

    return {
        "dataset": dataset,
        "table_path": str(table_path),
        "threshold_sweep_path": str(sweep_path),
        "pareto_path": str(pareto_path),
        "model_path": str(model_path),
        "threshold": float(threshold),
        "threshold_policy_type": deployed_policy_type,
        "per_step_thresholds": (
            _step_threshold_policy_to_list(cast(Dict[int, float], deployed_threshold), cfg.max_k)
            if deployed_policy_type == "per_step"
            else None
        ),
        "probe_summary": probe_row,
        "oracle_summary": oracle_summary,
        "best_fixed_f1": best_fixed,
        "probe_gain_over_best_fixed": probe_gain,
        "oracle_gap_to_probe": oracle_gap,
        "train_info": train_info,
        "threshold_diag": threshold_diag,
        "step_threshold_refine": step_refine_diag,
        "appendix_max_f1_operating_point": {
            "selection_rule": "dev_max_f1_global_threshold",
            "dev_threshold": appendix_max_f1_threshold,
            "dev_summary": appendix_max_f1_dev_row,
            "test_summary": appendix_max_f1_test_row,
        },
    }


def _artifact_href_for_doc(out_md: Path, artifact_str: str, repo_root: Path) -> str:
    """Markdown 写在 docs/ 下时，把仓库内路径转成相对该 .md 文件的路径。"""
    ap = Path(artifact_str)
    full = ap.resolve() if ap.is_absolute() else (repo_root / ap).resolve()
    return os.path.relpath(str(full), start=str(out_md.parent.resolve()))


def build_stage2_report(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    title_suffix: str = "",
    repo_root: Optional[Path] = None,
) -> Path:
    root = (repo_root or out_path.parent.parent).resolve()
    lines: List[str] = []
    lines.append("# Stage2 Report" + (f" ({title_suffix})" if title_suffix else ""))
    lines.append("")
    lines.append("## Probe Performance")
    lines.append("")
    for ds, info in results.items():
        p = info["probe_summary"]
        o = info["oracle_summary"]
        policy_type = info.get("threshold_policy_type", "global")
        policy_str = f"threshold={info['threshold']:.2f}"
        if policy_type == "per_step":
            policy_str = f"per_step_thresholds={info.get('per_step_thresholds')}"
        lines.append(
            f"- `{ds}`: policy={policy_type}, {policy_str}, "
            f"probe_f1={p['avg_f1']:.4f}, probe_steps={p['avg_steps']:.3f}, "
            f"oracle_f1={o['avg_f1']:.4f}, oracle_gap={info['oracle_gap_to_probe']:.4f}"
        )
    lines.append("")
    lines.append("## Phase C Diagnostics")
    lines.append("")
    for ds, info in results.items():
        diag = cast(Dict[str, Any], info.get("threshold_diag", {}))
        step_refine = cast(Dict[str, Any], info.get("step_threshold_refine", {}))
        matched = cast(List[Dict[str, Any]], diag.get("matched_budget_candidates", []))
        lines.append(
            f"- `{ds}`: chosen_lambda={diag.get('chosen_lambda')}, "
            f"gw_dev_avg_steps={diag.get('gw_dev_avg_steps')}, "
            f"feasible_threshold_count={diag.get('feasible_threshold_count')}, "
            f"matched_budget_candidates={len(matched)}, "
            f"step_refine_adopted={step_refine.get('adopted', False)}"
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
    lines.append("## Appendix Max-F1 Operating Point")
    lines.append("")
    for ds, info in results.items():
        appendix = cast(Dict[str, Any], info.get("appendix_max_f1_operating_point", {}))
        dev_row = cast(Optional[Dict[str, Any]], appendix.get("dev_summary"))
        test_row = cast(Optional[Dict[str, Any]], appendix.get("test_summary"))
        threshold = appendix.get("dev_threshold")
        if dev_row is None or test_row is None or threshold is None:
            lines.append(f"- `{ds}`: unavailable")
            continue
        lines.append(
            f"- `{ds}`: dev_max_f1_threshold={float(threshold):.2f}, "
            f"dev_f1={float(dev_row['avg_f1']):.4f}, dev_steps={float(dev_row['avg_steps']):.3f}, "
            f"test_f1={float(test_row['avg_f1']):.4f}, test_steps={float(test_row['avg_steps']):.3f}"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for ds, info in results.items():
        lines.append(
            f"- `{ds}`: table=`{_artifact_href_for_doc(out_path, info['table_path'], root)}`, "
            f"sweep=`{_artifact_href_for_doc(out_path, info['threshold_sweep_path'], root)}`, "
            f"pareto=`{_artifact_href_for_doc(out_path, info['pareto_path'], root)}`, "
            f"model=`{_artifact_href_for_doc(out_path, info['model_path'], root)}`"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def build_stage2_appendix_max_f1_summary(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> Path:
    rows: List[Dict[str, Any]] = []
    for ds, info in results.items():
        appendix = cast(Dict[str, Any], info.get("appendix_max_f1_operating_point", {}))
        dev_row = cast(Optional[Dict[str, Any]], appendix.get("dev_summary"))
        test_row = cast(Optional[Dict[str, Any]], appendix.get("test_summary"))
        threshold = appendix.get("dev_threshold")
        main_probe = cast(Dict[str, Any], info.get("probe_summary", {}))
        if dev_row is None or test_row is None or threshold is None:
            rows.append({"dataset": ds, "available": False})
            continue
        rows.append(
            {
                "dataset": ds,
                "available": True,
                "selection_rule": appendix.get("selection_rule", "dev_max_f1_global_threshold"),
                "dev_threshold": float(threshold),
                "dev_avg_f1": float(dev_row["avg_f1"]),
                "dev_avg_steps": float(dev_row["avg_steps"]),
                "test_avg_f1": float(test_row["avg_f1"]),
                "test_avg_steps": float(test_row["avg_steps"]),
                "main_probe_test_avg_f1": float(main_probe.get("avg_f1", 0.0)),
                "main_probe_test_avg_steps": float(main_probe.get("avg_steps", 0.0)),
                "delta_test_f1_vs_main": float(test_row["avg_f1"]) - float(main_probe.get("avg_f1", 0.0)),
                "delta_test_steps_vs_main": float(test_row["avg_steps"]) - float(main_probe.get("avg_steps", 0.0)),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
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
        choices=("last_token", "mean_pool", "last_mean_blend"),
        default="last_token",
        help="从 Stage1 的 .npz 取 hidden：last_token / mean_pool / last_mean_blend（0.5*last+0.5*mean）。",
    )
    parser.add_argument(
        "--hidden-branch-residual",
        action="store_true",
        help="ProbeMLP_v2：hidden 分支在 GELU 后加 compress_dim 残差 Linear（z+Linear(z)）。",
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
        "--train-margin-min-abs",
        type=float,
        default=0.0,
        help="训练集硬过滤：仅保留 |Oracle margin|≥该阈值的步；0 关闭。dev/test 不过滤。",
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
        "--probe-target",
        type=str,
        choices=("binary", "f1"),
        default=None,
        help="Probe 目标：binary / f1。省略时：无 --per-dataset-optimal 则默认为 binary；"
        "有 --per-dataset-optimal 则按 PER_DATASET_OPTIMAL 表 per-dataset（当前三数据集均为 binary）。"
        "显式传入时三数据集统一为该目标（用于对照消融，如 --probe-target f1 复现全 f1 基线）。",
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
        "--disable-step-threshold-refine",
        action="store_true",
        help="关闭 P1 的逐步阈值 refinement；默认开启，但只有在 dev 上确实改进 cost-aware 目标时才接管。",
    )
    parser.add_argument(
        "--step-threshold-refine-passes",
        type=int,
        default=3,
        help="P1：逐步阈值坐标搜索的最大轮数。",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="写入 probe 表 / checkpoint / meta 时的文件名后缀（避免并行或多配置覆盖）。",
    )
    parser.add_argument(
        "--load-artifact-suffix",
        type=str,
        default="",
        help="仅在 --rethreshold-only 下生效：checkpoint 加载后缀。优先级："
        "--load-artifact-suffix > --artifact-suffix > 空后缀（历史默认 probe_mlp*.pt）。",
    )
    parser.add_argument(
        "--rethreshold-only",
        action="store_true",
        help="不训练：从 artifacts/probe/<ds>/probe_mlp{tag}.pt 加载权重（tag 由"
        " --load-artifact-suffix / --artifact-suffix 决定；shallow 自动附加 _shallow），"
        "仅用当前 --gw-steps-cap-mult 等在 dev 上重选阈值并评估 test。输出文件名仍受 --artifact-suffix 影响。",
    )
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument(
        "--shallow-only",
        action="store_true",
        help="仅使用浅层特征（含 Delta）训练 Probe，不使用 Stage1 的 hidden states（w/o Deep Features 对照）。",
    )
    parser.add_argument(
        "--per-dataset-optimal",
        action="store_true",
        help="按消融结论为每个数据集单独设置 compress_dim、train_margin_min_abs 与 probe_target（覆盖 "
        "--compress-dim / --train-margin-min-abs / --probe-target）：hotpotqa→256/0+binary；"
        "musique→256/0+binary；2wiki→64/0.02+binary（2026-04-11 修正：三数据集均为 binary）。"
        "详见 docs/plan-04-10.md §二.2。",
    )
    parser.add_argument(
        "--seq-history-features",
        action="store_true",
        help="plan §二.3A：浅层拼接 oracle teacher 历史标量三维；dev/test 自回归用 pred 填维（需 MLP 探针，勿与 --sequence-gru 同开）。",
    )
    parser.add_argument(
        "--sequence-gru",
        action="store_true",
        help="plan §二.3B：轨迹级 GRU 探针（勿与 --seq-history-features 同开；需要 hidden states）。",
    )
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=128,
        help="ProbeSequenceGRU 隐状态维度。",
    )
    parser.add_argument(
        "--probe-feature-mode",
        type=str,
        choices=("full", "lite"),
        default="full",
        help="浅层特征：full=31 维（历史默认）；lite=15 维部署口径（无 NLI/熵/自洽/ROUGE 边际新颖度等）。",
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

    if bool(args.seq_history_features) and bool(args.sequence_gru):
        raise ValueError("不能同时指定 --seq-history-features 与 --sequence-gru。")
    if bool(args.sequence_gru) and bool(args.shallow_only):
        raise ValueError("--sequence-gru 需要 Stage1 hidden，请勿与 --shallow-only 同用。")

    cfg = Stage2Config(
        seed=args.seed,
        max_k=args.max_k,
        cost_per_step=args.cost_per_step,
        oracle_cost_metric=args.oracle_cost_metric,
        root_dir=Path(args.root_dir),
        hidden_state_key=args.hidden_state_key,
        probe_feature_mode=cast(Literal["full", "lite"], str(args.probe_feature_mode)),
        hidden_branch_residual=bool(args.hidden_branch_residual),
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
        train_margin_min_abs=float(args.train_margin_min_abs),
        focal_gamma=float(args.focal_gamma),
        focal_alpha=float(args.focal_alpha) if args.focal_alpha is not None else None,
        label_smoothing=float(args.label_smoothing),
        shallow_only=bool(args.shallow_only),
        seq_history_features=bool(args.seq_history_features),
        sequence_gru=bool(args.sequence_gru),
        gru_hidden_dim=int(args.gru_hidden_dim),
        probe_target=str(args.probe_target or "binary"),
        threshold_gw_steps_cap_mult=float(args.gw_steps_cap_mult),
        threshold_pareto_lambdas=pareto_lambdas,
        enable_step_threshold_refine=not bool(args.disable_step_threshold_refine),
        step_threshold_refine_passes=int(args.step_threshold_refine_passes),
        artifact_suffix=str(args.artifact_suffix or ""),
        load_artifact_suffix=str(args.load_artifact_suffix or ""),
        rethreshold_only=bool(args.rethreshold_only),
    )

    _set_seed(cfg.seed)
    _ensure_dirs(cfg, datasets)

    all_results: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        LOGGER.info("===== Stage2 dataset: %s =====", ds)
        run_cfg = cfg
        if bool(args.per_dataset_optimal):
            if ds not in PER_DATASET_OPTIMAL:
                raise ValueError(
                    f"--per-dataset-optimal 暂无 {ds!r} 的条目，当前仅支持 "
                    f"{sorted(PER_DATASET_OPTIMAL)}"
                )
            cd, tm, ptab, hbr = PER_DATASET_OPTIMAL[ds]
            ptab_s = str(ptab)
            if ptab_s not in {"binary", "f1"}:
                raise ValueError(f"PER_DATASET_OPTIMAL[{ds!r}] probe_target 非法：{ptab_s!r}")
            if args.probe_target is not None:
                # 显式 --probe-target：全数据集统一（用于复现「全 binary」类消融）
                run_cfg = replace(
                    cfg,
                    compress_dim=int(cd),
                    train_margin_min_abs=float(tm),
                    hidden_branch_residual=bool(hbr),
                )
                eff_pt = str(cfg.probe_target)
            else:
                eff_pt = ptab_s
                run_cfg = replace(
                    cfg,
                    compress_dim=int(cd),
                    train_margin_min_abs=float(tm),
                    probe_target=cast(Literal["binary", "f1"], eff_pt),
                    hidden_branch_residual=bool(hbr),
                )
            LOGGER.info(
                "per-dataset-optimal：%s → compress_dim=%d, margin=%g, probe_target=%s, hidden_branch_residual=%s",
                ds,
                cd,
                tm,
                eff_pt,
                bool(hbr),
            )
        all_results[ds] = run_dataset_stage2(run_cfg, ds)

    tag = _stage2_artifact_tag(cfg)
    report_name = f"stage2_report{tag}.md" if tag else "stage2_report.md"
    report_title = "Shallow-Only Probe" if cfg.shallow_only else ""
    if cfg.artifact_suffix.strip() and not cfg.shallow_only:
        report_title = (report_title + f" ({cfg.artifact_suffix.strip()})").strip()
    report_path = build_stage2_report(
        all_results,
        cfg.docs_dir / report_name,
        title_suffix=report_title,
        repo_root=cfg.root_dir.resolve(),
    )
    appendix_summary_name = (
        f"stage2_appendix_maxf1_summary{tag}.csv" if tag else "stage2_appendix_maxf1_summary.csv"
    )
    appendix_summary_path = build_stage2_appendix_max_f1_summary(
        all_results,
        cfg.results_dir / appendix_summary_name,
    )
    LOGGER.info(
        "Stage2 complete. Report: %s | Appendix max-F1 summary: %s",
        report_path,
        appendix_summary_path,
    )
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
