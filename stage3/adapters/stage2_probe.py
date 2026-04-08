"""
Stage2 探针加载与逐步「继续概率」推理 —— **唯一**建议大量 import ``stage2.run_stage2`` 内部实现的位置。

若 Stage2 修改了：
  - checkpoint 字段、Probe 结构、浅层特征拼接方式、或 StandardScaler 落盘格式，
请优先在本文件内对齐；Stage3 其余模块只使用返回的 ``continue_prob`` 与 numpy 特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from stage2.run_stage2 import (
    ProbeMLP,
    ProbeMLP_v2,
    Stage2Config,
    _build_step_feature_map,
    _infer_hidden_dim,
    _load_hidden_map,
    _load_trajectories,
    _precompute_probe_probs,
    _simulate_probe_policy_from_probs,
    _stage2_artifact_tag,
)


@dataclass(frozen=True)
class Stage2ProbeBundle:
    model: torch.nn.Module
    scaler: StandardScaler
    probe_threshold: float
    shallow_feature_dim: int
    dual_input: bool
    shallow_only: bool
    hidden_state_key: str


def default_probe_checkpoint_path(cfg: Stage2Config, dataset: str) -> Path:
    tag = _stage2_artifact_tag(cfg)
    return cfg.artifacts_probe_dir / dataset / f"probe_mlp{tag}.pt"


def load_stage2_probe_bundle(checkpoint: Path, device: torch.device) -> Stage2ProbeBundle:
    try:
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint), map_location=device)
    shallow_only = bool(ckpt.get("shallow_only", False))
    sdim = int(ckpt["shallow_feature_dim"])
    dropout = float(ckpt["dropout"])
    arch = str(ckpt.get("probe_arch", "mlp_v2"))

    if shallow_only:
        model = ProbeMLP(sdim, int(ckpt["mlp_hidden_dim"]), dropout).to(device)
    elif arch == "mlp_v2":
        model = ProbeMLP_v2(
            int(ckpt["stage1_hidden_dim"]),
            sdim,
            int(ckpt["compress_dim"]),
            int(ckpt["fuse_dim"]),
            dropout,
        ).to(device)
    else:
        raise ValueError(f"不支持的 Stage2 checkpoint probe_arch={arch!r}（仅支持 mlp / mlp_v2）。")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(scaler.mean_.shape[0])
    scaler.n_samples_seen_ = 1

    dual = isinstance(model, ProbeMLP_v2)
    return Stage2ProbeBundle(
        model=model,
        scaler=scaler,
        probe_threshold=float(ckpt["threshold"]),
        shallow_feature_dim=sdim,
        dual_input=dual,
        shallow_only=shallow_only,
        hidden_state_key=str(ckpt.get("hidden_state_key", "last_token")),
    )


def load_trajectories_and_step_features(
    cfg: Stage2Config,
    dataset: str,
    split: str,
    bundle: Stage2ProbeBundle,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], np.ndarray]]:
    trajs = _load_trajectories(cfg, dataset, split)
    if bundle.shallow_only:
        hd = 0
        hmap: Dict[Tuple[str, int], np.ndarray] = {}
    else:
        hd = _infer_hidden_dim(cfg, dataset, split, bundle.hidden_state_key)
        hmap = _load_hidden_map(cfg, dataset, split, hd, bundle.hidden_state_key)
    feat_map = _build_step_feature_map(trajs, hmap, hd, cfg)
    return trajs, feat_map


def precompute_continue_probabilities(
    feat_map: Dict[Tuple[str, int], np.ndarray],
    bundle: Stage2ProbeBundle,
    cfg: Stage2Config,
) -> Dict[Tuple[str, int], float]:
    return _precompute_probe_probs(feat_map, bundle.model, bundle.scaler, cfg)


def shallow_features_from_map(
    feat_map: Dict[Tuple[str, int], np.ndarray],
    shallow_dim: int,
) -> Dict[Tuple[str, int], np.ndarray]:
    """从 Stage2 逐步拼接向量中取出浅层子向量（供质量模型使用）。"""
    return {k: np.asarray(v[:shallow_dim], dtype=np.float32) for k, v in feat_map.items()}


def simulate_probe_baseline(
    trajectories: List[Dict[str, Any]],
    continue_probs: Dict[Tuple[str, int], float],
    probe_threshold: float,
    cfg: Stage2Config,
) -> List[Dict[str, Any]]:
    """无 E-value；与 Stage2 测试集 Probe 行为一致（给定同一组预计算概率）。"""
    return _simulate_probe_policy_from_probs(trajectories, probe_threshold, cfg, continue_probs)
