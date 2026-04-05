"""
Stage-2 pipeline for Pandora-RAG.

Covers:
  A. Build Oracle step labels from Stage-1 cached trajectories
  B. Train MLP Neural Probe with margin-weighted BCE loss
  C. Tune stop threshold on dev split
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from pretest.utils.weitzman import (
    compute_all_reservation_values,
    compute_trajectory_oracle,
    oracle_stopping_simulation,
    trajectory_cumulative_cost,
)

LOGGER = logging.getLogger(__name__)

SHALLOW_FEATURE_DIM = 9  # step 归一化索引、retrieval_score、semantic_entropy、self_consistency、ctx_overlap、nli_entail、nli_contra、log1p(token)、log1p(latency)


@dataclass
class Stage2Config:
    seed: int = 42
    max_k: int = 5
    cost_per_step: float = 0.05
    oracle_cost_metric: str = "fixed"
    root_dir: Path = Path(".")
    hidden_state_key: str = "last_token"

    # Training
    hidden_dim: int = 256
    dropout: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 35
    patience: int = 6
    margin_weight_floor: float = 0.1

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
    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.w = torch.tensor(w, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.w[idx]


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
) -> Dict[str, np.ndarray]:
    if hidden_dim <= 0:
        return {}
    feat_dir = cfg.features_dir / dataset / split / "hidden_states"
    if not feat_dir.exists():
        LOGGER.warning("%s 不存在，当前 split 使用零向量 hidden。", feat_dir)
        return {}

    out: Dict[str, np.ndarray] = {}
    bad = 0
    for fp in feat_dir.glob("*.npz"):
        sample_id = fp.stem
        try:
            with np.load(fp) as obj:
                if key not in obj:
                    bad += 1
                    continue
                vec = np.asarray(obj[key], dtype=np.float32).reshape(-1)
                if vec.size != hidden_dim:
                    bad += 1
                    continue
                out[sample_id] = vec
        except Exception:
            bad += 1
    if bad > 0:
        LOGGER.warning("%s/%s hidden states 异常文件数：%d", dataset, split, bad)
    return out


def _step_shallow_features(step: Dict[str, Any], k: int, cfg: Stage2Config) -> List[float]:
    cost = step.get("cost") or {}
    token_count = float(cost.get("token_count", 0) or 0.0)
    latency_ms = float(cost.get("latency_ms", 0.0) or 0.0)
    return [
        float(k) / float(max(1, cfg.max_k)),
        float(step.get("retrieval_score", 0.0) or 0.0),
        float(step.get("semantic_entropy", 0.0) or 0.0),
        float(step.get("self_consistency", 0.0) or 0.0),
        float(step.get("ctx_overlap", 0.0) or 0.0),
        float(step.get("nli_entail", 0.0) or 0.0),
        float(step.get("nli_contra", 0.0) or 0.0),
        math.log1p(max(0.0, token_count)),
        math.log1p(max(0.0, latency_ms)),
        # f1 已移除：推理时无 Ground Truth，使用 f1 会构成目标泄露。
    ]


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
    hidden_map: Dict[str, np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    x_rows: List[np.ndarray] = []
    y_vals: List[float] = []
    w_vals: List[float] = []

    n_missing_hidden = 0
    n_missing_label = 0
    total_steps = 0

    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)

    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        if not sample_id:
            continue
        step_targets = oracle_by_id.get(sample_id, {})
        hidden = hidden_map.get(sample_id, zero_hidden)
        if sample_id not in hidden_map:
            n_missing_hidden += 1

        steps = sorted(traj.get("steps") or [], key=lambda s: int(s.get("step", 0)))
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
            weight = max(cfg.margin_weight_floor, abs(margin))

            shallow = np.asarray(_step_shallow_features(step, k, cfg), dtype=np.float32)
            feat = np.concatenate([shallow, hidden], axis=0)
            x_rows.append(feat)
            y_vals.append(action_label)
            w_vals.append(weight)

    if not x_rows:
        raise RuntimeError("可训练样本为空，请确认 Stage1 轨迹与 Oracle 标签是否完整。")

    n_traj = max(1, len(trajectories))
    if n_missing_hidden / n_traj > 0.05:
        LOGGER.warning(
            "超过 5%% 的样本缺失 Hidden States（missing_hidden_ids=%d / trajectories=%d），"
            "请检查 Stage 1 的 hidden_states 缓存是否完整。",
            n_missing_hidden,
            len(trajectories),
        )

    stats = {
        "total_steps_seen": int(total_steps),
        "trainable_examples": int(len(x_rows)),
        "missing_hidden_ids": int(n_missing_hidden),
        "missing_step_labels": int(n_missing_label),
    }
    x = np.stack(x_rows).astype(np.float32)
    y = np.asarray(y_vals, dtype=np.float32)
    w = np.asarray(w_vals, dtype=np.float32)
    return x, y, w, stats


def _weighted_bce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weighted = bce * weights
    denom = torch.clamp(weights.sum(), min=1e-6)
    return weighted.sum() / denom


def _run_epoch(
    model: ProbeMLP,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_weight = 0.0
    for x, y, w in loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        logits = model(x)
        loss = _weighted_bce_loss(logits, y, w)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_weight = float(w.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight

    if total_weight <= 0:
        return 0.0
    return total_loss / total_weight


def _predict_probs(
    model: ProbeMLP,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
            prob = torch.sigmoid(model(xb)).detach().cpu().numpy().reshape(-1)
            out.append(prob)
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out, axis=0).astype(np.float32)


def _train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    w_dev: np.ndarray,
    cfg: Stage2Config,
) -> Tuple[ProbeMLP, Dict[str, Any], StandardScaler]:
    scaler = StandardScaler()
    # 仅对浅层特征做标准化，保留 LLM hidden states 的内在几何结构。
    sdim = SHALLOW_FEATURE_DIM
    x_train_s = np.concatenate(
        [scaler.fit_transform(x_train[:, :sdim]), x_train[:, sdim:]], axis=1
    )
    x_dev_s = np.concatenate(
        [scaler.transform(x_dev[:, :sdim]), x_dev[:, sdim:]], axis=1
    )

    train_ds = ProbeDataset(x_train_s, y_train, w_train)
    dev_ds = ProbeDataset(x_dev_s, y_dev, w_dev)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbeMLP(in_dim=x_train.shape[1], hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_dev_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    hist_rows: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        dev_loss = _run_epoch(model, dev_loader, None, device)
        hist_rows.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        LOGGER.info("Probe epoch %02d | train_loss=%.6f | dev_loss=%.6f", epoch, train_loss, dev_loss)

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

    dev_probs = _predict_probs(model, x_dev_s, cfg.batch_size, device)
    dev_pred = (dev_probs >= 0.5).astype(np.int32)
    dev_acc = float((dev_pred == y_dev.astype(np.int32)).mean()) if len(y_dev) > 0 else 0.0

    train_info = {
        "best_epoch": int(best_epoch),
        "best_dev_loss": float(best_dev_loss),
        "dev_acc_at_0.5": dev_acc,
        "history": hist_rows,
        "device": str(device),
    }
    return model, train_info, scaler


def _build_step_feature_map(
    trajectories: List[Dict[str, Any]],
    hidden_map: Dict[str, np.ndarray],
    hidden_dim: int,
    cfg: Stage2Config,
) -> Dict[Tuple[str, int], np.ndarray]:
    zero_hidden = np.zeros((hidden_dim,), dtype=np.float32)
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for traj in trajectories:
        sample_id = str(traj.get("id", ""))
        if not sample_id:
            continue
        hidden = hidden_map.get(sample_id, zero_hidden)
        for step in traj.get("steps") or []:
            k = int(step.get("step", 0))
            if k <= 0:
                continue
            shallow = np.asarray(_step_shallow_features(step, k, cfg), dtype=np.float32)
            out[(sample_id, k)] = np.concatenate([shallow, hidden], axis=0)
    return out


def _precompute_probe_probs(
    step_feature_map: Dict[Tuple[str, int], np.ndarray],
    model: ProbeMLP,
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
    feats_scaled = np.concatenate(
        [scaler.transform(feats_arr[:, :sdim]), feats_arr[:, sdim:]],
        axis=1,
    ).astype(np.float32)
    probs = _predict_probs(model, feats_scaled, cfg.batch_size, device)
    return {k: float(p) for k, p in zip(keys, probs)}


def _simulate_probe_policy(
    trajectories: List[Dict[str, Any]],
    step_feature_map: Dict[Tuple[str, int], np.ndarray],
    model: ProbeMLP,
    scaler: StandardScaler,
    threshold: float,
    cfg: Stage2Config,
    precomputed_probs: Optional[Dict[Tuple[str, int], float]] = None,
) -> List[Dict[str, Any]]:
    if precomputed_probs is None:
        precomputed_probs = _precompute_probe_probs(step_feature_map, model, scaler, cfg)

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


def _pick_best_threshold(
    dev_trajectories: List[Dict[str, Any]],
    dev_step_feature_map: Dict[Tuple[str, int], np.ndarray],
    model: ProbeMLP,
    scaler: StandardScaler,
    cfg: Stage2Config,
) -> Tuple[float, Dict[str, Any]]:
    candidates = [round(x, 3) for x in np.linspace(0.01, 0.99, 50)]
    dev_probs = _precompute_probe_probs(dev_step_feature_map, model, scaler, cfg)
    best_t = 0.5
    best_row: Optional[Dict[str, Any]] = None
    for t in candidates:
        rs = _simulate_probe_policy(
            dev_trajectories,
            dev_step_feature_map,
            model,
            scaler,
            t,
            cfg,
            precomputed_probs=dev_probs,
        )
        row = _summarize_results(rs, f"Probe@{t:.2f}")
        if best_row is None:
            best_t = t
            best_row = row
            continue
        # 主目标：F1 最大；次目标：成本更低。
        if (row["avg_f1"] > best_row["avg_f1"]) or (
            math.isclose(row["avg_f1"], best_row["avg_f1"], rel_tol=1e-8, abs_tol=1e-8)
            and row["avg_cost"] < best_row["avg_cost"]
        ):
            best_t = t
            best_row = row
    assert best_row is not None
    return float(best_t), best_row


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


def _plot_dataset_pareto(df: pd.DataFrame, dataset: str, out_path: Path) -> None:
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
    ax.set_title(f"Stage2 Probe Pareto - {dataset}")
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

    hidden_dim = _infer_hidden_dim(cfg, dataset, "train", cfg.hidden_state_key)
    train_hidden = _load_hidden_map(cfg, dataset, "train", hidden_dim, cfg.hidden_state_key)
    dev_hidden = _load_hidden_map(cfg, dataset, "dev", hidden_dim, cfg.hidden_state_key)
    test_hidden = _load_hidden_map(cfg, dataset, "test", hidden_dim, cfg.hidden_state_key)

    train_oracle_map, _train_oracle_rows = _make_oracle_maps(train_traj, cfg)
    dev_oracle_map, _dev_oracle_rows = _make_oracle_maps(dev_traj, cfg)
    _test_oracle_map, test_oracle_rows = _make_oracle_maps(test_traj, cfg)

    x_train, y_train, w_train, train_stats = _build_xyw(
        train_traj, train_oracle_map, train_hidden, hidden_dim, cfg
    )
    x_dev, y_dev, w_dev, dev_stats = _build_xyw(
        dev_traj, dev_oracle_map, dev_hidden, hidden_dim, cfg
    )

    model, train_info, scaler = _train_probe(x_train, y_train, w_train, x_dev, y_dev, w_dev, cfg)

    dev_step_feat_map = _build_step_feature_map(dev_traj, dev_hidden, hidden_dim, cfg)
    test_step_feat_map = _build_step_feature_map(test_traj, test_hidden, hidden_dim, cfg)
    threshold, best_dev_row = _pick_best_threshold(dev_traj, dev_step_feat_map, model, scaler, cfg)

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

    oracle_rows = _eval_oracle_rows(test_traj, test_oracle_rows, cfg)
    oracle_summary = _summarize_results(oracle_rows, "Oracle")
    rows.append(oracle_summary)

    table_df = pd.DataFrame(rows).sort_values(by=["avg_cost", "avg_f1"], ascending=[True, False])
    table_path = cfg.results_dir / f"stage2_probe_table_{dataset}.csv"
    table_df.to_csv(table_path, index=False)

    pareto_path = cfg.results_dir / f"stage2_probe_pareto_{dataset}.png"
    _plot_dataset_pareto(table_df, dataset, pareto_path)

    best_fixed = max((r["avg_f1"] for r in fixed_rows), default=0.0)
    probe_gain = float(probe_row["avg_f1"] - best_fixed)
    oracle_gap = float(oracle_summary["avg_f1"] - probe_row["avg_f1"])

    model_path = cfg.artifacts_probe_dir / dataset / "probe_mlp.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
            "threshold": float(threshold),
            "hidden_state_key": cfg.hidden_state_key,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "shallow_feature_dim": SHALLOW_FEATURE_DIM,
        },
        model_path,
    )

    meta_path = cfg.artifacts_probe_dir / dataset / "stage2_train_meta.json"
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
            "probe_summary": probe_row,
            "oracle_summary": oracle_summary,
            "best_fixed_f1": best_fixed,
            "probe_gain_over_best_fixed": probe_gain,
            "oracle_gap_to_probe": oracle_gap,
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


def build_stage2_report(results: Dict[str, Dict[str, Any]], out_path: Path) -> Path:
    lines: List[str] = []
    lines.append("# Stage2 Report")
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
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP 第一层宽度。")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--margin-weight-floor",
        type=float,
        default=0.1,
        help="margin 加权 BCE 的最小样本权重。",
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

    cfg = Stage2Config(
        seed=args.seed,
        max_k=args.max_k,
        cost_per_step=args.cost_per_step,
        oracle_cost_metric=args.oracle_cost_metric,
        root_dir=Path(args.root_dir),
        hidden_state_key=args.hidden_state_key,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        margin_weight_floor=args.margin_weight_floor,
    )

    _set_seed(cfg.seed)
    _ensure_dirs(cfg, datasets)

    all_results: Dict[str, Dict[str, Any]] = {}
    for ds in datasets:
        LOGGER.info("===== Stage2 dataset: %s =====", ds)
        all_results[ds] = run_dataset_stage2(cfg, ds)

    report_path = build_stage2_report(all_results, cfg.results_dir / "stage2_report.md")
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
