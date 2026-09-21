#!/usr/bin/env python3
"""
将 MuSiQue 的 answer_aliases 合并进 Stage1 processed JSONL，并对已有 trajectories.jsonl
离线重算每步 F1/EM（multi-gold max-F1 / any-EM）。

用法见 docs/reports/baselines/musique_alias_metric_alignment.md。完成后建议在同一 stage1-root 下执行：

  python -m stage1.run_stage1 --datasets musique --skip-prepare --skip-trajectories \\
    --root-dir <stage1-root>

以刷新 oracle 表、Pareto 图与 test oracle labels（无需重跑 LLM）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Dict, Iterator, List

# 与 stage1.run_stage1 一致：默认镜像
if os.environ.get("PANDORA_NO_CN_HF_MIRROR", "").lower() not in ("1", "true", "yes"):
    if not (os.environ.get("HF_ENDPOINT") or "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pretest.hf_env import init_pandora_hf_home

# 必须在 import huggingface_hub 之前固定 HF_HUB_CACHE，否则 hub 会沿用环境里不可写的缓存路径
init_pandora_hf_home()

from huggingface_hub import hf_hub_download

from qa_shared.metrics import build_gold_answers, compute_metrics_multi

MUSIQUE_REPO = "dgslibisey/MuSiQue"
# Hub 上的原始 JSONL（与 load_dataset 元数据无关，避免 datasets 3.x/4.x 的 List/Sequence 不兼容）
MUSIQUE_JSONL_FILES = (
    "musique_ans_v1.0_train.jsonl",
    "musique_ans_v1.0_dev.jsonl",
)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_musique_raw_by_id() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for fname in MUSIQUE_JSONL_FILES:
        local_path = hf_hub_download(
            repo_id=MUSIQUE_REPO,
            filename=fname,
            repo_type="dataset",
        )
        with Path(local_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                rid = str(ex.get("id") or ex.get("_id") or "").strip()
                if rid:
                    idx[rid] = ex
    return idx


def patch_processed_jsonl(path: Path, raw_by_id: Dict[str, Dict[str, Any]]) -> tuple[int, int]:
    rows = list(_iter_jsonl(path))
    n = 0
    missing = 0
    for r in rows:
        rid = str(r.get("id", ""))
        raw = raw_by_id.get(rid)
        if not raw:
            missing += 1
            continue
        g = build_gold_answers(raw)
        r["answer"] = g[0]
        r["answer_aliases"] = g[1:]
        r["gold_answers"] = g
        n += 1
    if n:
        _write_jsonl(path, rows)
    return n, missing


def rescore_trajectories_jsonl(path: Path, raw_by_id: Dict[str, Dict[str, Any]]) -> int:
    rows = list(_iter_jsonl(path))
    touched = 0
    for r in rows:
        rid = str(r.get("id", ""))
        raw = raw_by_id.get(rid) if raw_by_id else None
        if raw:
            gold_answers = build_gold_answers(raw)
            r["gold_answer"] = gold_answers[0]
            r["answer_aliases"] = gold_answers[1:]
            r["gold_answers"] = gold_answers
        else:
            gold_answers = r.get("gold_answers") or [r.get("gold_answer") or ""]
        for s in r.get("steps") or []:
            pred = str(s.get("current_answer", s.get("answer", "")) or "")
            f1, em = compute_metrics_multi(pred, gold_answers)
            new_f1 = round(float(f1), 4)
            new_em = bool(em)
            if s.get("f1") != new_f1 or bool(s.get("em")) != new_em:
                touched += 1
            s["f1"] = new_f1
            s["em"] = new_em
    _write_jsonl(path, rows)
    return touched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage1-root",
        type=Path,
        required=True,
        help="Stage1 根目录（含 data/processed/musique 与可选 cache/trajectories/musique）",
    )
    ap.add_argument(
        "--skip-hf",
        action="store_true",
        help="不从 Hub 拉取 MuSiQue；仅根据每条轨迹/行内已有 gold_answers 重算 F1/EM",
    )
    args = ap.parse_args()
    root: Path = args.stage1_root.resolve()
    proc_dir = root / "data" / "processed" / "musique"
    traj_root = root / "cache" / "trajectories" / "musique"

    raw_by_id: Dict[str, Dict[str, Any]] = {}
    if not args.skip_hf:
        raw_by_id = load_musique_raw_by_id()

    patched_rows = 0
    missing_raw = 0
    if proc_dir.is_dir() and raw_by_id:
        for split in ("train", "calib", "dev", "test"):
            p = proc_dir / f"{split}.jsonl"
            if p.exists():
                pr, miss = patch_processed_jsonl(p, raw_by_id)
                patched_rows += pr
                missing_raw += miss
    elif proc_dir.is_dir() and not args.skip_hf:
        print("警告：HF 索引为空，跳过 processed 修补。", file=sys.stderr)

    traj_steps = 0
    if traj_root.is_dir():
        for split_dir in sorted(traj_root.iterdir()):
            if not split_dir.is_dir():
                continue
            tpath = split_dir / "trajectories.jsonl"
            if tpath.exists():
                traj_steps += rescore_trajectories_jsonl(tpath, raw_by_id)
    else:
        print(f"未找到轨迹目录（可忽略）：{traj_root}", file=sys.stderr)

    print(f"processed 中按 raw 更新 gold 字段的样本数: {patched_rows}")
    if missing_raw:
        print(f"processed 中未在 Hub JSONL 找到 id 的条数（未改写）: {missing_raw}")
    print(f"trajectories 中发生变化的 step 计数字段数: {traj_steps}")


if __name__ == "__main__":
    main()
