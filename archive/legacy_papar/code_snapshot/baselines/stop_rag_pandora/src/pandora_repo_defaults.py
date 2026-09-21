"""Pandora-RAG repo layout helpers for Stop-RAG defaults."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Upstream Stop-RAG Hub default (when no local weights and no env override).
_FALLBACK_VLLM_HUB_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Same directory name as Stage1 (`stage1/run_stage1._resolve_local_llama_weights_dir`).
_LOCAL_LLAMA_DIR = "Meta-Llama-3.1-8B-Instruct"


def _stop_rag_src_dir() -> Path:
    """Directory containing this module (`.../baselines/Stop-RAG/src`)."""
    return Path(__file__).resolve().parent


def _stop_rag_root() -> Path:
    return _stop_rag_src_dir().parent


def _pandora_repo_root() -> Path:
    """`Pandora-RAG` root when Stop-RAG lives in `baselines/Stop-RAG`."""
    return _stop_rag_root().parent.parent


def ensure_pandora_repo_root_on_path() -> Path:
    """Add Pandora-RAG repo root to ``sys.path`` for shared modules."""
    root = _pandora_repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def resolve_default_vllm_model_id() -> str:
    """Prefer ``STOP_RAG_VLLM_MODEL``, then local ``models/Meta-Llama-3.1-8B-Instruct``."""
    env = (os.environ.get("STOP_RAG_VLLM_MODEL") or "").strip()
    if env:
        return env

    candidates = (
        _pandora_repo_root() / "models" / _LOCAL_LLAMA_DIR,
        _stop_rag_root() / "models" / _LOCAL_LLAMA_DIR,
    )
    for local in candidates:
        if (local / "config.json").is_file():
            return str(local)
    return _FALLBACK_VLLM_HUB_ID
