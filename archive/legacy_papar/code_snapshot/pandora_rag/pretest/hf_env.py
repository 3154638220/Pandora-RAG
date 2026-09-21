"""将 Hugging Face 缓存固定到仓库根目录 `.hf_cache`，避免沿用已损坏或不可用的系统 HF_HOME。"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def init_pandora_hf_home(*, root: Path | None = None) -> Path:
    """在 import datasets / transformers 之前调用。

    若设置 ``PANDORA_USE_SYSTEM_HF_HOME=1``（或 true/yes），则保留环境变量中的 ``HF_HOME``。
    """
    r = root if root is not None else repo_root()
    target = (r / ".hf_cache").resolve()
    target.mkdir(parents=True, exist_ok=True)
    if os.environ.get("PANDORA_USE_SYSTEM_HF_HOME", "").lower() not in ("1", "true", "yes"):
        os.environ["HF_HOME"] = str(target)
    return target
