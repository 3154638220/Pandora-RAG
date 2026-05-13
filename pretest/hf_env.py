"""将 Hugging Face 缓存固定到仓库根目录 `.hf_cache`，避免沿用已损坏或不可用的系统 HF_HOME。"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def init_pandora_hf_home(*, root: Path | None = None) -> Path:
    """在 import datasets / transformers / huggingface_hub 之前调用。

    默认将 ``HF_HOME`` 与 ``HF_HUB_CACHE`` 指到仓库 ``.hf_cache``（及 ``.hf_cache/hub``），
    避免沿用环境里不可写的缓存路径。

    若设置 ``PANDORA_USE_SYSTEM_HF_HOME=1``（或 true/yes），则**不**改写上述变量。
    """
    r = root if root is not None else repo_root()
    target = (r / ".hf_cache").resolve()
    target.mkdir(parents=True, exist_ok=True)
    if os.environ.get("PANDORA_USE_SYSTEM_HF_HOME", "").lower() not in ("1", "true", "yes"):
        os.environ["HF_HOME"] = str(target)
        # hf_hub_download 优先读 HF_HUB_CACHE；若环境指向不可写目录会导致下载失败
        hub = target / "hub"
        hub.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hub)
    return target
