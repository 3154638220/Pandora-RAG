#!/usr/bin/env python3
"""
预下载第一阶段资源：
  - 三大多跳 QA 数据集（Hugging Face Parquet）
  - Llama-3.1-8B-Instruct 权重（需 HF 账号接受 Meta 许可并设置 HF_TOKEN）

数据集默认走国内镜像 https://hf-mirror.com。
模型权重：默认先连官方 https://huggingface.co（需能访问；可设 HTTPS_PROXY）。若超时或目录里只有配置没有
model-*.safetensors，会自动改用 ModelScope（LLM-Research/Meta-Llama-3.1-8B-Instruct，国内通常更快）。
也可直接：python ... --model --via-modelscope

用法:
  python -u scripts/download_stage1_assets.py --datasets
  python -u scripts/download_stage1_assets.py --model
  python -u scripts/download_stage1_assets.py --model --via-modelscope
  python -u scripts/download_stage1_assets.py --all
  # 更快（需 pip install hf_transfer）:
  python -u scripts/download_stage1_assets.py --all --hf-transfer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 与 stage1.run_stage1 保持一致
# Hotpot distractor 与 MuSiQue 在 Hub 上仅提供 train/validation；test 由 prepare_data 从 validation 尾部切分；
# Calib+Dev 从剩余 validation 池中 seed 打乱后各取配额，与 Test 的 id 正交。
# 若 validation 偏小（如 MuSiQue），prepare_data 会收窄 test，优先保证 Calib+Dev 满额（见 docs/experiments.md A2）。
HF_SPECS = {
    "hotpotqa": ("hotpot_qa", None, ("train", "validation"), "refs/convert/parquet", "distractor"),
    "musique": ("dgslibisey/MuSiQue", None, ("train", "validation"), None, None),
    "2wiki": ("framolfese/2WikiMultihopQA", None, ("train", "validation", "test"), None, None),
}

LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# 与 HF 权重布局兼容，供无法直连 huggingface.co 时使用
MODELSCOPE_LLAMA_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"

# 国内常用 HF 镜像（hf-mirror 与 Hub API 路径兼容，datasets/hub 均识别 HF_ENDPOINT）
DEFAULT_CN_HF_ENDPOINT = "https://hf-mirror.com"
# 门控模型：镜像站通常 403，必须用官方 endpoint + token
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"

# 分片文件若小于此值视为未下完/损坏（8B 每片约数 GB）
_MIN_SHARD_BYTES = 100 * 1024 * 1024


def _configure_download_env(*, cn_mirror: bool, hf_transfer: bool) -> None:
    """必须在 import datasets / huggingface_hub 之前调用。"""
    if cn_mirror:
        os.environ["HF_ENDPOINT"] = DEFAULT_CN_HF_ENDPOINT
        print(f"[hub] 使用国内镜像 HF_ENDPOINT={DEFAULT_CN_HF_ENDPOINT}")
    if hf_transfer:
        try:
            import hf_transfer  # noqa: F401

            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("[hub] 已启用 hf_transfer 并行下载")
        except ImportError:
            print(
                "[hub] 未安装 hf_transfer，跳过并行加速（可执行: pip install hf_transfer）",
                file=sys.stderr,
            )


def download_datasets() -> None:
    from datasets import load_dataset

    for name, (repo, config, splits, revision, data_dir) in HF_SPECS.items():
        print(f"[datasets] {name} <- {repo}" + (f" ({config})" if config else ""))
        if revision:
            print(f"  revision={revision}")
        if data_dir:
            print(f"  data_dir={data_dir}")
        for sp in splits:
            try:
                kw: dict = {}
                if revision:
                    kw["revision"] = revision
                if data_dir:
                    kw["data_dir"] = data_dir
                if config:
                    load_dataset(repo, config, split=sp, **kw)
                else:
                    load_dataset(repo, split=sp, **kw)
                print(f"  ok split={sp}")
            except Exception as e:
                print(f"  FAIL split={sp}: {e}", file=sys.stderr)
                raise


def _expected_safetensor_shards(local_dir: Path) -> set[str]:
    idx = local_dir / "model.safetensors.index.json"
    if not idx.is_file():
        return set()
    try:
        data = json.loads(idx.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    wm = data.get("weight_map")
    if not isinstance(wm, dict):
        return set()
    return {str(v) for v in wm.values() if isinstance(v, str) and v.endswith(".safetensors")}


def llama_weights_complete(local_dir: Path) -> bool:
    """检查是否存在可用的 Llama safetensors 分片（避免 Hub 超时后误报成功）。"""
    local_dir = local_dir.resolve()
    single = local_dir / "model.safetensors"
    if single.is_file() and single.stat().st_size >= _MIN_SHARD_BYTES:
        return True
    shards = _expected_safetensor_shards(local_dir)
    if shards:
        for name in shards:
            p = local_dir / name
            if not p.is_file() or p.stat().st_size < _MIN_SHARD_BYTES:
                return False
        return True
    for p in local_dir.glob("model-*-of-*.safetensors"):
        if p.is_file() and p.stat().st_size >= _MIN_SHARD_BYTES:
            return True
    return False


def _download_model_hf(local_dir: Path, token: str | bool) -> None:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    print(f"[model] Hugging Face: {LLAMA_ID} -> {local_dir}")
    print(f"[model] endpoint={OFFICIAL_HF_ENDPOINT}（可设 HTTPS_PROXY 以穿越网络限制）")
    if token is True:
        print("[model] 未设置 HF_TOKEN 环境变量，将使用 HF_HOME 下已保存的令牌")
    snapshot_download(
        repo_id=LLAMA_ID,
        local_dir=str(local_dir),
        token=token,
        endpoint=OFFICIAL_HF_ENDPOINT,
    )


def _download_model_modelscope(local_dir: Path) -> None:
    from modelscope import snapshot_download as ms_snapshot_download

    print(f"[model] ModelScope: {MODELSCOPE_LLAMA_ID} -> {local_dir}")
    ms_snapshot_download(MODELSCOPE_LLAMA_ID, local_dir=str(local_dir))


def download_model(local_dir: Path, *, via_modelscope: bool) -> None:
    raw = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token: str | bool = raw.strip() if (raw or "").strip() else True
    local_dir = local_dir.resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    if via_modelscope:
        _download_model_modelscope(local_dir)
    else:
        try:
            _download_model_hf(local_dir, token)
        except Exception as e:
            print(f"[model] Hugging Face 下载异常（将尝试 ModelScope）: {e}", file=sys.stderr)
        if not llama_weights_complete(local_dir):
            print(
                "[model] 未检测到完整 safetensors 分片（常见于 huggingface.co 连接超时），"
                "改用 ModelScope 拉取权重…",
                file=sys.stderr,
            )
            _download_model_modelscope(local_dir)

    if not llama_weights_complete(local_dir):
        raise RuntimeError(
            "模型权重仍不完整。可尝试："
            "1) export HTTPS_PROXY=... 后重试官方 Hub；"
            "2) python -u scripts/download_stage1_assets.py --model --via-modelscope；"
            "3) pip install -U modelscope"
        )
    print("  ok")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download Stage-1 datasets and Llama 3.1 8B weights")
    parser.add_argument("--datasets", action="store_true", help="Download HotpotQA, MuSiQue, 2WikiMultihopQA")
    parser.add_argument("--model", action="store_true", help=f"Download {LLAMA_ID} (needs HF_TOKEN + license)")
    parser.add_argument("--all", action="store_true", help="Both --datasets and --model")
    parser.add_argument(
        "--no-cn-mirror",
        action="store_true",
        help="不强制国内镜像，使用官方 huggingface.co（或你已导出的 HF_ENDPOINT）",
    )
    parser.add_argument(
        "--hf-transfer",
        action="store_true",
        help="启用 huggingface 的 hf_transfer 多连接下载（需 pip install hf_transfer）",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(root / "models" / "Meta-Llama-3.1-8B-Instruct"),
        help="Directory for Llama weights",
    )
    parser.add_argument(
        "--via-modelscope",
        action="store_true",
        help="仅从 ModelScope 下载（跳过 Hugging Face，适合无法访问 huggingface.co 的环境）",
    )
    args = parser.parse_args()
    if not args.datasets and not args.model and not args.all:
        parser.print_help()
        sys.exit(1)

    from pretest.hf_env import init_pandora_hf_home

    init_pandora_hf_home(root=root)

    _configure_download_env(cn_mirror=not args.no_cn_mirror, hf_transfer=args.hf_transfer)

    if args.all or args.datasets:
        download_datasets()
    if args.all or args.model:
        try:
            download_model(Path(args.model_dir), via_modelscope=args.via_modelscope)
        except Exception as e:
            print(
                "\n模型下载失败。\n"
                "- 若在国内无法访问 huggingface.co：请使用\n"
                f"    python -u scripts/download_stage1_assets.py --model --via-modelscope --model-dir {args.model_dir}\n"
                "- 若走官方 Hub：需能访问 huggingface.co（可设 HTTPS_PROXY），并在\n"
                "  https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct 完成许可，且 export HF_TOKEN=…\n"
                f"详情: {e}\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
