#!/usr/bin/env python3
"""
预下载第一阶段资源：
  - 三大多跳 QA 数据集（Hugging Face Parquet 镜像）
  - Llama-3.1-8B-Instruct 权重（需 HF 账号接受 Meta 许可并设置 HF_TOKEN）

用法:
  python scripts/download_stage1_assets.py --datasets
  python scripts/download_stage1_assets.py --model
  python scripts/download_stage1_assets.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 与 stage1.run_stage1 保持一致
# Hotpot distractor 与 MuSiQue 在 Hub 上仅提供 train/validation；test 由 prepare_data 从 validation 切分。
HF_SPECS = {
    "hotpotqa": ("hotpot_qa", "distractor", ("train", "validation")),
    "musique": ("dgslibisey/MuSiQue", None, ("train", "validation")),
    "2wiki": ("framolfese/2WikiMultihopQA", None, ("train", "validation", "test")),
}

LLAMA_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def download_datasets() -> None:
    from datasets import load_dataset

    for name, (repo, config, splits) in HF_SPECS.items():
        print(f"[datasets] {name} <- {repo}" + (f" ({config})" if config else ""))
        for sp in splits:
            try:
                if config:
                    load_dataset(repo, config, split=sp)
                else:
                    load_dataset(repo, split=sp)
                print(f"  ok split={sp}")
            except Exception as e:
                print(f"  FAIL split={sp}: {e}", file=sys.stderr)
                raise


def download_model(local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local_dir = local_dir.resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model] {LLAMA_ID} -> {local_dir}")
    snapshot_download(repo_id=LLAMA_ID, local_dir=str(local_dir), token=token)
    print("  ok")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download Stage-1 datasets and Llama 3.1 8B weights")
    parser.add_argument("--datasets", action="store_true", help="Download HotpotQA, MuSiQue, 2WikiMultihopQA")
    parser.add_argument("--model", action="store_true", help=f"Download {LLAMA_ID} (needs HF_TOKEN + license)")
    parser.add_argument("--all", action="store_true", help="Both --datasets and --model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(root / "models" / "Meta-Llama-3.1-8B-Instruct"),
        help="Directory for Llama weights",
    )
    args = parser.parse_args()
    if not args.datasets and not args.model and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all or args.datasets:
        download_datasets()
    if args.all or args.model:
        try:
            download_model(Path(args.model_dir))
        except Exception as e:
            print(
                "\n模型下载失败（常见原因：未设置 HF_TOKEN，或未在 Hugging Face 接受 Meta Llama 许可）。\n"
                "请访问 https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct 完成授权后执行：\n"
                "  export HF_TOKEN=hf_...\n"
                f"  python scripts/download_stage1_assets.py --model --model-dir {args.model_dir}\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
