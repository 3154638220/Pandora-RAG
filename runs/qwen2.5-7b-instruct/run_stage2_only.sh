#!/usr/bin/env bash
# Qwen 对比实验 Stage2：依赖同目录下已完成 Stage1（cache/trajectories、artifacts/oracle）。
# 默认在 GPU1 上训练探针，避免与 GPU0 上 vLLM 冲突：可 export CUDA_VISIBLE_DEVICES=… 覆盖。
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate pandora-rag
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
REPO="/home/x12dpg/hjx/Pandora-RAG"
cd "$REPO"
export HF_HOME="$REPO/.hf_cache"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
ROOT_DIR="${QWEN_ROOT_DIR:-$REPO/runs/qwen2.5-7b-instruct}"
DS="${QWEN_DATASETS:-hotpotqa,musique,2wiki}"
exec python -m stage2.run_stage2 --root-dir "$ROOT_DIR" --datasets "$DS" --per-dataset-optimal
