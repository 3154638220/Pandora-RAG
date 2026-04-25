#!/usr/bin/env bash
# Qwen 对比实验 Stage3：依赖同目录下已完成 Stage2（artifacts/probe/*/probe_mlp.pt）。
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
G="${QWEN_GAMMAS:-0.5}"
exec python -m stage3.run_stage3 --root-dir "$ROOT_DIR" --datasets "$DS" --gammas "$G"
