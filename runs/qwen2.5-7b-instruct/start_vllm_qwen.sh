#!/usr/bin/env bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate pandora-rag
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
REPO="/home/x12dpg/hjx/Pandora-RAG"
cd "$REPO"
# 与 stage1 一致，强制可写缓存（系统环境里的 HF_HOME 可能指向无权限目录）
export HF_HOME="$REPO/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME"
# 已整仓下完时设置 QWEN_MODEL_DIR 指向含 config + 4×safetensors 的目录，可避免 Hub 断点受网络影响
QWEN_MODEL_PATH="${QWEN_MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct}"
export CUDA_VISIBLE_DEVICES=0
exec python -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_MODEL_PATH" \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --attention-backend FLASHINFER \
  --served-model-name Qwen/Qwen2.5-7B-Instruct
