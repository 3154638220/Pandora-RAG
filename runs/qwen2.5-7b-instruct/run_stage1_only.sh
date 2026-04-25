#!/usr/bin/env bash
# 在 vLLM（Qwen2.5-7B）已监听 OPENAI_API_BASE 时，仅跑 Stage1；与主实验 cache 隔离。
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate pandora-rag
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
REPO="/home/x12dpg/hjx/Pandora-RAG"
cd "$REPO"
export HF_HOME="$REPO/.hf_cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# vLLM 在物理 GPU0 时，本进程只暴露 GPU1 给 transformers hidden（避免与 vLLM 争显存）
export CUDA_VISIBLE_DEVICES="${QWEN_STAGE1_CUDA_VISIBLE:-1}"
export HIDDEN_STATE_DEVICE="${HIDDEN_STATE_DEVICE:-cuda:0}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export LLM_MODEL="${QWEN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
SNAP="${QWEN_MODEL_DIR:-$REPO/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
ROOT_DIR="${QWEN_ROOT_DIR:-$REPO/runs/qwen2.5-7b-instruct}"
DS="${QWEN_DATASETS:-hotpotqa,musique,2wiki}"
S1_EXTRA=()
if [[ "${PANDORA_STAGE1_SKIP_SELF_EVAL:-}" =~ ^(1|true|yes)$ ]]; then
  S1_EXTRA+=(--skip-self-eval)
fi
exec python -m stage1.run_stage1 \
  --root-dir "$ROOT_DIR" \
  --datasets "$DS" \
  --train-quota "${QWEN_TRAIN_QUOTA:-4000}" \
  --calib-quota "${QWEN_CALIB_QUOTA:-1000}" \
  --dev-quota "${QWEN_DEV_QUOTA:-1000}" \
  --test-quota "${QWEN_TEST_QUOTA:-1000}" \
  --hidden-state-model "$SNAP" \
  --collect-splits train,calib,dev,test \
  --num-workers "${PANDORA_NUM_WORKERS:-4}" \
  "${S1_EXTRA[@]}"
