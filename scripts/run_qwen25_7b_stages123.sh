#!/usr/bin/env bash
# 使用 Qwen2.5-7B-Instruct 作为生成器 + 逐步 hidden backbone，在独立 --root-dir 下跑 Stage1→2→3（对比主实验 Llama-3.1-8B）。
# 与主目录 cache 隔离，避免覆盖；可复用已准备的 data/processed（见 QWEN_COPY_DATA_FROM）。
#
# 前置（终端 A — vLLM，独占一张 GPU，例如物理 GPU0）：
#   推荐：conda activate pandora-rag 后执行
#     runs/qwen2.5-7b-instruct/start_vllm_qwen.sh
#   （含 HF_HOME、FLASHINFER、enforce-eager；若权重已整仓下到本地可 export QWEN_MODEL_DIR=.../snapshots/<hash>）
#   权重未下全时：pip install hf_transfer && export HF_HUB_ENABLE_HF_TRANSFER=1
#   && huggingface-cli download Qwen/Qwen2.5-7B-Instruct --resume-download
#
# 本脚本（终端 B，建议 Stage1 hidden 用另一张卡：export CUDA_VISIBLE_DEVICES=1 再设 HIDDEN_STATE_DEVICE=cuda:0）在仓库根执行：

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# 与 `pretest.hf_env` 一致，避免未设置 HF_HOME 时落盘到无权限目录
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"

# 与 LLM 服务注册名、hidden 侧模型路径一致。若已下完整权重目录，可 export QWEN_MODEL_DIR=.../snapshots/<hash> 以加速且稳定。
export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
HIDDEN_PATH="${QWEN_MODEL_DIR:-$QWEN_MODEL_ID}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export LLM_MODEL="${QWEN_MODEL_ID}"
# 与 vLLM 同一张 GPU 时易 OOM：建议 hidden 在另一张卡（在仅暴露单卡的进程里为 cuda:0）
export HIDDEN_STATE_DEVICE="${HIDDEN_STATE_DEVICE:-cuda:0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ROOT_DIR="${QWEN_ROOT_DIR:-${REPO_ROOT}/runs/qwen2.5-7b-instruct}"
DATASETS="${QWEN_DATASETS:-hotpotqa,musique,2wiki}"

TRAIN_Q="${QWEN_TRAIN_QUOTA:-4000}"
CALIB_Q="${QWEN_CALIB_QUOTA:-1000}"
DEV_Q="${QWEN_DEV_QUOTA:-1000}"
TEST_Q="${QWEN_TEST_QUOTA:-1000}"

mkdir -p "${ROOT_DIR}/"{results,docs,artifacts,cache,logs}

# 复用主仓库已 prepare 的 data/（不复制 cache）
if [[ -n "${QWEN_COPY_DATA_FROM:-}" ]]; then
  if [[ ! -d "${ROOT_DIR}/data/processed" ]]; then
    echo "==> 复制 data 自 ${QWEN_COPY_DATA_FROM} -> ${ROOT_DIR}/"
    cp -a "${QWEN_COPY_DATA_FROM}/data" "${ROOT_DIR}/"
  else
    echo "==> 已存在 ${ROOT_DIR}/data/processed，跳过复制"
  fi
else
  if [[ ! -d "${ROOT_DIR}/data/processed" ]]; then
    echo "错误：未找到 ${ROOT_DIR}/data/processed。请先在本仓库跑过 stage1 prepare，或设置："
    echo "  export QWEN_COPY_DATA_FROM=${REPO_ROOT}"
    exit 1
  fi
fi

S1_EXTRA=()
if [[ "${PANDORA_STAGE1_SKIP_SELF_EVAL:-}" == 1 ]]; then
  S1_EXTRA+=(--skip-self-eval)
fi
if [[ -n "${PANDORA_NUM_WORKERS:-}" ]]; then
  S1_EXTRA+=(--num-workers "${PANDORA_NUM_WORKERS}")
fi

echo "==> Stage1  root=${ROOT_DIR}  LLM=${LLM_MODEL}  hidden=${HIDDEN_PATH}"
python -m stage1.run_stage1 \
  --root-dir "${ROOT_DIR}" \
  --datasets "${DATASETS}" \
  --train-quota "${TRAIN_Q}" \
  --calib-quota "${CALIB_Q}" \
  --dev-quota "${DEV_Q}" \
  --test-quota "${TEST_Q}" \
  --hidden-state-model "${HIDDEN_PATH}" \
  --collect-splits train,calib,dev,test \
  "${S1_EXTRA[@]}"

echo "==> Stage2  --per-dataset-optimal --root-dir ${ROOT_DIR}"
python -m stage2.run_stage2 \
  --root-dir "${ROOT_DIR}" \
  --datasets "${DATASETS}" \
  --per-dataset-optimal

echo "==> Stage3  --root-dir ${ROOT_DIR}"
python -m stage3.run_stage3 \
  --root-dir "${ROOT_DIR}" \
  --datasets "${DATASETS}" \
  --gammas 0.5

echo "==> 完成。产出在 ${ROOT_DIR}/results 与 ${ROOT_DIR}/docs"
