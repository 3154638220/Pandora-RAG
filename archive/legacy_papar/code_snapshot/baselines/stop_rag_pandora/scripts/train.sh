#!/usr/bin/env bash

DATASET="${1:-}"
RETRIEVER="${2:-}"
METHOD="${3:-}"

if [ -z "$DATASET" ] || [ -z "$RETRIEVER" ] || [ -z "$METHOD" ]; then
    echo "Usage: $0 <dataset> <retriever> <method>"
    exit 1
fi

TRAIN_PATH="data/processed/${DATASET}/${METHOD}/${RETRIEVER}/train/train.jsonl"
EVAL_PATH="data/processed/${DATASET}/${METHOD}/${RETRIEVER}/train/eval.jsonl"

if [ ! -f "$TRAIN_PATH" ]; then
    echo "Train data not found: $TRAIN_PATH" >&2
    exit 1
fi

if [ ! -f "$EVAL_PATH" ]; then
    echo "Eval data not found: $EVAL_PATH" >&2
    exit 1
fi

OUTPUT_DIR="outputs/${DATASET}_${METHOD}_${RETRIEVER}"
RUN_NAME="${DATASET}_${METHOD}_${RETRIEVER}"
# 避免 PATH 的 torchrun 指向 base conda，与 STOP_RAG_PYTHON（如 pandora-rag）不一致导致缺包
if [ -z "${STOP_RAG_TORCHRUN:-}" ] && [ -n "${STOP_RAG_PYTHON:-}" ]; then
    _tr_dir="$(dirname "${STOP_RAG_PYTHON}")"
    if [ -x "${_tr_dir}/torchrun" ]; then
        STOP_RAG_TORCHRUN="${_tr_dir}/torchrun"
    fi
fi
TORCHRUN_BIN="${STOP_RAG_TORCHRUN:-torchrun}"
MODEL_ID="${STOP_RAG_ENCODER_MODEL:-microsoft/deberta-v3-large}"
NPROC_PER_NODE="${STOP_RAG_NPROC_PER_NODE:-1}"
TRAIN_BATCH_SIZE="${STOP_RAG_TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${STOP_RAG_GRAD_ACCUM:-16}"
MAX_LENGTH="${STOP_RAG_MAX_LENGTH:-2048}"
DEEPSPEED_CONFIG="${STOP_RAG_DEEPSPEED_CONFIG:-}"
CMD=(
    "${TORCHRUN_BIN}" "--nproc_per_node" "${NPROC_PER_NODE}" -m src.train.stop_rag_train
    --model-id "${MODEL_ID}"
    --model-arch "encoder_only"
    --train-data-path "${TRAIN_PATH}"
    --eval-data-path "${EVAL_PATH}"
    --trainer-output-dir "${OUTPUT_DIR}"
    --run-name "${RUN_NAME}"
    --target-type "tdlambda"
    --target-label "f1"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRAD_ACCUM}"
    --max-length "${MAX_LENGTH}"
)

mkdir -p "$OUTPUT_DIR"

if [ "${STOP_RAG_BF16:-true}" = "true" ]; then
    CMD+=(--bf16)
fi

# 始终显式传入 --deepspeed-config：为空时传 ""，Python 侧 "" or None = None 可禁用 deepspeed
CMD+=(--deepspeed-config "${DEEPSPEED_CONFIG}")

if [ "${WANDB_DISABLED:-true}" = "true" ]; then
    CMD+=(--disable-wandb)
fi

if [ "${STOP_RAG_GRADIENT_CHECKPOINTING:-false}" = "true" ]; then
    CMD+=(--gradient-checkpointing)
fi

if [ -n "${STOP_RAG_RESUME_FROM_CHECKPOINT:-}" ]; then
    CMD+=(--resume-from-checkpoint "${STOP_RAG_RESUME_FROM_CHECKPOINT}")
fi

"${CMD[@]}"
