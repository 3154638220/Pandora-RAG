#!/usr/bin/env bash

set -eu

DATASET="${1:-}"
RETRIEVER="${2:-}"
METHOD="${3:-}"

if [ "$DATASET" != "hotpotqa" ] && [ "$DATASET" != "2wikimultihopqa" ] && [ "$DATASET" != "musique" ]; then
    echo "Invalid dataset. Use 'hotpotqa', '2wikimultihopqa', or 'musique'." >&2
    exit 1
fi

if [ "$RETRIEVER" != "contriever" ] && [ "$RETRIEVER" != "bm25" ]; then
    echo "Invalid retriever. Use 'contriever' or 'bm25'." >&2
    exit 1
fi

if [ "$METHOD" != "ours" ] && [ "$METHOD" != "corag" ]; then
    echo "Invalid method. Use 'ours' or 'corag'." >&2
    exit 1
fi

BASE_DATA_DIR="data/processed/${DATASET}/${METHOD}/${RETRIEVER}"
TRAIN_OUTPUT_DIR="outputs/${DATASET}_${METHOD}_${RETRIEVER}"
TEST_RESULTS_DIR="results/${DATASET}_${METHOD}_${RETRIEVER}"
PYTHON_BIN="${STOP_RAG_PYTHON:-python}"

if [ ! -d "$TRAIN_OUTPUT_DIR" ]; then
    echo "Training output directory not found: $TRAIN_OUTPUT_DIR" >&2
    exit 1
fi

COMPUTE_SCORES_DIR="${TEST_RESULTS_DIR}/compute_scores"
mkdir -p "${COMPUTE_SCORES_DIR}/eval"
mkdir -p "${COMPUTE_SCORES_DIR}/test"
SD_MAX_LENGTH="${STOP_RAG_SD_MAX_LENGTH:-2048}"
# 单卡跑 deberta-v3-large 评分时 batch 过大易 OOM（device_map=auto 已弃用以避免多卡碎片）
SCORE_BATCH="${STOP_RAG_COMPUTE_SCORES_BATCH_SIZE:-8}"

EVAL_TRACES_PATH="${BASE_DATA_DIR}/eval_subsampled_traces/eval_subsampled_partial_traces_labeled.jsonl"
TEST_TRACES_PATH="${BASE_DATA_DIR}/test_subsampled_traces/test_subsampled_partial_traces_labeled.jsonl"

for ckpt_path in "${TRAIN_OUTPUT_DIR}"/checkpoint-*; do
    if [ ! -d "$ckpt_path" ]; then
        continue
    fi

    ckpt=$(basename "$ckpt_path" | cut -d'-' -f2)

    if [ -f "$EVAL_TRACES_PATH" ]; then
        "${PYTHON_BIN}" -m src.test.stop_rag_compute_scores \
            --input-path "$EVAL_TRACES_PATH" \
            --output-path "${COMPUTE_SCORES_DIR}/eval/${DATASET}_eval_ckpt${ckpt}.jsonl" \
            --checkpoint-path "$ckpt_path" \
            --max-length "$SD_MAX_LENGTH" \
            --batch-size "$SCORE_BATCH" \
            --bf16
    fi

    if [ -f "$TEST_TRACES_PATH" ]; then
        "${PYTHON_BIN}" -m src.test.stop_rag_compute_scores \
            --input-path "$TEST_TRACES_PATH" \
            --output-path "${COMPUTE_SCORES_DIR}/test/${DATASET}_test_ckpt${ckpt}.jsonl" \
            --checkpoint-path "$ckpt_path" \
            --max-length "$SD_MAX_LENGTH" \
            --batch-size "$SCORE_BATCH" \
            --bf16
    fi
done

"${PYTHON_BIN}" -m src.test.stop_rag_find_best \
    --input-dir "${COMPUTE_SCORES_DIR}/eval" \
    --thresholds -0.20 -0.19 -0.18 -0.17 -0.16 -0.15 -0.14 -0.13 -0.12 -0.11 -0.10 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02 -0.01 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 \
    --target-label "f1"
