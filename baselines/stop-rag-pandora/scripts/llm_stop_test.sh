#!/usr/bin/env bash

set -eu

# shellcheck source=pandora_resolve_vllm.sh
. "$(cd "$(dirname "$0")" && pwd)/pandora_resolve_vllm.sh"

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
TEST_TRACES_PATH="${BASE_DATA_DIR}/test_subsampled_traces/test_subsampled_partial_traces_labeled.jsonl"
PYTHON_BIN="${STOP_RAG_PYTHON:-python}"
VLLM_MODEL_ID="${STOP_RAG_VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
VLLM_TP_SIZE="${STOP_RAG_VLLM_TP_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_MAX_MODEL_LEN="${STOP_RAG_VLLM_MAX_MODEL_LEN:-2048}"

if [ ! -f "$TEST_TRACES_PATH" ]; then
    echo "Test traces not found: $TEST_TRACES_PATH" >&2
    exit 1
fi

"${PYTHON_BIN}" -m src.test.llm_stop_test \
    --input-path "$TEST_TRACES_PATH" \
    --vllm-model-id "$VLLM_MODEL_ID" \
    --vllm-tp-size "$VLLM_TP_SIZE" \
    --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
