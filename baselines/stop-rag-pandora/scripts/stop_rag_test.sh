#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Prefer Pandora repo-local Llama when STOP_RAG_VLLM_MODEL is unset.
# shellcheck source=pandora_resolve_vllm.sh
. "${SCRIPT_DIR}/pandora_resolve_vllm.sh"

DATASET_SPEC="${1:-}"
RETRIEVER="${2:-}"
METHOD="${3:-}"
CKPT_SPEC="${4:-}"
THRESHOLD_SPEC="${5:-}"

usage() {
    echo "Usage: $0 <dataset|dataset1,dataset2|all> <retriever> <method> <ckpt|ckpt1,ckpt2> <threshold|thr1,thr2>" >&2
}

split_csv() {
    local raw="${1// /}"
    local -n out_ref=$2
    IFS=',' read -r -a out_ref <<< "$raw"
}

if [ -z "$DATASET_SPEC" ] || [ -z "$RETRIEVER" ] || [ -z "$METHOD" ] || [ -z "$CKPT_SPEC" ] || [ -z "$THRESHOLD_SPEC" ]; then
    usage
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

DATASETS=()
if [ "$DATASET_SPEC" = "all" ]; then
    DATASETS=(hotpotqa 2wikimultihopqa musique)
else
    split_csv "$DATASET_SPEC" DATASETS
fi

for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" != "hotpotqa" ] && [ "$DATASET" != "2wikimultihopqa" ] && [ "$DATASET" != "musique" ]; then
        echo "Invalid dataset. Use 'hotpotqa', '2wikimultihopqa', or 'musique'." >&2
        exit 1
    fi
done

CKPTS=()
THRESHOLDS=()
split_csv "$CKPT_SPEC" CKPTS
split_csv "$THRESHOLD_SPEC" THRESHOLDS

if [ "${#CKPTS[@]}" -ne 1 ] && [ "${#CKPTS[@]}" -ne "${#DATASETS[@]}" ]; then
    echo "Checkpoint count must be 1 or match dataset count (${#DATASETS[@]})." >&2
    exit 1
fi

if [ "${#THRESHOLDS[@]}" -ne 1 ] && [ "${#THRESHOLDS[@]}" -ne "${#DATASETS[@]}" ]; then
    echo "Threshold count must be 1 or match dataset count (${#DATASETS[@]})." >&2
    exit 1
fi

run_one() {
    local DATASET="$1"
    local CKPT="$2"
    local THRESHOLD="$3"

    PYTHON_BIN="${STOP_RAG_PYTHON:-python}"
    TRAIN_OUTPUT_DIR="outputs/${DATASET}_${METHOD}_${RETRIEVER}"
    CKPT_PATH="${TRAIN_OUTPUT_DIR}/checkpoint-${CKPT}"
    TEST_RESULTS_DIR="results/${DATASET}_${METHOD}_${RETRIEVER}"
    ONLINE_TEST_DIR="${TEST_RESULTS_DIR}/online_test"
    TRACES_PATH="${ONLINE_TEST_DIR}/${DATASET}_test_ckpt${CKPT}_thr${THRESHOLD}.jsonl"
    OUTPUT_PATH="${ONLINE_TEST_DIR}/${DATASET}_test_ckpt${CKPT}_thr${THRESHOLD}.metrics.json"
    STOP_LOG_PATH="${ONLINE_TEST_DIR}/${DATASET}_test_ckpt${CKPT}_thr${THRESHOLD}.stop_log.jsonl"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "Checkpoint directory not found: $CKPT_PATH" >&2
        exit 1
    fi

    VLLM_MODEL_ID="${STOP_RAG_VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

    _pandora_models="$(cd "${SCRIPT_DIR}/../../.." && pwd)/models"
    _local_reranker="${_pandora_models}/bge-reranker-v2-m3"
    _local_contriever="${_pandora_models}/contriever-msmarco"
    if [ -z "${STOP_RAG_RERANKER_MODEL:-}" ] && [ -f "${_local_reranker}/config.json" ]; then
        STOP_RAG_RERANKER_MODEL="${_local_reranker}"
    fi
    if [ -z "${STOP_RAG_CONTRIEVER_MODEL:-}" ] && [ -f "${_local_contriever}/config.json" ]; then
        STOP_RAG_CONTRIEVER_MODEL="${_local_contriever}"
    fi
    unset _pandora_models _local_reranker _local_contriever

    RERANKER_MODEL_ID="${STOP_RAG_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
    CONTRIEVER_MODEL_PATH="${STOP_RAG_CONTRIEVER_MODEL:-facebook/contriever-msmarco}"
    VLLM_TP_SIZE="${STOP_RAG_VLLM_TP_SIZE:-1}"
    VLLM_GPU_MEMORY_UTILIZATION="${STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
    VLLM_MAX_MODEL_LEN="${STOP_RAG_VLLM_MAX_MODEL_LEN:-8192}"
    VLLM_ENFORCE_EAGER_ARGS=()
    if [ "${STOP_RAG_VLLM_ENFORCE_EAGER:-true}" != "false" ]; then VLLM_ENFORCE_EAGER_ARGS=(--vllm-enforce-eager); fi

    # 与 dataset.sh 一致：vLLM 占 cuda:0..(TP-1)；Contriever / Reranker 在后续逻辑卡（TP=2 且可见 3 卡时常为同一张 cuda:2）
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        _num_gpus="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
        _num_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
    if [ -z "${_num_gpus}" ] || [ "${_num_gpus}" -lt 1 ]; then
        _num_gpus=1
    fi
    if [ "${VLLM_TP_SIZE}" -ge "${_num_gpus}" ]; then
        echo "stop_rag_test.sh: STOP_RAG_VLLM_TP_SIZE=${VLLM_TP_SIZE} 必须小于可见 GPU 数（当前 ${_num_gpus}），否则没有空闲卡加载 Contriever/Reranker。" >&2
        exit 1
    fi
    _free_gpu="${VLLM_TP_SIZE}"
    if [ "${VLLM_TP_SIZE}" = "1" ]; then
        export STOP_RAG_CONTRIEVER_DEVICE="${STOP_RAG_CONTRIEVER_DEVICE:-cuda:1}"
        export STOP_RAG_RERANKER_DEVICE="${STOP_RAG_RERANKER_DEVICE:-cuda:2}"
    else
        export STOP_RAG_CONTRIEVER_DEVICE="${STOP_RAG_CONTRIEVER_DEVICE:-cuda:${_free_gpu}}"
        _rerank_idx=$((_free_gpu + 1))
        if [ "${_rerank_idx}" -ge "${_num_gpus}" ]; then
            export STOP_RAG_RERANKER_DEVICE="${STOP_RAG_RERANKER_DEVICE:-cuda:${_free_gpu}}"
        else
            export STOP_RAG_RERANKER_DEVICE="${STOP_RAG_RERANKER_DEVICE:-cuda:${_rerank_idx}}"
        fi
    fi
    export STOP_RAG_SD_DEVICE="${STOP_RAG_SD_DEVICE:-$STOP_RAG_CONTRIEVER_DEVICE}"

    PIPELINE_BATCH_SIZE="${STOP_RAG_PIPELINE_BATCH_SIZE:-128}"
    SEARCH_BATCH_SIZE="${STOP_RAG_SEARCH_BATCH_SIZE:-16}"
    STOP_RAG_MAX_ITERATIONS="${STOP_RAG_MAX_ITERATIONS:-5}"
    STOP_RAG_SD_MAX_LENGTH="${STOP_RAG_SD_MAX_LENGTH:-2048}"
    EXTRA_PIPELINE_ARGS=()
    if [ "${STOP_RAG_RERANKER_DISABLE:-false}" = "true" ]; then
        EXTRA_PIPELINE_ARGS+=(--reranker-disable)
    fi
    if [ "${STOP_RAG_SD_BF16:-true}" = "true" ]; then
        EXTRA_PIPELINE_ARGS+=(--sd-bf16)
    fi

    mkdir -p "$ONLINE_TEST_DIR"

    echo "Running online stop-rag test for dataset=${DATASET}, retriever=${RETRIEVER}, method=${METHOD}, ckpt=${CKPT}, threshold=${THRESHOLD}"

    if [ "$METHOD" = "ours" ]; then
        "${PYTHON_BIN}" -m src.pipeline.pipeline \
            --dataset "$DATASET" \
            --dataset-type "test_subsampled" \
            --retriever-type "$RETRIEVER" \
            --sd-provider "multihead" \
            --sd-checkpoint-path "$CKPT_PATH" \
            --sd-threshold "$THRESHOLD" \
            --sd-max-length "$STOP_RAG_SD_MAX_LENGTH" \
            --traces-path "$TRACES_PATH" \
            --stop-log-path "$STOP_LOG_PATH" \
            --output-path "$OUTPUT_PATH" \
            --vllm-model-id "$VLLM_MODEL_ID" \
            --vllm-tp-size "$VLLM_TP_SIZE" \
            --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
            "${VLLM_ENFORCE_EAGER_ARGS[@]}" \
            --reranker-model-id "$RERANKER_MODEL_ID" \
            --contriever-model-path "$CONTRIEVER_MODEL_PATH" \
            --batch-size "$PIPELINE_BATCH_SIZE" \
            --search-batch-size "$SEARCH_BATCH_SIZE" \
            --max-iterations "$STOP_RAG_MAX_ITERATIONS" \
            "${EXTRA_PIPELINE_ARGS[@]}"
    else
        "${PYTHON_BIN}" -m src.baselines.corag.pipeline \
            --dataset "$DATASET" \
            --dataset-type "test_subsampled" \
            --retriever-type "$RETRIEVER" \
            --sd-provider "multihead" \
            --sd-checkpoint-path "$CKPT_PATH" \
            --sd-threshold "$THRESHOLD" \
            --sd-max-length "$STOP_RAG_SD_MAX_LENGTH" \
            --traces-path "$TRACES_PATH" \
            --output-path "$OUTPUT_PATH" \
            --vllm-model-id "$VLLM_MODEL_ID" \
            --vllm-tp-size "$VLLM_TP_SIZE" \
            --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
            "${VLLM_ENFORCE_EAGER_ARGS[@]}" \
            --reranker-model-id "$RERANKER_MODEL_ID" \
            --contriever-model-path "$CONTRIEVER_MODEL_PATH" \
            --batch-size "$PIPELINE_BATCH_SIZE" \
            --search-batch-size "$SEARCH_BATCH_SIZE" \
            --max-iterations "$STOP_RAG_MAX_ITERATIONS" \
            "${EXTRA_PIPELINE_ARGS[@]}"
    fi

    echo "Online stop-rag test finished for ${DATASET}."
    echo "Metrics saved to: $OUTPUT_PATH"
    echo "Traces saved to: $TRACES_PATH"
}

for i in "${!DATASETS[@]}"; do
    CKPT_INDEX=0
    THRESHOLD_INDEX=0
    if [ "${#CKPTS[@]}" -gt 1 ]; then
        CKPT_INDEX=$i
    fi
    if [ "${#THRESHOLDS[@]}" -gt 1 ]; then
        THRESHOLD_INDEX=$i
    fi

    run_one "${DATASETS[$i]}" "${CKPTS[$CKPT_INDEX]}" "${THRESHOLDS[$THRESHOLD_INDEX]}"
done
