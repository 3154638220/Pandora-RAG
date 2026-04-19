#!/usr/bin/env bash

set -eu

# Prefer Pandora repo-local Llama when STOP_RAG_VLLM_MODEL is unset.
# shellcheck source=pandora_resolve_vllm.sh
. "$(cd "$(dirname "$0")" && pwd)/pandora_resolve_vllm.sh"

# 与 Pandora 主实验对齐：检索器选 contriever（facebook/contriever-msmarco，见 STOP_RAG_CONTRIEVER_MODEL），
# 精排默认开启 BAAI/bge-reranker-v2-m3（见 STOP_RAG_RERANKER_MODEL）；仅调试用 STOP_RAG_RERANKER_DISABLE=true。

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

BASE_TRACES_DIR="data/processed/${DATASET}/${METHOD}/${RETRIEVER}"
ICL_EXAMPLES_PATH="src/pipeline/ircot_prompts/${DATASET}/gold_with_3_distractors_context_cot_qa_flan_t5.txt"
PYTHON_BIN="${STOP_RAG_PYTHON:-python}"
VLLM_MODEL_ID="${STOP_RAG_VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# 优先使用 Pandora 仓库本地模型目录，避免从 HuggingFace 下载
_scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_pandora_models="$(cd "${_scripts_dir}/../../.." && pwd)/models"
_local_reranker="${_pandora_models}/bge-reranker-v2-m3"
_local_contriever="${_pandora_models}/contriever-msmarco"
if [ -z "${STOP_RAG_RERANKER_MODEL:-}" ] && [ -f "${_local_reranker}/config.json" ]; then
    STOP_RAG_RERANKER_MODEL="${_local_reranker}"
fi
if [ -z "${STOP_RAG_CONTRIEVER_MODEL:-}" ] && [ -f "${_local_contriever}/config.json" ]; then
    STOP_RAG_CONTRIEVER_MODEL="${_local_contriever}"
fi
unset _scripts_dir _pandora_models _local_reranker _local_contriever

RERANKER_MODEL_ID="${STOP_RAG_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
CONTRIEVER_MODEL_PATH="${STOP_RAG_CONTRIEVER_MODEL:-facebook/contriever-msmarco}"
VLLM_TP_SIZE="${STOP_RAG_VLLM_TP_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
# max_model_len=8192：prefix cache 命中后每请求只需 prefill ~850 tokens（history+Q），并发大幅提升。
# 默认开启 enforce_eager（禁用 CUDA graph / torch.compile 捕获），避免多进程 worker 在部分机器上 OOM 或长时间卡编译。
VLLM_MAX_MODEL_LEN="${STOP_RAG_VLLM_MAX_MODEL_LEN:-8192}"
VLLM_ENFORCE_EAGER_ARGS=()
if [ "${STOP_RAG_VLLM_ENFORCE_EAGER:-true}" != "false" ]; then VLLM_ENFORCE_EAGER_ARGS=(--vllm-enforce-eager)
fi
# vLLM 占用 cuda:0 .. cuda:(TP-1)；Contriever/Reranker 需放到其余卡，否则会与 TP 分片抢显存 OOM。
# TP=1：默认检索 cuda:1、精排 cuda:2；TP>=2：Contriever 用 cuda:TP，Reranker 尝试 cuda:TP+1，
#   若可见 GPU 不够则退回与 Contriever 共享同一卡（两者顺序执行，不会并发抢显存）。
# 用 CUDA_VISIBLE_DEVICES 中的数量（而非物理 GPU 总数）判断可见 GPU 数，避免逻辑设备序号越界。
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _num_gpus="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
else
    _num_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
if [ -z "${_num_gpus}" ] || [ "${_num_gpus}" -lt 1 ]; then
    _num_gpus=1
fi
if [ "${VLLM_TP_SIZE}" -ge "${_num_gpus}" ]; then
    echo "stop-rag dataset.sh: STOP_RAG_VLLM_TP_SIZE=${VLLM_TP_SIZE} 必须小于可见 GPU 数（当前 ${_num_gpus}），否则没有空闲卡加载 Contriever/Reranker；请降低 TP 或改用 CPU检索（STOP_RAG_CONTRIEVER_DEVICE=cpu）。" >&2
    exit 1
fi
# Llama-3.1-8B 等模型的 head 数须能被 TP 整除（32 可整除 1/2/4/8，不能 3）。
_free_gpu="${VLLM_TP_SIZE}"
if [ "${VLLM_TP_SIZE}" = "1" ]; then
    export STOP_RAG_CONTRIEVER_DEVICE="${STOP_RAG_CONTRIEVER_DEVICE:-cuda:1}"
    export STOP_RAG_RERANKER_DEVICE="${STOP_RAG_RERANKER_DEVICE:-cuda:2}"
else
    export STOP_RAG_CONTRIEVER_DEVICE="${STOP_RAG_CONTRIEVER_DEVICE:-cuda:${_free_gpu}}"
    _rerank_idx=$((_free_gpu + 1))
    if [ "${_rerank_idx}" -ge "${_num_gpus}" ]; then
        echo "stop-rag dataset.sh: 可见 GPU 数（${_num_gpus}）不足以为 Reranker 分配独立卡，Reranker 与 Contriever 共享 cuda:${_free_gpu}（两者顺序执行，不影响正确性）。如需独立卡请增加 CUDA_VISIBLE_DEVICES 或手动指定 STOP_RAG_RERANKER_DEVICE。" >&2
        export STOP_RAG_RERANKER_DEVICE="${STOP_RAG_RERANKER_DEVICE:-cuda:${_free_gpu}}"
    else
        export STOP_RAG_RERANKER_DEVICE="${STOP_RAG_RERANKER_DEVICE:-cuda:${_rerank_idx}}"
    fi
fi
PIPELINE_BATCH_SIZE="${STOP_RAG_PIPELINE_BATCH_SIZE:-128}"
SEARCH_BATCH_SIZE="${STOP_RAG_SEARCH_BATCH_SIZE:-16}"
# 与 Pandora / README 对齐：partial trace 步数与 F1 估计重复次数
STOP_RAG_MAX_ITERATIONS="${STOP_RAG_MAX_ITERATIONS:-5}"
STOP_RAG_REPEAT_SIZE="${STOP_RAG_REPEAT_SIZE:-1}"
EXTRA_PIPELINE_ARGS=()
if [ "${STOP_RAG_RERANKER_DISABLE:-false}" = "true" ]; then
    EXTRA_PIPELINE_ARGS+=(--reranker-disable)
fi

DATASET_TYPES="${STOP_RAG_DATASET_TYPES:-train eval_subsampled test_subsampled}"

for DATASET_TYPE in $DATASET_TYPES; do
    TRACES_BASE_PATH="${BASE_TRACES_DIR}/${DATASET_TYPE}_traces"
    TRACES_FILE="${TRACES_BASE_PATH}/${DATASET_TYPE}_traces.jsonl"
    
    mkdir -p "$TRACES_BASE_PATH"

    if [ "$METHOD" = "ours" ]; then

        "${PYTHON_BIN}" -m src.pipeline.pipeline \
            --dataset "$DATASET" \
            --dataset-type "$DATASET_TYPE" \
            --retriever-type "$RETRIEVER" \
            --sd-provider "nostop" \
            --traces-path "$TRACES_FILE" \
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

        PARTIAL_TRACES_FILE="${TRACES_BASE_PATH}/${DATASET_TYPE}_partial_traces.jsonl"
        "${PYTHON_BIN}" -m src.train.preprocessing.extract_partial_traces \
            --input-path "$TRACES_FILE" \
            --output-path "$PARTIAL_TRACES_FILE" \
            --max-iterations "$STOP_RAG_MAX_ITERATIONS"

        LABELED_TRACES_FILE="${TRACES_BASE_PATH}/${DATASET_TYPE}_partial_traces_labeled.jsonl"
        "${PYTHON_BIN}" -m src.train.preprocessing.compute_metrics \
            --input-path "$PARTIAL_TRACES_FILE" \
            --output-path "$LABELED_TRACES_FILE" \
            --repeat-size "$STOP_RAG_REPEAT_SIZE" \
            --icl-examples-path "$ICL_EXAMPLES_PATH" \
            --vllm-model-id "$VLLM_MODEL_ID" \
            --vllm-tp-size "$VLLM_TP_SIZE" \
            --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
            "${VLLM_ENFORCE_EAGER_ARGS[@]}"

        "${PYTHON_BIN}" -m src.train.preprocessing.compute_labels \
            --input-path "$LABELED_TRACES_FILE" \
            --output-path "$LABELED_TRACES_FILE" \
            --target-label "f1" \
            --max-iterations "$STOP_RAG_MAX_ITERATIONS"

    elif [ "$METHOD" = "corag" ]; then
        "${PYTHON_BIN}" -m src.baselines.corag.pipeline \
            --dataset "$DATASET" \
            --dataset-type "$DATASET_TYPE" \
            --retriever-type "$RETRIEVER" \
            --traces-path "$TRACES_FILE" \
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

        CORAG_PARTIAL_TRACES_FILE="${TRACES_BASE_PATH}/${DATASET_TYPE}_partial_traces.jsonl"
        "${PYTHON_BIN}" -m src.baselines.corag.extract_partial_traces \
            --input-path "$TRACES_FILE" \
            --output-path "$CORAG_PARTIAL_TRACES_FILE" \
            --max-iterations "$STOP_RAG_MAX_ITERATIONS"

        CORAG_LABELED_FILE="${TRACES_BASE_PATH}/${DATASET_TYPE}_partial_traces_labeled.jsonl"
        "${PYTHON_BIN}" -m src.baselines.corag.compute_metrics \
            --input-path "$CORAG_PARTIAL_TRACES_FILE" \
            --output-path "$CORAG_LABELED_FILE" \
            --repeat-size "$STOP_RAG_REPEAT_SIZE" \
            --vllm-model-id "$VLLM_MODEL_ID" \
            --vllm-tp-size "$VLLM_TP_SIZE" \
            --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
            "${VLLM_ENFORCE_EAGER_ARGS[@]}"

        "${PYTHON_BIN}" -m src.train.preprocessing.compute_labels \
            --input-path "$CORAG_LABELED_FILE" \
            --output-path "$CORAG_LABELED_FILE" \
            --target-label "f1" \
            --max-iterations "$STOP_RAG_MAX_ITERATIONS"
    fi
done

SPLIT_DATA_DIR="${BASE_TRACES_DIR}/train"
mkdir -p "$SPLIT_DATA_DIR"

TRAIN_LABELED_FILE="${BASE_TRACES_DIR}/train_traces/train_partial_traces_labeled.jsonl"
"${PYTHON_BIN}" -m src.train.preprocessing.split_data \
    --input-path "$TRAIN_LABELED_FILE" \
    --output-dir "$SPLIT_DATA_DIR"
