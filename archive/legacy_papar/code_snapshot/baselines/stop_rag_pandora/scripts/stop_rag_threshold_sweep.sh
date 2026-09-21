#!/usr/bin/env bash

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET_SPEC="${1:-}"
RETRIEVER="${2:-}"
METHOD="${3:-}"
CKPT_SPEC="${4:-}"
THRESHOLD_SPEC="${5:-}"

usage() {
    echo "Usage: $0 <dataset|dataset1,dataset2|all> <retriever> <method> <ckpt|ckpt1,ckpt2> <thr1,thr2,...>" >&2
    echo "Example: $0 hotpotqa contriever ours 1000 '-0.20,-0.16,-0.12,-0.09,-0.06,-0.03,0.00,0.03,0.06'" >&2
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

DATASETS=()
if [ "$DATASET_SPEC" = "all" ]; then
    DATASETS=(hotpotqa 2wikimultihopqa musique)
else
    split_csv "$DATASET_SPEC" DATASETS
fi

CKPTS=()
THRESHOLDS=()
split_csv "$CKPT_SPEC" CKPTS
split_csv "$THRESHOLD_SPEC" THRESHOLDS

if [ "${#CKPTS[@]}" -ne 1 ] && [ "${#CKPTS[@]}" -ne "${#DATASETS[@]}" ]; then
    echo "Checkpoint count must be 1 or match dataset count (${#DATASETS[@]})." >&2
    exit 1
fi

echo "Running true online Stop-RAG threshold sweep."
echo "Datasets: ${DATASETS[*]}"
echo "Thresholds: ${THRESHOLDS[*]}"

for i in "${!DATASETS[@]}"; do
    CKPT_INDEX=0
    if [ "${#CKPTS[@]}" -gt 1 ]; then
        CKPT_INDEX=$i
    fi

    for THRESHOLD in "${THRESHOLDS[@]}"; do
        "${SCRIPT_DIR}/stop_rag_test.sh" \
            "${DATASETS[$i]}" \
            "$RETRIEVER" \
            "$METHOD" \
            "${CKPTS[$CKPT_INDEX]}" \
            "$THRESHOLD"
    done
done

echo "Sweep finished. Summarize from the Pandora repo root with:"
echo "  python scripts/stop_rag_make_pareto.py"
