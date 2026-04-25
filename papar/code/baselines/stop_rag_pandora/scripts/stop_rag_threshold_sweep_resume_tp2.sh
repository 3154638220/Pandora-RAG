#!/usr/bin/env bash
# 从暂停处续跑 Stop-RAG 阈值扫描：vLLM 使用 TP=2，环境与本仓库 `run_stop_rag_find_best_and_test_only.sh` 对齐。
# 默认续跑范围（与此前 `all` + 全阈值扫描衔接）：
#   1) HotpotQA：仅剩余 0.03, 0.06, 0.10（假设 -0.20…0.00 已跑完）
#   2) 2Wiki：完整 10 个阈值
#   3) MuSiQue：完整 10 个阈值
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# 勿用外层已 export 的 HF_HOME（可能指向不可用路径）；显式指定：STOP_RAG_HF_HOME=/path
export HF_HOME="${STOP_RAG_HF_HOME:-${REPO_ROOT}/.hf_cache}"
# 四卡：TP=2 占逻辑 cuda:0–1，Contriever→cuda:2，Reranker→cuda:3（与 stop_rag_test.sh 默认一致）。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export STOP_RAG_VLLM_TP_SIZE="${STOP_RAG_VLLM_TP_SIZE:-2}"
unset STOP_RAG_CONTRIEVER_DEVICE STOP_RAG_RERANKER_DEVICE 2>/dev/null || true

export STOP_RAG_PYTHON="${STOP_RAG_PYTHON:-/home/x12dpg/miniconda3/envs/pandora-rag/bin/python}"
export LD_LIBRARY_PATH="/home/x12dpg/miniconda3/envs/pandora-rag/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export STOP_RAG_VLLM_MODEL="${STOP_RAG_VLLM_MODEL:-${REPO_ROOT}/models/Meta-Llama-3.1-8B-Instruct}"
export STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION="${STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
export STOP_RAG_VLLM_MAX_MODEL_LEN="${STOP_RAG_VLLM_MAX_MODEL_LEN:-8192}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

THR_FULL="${THR_FULL:--0.20,-0.16,-0.12,-0.09,-0.06,-0.03,0.00,0.03,0.06,0.10}"
HOTPOT_THR_REMAINING="${HOTPOT_THR_REMAINING:-0.03,0.06,0.10}"

PARTIAL="${PROJECT_ROOT}/results/hotpotqa_ours_contriever/online_test/hotpotqa_test_ckpt1000_thr0.03"
if [ -f "${PARTIAL}.jsonl" ] || [ -f "${PARTIAL}.stop_log.jsonl" ] || [ -f "${PARTIAL}.metrics.json" ]; then
  echo "Removing partial HotpotQA thr=0.03 artifacts before rerun..."
  rm -f "${PARTIAL}.jsonl" "${PARTIAL}.stop_log.jsonl" "${PARTIAL}.metrics.json"
fi

echo "=== Resume 1/3: HotpotQA (ckpt 1000), thresholds ${HOTPOT_THR_REMAINING} ==="
"${SCRIPT_DIR}/stop_rag_threshold_sweep.sh" hotpotqa contriever ours 1000 "${HOTPOT_THR_REMAINING}"

echo "=== Resume 2/3: 2WikiMultiHopQA (ckpt 2400), full grid ==="
"${SCRIPT_DIR}/stop_rag_threshold_sweep.sh" 2wikimultihopqa contriever ours 2400 "${THR_FULL}"

echo "=== Resume 3/3: MuSiQue (ckpt 1200), full grid ==="
"${SCRIPT_DIR}/stop_rag_threshold_sweep.sh" musique contriever ours 1200 "${THR_FULL}"

echo "Sweep finished. Summarize from Pandora repo root with:"
echo "  python scripts/stop_rag_make_pareto.py"
