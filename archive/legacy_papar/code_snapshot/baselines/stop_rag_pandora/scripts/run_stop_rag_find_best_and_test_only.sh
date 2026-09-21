#!/usr/bin/env bash
# 训练已全部完成时：仅 find_best → stop_rag_test（与 run_stop_rag_from_train_resume.sh 环境一致）
set -euo pipefail
SR=/home/x12dpg/hjx/Pandora-RAG/baselines/Stop-RAG
cd "$SR"
export HF_HOME=/home/x12dpg/hjx/Pandora-RAG/.hf_cache
export STOP_RAG_PYTHON=/home/x12dpg/miniconda3/envs/pandora-rag/bin/python
export CUDA_VISIBLE_DEVICES=1,2,3
export STOP_RAG_VLLM_TP_SIZE=2
unset STOP_RAG_CONTRIEVER_DEVICE STOP_RAG_RERANKER_DEVICE 2>/dev/null || true
export STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION="${STOP_RAG_VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
export STOP_RAG_VLLM_MAX_MODEL_LEN=8192
export STOP_RAG_PIPELINE_BATCH_SIZE="${STOP_RAG_PIPELINE_BATCH_SIZE:-64}"
export LD_LIBRARY_PATH="/home/x12dpg/miniconda3/envs/pandora-rag/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export STOP_RAG_SD_MAX_LENGTH="${STOP_RAG_SD_MAX_LENGTH:-1024}"

METHOD=ours
RETRIEVER=contriever
LOG="${STOP_RAG_FIND_BEST_ONLY_LOG:-/home/x12dpg/hjx/Pandora-RAG/results/stop_rag_from_train_find_best_only_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== Stop-RAG 仅 find_best + test | $LOG ==="

CKPTS=()
THRS=()
for ds in hotpotqa 2wikimultihopqa musique; do
  echo "=== stop_rag_find_best.sh ${ds} ==="
  tmp=$(mktemp)
  set +e
  ./scripts/stop_rag_find_best.sh "$ds" "$RETRIEVER" "$METHOD" 2>&1 | tee "$tmp"
  ec=${PIPESTATUS[0]}
  set -e
  if [ "$ec" -ne 0 ]; then
    echo "find_best failed for ${ds} exit=${ec}" >&2
    rm -f "$tmp"
    exit "$ec"
  fi
  line=$(grep "Best threshold:" "$tmp" | tail -1)
  read -r ckpt thr < <("$STOP_RAG_PYTHON" -c "
import re, sys
m = re.search(r\"ckpt(\\d+)\\.jsonl'\\s+([-0-9.]+)\", sys.argv[1])
if not m:
    sys.exit('parse failed: ' + sys.argv[1])
print(m.group(1), m.group(2))
" "$line")
  CKPTS+=("$ckpt")
  THRS+=("$thr")
  rm -f "$tmp"
  echo "picked ds=${ds} ckpt=${ckpt} threshold=${thr}"
done

CKPT_JOIN=$(IFS=,; echo "${CKPTS[*]}")
THR_JOIN=$(IFS=,; echo "${THRS[*]}")

echo "=== stop_rag_test.sh ckpts=${CKPT_JOIN} thrs=${THR_JOIN} ==="
./scripts/stop_rag_test.sh "hotpotqa,2wikimultihopqa,musique" "$RETRIEVER" "$METHOD" "$CKPT_JOIN" "$THR_JOIN"

echo "=== ALL DONE ==="
