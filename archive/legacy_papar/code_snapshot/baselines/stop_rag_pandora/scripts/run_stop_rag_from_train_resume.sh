#!/usr/bin/env bash
# 数据已就绪时：train → find_best → stop_rag_test（三数据集）
# 单 GPU 训练（本机 GPU 无 P2P，多卡 NCCL 易挂）；vLLM 仍用 TP=2
# OOM 对策：batch=1 + grad_accum=16 + gradient checkpointing + expandable segments
set -euo pipefail
SR=/home/x12dpg/hjx/Pandora-RAG/baselines/Stop-RAG
cd "$SR"
export HF_HOME=/home/x12dpg/hjx/Pandora-RAG/.hf_cache
export STOP_RAG_PYTHON=/home/x12dpg/miniconda3/envs/pandora-rag/bin/python
export STOP_RAG_TORCHRUN=/home/x12dpg/miniconda3/envs/pandora-rag/bin/torchrun
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

export STOP_RAG_NPROC_PER_NODE=1
export STOP_RAG_TRAIN_BATCH_SIZE=1
export STOP_RAG_GRAD_ACCUM=16
# max_length=1024：注意力矩阵 O(seq²)，从 2048→1024 省 4 倍显存，彻底解决长序列 OOM
# 同时将推理/评分的 sd_max_length 对齐为 1024，避免训推分布偏移损害公平性
export STOP_RAG_MAX_LENGTH=1024
export STOP_RAG_SD_MAX_LENGTH=1024
# 勿开 gradient checkpointing：MultiheadModel 双头 + 自定义 loss 与 HF checkpoint 反向不兼容（double backward）
export STOP_RAG_GRADIENT_CHECKPOINTING=false
export STOP_RAG_DEEPSPEED_CONFIG=""

METHOD=ours
RETRIEVER=contriever
LOG=/home/x12dpg/hjx/Pandora-RAG/results/stop_rag_from_train_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stop-RAG 续跑 | 单卡 batch=1 gc=on tp=2 | $LOG ==="

for ds in hotpotqa 2wikimultihopqa musique; do
  td="data/processed/${ds}/${METHOD}/${RETRIEVER}/train"
  mkdir -p "$td"
  ln -sf train_partial_traces_labeled_train.jsonl "$td/train.jsonl"
  ln -sf train_partial_traces_labeled_eval.jsonl "$td/eval.jsonl"
done

# hotpotqa 已用 max_length=2048 成功完成，保留其 checkpoint，不重训
# 2wikimultihopqa / musique 改用 max_length=1024 从头训（删除残留 checkpoint 避免混用）
# 续跑 find_best/test 时设 STOP_RAG_FROM_TRAIN_RESUME_NO_CLEAN=1，避免误删已训好的 checkpoint
if [ "${STOP_RAG_FROM_TRAIN_RESUME_NO_CLEAN:-0}" = "1" ]; then
  echo "=== 跳过清理 outputs（STOP_RAG_FROM_TRAIN_RESUME_NO_CLEAN=1）==="
else
  echo "=== 清理 2wikimultihopqa/musique 不完整输出 ==="
  rm -rf "outputs/2wikimultihopqa_ours_contriever" "outputs/musique_ours_contriever"
fi

for ds in hotpotqa 2wikimultihopqa musique; do
  if [ "$ds" = "hotpotqa" ] && [ -d "outputs/hotpotqa_ours_contriever/checkpoint-2391" ]; then
    echo "=== train.sh hotpotqa：已完成，跳过 ==="
    continue
  fi
  echo "=== train.sh ${ds} ==="
  ./scripts/train.sh "$ds" "$RETRIEVER" "$METHOD"
done

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
