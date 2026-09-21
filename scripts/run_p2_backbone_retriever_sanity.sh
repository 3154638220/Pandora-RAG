#!/usr/bin/env bash
# P2: 小规模跨检索器 sanity check（bm25 vs Contriever+BGE），与主实验同生成器（默认 Llama-3.1-8B）。
# 使用互相隔离的 --root-dir，避免 cache/trajectories 与主 run 混用。
#
# 前置：vLLM 已起在 OPENAI_API_BASE（默认 http://127.0.0.1:8000/v1）；Contriever+BGE 需 GPU（RETRIEVER_DEVICE）。
#
# 用法：
#   chmod +x scripts/run_p2_backbone_retriever_sanity.sh
#   ./scripts/run_p2_backbone_retriever_sanity.sh
#
# 环境变量（可选）：
#   P2_REPO_ROOT        仓库根（默认：脚本所在目录的父目录）
#   P2_BASE             隔离运行根目录（默认：runs/p2_backbone_sanity）
#   P2_DATASETS         逗号分隔（默认：hotpotqa；可加 2wiki）
#   P2_TRAIN_QUOTA 等    小样本配额，默认 train=200 calib=80 dev=80 test=120
#   P2_COPY_DATA_FROM   若设为当前仓库根，则先 cp -a data 到各 root-dir，避免重复下数据（仍需各自 cache）
#   RETRIEVER_DEVICE    contriever_bge 用，默认 cuda:2
#   HIDDEN_STATE_DEVICE 逐步 hidden，默认 cuda:1（与 RETRIEVER_DEVICE 错开）

set -euo pipefail

P2_REPO_ROOT="${P2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${P2_REPO_ROOT}"

P2_BASE="${P2_BASE:-runs/p2_backbone_sanity}"
P2_DATASETS="${P2_DATASETS:-hotpotqa}"
TRAIN_Q="${P2_TRAIN_QUOTA:-200}"
CALIB_Q="${P2_CALIB_QUOTA:-80}"
DEV_Q="${P2_DEV_QUOTA:-80}"
TEST_Q="${P2_TEST_QUOTA:-120}"

ROOT_BM25="${P2_BASE}/bm25"
ROOT_DENSE="${P2_BASE}/contriever_bge"

if [[ -n "${P2_COPY_DATA_FROM:-}" ]]; then
  for R in "${ROOT_BM25}" "${ROOT_DENSE}"; do
    mkdir -p "${R}"
    if [[ ! -d "${R}/data/processed" ]]; then
      echo "Copying data from ${P2_COPY_DATA_FROM} -> ${R}/data"
      cp -a "${P2_COPY_DATA_FROM}/data" "${R}/"
    fi
  done
fi

export PYTHONPATH="${P2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# vLLM 建议独占一张物理 GPU；Contriever+BGE 与逐步 hidden 勿同卡（易 OOM）。
# 默认：检索 cuda:2，hidden cuda:1（若未显式设置）。
export RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda:2}"
export HIDDEN_STATE_DEVICE="${HIDDEN_STATE_DEVICE:-cuda:1}"

run_stage1() {
  local root_dir="$1"
  local backend="$2"
  echo "=== Stage1 root=${root_dir} retriever-backend=${backend} ==="
  python -m stage1.run_stage1 \
    --root-dir "${root_dir}" \
    --datasets "${P2_DATASETS}" \
    --train-quota "${TRAIN_Q}" \
    --calib-quota "${CALIB_Q}" \
    --dev-quota "${DEV_Q}" \
    --test-quota "${TEST_Q}" \
    --retriever-backend "${backend}" \
    --skip-self-eval \
    --collect-splits train,calib,dev,test
}

run_stage1 "${ROOT_BM25}" "bm25"
run_stage1 "${ROOT_DENSE}" "contriever_bge"

echo "=== Aggregate P2 table ==="
python scripts/p2_backbone_retriever_aggregate.py \
  --bm25-root "${ROOT_BM25}" \
  --dense-root "${ROOT_DENSE}" \
  --datasets "${P2_DATASETS}" \
  --out results/p2_backbone_retriever_sanity.csv

echo "Done. See results/p2_backbone_retriever_sanity.csv and docs/reports/baselines/p2_backbone_retriever_sanity.md"
