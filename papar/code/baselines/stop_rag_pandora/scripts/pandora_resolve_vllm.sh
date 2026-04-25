#!/usr/bin/env bash
# When STOP_RAG_VLLM_MODEL is unset, use Pandora-RAG repo-local Llama if present
# (same layout as Stage1: <Pandora-RAG>/models/Meta-Llama-3.1-8B-Instruct).

if [ -z "${STOP_RAG_VLLM_MODEL:-}" ]; then
  _sr_scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _sr_root="$(cd "${_sr_scripts_dir}/.." && pwd)"
  _pandora_root="$(cd "${_sr_root}/../.." && pwd)"
  _local_llama="${_pandora_root}/models/Meta-Llama-3.1-8B-Instruct"
  if [ -f "${_local_llama}/config.json" ]; then
    STOP_RAG_VLLM_MODEL="${_local_llama}"
    export STOP_RAG_VLLM_MODEL
  elif [ -f "${_sr_root}/models/Meta-Llama-3.1-8B-Instruct/config.json" ]; then
    STOP_RAG_VLLM_MODEL="${_sr_root}/models/Meta-Llama-3.1-8B-Instruct"
    export STOP_RAG_VLLM_MODEL
  fi
  unset _sr_scripts_dir _sr_root _pandora_root _local_llama
fi
