# Pandora-RAG Source Map

This folder contains the cleaned core code for the paper.

## Main Entry Points

- `stage1/run_stage1.py`: data preparation, trajectory collection, oracle labels, and Stage 1 reporting.
- `stage2/run_stage2.py`: probe training, threshold tuning, and Stage 2 evaluation.
- `stage2/run_feature_diagnostic.py`: feature diagnostics and ablations.
- `stage2/run_deployable_gw_eval.py`: deployable Weitzman evaluation.
- `stage3/run_stage3.py`: E-value monitoring, CP comparison, and Stage 3 plots/reports.

## Shared Modules

- `pretest/hf_env.py`: Hugging Face cache and environment setup.
- `pretest/utils/llm_client.py`: OpenAI-compatible LLM client and feature extraction.
- `pretest/utils/retriever.py`: BM25 and Contriever+BGE retrieval backends.
- `pretest/utils/weitzman.py`: Pandora / DP oracle / reservation-value utilities.
- `qa_shared/metrics.py`: QA evaluation metrics.
- `qa_shared/prompts.py`: prompt templates shared by the iterative RAG pipeline.

## Running Hint

Use `python -m stage1.run_stage1`, `python -m stage2.run_stage2`, and `python -m stage3.run_stage3` from this directory.
