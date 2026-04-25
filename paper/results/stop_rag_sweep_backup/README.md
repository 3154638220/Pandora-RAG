# Stop-RAG Sweep Backup

Created on 2026-04-20 to preserve the Stop-RAG true-online threshold sweep
artifacts inside the main `results/` tree before large Stop-RAG checkpoints are
cleaned.

Contents:

- `raw_online_test/`: copied from `baselines/stop-rag-pandora/results/*/online_test/`.
  It contains the raw true-online `jsonl`, `metrics.json`, and `stop_log.jsonl`
  files for the aligned Stop-RAG threshold sweep.
- `summary/`: copied summary CSVs used for paper tables and matched-budget
  comparisons.
- `figures/`: copied F1/EM Pareto plots for HotpotQA, MuSiQue, and 2Wiki.
- `docs/`: copied generated sweep report.

This backup is enough to regenerate the Stop-RAG sweep tables and figures from
existing outputs. It intentionally does not include Stop-RAG model checkpoints;
rerunning online tests still requires the corresponding checkpoint directories.
