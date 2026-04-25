# Paper Code Bundle

This directory is a cleaned code bundle for writing and packaging the Pandora-RAG paper on another machine.
It keeps the method code, analysis utilities, and the aligned Stop-RAG baseline, while leaving out caches, checkpoints, datasets, logs, and result dumps.

## Layout

- `pandora_rag/`: main Pandora-RAG source code used by the paper.
- `analysis_tools/`: helper scripts for table export, significance tests, Pareto aggregation, and cost accounting.
- `baselines/stop_rag_pandora/`: the repo-aligned Stop-RAG baseline code used for head-to-head comparison.
- `requirements.txt`: lightweight dependency list for the Pandora-RAG main pipeline.
- `.env.example`: environment variable template copied from the main repo.

## Recommended Starting Points

- Main method code: `pandora_rag/stage1/run_stage1.py`
- Probe training: `pandora_rag/stage2/run_stage2.py`
- E-value evaluation: `pandora_rag/stage3/run_stage3.py`
- Paper analysis scripts: `analysis_tools/stage3_export_tables.py`, `analysis_tools/stage3_significance.py`
- Stop-RAG comparison code: `baselines/stop_rag_pandora/README.md`

## Notes

- Run Pandora-RAG commands from inside `pandora_rag/` so sibling imports like `pretest.*` and `qa_shared.*` work as expected.
- This bundle is organized for paper writing and code appendix use, not for one-click full reproduction.
- Large artifacts intentionally remain in the original repository only.
