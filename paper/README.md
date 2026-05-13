# Paper Material Index

This directory is a lightweight bundle for writing the paper on another
machine. It intentionally excludes model weights, caches, raw datasets, and
long run logs.

## Start Here

- `source/Pandora-RAG.tex`: current LaTeX draft.
- `source/neurips_project_brief.md`: compact project framing for NeurIPS-style writing.
- `source/Nips-suggestion.md`: writing and positioning suggestions.
- `docs/stage3_results_summary.md`: concise Stage 3 result summary.
- `docs/stage3_narrative.md`: narrative text for the final stage.
- `docs/stage3_significance_report.md`: significance testing notes (default `pdopt_best` probes).
- `docs/stage3_significance_report_alias_metric_aligned.md`: same tests for alias-metric–aligned main-text checkpoints (`--artifact-suffix alias_metric_aligned`).
- `docs/stage2_final.md`: Stage 2 final report.
- `docs/stage1_report.md`: Stage 1 report.
- `docs/成果总结.md`: Chinese summary of the overall work.

## Layout

- `source/`: top-level writing material, method/math/experiment notes, current `.tex`, and `Stop-RAG.pdf`.
- `docs/`: project reports, plans, ablation summaries, dataset/stage reports, and closeout notes.
- `results/`: paper-useful figures, CSV tables, and JSON summaries copied from top-level `results/`.
- `results/e2_predictive/`: predictive Stage 3 figure set and evaluation JSON.
- `results/e4_fixed/`: fixed-threshold Stage 3 figure set and evaluation JSON.
- `results/stop_rag_sweep_backup/`: Stop-RAG sweep summary CSVs and figures only, without raw online-test JSONL files.
- `repro/`: dependency/env hints and data split manifests.
- `baselines/`: baseline README and experiment-record notes.

## Notes

- The bundle favors writing and figure/table lookup over full reproduction.
- Full reproduction still depends on the original repository data, caches, checkpoints, and model directories.
- Large raw files were left out deliberately to keep this directory portable.
