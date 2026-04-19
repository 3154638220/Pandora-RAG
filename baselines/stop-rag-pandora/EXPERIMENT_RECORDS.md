# Stop-RAG Pandora Snapshot

This directory is a Git-trackable snapshot of the local `baselines/Stop-RAG`
worktree. The original `baselines/Stop-RAG` directory remains an external Git
repository and is intentionally ignored by the outer Pandora-RAG repository.

Included:

- `README.md`, `pyproject.toml`, `uv.lock`
- `scripts/`
- `src/`
- `stop_rag_dataset_run.log`
- lightweight online-test metrics under `results/**/online_test/*.metrics.json`
- checkpoint metadata under `outputs/**/config.json` and `outputs/**/trainer_state.json`

Excluded:

- nested `.git/`
- raw and processed datasets
- full checkpoint weights and optimizer states
- full `results/**/*.jsonl` traces
- Python bytecode caches

The full local Stop-RAG worktree currently contains large generated artifacts
that should not be committed directly: about 147G under `outputs/`, about 2.2G
under `data/`, and about 1.4G under `results/`.
