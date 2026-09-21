# Paper Material Index

`paper/` is the writing and paper-material workspace. It deliberately excludes
model weights, caches, raw datasets, and long run logs.

## Start Here

- [`source/www2027/`](source/www2027/): the only active WWW 2027 writing entry.
- [`source/legacy_neurips/`](source/legacy_neurips/): archived NeurIPS drafts and historical writing suggestions.
- [`source/notes/`](source/notes/): method, mathematics, experimental setup, comparison, and project-framing notes.
- [`figs/`](figs/): figures selected for paper use.
- [`results/`](results/): compact paper-facing result bundle.
- [`repro/`](repro/): dependency hints and fixed split manifests.
- [`baselines/`](baselines/): baseline documentation and records, including Stop-RAG.

## Source of truth

The full experiment code remains at the repository root (`stage1/`, `stage2/`,
`stage3/`, `pretest/`, `qa_shared/`, and `scripts/`). The full experiment
reports live under [`../docs/`](../docs/), while raw outputs remain under
[`../results/`](../results/). Files under `paper/results/` and `paper/figs/`
are curated copies for writing and should not be treated as a second execution
source.

## WWW 2027 convention

New manuscript files, appendices, submission notes, and version records belong
under [`source/www2027/`](source/www2027/). Do not create new paper drafts in
the repository root or revive the archived `paper/docs/` snapshot.
