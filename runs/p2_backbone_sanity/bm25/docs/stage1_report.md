# Stage1 Report

## Data Statistics

- `hotpotqa`: train=200, calib=80, dev=80, test=120

## Cache Integrity

- `hotpotqa`: trajectory_cached=480, hidden_npz_total=2397 (train=997, calib=400, dev=400, test=600), bad_npz_total=0

## Feature Distribution

- `hotpotqa`: entropy_mean=0.0000, self_consistency_mean=1.0000, overlap_mean=0.0891

## Oracle Frontier

- `hotpotqa` pareto: `../runs/p2_backbone_sanity/bm25/results/stage1_oracle_pareto_hotpotqa.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `hotpotqa`: known_gt=120, unknown_gt=0, exact_match_over_known=26.67%; 2-hop 32/120 (26.67%)

## Go/No-Go Checks

- `hotpotqa`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1257, pass=True
