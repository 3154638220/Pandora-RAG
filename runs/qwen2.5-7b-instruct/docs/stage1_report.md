# Stage1 Report

## Data Statistics

- `musique`: train=4000, calib=1000, dev=1000, test=417

## Cache Integrity

- `musique`: trajectory_cached=6417, hidden_npz_total=32085 (train=20000, calib=5000, dev=5000, test=2085), bad_npz_total=0

## Feature Distribution

- `musique`: entropy_mean=0.0000, self_consistency_mean=1.0000, overlap_mean=0.0652

## Oracle Frontier

- `musique` pareto: `../results/stage1_oracle_pareto_musique.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `musique`: known_gt=417, unknown_gt=0, exact_match_over_known=9.35%; 2-hop 5/73 (6.85%); 3-hop 11/113 (9.73%); 4-hop 23/231 (9.96%)

## Go/No-Go Checks

- `musique`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0682, pass=True
