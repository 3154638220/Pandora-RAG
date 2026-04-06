# Stage1 Report

## Data Statistics

- `hotpotqa`: train=4000, calib=1000, dev=1000, test=1000
- `musique`: train=4000, calib=1000, dev=1000, test=417
- `2wiki`: train=4000, calib=1000, dev=1000, test=1000

## Cache Integrity

- `hotpotqa`: trajectory_cached=7000, hidden_state_files=5000, bad_feature_files=0
- `musique`: trajectory_cached=6417, hidden_state_files=2085, bad_feature_files=0
- `2wiki`: trajectory_cached=7000, hidden_state_files=5000, bad_feature_files=0

## Feature Distribution

- `hotpotqa`: entropy_mean=0.5215, self_consistency_mean=0.7790, overlap_mean=0.0869
- `musique`: entropy_mean=1.0594, self_consistency_mean=0.5727, overlap_mean=0.0666
- `2wiki`: entropy_mean=0.6196, self_consistency_mean=0.7343, overlap_mean=0.0722

## Oracle Frontier

- `hotpotqa` pareto: `results/stage1_oracle_pareto_hotpotqa.png`
- `musique` pareto: `results/stage1_oracle_pareto_musique.png`
- `2wiki` pareto: `results/stage1_oracle_pareto_2wiki.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `hotpotqa`: known_gt=1000, unknown_gt=0, exact_match_over_known=14.20%; 2-hop 142/1000 (14.20%)
- `musique`: known_gt=417, unknown_gt=0, exact_match_over_known=5.52%; 2-hop 7/73 (9.59%); 3-hop 7/113 (6.19%); 4-hop 9/231 (3.90%)
- `2wiki`: known_gt=1000, unknown_gt=0, exact_match_over_known=11.80%; 2-hop 108/796 (13.57%); 4-hop 10/204 (4.90%)

## Go/No-Go Checks

- `hotpotqa`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1453, pass=True
- `musique`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0938, pass=True
- `2wiki`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1721, pass=True