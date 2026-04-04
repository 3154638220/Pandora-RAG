# Stage1 Report

## Data Statistics

- `hotpotqa`: train=5000, dev=1000, test=1000
- `musique`: train=5000, dev=1000, test=1000
- `2wiki`: train=5000, dev=1000, test=1000

## Cache Integrity

- `hotpotqa`: trajectory_cached=7000, hidden_state_files=1000, bad_feature_files=0
- `musique`: trajectory_cached=7000, hidden_state_files=1000, bad_feature_files=0
- `2wiki`: trajectory_cached=7000, hidden_state_files=1000, bad_feature_files=0

## Feature Distribution

- `hotpotqa`: entropy_mean=0.3475, self_consistency_mean=0.8602, overlap_mean=0.0868
- `musique`: entropy_mean=0.1818, self_consistency_mean=0.9267, overlap_mean=0.0660
- `2wiki`: entropy_mean=0.6015, self_consistency_mean=0.7599, overlap_mean=0.0719

## Oracle Frontier

- `hotpotqa` pareto: `results/stage1_oracle_pareto_hotpotqa.png`
- `musique` pareto: `results/stage1_oracle_pareto_musique.png`
- `2wiki` pareto: `results/stage1_oracle_pareto_2wiki.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `hotpotqa`: known_gt=1000, unknown_gt=0, exact_match_over_known=13.70%; 2-hop 137/1000 (13.70%)
- `musique`: known_gt=1000, unknown_gt=0, exact_match_over_known=5.60%; 2-hop 7/73 (9.59%); 3-hop 33/522 (6.32%); 4-hop 16/405 (3.95%)
- `2wiki`: known_gt=0, unknown_gt=1000 (缺少可用 hop 标签)

## Go/No-Go Checks

- `hotpotqa`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1096, pass=True
- `musique`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0916, pass=True
- `2wiki`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0000, pass=False
