# Stage1 Report

## Data Statistics

- `hotpotqa`: train=4000, calib=1000, dev=1000, test=1000
- `musique`: train=4000, calib=1000, dev=1000, test=417
- `2wiki`: train=4000, calib=1000, dev=1000, test=1000

## Cache Integrity

- `hotpotqa`: trajectory_cached=7000, hidden_npz_total=34896 (train=19929, calib=4986, dev=4990, test=4991), bad_npz_total=0
- `musique`: trajectory_cached=6417, hidden_npz_total=32085 (train=20000, calib=5000, dev=5000, test=2085), bad_npz_total=0
- `2wiki`: trajectory_cached=7000, hidden_npz_total=35000 (train=20000, calib=5000, dev=5000, test=5000), bad_npz_total=0

## Feature Distribution

- `hotpotqa`: entropy_mean=0.0000, self_consistency_mean=1.0000, overlap_mean=0.0851
- `musique`: entropy_mean=0.0000, self_consistency_mean=1.0000, overlap_mean=0.0652
- `2wiki`: entropy_mean=0.0000, self_consistency_mean=1.0000, overlap_mean=0.0700

## Oracle Frontier

- `hotpotqa` pareto: `../results/stage1_oracle_pareto_hotpotqa.png`
- `musique` pareto: `../results/stage1_oracle_pareto_musique.png`
- `2wiki` pareto: `../results/stage1_oracle_pareto_2wiki.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `hotpotqa`: known_gt=1000, unknown_gt=0, exact_match_over_known=23.30%; 2-hop 233/1000 (23.30%)
- `musique`: known_gt=417, unknown_gt=0, exact_match_over_known=9.59%; 2-hop 6/73 (8.22%); 3-hop 11/113 (9.73%); 4-hop 23/231 (9.96%)
- `2wiki`: known_gt=1000, unknown_gt=0, exact_match_over_known=21.50%; 2-hop 211/796 (26.51%); 4-hop 4/204 (1.96%)

## Go/No-Go Checks

- `hotpotqa`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0874, pass=True
- `musique`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0674, pass=True
- `2wiki`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1357, pass=True
