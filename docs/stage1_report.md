# Stage1 Report

## Data Statistics

- `hotpotqa`: train=4000, calib=1000, dev=1000, test=1000
- `musique`: train=4000, calib=1000, dev=1000, test=417
- `2wiki`: train=4000, calib=1000, dev=1000, test=1000

## Cache Integrity

（`hidden_npz_total` 为四 split 下 `{id}_step{k}.npz` 总数；每步一个文件，故约为「各 split 样本数 × max_k」。旧版报告仅列出 test split 的 npz 数，MuSiQue test 仅 417 条时会显示 2085，易被误读为「缺 hidden」。）

- `hotpotqa`: trajectory_cached=7000, hidden_npz_total=35000 (train=20000, calib=5000, dev=5000, test=5000), bad_npz_total=0
- `musique`: trajectory_cached=6417, hidden_npz_total=32085 (train=20000, calib=5000, dev=5000, test=2085), bad_npz_total=0
- `2wiki`: trajectory_cached=7000, hidden_npz_total=35000 (train=20000, calib=5000, dev=5000, test=5000), bad_npz_total=0

## Feature Distribution

- `hotpotqa`: entropy_mean=0.5285, self_consistency_mean=0.7757, overlap_mean=0.0867
- `musique`: entropy_mean=1.0560, self_consistency_mean=0.5769, overlap_mean=0.0662
- `2wiki`: entropy_mean=0.6327, self_consistency_mean=0.7296, overlap_mean=0.0719

## Oracle Frontier

- `hotpotqa` pareto: `../results/stage1_oracle_pareto_hotpotqa.png`
- `musique` pareto: `../results/stage1_oracle_pareto_musique.png`
- `2wiki` pareto: `../results/stage1_oracle_pareto_2wiki.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `hotpotqa`: known_gt=1000, unknown_gt=0, exact_match_over_known=14.20%; 2-hop 142/1000 (14.20%)
- `musique`: known_gt=417, unknown_gt=0, exact_match_over_known=6.71%; 2-hop 8/73 (10.96%); 3-hop 11/113 (9.73%); 4-hop 9/231 (3.90%)
- `2wiki`: known_gt=1000, unknown_gt=0, exact_match_over_known=12.20%; 2-hop 110/796 (13.82%); 4-hop 12/204 (5.88%)

## Go/No-Go Checks

- `hotpotqa`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1468, pass=True
- `musique`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1146, pass=True
- `2wiki`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1754, pass=True
