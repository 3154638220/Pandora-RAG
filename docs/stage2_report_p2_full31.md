# Stage2 Report ((p2_full31))

## Probe Performance

- `hotpotqa`: policy=per_step, per_step_thresholds=[0.71, 0.67, 0.71, 0.71, 0.71], probe_f1=0.6530, probe_steps=1.706, oracle_f1=0.7810, oracle_gap=0.1280
- `musique`: policy=per_step, per_step_thresholds=[0.47, 0.61, 0.53, 0.57, 0.57], probe_f1=0.3934, probe_steps=3.410, oracle_f1=0.4966, oracle_gap=0.1033
- `2wiki`: policy=per_step, per_step_thresholds=[0.71, 0.77, 0.63, 0.77, 0.71], probe_f1=0.5725, probe_steps=1.801, oracle_f1=0.6954, oracle_gap=0.1229

## Phase C Diagnostics

- `hotpotqa`: chosen_lambda=0.1, gw_dev_avg_steps=1.644, feasible_threshold_count=15, matched_budget_candidates=5, step_refine_adopted=True
- `musique`: chosen_lambda=0.1, gw_dev_avg_steps=2.965, feasible_threshold_count=23, matched_budget_candidates=5, step_refine_adopted=True
- `2wiki`: chosen_lambda=0.1, gw_dev_avg_steps=1.773, feasible_threshold_count=15, matched_budget_candidates=5, step_refine_adopted=True

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.6775, probe_gain=-0.0246
- `musique`: best_fixed_f1=0.4022, probe_gain=-0.0089
- `2wiki`: best_fixed_f1=0.5908, probe_gain=-0.0183

## Appendix Max-F1 Operating Point

- `hotpotqa`: dev_max_f1_threshold=0.53, dev_f1=0.6828, dev_steps=3.063, test_f1=0.6930, test_steps=3.056
- `musique`: dev_max_f1_threshold=0.53, dev_f1=0.4369, dev_steps=3.340, test_f1=0.4198, test_steps=3.873
- `2wiki`: dev_max_f1_threshold=0.63, dev_f1=0.5996, dev_steps=2.248, test_f1=0.5976, test_steps=2.248

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_p2_full31.csv`, sweep=`../results/stage2_threshold_sweep_hotpotqa_p2_full31.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_p2_full31.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_p2_full31.pt`
- `musique`: table=`../results/stage2_probe_table_musique_p2_full31.csv`, sweep=`../results/stage2_threshold_sweep_musique_p2_full31.csv`, pareto=`../results/stage2_probe_pareto_musique_p2_full31.png`, model=`../artifacts/probe/musique/probe_mlp_p2_full31.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_p2_full31.csv`, sweep=`../results/stage2_threshold_sweep_2wiki_p2_full31.csv`, pareto=`../results/stage2_probe_pareto_2wiki_p2_full31.png`, model=`../artifacts/probe/2wiki/probe_mlp_p2_full31.pt`
