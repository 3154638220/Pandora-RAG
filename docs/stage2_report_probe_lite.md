# Stage2 Report ((probe_lite))

## Probe Performance

- `hotpotqa`: policy=per_step, per_step_thresholds=[0.73, 0.71, 0.69, 0.75, 0.73], probe_f1=0.6531, probe_steps=1.682, oracle_f1=0.7810, oracle_gap=0.1279
- `musique`: policy=per_step, per_step_thresholds=[0.61, 0.63, 0.53, 0.55, 0.55], probe_f1=0.3918, probe_steps=3.391, oracle_f1=0.4966, oracle_gap=0.1049
- `2wiki`: policy=per_step, per_step_thresholds=[0.65, 0.69, 0.79, 0.69, 0.69], probe_f1=0.5855, probe_steps=1.829, oracle_f1=0.6954, oracle_gap=0.1099

## Phase C Diagnostics

- `hotpotqa`: chosen_lambda=0.1, gw_dev_avg_steps=1.644, feasible_threshold_count=14, matched_budget_candidates=5, step_refine_adopted=True
- `musique`: chosen_lambda=0.1, gw_dev_avg_steps=2.965, feasible_threshold_count=23, matched_budget_candidates=5, step_refine_adopted=True
- `2wiki`: chosen_lambda=0.1, gw_dev_avg_steps=1.773, feasible_threshold_count=17, matched_budget_candidates=5, step_refine_adopted=True

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.6775, probe_gain=-0.0245
- `musique`: best_fixed_f1=0.4022, probe_gain=-0.0105
- `2wiki`: best_fixed_f1=0.5908, probe_gain=-0.0053

## Appendix Max-F1 Operating Point

- `hotpotqa`: dev_max_f1_threshold=0.61, dev_f1=0.6819, dev_steps=2.578, test_f1=0.7024, test_steps=2.556
- `musique`: dev_max_f1_threshold=0.53, dev_f1=0.4450, dev_steps=3.255, test_f1=0.4130, test_steps=3.887
- `2wiki`: dev_max_f1_threshold=0.59, dev_f1=0.6085, dev_steps=2.202, test_f1=0.5979, test_steps=2.229

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_probe_lite.csv`, sweep=`../results/stage2_threshold_sweep_hotpotqa_probe_lite.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_probe_lite.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_probe_lite.pt`
- `musique`: table=`../results/stage2_probe_table_musique_probe_lite.csv`, sweep=`../results/stage2_threshold_sweep_musique_probe_lite.csv`, pareto=`../results/stage2_probe_pareto_musique_probe_lite.png`, model=`../artifacts/probe/musique/probe_mlp_probe_lite.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_probe_lite.csv`, sweep=`../results/stage2_threshold_sweep_2wiki_probe_lite.csv`, pareto=`../results/stage2_probe_pareto_2wiki_probe_lite.png`, model=`../artifacts/probe/2wiki/probe_mlp_probe_lite.pt`
