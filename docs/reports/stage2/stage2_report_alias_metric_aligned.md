# Stage2 Report ((alias_metric_aligned))

## Probe Performance

- `hotpotqa`: policy=per_step, per_step_thresholds=[0.71, 0.71, 0.73, 0.73, 0.71], probe_f1=0.6579, probe_steps=1.701, oracle_f1=0.7810, oracle_gap=0.1231
- `musique`: policy=per_step, per_step_thresholds=[0.43, 0.61, 0.57, 0.65, 0.57], probe_f1=0.4016, probe_steps=3.324, oracle_f1=0.5156, oracle_gap=0.1140
- `2wiki`: policy=per_step, per_step_thresholds=[0.67, 0.79, 0.71, 0.81, 0.71], probe_f1=0.5893, probe_steps=1.850, oracle_f1=0.6954, oracle_gap=0.1061

## Phase C Diagnostics

- `hotpotqa`: chosen_lambda=0.1, gw_dev_avg_steps=1.644, feasible_threshold_count=15, matched_budget_candidates=5, step_refine_adopted=True
- `musique`: chosen_lambda=0.1, gw_dev_avg_steps=2.897, feasible_threshold_count=22, matched_budget_candidates=5, step_refine_adopted=True
- `2wiki`: chosen_lambda=0.1, gw_dev_avg_steps=1.773, feasible_threshold_count=15, matched_budget_candidates=5, step_refine_adopted=True

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.6775, probe_gain=-0.0197
- `musique`: best_fixed_f1=0.4145, probe_gain=-0.0129
- `2wiki`: best_fixed_f1=0.5908, probe_gain=-0.0015

## Appendix Max-F1 Operating Point

- `hotpotqa`: dev_max_f1_threshold=0.55, dev_f1=0.6826, dev_steps=2.840, test_f1=0.6964, test_steps=2.826
- `musique`: dev_max_f1_threshold=0.57, dev_f1=0.4701, dev_steps=2.952, test_f1=0.4245, test_steps=3.564
- `2wiki`: dev_max_f1_threshold=0.59, dev_f1=0.6062, dev_steps=2.432, test_f1=0.5993, test_steps=2.434

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_alias_metric_aligned.csv`, sweep=`../results/stage2_threshold_sweep_hotpotqa_alias_metric_aligned.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_alias_metric_aligned.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_alias_metric_aligned.pt`
- `musique`: table=`../results/stage2_probe_table_musique_alias_metric_aligned.csv`, sweep=`../results/stage2_threshold_sweep_musique_alias_metric_aligned.csv`, pareto=`../results/stage2_probe_pareto_musique_alias_metric_aligned.png`, model=`../artifacts/probe/musique/probe_mlp_alias_metric_aligned.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_alias_metric_aligned.csv`, sweep=`../results/stage2_threshold_sweep_2wiki_alias_metric_aligned.csv`, pareto=`../results/stage2_probe_pareto_2wiki_alias_metric_aligned.png`, model=`../artifacts/probe/2wiki/probe_mlp_alias_metric_aligned.pt`
