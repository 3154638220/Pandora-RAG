# Stage2 Report ((pdopt_best))

## Probe Performance

- `hotpotqa`: policy=per_step, per_step_thresholds=[0.73, 0.69, 0.81, 0.73, 0.73], probe_f1=0.6544, probe_steps=1.732, oracle_f1=0.7810, oracle_gap=0.1266
- `musique`: policy=per_step, per_step_thresholds=[0.61, 0.63, 0.63, 0.69, 0.63], probe_f1=0.3969, probe_steps=3.305, oracle_f1=0.4966, oracle_gap=0.0997
- `2wiki`: policy=per_step, per_step_thresholds=[0.61, 0.79, 0.73, 0.77, 0.67], probe_f1=0.5941, probe_steps=1.822, oracle_f1=0.6954, oracle_gap=0.1013

## Phase C Diagnostics

- `hotpotqa`: chosen_lambda=0.1, gw_dev_avg_steps=1.644, feasible_threshold_count=14, matched_budget_candidates=5, step_refine_adopted=True
- `musique`: chosen_lambda=0.1, gw_dev_avg_steps=2.965, feasible_threshold_count=20, matched_budget_candidates=5, step_refine_adopted=True
- `2wiki`: chosen_lambda=0.1, gw_dev_avg_steps=1.773, feasible_threshold_count=17, matched_budget_candidates=5, step_refine_adopted=True

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.6775, probe_gain=-0.0231
- `musique`: best_fixed_f1=0.4022, probe_gain=-0.0053
- `2wiki`: best_fixed_f1=0.5908, probe_gain=+0.0033

## Appendix Max-F1 Operating Point

- `hotpotqa`: dev_max_f1_threshold=0.47, dev_f1=0.6877, dev_steps=2.948, test_f1=0.6884, test_steps=2.981
- `musique`: dev_max_f1_threshold=0.55, dev_f1=0.4395, dev_steps=3.509, test_f1=0.4140, test_steps=4.192
- `2wiki`: dev_max_f1_threshold=0.57, dev_f1=0.5950, dev_steps=2.562, test_f1=0.5918, test_steps=2.629

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_best.csv`, sweep=`../results/stage2_threshold_sweep_hotpotqa_pdopt_best.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_best.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_best.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_best.csv`, sweep=`../results/stage2_threshold_sweep_musique_pdopt_best.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_best.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_best.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_best.csv`, sweep=`../results/stage2_threshold_sweep_2wiki_pdopt_best.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_best.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_best.pt`
