# Stage2 Report

## Probe Performance

- `hotpotqa`: policy=per_step, per_step_thresholds=[0.61, 0.71, 0.57, 0.67, 0.67], probe_f1=0.5835, probe_steps=2.477, oracle_f1=0.6888, oracle_gap=0.1053
- `musique`: policy=per_step, per_step_thresholds=[0.65, 0.55, 0.61, 0.63, 0.59], probe_f1=0.2532, probe_steps=3.612, oracle_f1=0.3414, oracle_gap=0.0883
- `2wiki`: policy=per_step, per_step_thresholds=[0.49, 0.63, 0.61, 0.61, 0.61], probe_f1=0.4569, probe_steps=2.555, oracle_f1=0.5995, oracle_gap=0.1426

## Phase C Diagnostics

- `hotpotqa`: chosen_lambda=0.1, gw_dev_avg_steps=2.374, feasible_threshold_count=17, matched_budget_candidates=5, step_refine_adopted=True
- `musique`: chosen_lambda=0.1, gw_dev_avg_steps=3.87, feasible_threshold_count=22, matched_budget_candidates=5, step_refine_adopted=True
- `2wiki`: chosen_lambda=0.1, gw_dev_avg_steps=2.682, feasible_threshold_count=21, matched_budget_candidates=5, step_refine_adopted=True

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.6014, probe_gain=-0.0179
- `musique`: best_fixed_f1=0.2740, probe_gain=-0.0209
- `2wiki`: best_fixed_f1=0.4637, probe_gain=-0.0068

## Appendix Max-F1 Operating Point

- `hotpotqa`: dev_max_f1_threshold=0.51, dev_f1=0.6280, dev_steps=3.905, test_f1=0.6060, test_steps=3.908
- `musique`: dev_max_f1_threshold=0.55, dev_f1=0.3014, dev_steps=4.162, test_f1=0.2747, test_steps=4.410
- `2wiki`: dev_max_f1_threshold=0.61, dev_f1=0.4609, dev_steps=2.388, test_f1=0.4518, test_steps=2.440

## Artifacts

- `hotpotqa`: table=`../runs/qwen2.5-7b-instruct/results/stage2_probe_table_hotpotqa.csv`, sweep=`../runs/qwen2.5-7b-instruct/results/stage2_threshold_sweep_hotpotqa.csv`, pareto=`../runs/qwen2.5-7b-instruct/results/stage2_probe_pareto_hotpotqa.png`, model=`../runs/qwen2.5-7b-instruct/artifacts/probe/hotpotqa/probe_mlp.pt`
- `musique`: table=`../runs/qwen2.5-7b-instruct/results/stage2_probe_table_musique.csv`, sweep=`../runs/qwen2.5-7b-instruct/results/stage2_threshold_sweep_musique.csv`, pareto=`../runs/qwen2.5-7b-instruct/results/stage2_probe_pareto_musique.png`, model=`../runs/qwen2.5-7b-instruct/artifacts/probe/musique/probe_mlp.pt`
- `2wiki`: table=`../runs/qwen2.5-7b-instruct/results/stage2_probe_table_2wiki.csv`, sweep=`../runs/qwen2.5-7b-instruct/results/stage2_threshold_sweep_2wiki.csv`, pareto=`../runs/qwen2.5-7b-instruct/results/stage2_probe_pareto_2wiki.png`, model=`../runs/qwen2.5-7b-instruct/artifacts/probe/2wiki/probe_mlp.pt`
