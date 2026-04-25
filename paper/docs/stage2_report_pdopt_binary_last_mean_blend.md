# Stage2 Report ((pdopt_binary_last_mean_blend))

## Probe Performance

- `hotpotqa`: threshold=0.67, probe_f1=0.5142, probe_steps=2.422, oracle_f1=0.6232, oracle_gap=0.1090
- `musique`: threshold=0.61, probe_f1=0.1672, probe_steps=3.381, oracle_f1=0.2526, oracle_gap=0.0854
- `2wiki`: threshold=0.65, probe_f1=0.3598, probe_steps=2.674, oracle_f1=0.5336, oracle_gap=0.1737

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0378
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0292
- `2wiki`: best_fixed_f1=0.3582, probe_gain=+0.0016

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_binary_last_mean_blend.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_binary_last_mean_blend.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_binary_last_mean_blend.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_binary_last_mean_blend.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_binary_last_mean_blend.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_binary_last_mean_blend.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_binary_last_mean_blend.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_binary_last_mean_blend.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_binary_last_mean_blend.pt`
