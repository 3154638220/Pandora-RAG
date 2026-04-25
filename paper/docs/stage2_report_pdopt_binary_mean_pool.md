# Stage2 Report ((pdopt_binary_mean_pool))

## Probe Performance

- `hotpotqa`: threshold=0.65, probe_f1=0.4856, probe_steps=2.627, oracle_f1=0.6232, oracle_gap=0.1376
- `musique`: threshold=0.65, probe_f1=0.1216, probe_steps=1.986, oracle_f1=0.2526, oracle_gap=0.1310
- `2wiki`: threshold=0.67, probe_f1=0.3179, probe_steps=2.372, oracle_f1=0.5336, oracle_gap=0.2156

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0092
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0164
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0403

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_binary_mean_pool.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_binary_mean_pool.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_binary_mean_pool.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_binary_mean_pool.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_binary_mean_pool.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_binary_mean_pool.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_binary_mean_pool.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_binary_mean_pool.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_binary_mean_pool.pt`
