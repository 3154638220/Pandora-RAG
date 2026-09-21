# Stage2 Report ((ablate_hidden_mean_pool))

## Probe Performance

- `hotpotqa`: threshold=0.83, probe_f1=0.4485, probe_steps=2.889, oracle_f1=0.6232, oracle_gap=0.1747
- `musique`: threshold=0.49, probe_f1=0.1331, probe_steps=2.156, oracle_f1=0.2526, oracle_gap=0.1195
- `2wiki`: threshold=0.83, probe_f1=0.3158, probe_steps=2.921, oracle_f1=0.5336, oracle_gap=0.2177

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0279
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0049
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0424

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_hidden_mean_pool.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_hidden_mean_pool.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_hidden_mean_pool.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_hidden_mean_pool.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_hidden_mean_pool.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_hidden_mean_pool.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_hidden_mean_pool.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_hidden_mean_pool.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_hidden_mean_pool.pt`
