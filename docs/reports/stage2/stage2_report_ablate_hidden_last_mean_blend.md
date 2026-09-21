# Stage2 Report ((ablate_hidden_last_mean_blend))

## Probe Performance

- `hotpotqa`: threshold=0.87, probe_f1=0.4392, probe_steps=2.538, oracle_f1=0.6232, oracle_gap=0.1840
- `musique`: threshold=0.61, probe_f1=0.1181, probe_steps=1.664, oracle_f1=0.2526, oracle_gap=0.1345
- `2wiki`: threshold=0.75, probe_f1=0.3371, probe_steps=3.473, oracle_f1=0.5336, oracle_gap=0.1964

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0372
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0199
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0211

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_hidden_last_mean_blend.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_hidden_last_mean_blend.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_hidden_last_mean_blend.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_hidden_last_mean_blend.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_hidden_last_mean_blend.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_hidden_last_mean_blend.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_hidden_last_mean_blend.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_hidden_last_mean_blend.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_hidden_last_mean_blend.pt`
