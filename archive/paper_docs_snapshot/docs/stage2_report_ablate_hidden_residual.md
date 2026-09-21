# Stage2 Report ((ablate_hidden_residual))

## Probe Performance

- `hotpotqa`: threshold=0.85, probe_f1=0.4569, probe_steps=2.821, oracle_f1=0.6232, oracle_gap=0.1663
- `musique`: threshold=0.25, probe_f1=0.1428, probe_steps=2.177, oracle_f1=0.2526, oracle_gap=0.1098
- `2wiki`: threshold=0.81, probe_f1=0.2924, probe_steps=3.100, oracle_f1=0.5336, oracle_gap=0.2412

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0195
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0048
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0658

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_hidden_residual.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_hidden_residual.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_hidden_residual.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_hidden_residual.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_hidden_residual.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_hidden_residual.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_hidden_residual.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_hidden_residual.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_hidden_residual.pt`
