# Stage2 Report ((pdopt_binary_hidden_residual))

## Probe Performance

- `hotpotqa`: threshold=0.59, probe_f1=0.5273, probe_steps=3.003, oracle_f1=0.6232, oracle_gap=0.0959
- `musique`: threshold=0.61, probe_f1=0.1743, probe_steps=3.170, oracle_f1=0.2526, oracle_gap=0.0783
- `2wiki`: threshold=0.63, probe_f1=0.3738, probe_steps=3.371, oracle_f1=0.5336, oracle_gap=0.1597

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0509
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0363
- `2wiki`: best_fixed_f1=0.3582, probe_gain=+0.0156

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_binary_hidden_residual.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_binary_hidden_residual.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_binary_hidden_residual.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_binary_hidden_residual.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_binary_hidden_residual.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_binary_hidden_residual.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_binary_hidden_residual.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_binary_hidden_residual.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_binary_hidden_residual.pt`
