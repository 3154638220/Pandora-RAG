# Stage2 Report ((pdopt_f1_baseline))

## Probe Performance

- `hotpotqa`: threshold=0.83, probe_f1=0.4561, probe_steps=2.874, oracle_f1=0.6232, oracle_gap=0.1671
- `musique`: threshold=0.47, probe_f1=0.1302, probe_steps=2.103, oracle_f1=0.2526, oracle_gap=0.1224
- `2wiki`: threshold=0.75, probe_f1=0.3287, probe_steps=3.409, oracle_f1=0.5336, oracle_gap=0.2048

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0203
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0078
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0295

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_f1_baseline.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_f1_baseline.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_f1_baseline.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_f1_baseline.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_f1_baseline.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_f1_baseline.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_f1_baseline.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_f1_baseline.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_f1_baseline.pt`