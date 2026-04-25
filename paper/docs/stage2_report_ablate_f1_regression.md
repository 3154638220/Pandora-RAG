# Stage2 Report ((ablate_f1_regression))

## Probe Performance

- `hotpotqa`: threshold=0.83, probe_f1=0.4496, probe_steps=2.913, oracle_f1=0.6232, oracle_gap=0.1736
- `musique`: threshold=0.45, probe_f1=0.1232, probe_steps=1.839, oracle_f1=0.2526, oracle_gap=0.1294
- `2wiki`: threshold=0.81, probe_f1=0.3131, probe_steps=3.239, oracle_f1=0.5336, oracle_gap=0.2204

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0267
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0148
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0451

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_f1_regression.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_f1_regression.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_f1_regression.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_f1_regression.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_f1_regression.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_f1_regression.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_f1_regression.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_f1_regression.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_f1_regression.pt`
