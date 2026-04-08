# Stage2 Report

## Probe Performance

- `hotpotqa`: threshold=0.63, probe_f1=0.5148, probe_steps=2.532, oracle_f1=0.6232, oracle_gap=0.1084
- `musique`: threshold=0.63, probe_f1=0.1794, probe_steps=3.089, oracle_f1=0.2526, oracle_gap=0.0732
- `2wiki`: threshold=0.65, probe_f1=0.3484, probe_steps=2.411, oracle_f1=0.5336, oracle_gap=0.1852

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0384
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0414
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0098

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa.png`, model=`../artifacts/probe/hotpotqa/probe_mlp.pt`
- `musique`: table=`../results/stage2_probe_table_musique.csv`, pareto=`../results/stage2_probe_pareto_musique.png`, model=`../artifacts/probe/musique/probe_mlp.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki.csv`, pareto=`../results/stage2_probe_pareto_2wiki.png`, model=`../artifacts/probe/2wiki/probe_mlp.pt`