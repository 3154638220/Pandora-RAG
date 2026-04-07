# Stage2 Report

## Probe Performance

- `hotpotqa`: threshold=0.61, probe_f1=0.5256, probe_steps=2.892, oracle_f1=0.6283, oracle_gap=0.1027
- `musique`: threshold=0.59, probe_f1=0.1785, probe_steps=3.513, oracle_f1=0.2430, oracle_gap=0.0645
- `2wiki`: threshold=0.63, probe_f1=0.3580, probe_steps=3.109, oracle_f1=0.5264, oracle_gap=0.1684

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4831, probe_gain=+0.0426
- `musique`: best_fixed_f1=0.1492, probe_gain=+0.0293
- `2wiki`: best_fixed_f1=0.3543, probe_gain=+0.0037

## Artifacts

- `hotpotqa`: table=`results/stage2_probe_table_hotpotqa.csv`, pareto=`results/stage2_probe_pareto_hotpotqa.png`, model=`artifacts/probe/hotpotqa/probe_mlp.pt`
- `musique`: table=`results/stage2_probe_table_musique.csv`, pareto=`results/stage2_probe_pareto_musique.png`, model=`artifacts/probe/musique/probe_mlp.pt`
- `2wiki`: table=`results/stage2_probe_table_2wiki.csv`, pareto=`results/stage2_probe_pareto_2wiki.png`, model=`artifacts/probe/2wiki/probe_mlp.pt`
