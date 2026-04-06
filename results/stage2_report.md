# Stage2 Report

## Probe Performance

- `hotpotqa`: threshold=0.35, probe_f1=0.5327, probe_steps=3.208, oracle_f1=0.6283, oracle_gap=0.0956
- `musique`: threshold=0.25, probe_f1=0.1609, probe_steps=4.129, oracle_f1=0.2430, oracle_gap=0.0822
- `2wiki`: threshold=0.23, probe_f1=0.3731, probe_steps=4.391, oracle_f1=0.5264, oracle_gap=0.1533

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4831, probe_gain=+0.0496
- `musique`: best_fixed_f1=0.1492, probe_gain=+0.0117
- `2wiki`: best_fixed_f1=0.3543, probe_gain=+0.0188

## Artifacts

- `hotpotqa`: table=`results/stage2_probe_table_hotpotqa.csv`, pareto=`results/stage2_probe_pareto_hotpotqa.png`, model=`artifacts/probe/hotpotqa/probe_mlp.pt`
- `musique`: table=`results/stage2_probe_table_musique.csv`, pareto=`results/stage2_probe_pareto_musique.png`, model=`artifacts/probe/musique/probe_mlp.pt`
- `2wiki`: table=`results/stage2_probe_table_2wiki.csv`, pareto=`results/stage2_probe_pareto_2wiki.png`, model=`artifacts/probe/2wiki/probe_mlp.pt`
