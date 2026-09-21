# Stage2 Report (Shallow-Only Probe)

## Probe Performance

- `hotpotqa`: threshold=0.41, probe_f1=0.4829, probe_steps=3.448, oracle_f1=0.6283, oracle_gap=0.1454
- `musique`: threshold=0.29, probe_f1=0.1394, probe_steps=4.336, oracle_f1=0.2430, oracle_gap=0.1036
- `2wiki`: threshold=0.31, probe_f1=0.3552, probe_steps=4.420, oracle_f1=0.5264, oracle_gap=0.1712

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4831, probe_gain=-0.0001
- `musique`: best_fixed_f1=0.1492, probe_gain=-0.0098
- `2wiki`: best_fixed_f1=0.3543, probe_gain=+0.0009

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_shallow.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_shallow.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_shallow.pt`
- `musique`: table=`../results/stage2_probe_table_musique_shallow.csv`, pareto=`../results/stage2_probe_pareto_musique_shallow.png`, model=`../artifacts/probe/musique/probe_mlp_shallow.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_shallow.csv`, pareto=`../results/stage2_probe_pareto_2wiki_shallow.png`, model=`../artifacts/probe/2wiki/probe_mlp_shallow.pt`