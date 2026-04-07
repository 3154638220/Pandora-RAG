# Stage2 Report ((d3_exp2_focal1))

## Probe Performance

- `hotpotqa`: threshold=0.59, probe_f1=0.5260, probe_steps=2.999, oracle_f1=0.6283, oracle_gap=0.1023
- `musique`: threshold=0.65, probe_f1=0.1811, probe_steps=3.127, oracle_f1=0.2430, oracle_gap=0.0619
- `2wiki`: threshold=0.67, probe_f1=0.3582, probe_steps=3.043, oracle_f1=0.5264, oracle_gap=0.1682

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4831, probe_gain=+0.0430
- `musique`: best_fixed_f1=0.1492, probe_gain=+0.0319
- `2wiki`: best_fixed_f1=0.3543, probe_gain=+0.0039

## Artifacts

- `hotpotqa`: table=`results/stage2_probe_table_hotpotqa_d3_exp2_focal1.csv`, pareto=`results/stage2_probe_pareto_hotpotqa_d3_exp2_focal1.png`, model=`artifacts/probe/hotpotqa/probe_mlp_d3_exp2_focal1.pt`
- `musique`: table=`results/stage2_probe_table_musique_d3_exp2_focal1.csv`, pareto=`results/stage2_probe_pareto_musique_d3_exp2_focal1.png`, model=`artifacts/probe/musique/probe_mlp_d3_exp2_focal1.pt`
- `2wiki`: table=`results/stage2_probe_table_2wiki_d3_exp2_focal1.csv`, pareto=`results/stage2_probe_pareto_2wiki_d3_exp2_focal1.png`, model=`artifacts/probe/2wiki/probe_mlp_d3_exp2_focal1.pt`
