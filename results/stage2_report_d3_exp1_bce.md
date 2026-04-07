# Stage2 Report ((d3_exp1_bce))

## Probe Performance

- `hotpotqa`: threshold=0.77, probe_f1=0.5235, probe_steps=2.878, oracle_f1=0.6283, oracle_gap=0.1049
- `musique`: threshold=0.73, probe_f1=0.1778, probe_steps=3.573, oracle_f1=0.2430, oracle_gap=0.0652
- `2wiki`: threshold=0.83, probe_f1=0.3446, probe_steps=2.471, oracle_f1=0.5264, oracle_gap=0.1818

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4831, probe_gain=+0.0404
- `musique`: best_fixed_f1=0.1492, probe_gain=+0.0286
- `2wiki`: best_fixed_f1=0.3543, probe_gain=-0.0097

## Artifacts

- `hotpotqa`: table=`results/stage2_probe_table_hotpotqa_d3_exp1_bce.csv`, pareto=`results/stage2_probe_pareto_hotpotqa_d3_exp1_bce.png`, model=`artifacts/probe/hotpotqa/probe_mlp_d3_exp1_bce.pt`
- `musique`: table=`results/stage2_probe_table_musique_d3_exp1_bce.csv`, pareto=`results/stage2_probe_pareto_musique_d3_exp1_bce.png`, model=`artifacts/probe/musique/probe_mlp_d3_exp1_bce.pt`
- `2wiki`: table=`results/stage2_probe_table_2wiki_d3_exp1_bce.csv`, pareto=`results/stage2_probe_pareto_2wiki_d3_exp1_bce.png`, model=`artifacts/probe/2wiki/probe_mlp_d3_exp1_bce.pt`
