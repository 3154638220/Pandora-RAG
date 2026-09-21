# Stage2 Report ((d3_exp3_focal2_nols))

## Probe Performance

- `hotpotqa`: threshold=0.61, probe_f1=0.5261, probe_steps=2.921, oracle_f1=0.6283, oracle_gap=0.1023
- `musique`: threshold=0.61, probe_f1=0.1748, probe_steps=3.094, oracle_f1=0.2430, oracle_gap=0.0682
- `2wiki`: threshold=0.63, probe_f1=0.3559, probe_steps=2.987, oracle_f1=0.5264, oracle_gap=0.1704

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4831, probe_gain=+0.0430
- `musique`: best_fixed_f1=0.1492, probe_gain=+0.0256
- `2wiki`: best_fixed_f1=0.3543, probe_gain=+0.0017

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_d3_exp3_focal2_nols.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_d3_exp3_focal2_nols.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_d3_exp3_focal2_nols.pt`
- `musique`: table=`../results/stage2_probe_table_musique_d3_exp3_focal2_nols.csv`, pareto=`../results/stage2_probe_pareto_musique_d3_exp3_focal2_nols.png`, model=`../artifacts/probe/musique/probe_mlp_d3_exp3_focal2_nols.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_d3_exp3_focal2_nols.csv`, pareto=`../results/stage2_probe_pareto_2wiki_d3_exp3_focal2_nols.png`, model=`../artifacts/probe/2wiki/probe_mlp_d3_exp3_focal2_nols.pt`
