# Stage2 Report ((ablate_margin_m02))

## Probe Performance

- `hotpotqa`: threshold=0.63, probe_f1=0.5167, probe_steps=2.450, oracle_f1=0.6232, oracle_gap=0.1065
- `musique`: threshold=0.65, probe_f1=0.1687, probe_steps=3.038, oracle_f1=0.2526, oracle_gap=0.0839
- `2wiki`: threshold=0.61, probe_f1=0.3833, probe_steps=3.322, oracle_f1=0.5336, oracle_gap=0.1503

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0403
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0307
- `2wiki`: best_fixed_f1=0.3582, probe_gain=+0.0251

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_margin_m02.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_margin_m02.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_margin_m02.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_margin_m02.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_margin_m02.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_margin_m02.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_margin_m02.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_margin_m02.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_margin_m02.pt`
