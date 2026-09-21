# Stage2 Report ((ablate_margin_m10))

## Probe Performance

- `hotpotqa`: threshold=0.43, probe_f1=0.5196, probe_steps=2.905, oracle_f1=0.6232, oracle_gap=0.1036
- `musique`: threshold=0.51, probe_f1=0.1576, probe_steps=3.942, oracle_f1=0.2526, oracle_gap=0.0950
- `2wiki`: threshold=0.49, probe_f1=0.3492, probe_steps=3.486, oracle_f1=0.5336, oracle_gap=0.1844

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0432
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0196
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0090

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_margin_m10.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_margin_m10.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_margin_m10.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_margin_m10.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_margin_m10.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_margin_m10.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_margin_m10.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_margin_m10.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_margin_m10.pt`
