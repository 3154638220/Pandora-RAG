# Stage2 Report ((ablate_margin_m05))

## Probe Performance

- `hotpotqa`: threshold=0.63, probe_f1=0.5151, probe_steps=2.563, oracle_f1=0.6232, oracle_gap=0.1081
- `musique`: threshold=0.63, probe_f1=0.1793, probe_steps=3.424, oracle_f1=0.2526, oracle_gap=0.0733
- `2wiki`: threshold=0.67, probe_f1=0.3628, probe_steps=2.700, oracle_f1=0.5336, oracle_gap=0.1708

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0387
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0413
- `2wiki`: best_fixed_f1=0.3582, probe_gain=+0.0046

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_margin_m05.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_margin_m05.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_margin_m05.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_margin_m05.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_margin_m05.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_margin_m05.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_margin_m05.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_margin_m05.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_margin_m05.pt`
