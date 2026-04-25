# Stage2 Report ((pdopt_true_baseline))

## Probe Performance

- `hotpotqa`: threshold=0.63, probe_f1=0.5273, probe_steps=2.854, oracle_f1=0.6232, oracle_gap=0.0959
- `musique`: threshold=0.59, probe_f1=0.1826, probe_steps=3.252, oracle_f1=0.2526, oracle_gap=0.0700
- `2wiki`: threshold=0.83, probe_f1=0.2759, probe_steps=1.990, oracle_f1=0.5336, oracle_gap=0.2576

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0509
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0446
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0823

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_true_baseline.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_true_baseline.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_true_baseline.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_true_baseline.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_true_baseline.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_true_baseline.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_true_baseline.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_true_baseline.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_true_baseline.pt`