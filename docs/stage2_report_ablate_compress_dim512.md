# Stage2 Report ((ablate_compress_dim512))

## Probe Performance

- `hotpotqa`: threshold=0.61, probe_f1=0.5251, probe_steps=2.885, oracle_f1=0.6232, oracle_gap=0.0981
- `musique`: threshold=0.61, probe_f1=0.1818, probe_steps=3.590, oracle_f1=0.2526, oracle_gap=0.0708
- `2wiki`: threshold=0.69, probe_f1=0.3047, probe_steps=1.975, oracle_f1=0.5336, oracle_gap=0.2289

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0487
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0438
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0535

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_ablate_compress_dim512.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_ablate_compress_dim512.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_ablate_compress_dim512.pt`
- `musique`: table=`../results/stage2_probe_table_musique_ablate_compress_dim512.csv`, pareto=`../results/stage2_probe_pareto_musique_ablate_compress_dim512.png`, model=`../artifacts/probe/musique/probe_mlp_ablate_compress_dim512.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_ablate_compress_dim512.csv`, pareto=`../results/stage2_probe_pareto_2wiki_ablate_compress_dim512.png`, model=`../artifacts/probe/2wiki/probe_mlp_ablate_compress_dim512.pt`
