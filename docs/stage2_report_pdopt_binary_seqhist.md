# Stage2 Report ((pdopt_binary_seqhist))

## Probe Performance

- `hotpotqa`: threshold=0.59, probe_f1=0.5109, probe_steps=2.831, oracle_f1=0.6232, oracle_gap=0.1123
- `musique`: threshold=0.63, probe_f1=0.1586, probe_steps=2.868, oracle_f1=0.2526, oracle_gap=0.0940
- `2wiki`: threshold=0.67, probe_f1=0.3502, probe_steps=2.796, oracle_f1=0.5336, oracle_gap=0.1833

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0345
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0207
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0080

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_binary_seqhist.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_binary_seqhist.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_binary_seqhist.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_binary_seqhist.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_binary_seqhist.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_binary_seqhist.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_binary_seqhist.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_binary_seqhist.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_binary_seqhist.pt`
