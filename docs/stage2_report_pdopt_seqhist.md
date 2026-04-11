# Stage2 Report ((pdopt_seqhist))

## Probe Performance

- `hotpotqa`: threshold=0.85, probe_f1=0.4453, probe_steps=2.937, oracle_f1=0.6232, oracle_gap=0.1779
- `musique`: threshold=0.71, probe_f1=0.0890, probe_steps=1.705, oracle_f1=0.2526, oracle_gap=0.1636
- `2wiki`: threshold=0.85, probe_f1=0.2804, probe_steps=2.243, oracle_f1=0.5336, oracle_gap=0.2531

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0311
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0490
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0778

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_seqhist.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_seqhist.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_seqhist.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_seqhist.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_seqhist.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_seqhist.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_seqhist.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_seqhist.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_seqhist.pt`