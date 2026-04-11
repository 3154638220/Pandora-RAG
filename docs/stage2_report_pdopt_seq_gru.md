# Stage2 Report ((pdopt_seq_gru))

## Probe Performance

- `hotpotqa`: threshold=0.85, probe_f1=0.4752, probe_steps=2.898, oracle_f1=0.6232, oracle_gap=0.1480
- `musique`: threshold=0.67, probe_f1=0.1005, probe_steps=1.391, oracle_f1=0.2526, oracle_gap=0.1521
- `2wiki`: threshold=0.69, probe_f1=0.3437, probe_steps=3.311, oracle_f1=0.5336, oracle_gap=0.1899

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=-0.0012
- `musique`: best_fixed_f1=0.1380, probe_gain=-0.0374
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0145

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_seq_gru.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_seq_gru.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_seq_gru.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_seq_gru.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_seq_gru.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_seq_gru.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_seq_gru.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_seq_gru.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_seq_gru.pt`