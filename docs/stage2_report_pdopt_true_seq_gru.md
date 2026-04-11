# Stage2 Report ((pdopt_true_seq_gru))

## Probe Performance

- `hotpotqa`: threshold=0.63, probe_f1=0.5150, probe_steps=2.654, oracle_f1=0.6232, oracle_gap=0.1082
- `musique`: threshold=0.63, probe_f1=0.1719, probe_steps=2.993, oracle_f1=0.2526, oracle_gap=0.0807
- `2wiki`: threshold=0.77, probe_f1=0.3340, probe_steps=3.168, oracle_f1=0.5336, oracle_gap=0.1995

## Probe vs Best Fixed-K

- `hotpotqa`: best_fixed_f1=0.4764, probe_gain=+0.0386
- `musique`: best_fixed_f1=0.1380, probe_gain=+0.0339
- `2wiki`: best_fixed_f1=0.3582, probe_gain=-0.0242

## Artifacts

- `hotpotqa`: table=`../results/stage2_probe_table_hotpotqa_pdopt_true_seq_gru.csv`, pareto=`../results/stage2_probe_pareto_hotpotqa_pdopt_true_seq_gru.png`, model=`../artifacts/probe/hotpotqa/probe_mlp_pdopt_true_seq_gru.pt`
- `musique`: table=`../results/stage2_probe_table_musique_pdopt_true_seq_gru.csv`, pareto=`../results/stage2_probe_pareto_musique_pdopt_true_seq_gru.png`, model=`../artifacts/probe/musique/probe_mlp_pdopt_true_seq_gru.pt`
- `2wiki`: table=`../results/stage2_probe_table_2wiki_pdopt_true_seq_gru.csv`, pareto=`../results/stage2_probe_pareto_2wiki_pdopt_true_seq_gru.png`, model=`../artifacts/probe/2wiki/probe_mlp_pdopt_true_seq_gru.pt`
