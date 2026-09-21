# Stop-RAG Threshold Sweep / Pareto Frontier

This report is generated from true-online Stop-RAG test artifacts under `baselines/stop-rag-pandora/results/*/online_test/`.
Offline `compute_scores` replay files are intentionally excluded.

## Coverage

- 2Wiki: 10/10 online points have avg-step metadata.
- HotpotQA: 10/10 online points have avg-step metadata.
- MuSiQue: 10/11 online points have avg-step metadata.

## Matched Budget

| Dataset | Pandora F1 | Pandora steps | Stop-RAG F1 @ Pandora budget | Stop-RAG steps to reach Pandora F1 |
| --- | ---: | ---: | ---: | ---: |
| 2Wiki | 0.5728 | 1.905 |  |  |
| HotpotQA | 0.6654 | 1.807 | 0.4110 |  |
| MuSiQue | 0.4127 | 3.410 | 0.2397 |  |

## Pareto Points

| Dataset | Checkpoint | Threshold | F1 | EM | Avg steps | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 2400 | -0.20 | 0.5286 | 0.4470 | 1.958 | 1000 |
| 2Wiki | 2400 | -0.16 | 0.5377 | 0.4560 | 2.062 | 1000 |
| 2Wiki | 2400 | -0.12 | 0.5429 | 0.4640 | 2.196 | 1000 |
| 2Wiki | 2400 | -0.09 | 0.5439 | 0.4620 | 2.323 | 1000 |
| 2Wiki | 2400 | -0.06 | 0.5552 | 0.4720 | 2.536 | 1000 |
| HotpotQA | 1000 | -0.12 | 0.4110 | 0.3080 | 1.000 | 1000 |
| HotpotQA | 1000 | -0.03 | 0.6100 | 0.4750 | 5.000 | 1000 |
| MuSiQue | 1200 | -0.20 | 0.2397 | 0.1727 | 3.204 | 417 |
| MuSiQue | 1200 | -0.16 | 0.2452 | 0.1703 | 3.477 | 417 |
| MuSiQue | 1200 | -0.12 | 0.2629 | 0.1823 | 3.731 | 417 |
| MuSiQue | 1200 | -0.06 | 0.2687 | 0.1990 | 4.329 | 417 |
| MuSiQue | 1200 | -0.03 | 0.2731 | 0.2038 | 4.571 | 417 |

## Pareto Points (EM objective)

| Dataset | Checkpoint | Threshold | F1 | EM | Avg steps | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2Wiki | 2400 | -0.20 | 0.5286 | 0.4470 | 1.958 | 1000 |
| 2Wiki | 2400 | -0.16 | 0.5377 | 0.4560 | 2.062 | 1000 |
| 2Wiki | 2400 | -0.12 | 0.5429 | 0.4640 | 2.196 | 1000 |
| 2Wiki | 2400 | -0.06 | 0.5552 | 0.4720 | 2.536 | 1000 |
| HotpotQA | 1000 | -0.12 | 0.4110 | 0.3080 | 1.000 | 1000 |
| HotpotQA | 1000 | -0.03 | 0.6100 | 0.4750 | 5.000 | 1000 |
| MuSiQue | 1200 | -0.20 | 0.2397 | 0.1727 | 3.204 | 417 |
| MuSiQue | 1200 | -0.12 | 0.2629 | 0.1823 | 3.731 | 417 |
| MuSiQue | 1200 | -0.09 | 0.2535 | 0.1871 | 4.000 | 417 |
| MuSiQue | 1200 | -0.06 | 0.2687 | 0.1990 | 4.329 | 417 |
| MuSiQue | 1200 | -0.03 | 0.2731 | 0.2038 | 4.571 | 417 |

## Figures

- `results/stop_rag_pareto_2wiki.png`
- `results/stop_rag_pareto_em_2wiki.png`
- `results/stop_rag_pareto_hotpotqa.png`
- `results/stop_rag_pareto_em_hotpotqa.png`
- `results/stop_rag_pareto_musique.png`
- `results/stop_rag_pareto_em_musique.png`
