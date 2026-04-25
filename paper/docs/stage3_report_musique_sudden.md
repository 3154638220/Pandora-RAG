# Stage3 E-value 报告：musique

- Probe checkpoint: `artifacts/probe/musique/probe_mlp_pdopt_best.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: sudden

## γ = 0.5

Quality Model — Brier: 0.1973, ECE: 0.1219, Pos rate: 0.34

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.2341 | 0.7914 | 3.53 | — |
| 0.1 | Probe+E-value | 0.2341 | 0.7914 | 3.55 | 0.276 |
| 0.1 | Probe+CP | 0.2271 | 0.7962 | 4.94 | 0.770 |

  E-wealth final=10.0000, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.225, selective_acc=0.351, abstained=323, first_abstain_idx=3

| 0.2 | Probe | 0.2341 | 0.7914 | 3.53 | — |
| 0.2 | Probe+E-value | 0.2341 | 0.7914 | 3.59 | 0.392 |
| 0.2 | Probe+CP | 0.2319 | 0.7914 | 4.79 | 0.701 |

  E-wealth final=5.0000, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.252, selective_acc=0.352, abstained=312, first_abstain_idx=3

