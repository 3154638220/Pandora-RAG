# Stage3 E-value 报告：musique

- Probe checkpoint: `artifacts/probe/musique/probe_mlp_pdopt_best.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: periodic

## γ = 0.5

Quality Model — Brier: 0.1973, ECE: 0.1219, Pos rate: 0.34

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.4152 | 0.5803 | 3.39 | — |
| 0.1 | Probe+E-value | 0.4127 | 0.5827 | 3.41 | 0.276 |
| 0.1 | Probe+CP | 0.4015 | 0.5971 | 4.85 | 0.770 |

  E-wealth final=10.0000, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.499, selective_acc=0.663, abstained=209, first_abstain_idx=6

| 0.2 | Probe | 0.4152 | 0.5803 | 3.39 | — |
| 0.2 | Probe+E-value | 0.4118 | 0.5827 | 3.45 | 0.392 |
| 0.2 | Probe+CP | 0.4047 | 0.5923 | 4.67 | 0.701 |

  E-wealth final=5.0000, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.566, selective_acc=0.606, abstained=181, first_abstain_idx=13

