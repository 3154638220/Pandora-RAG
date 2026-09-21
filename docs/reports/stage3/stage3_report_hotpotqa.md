# Stage3 E-value 报告：hotpotqa

- Probe checkpoint: `artifacts/probe/hotpotqa/probe_mlp_pdopt_best.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1957, ECE: 0.1051, Pos rate: 0.63

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.6569 | 0.3030 | 1.70 | — |
| 0.1 | Probe+E-value | 0.6654 | 0.2950 | 1.81 | 0.283 |
| 0.1 | Probe+CP | 0.6716 | 0.2960 | 4.11 | 0.752 |

  E-wealth final=0.0145, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.931, selective_acc=0.705, abstained=69, first_abstain_idx=5

| 0.2 | Probe | 0.6569 | 0.3030 | 1.70 | — |
| 0.2 | Probe+E-value | 0.6814 | 0.2750 | 1.89 | 0.383 |
| 0.2 | Probe+CP | 0.6733 | 0.2940 | 3.54 | 0.673 |

  E-wealth final=0.0000, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=1.000, selective_acc=0.725, abstained=0, first_abstain_idx=None

