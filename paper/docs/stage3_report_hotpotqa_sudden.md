# Stage3 E-value 报告：hotpotqa

- Probe checkpoint: `artifacts/probe/hotpotqa/probe_mlp_pdopt_best.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: sudden

## γ = 0.5

Quality Model — Brier: 0.1957, ECE: 0.1051, Pos rate: 0.63

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.5551 | 0.3900 | 1.73 | — |
| 0.1 | Probe+E-value | 0.5592 | 0.3860 | 1.85 | 0.283 |
| 0.1 | Probe+CP | 0.5659 | 0.3910 | 4.24 | 0.752 |

  E-wealth final=10.0000, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.787, selective_acc=0.633, abstained=213, first_abstain_idx=5

| 0.2 | Probe | 0.5551 | 0.3900 | 1.73 | — |
| 0.2 | Probe+E-value | 0.5757 | 0.3640 | 1.94 | 0.383 |
| 0.2 | Probe+CP | 0.5681 | 0.3880 | 3.69 | 0.673 |

  E-wealth final=2.2185, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.989, selective_acc=0.638, abstained=11, first_abstain_idx=928

