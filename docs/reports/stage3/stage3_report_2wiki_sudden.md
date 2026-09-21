# Stage3 E-value 报告：2wiki

- Probe checkpoint: `artifacts/probe/2wiki/probe_mlp_pdopt_best.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: sudden

## γ = 0.5

Quality Model — Brier: 0.1943, ECE: 0.0302, Pos rate: 0.53

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.4471 | 0.5160 | 1.87 | — |
| 0.1 | Probe+E-value | 0.4521 | 0.5120 | 1.96 | 0.231 |
| 0.1 | Probe+CP | 0.4285 | 0.5370 | 4.38 | 0.782 |

  E-wealth final=10.0000, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.561, selective_acc=0.513, abstained=439, first_abstain_idx=11

| 0.2 | Probe | 0.4471 | 0.5160 | 1.87 | — |
| 0.2 | Probe+E-value | 0.4594 | 0.5040 | 2.06 | 0.355 |
| 0.2 | Probe+CP | 0.4393 | 0.5250 | 3.84 | 0.693 |

  E-wealth final=5.0000, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.666, selective_acc=0.523, abstained=334, first_abstain_idx=20

