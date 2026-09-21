# Stage3 E-value 报告：2wiki

- Probe checkpoint: `artifacts/probe/2wiki/probe_mlp_pdopt_best.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1943, ECE: 0.0302, Pos rate: 0.53

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.5639 | 0.4090 | 1.82 | — |
| 0.1 | Probe+E-value | 0.5728 | 0.4010 | 1.91 | 0.231 |
| 0.1 | Probe+CP | 0.5383 | 0.4370 | 4.32 | 0.782 |

  E-wealth final=10.0000, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.706, selective_acc=0.581, abstained=294, first_abstain_idx=11

| 0.2 | Probe | 0.5639 | 0.4090 | 1.82 | — |
| 0.2 | Probe+E-value | 0.5820 | 0.3910 | 2.00 | 0.355 |
| 0.2 | Probe+CP | 0.5536 | 0.4210 | 3.75 | 0.693 |

  E-wealth final=0.0203, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.853, selective_acc=0.600, abstained=147, first_abstain_idx=20

