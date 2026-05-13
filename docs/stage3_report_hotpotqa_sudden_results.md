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

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.6111, steps=1.75, coverage=0.004, abstain_n=996, final_wealth=10.0000
    - fixed_k: F1=0.5660, steps=2.49, coverage=1.000, abstain_n=0, final_wealth=10.0000
    - raise_budget: F1=0.5720, steps=2.08, coverage=1.000, abstain_n=0, final_wealth=10.0000

