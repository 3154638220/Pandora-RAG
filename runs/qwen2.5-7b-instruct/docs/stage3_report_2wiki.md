# Stage3 E-value 报告：2wiki

- Probe checkpoint: `runs/qwen2.5-7b-instruct/artifacts/probe/2wiki/probe_mlp.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1959, ECE: 0.0635, Pos rate: 0.42

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.4518 | 0.5390 | 2.44 | — |
| 0.1 | Probe+E-value | 0.4538 | 0.5370 | 2.45 | 0.218 |
| 0.1 | Probe+CP | 0.4732 | 0.5150 | 4.36 | 0.765 |

  E-wealth final=9.6948, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.516, selective_acc=0.461, abstained=484, first_abstain_idx=7

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.4445, steps=2.50, coverage=0.006, abstain_n=994, final_wealth=10.0000
    - fixed_k: F1=0.4636, steps=3.67, coverage=1.000, abstain_n=0, final_wealth=10.0000
    - raise_budget: F1=0.4621, steps=2.89, coverage=1.000, abstain_n=0, final_wealth=10.0000

| 0.2 | Probe | 0.4518 | 0.5390 | 2.44 | — |
| 0.2 | Probe+E-value | 0.4590 | 0.5320 | 2.50 | 0.356 |
| 0.2 | Probe+CP | 0.4730 | 0.5150 | 3.99 | 0.702 |

  E-wealth final=2.5115, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.594, selective_acc=0.473, abstained=406, first_abstain_idx=7

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.4445, steps=2.50, coverage=0.006, abstain_n=994, final_wealth=5.0000
    - fixed_k: F1=0.4695, steps=3.50, coverage=1.000, abstain_n=0, final_wealth=3.1613
    - raise_budget: F1=0.4640, steps=2.85, coverage=1.000, abstain_n=0, final_wealth=3.2872

