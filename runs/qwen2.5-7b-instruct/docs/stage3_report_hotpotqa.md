# Stage3 E-value 报告：hotpotqa

- Probe checkpoint: `runs/qwen2.5-7b-instruct/artifacts/probe/hotpotqa/probe_mlp.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1776, ECE: 0.0424, Pos rate: 0.54

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.5857 | 0.3890 | 2.41 | — |
| 0.1 | Probe+E-value | 0.5894 | 0.3850 | 2.42 | 0.279 |
| 0.1 | Probe+CP | 0.6055 | 0.3730 | 4.21 | 0.777 |

  E-wealth final=5.8417, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.739, selective_acc=0.629, abstained=261, first_abstain_idx=9

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.4408, steps=1.38, coverage=0.008, abstain_n=992, final_wealth=10.0000
    - fixed_k: F1=0.5968, steps=3.09, coverage=1.000, abstain_n=0, final_wealth=7.3274
    - raise_budget: F1=0.5944, steps=2.67, coverage=1.000, abstain_n=0, final_wealth=7.1621

| 0.2 | Probe | 0.5857 | 0.3890 | 2.41 | — |
| 0.2 | Probe+E-value | 0.5931 | 0.3810 | 2.48 | 0.397 |
| 0.2 | Probe+CP | 0.6042 | 0.3740 | 3.92 | 0.727 |

  E-wealth final=0.2100, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.868, selective_acc=0.627, abstained=132, first_abstain_idx=9

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.4408, steps=1.38, coverage=0.008, abstain_n=992, final_wealth=5.0000
    - fixed_k: F1=0.5915, steps=2.79, coverage=1.000, abstain_n=0, final_wealth=0.2438
    - raise_budget: F1=0.5956, steps=2.59, coverage=1.000, abstain_n=0, final_wealth=0.2301

