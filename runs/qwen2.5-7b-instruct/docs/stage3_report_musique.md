# Stage3 E-value 报告：musique

- Probe checkpoint: `runs/qwen2.5-7b-instruct/artifacts/probe/musique/probe_mlp.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1793, ECE: 0.1771, Pos rate: 0.24

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.2680 | 0.7266 | 3.86 | — |
| 0.1 | Probe+E-value | 0.2680 | 0.7266 | 3.86 | 0.055 |
| 0.1 | Probe+CP | 0.2735 | 0.7290 | 4.79 | 0.816 |

  E-wealth final=8.4379, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.288, selective_acc=0.267, abstained=297, first_abstain_idx=4

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.3996, steps=5.00, coverage=0.007, abstain_n=414, final_wealth=10.0000
    - fixed_k: F1=0.2779, steps=4.76, coverage=1.000, abstain_n=0, final_wealth=7.9710
    - raise_budget: F1=0.2777, steps=4.06, coverage=1.000, abstain_n=0, final_wealth=8.6369

| 0.2 | Probe | 0.2680 | 0.7266 | 3.86 | — |
| 0.2 | Probe+E-value | 0.2668 | 0.7290 | 3.88 | 0.241 |
| 0.2 | Probe+CP | 0.2713 | 0.7266 | 4.62 | 0.753 |

  E-wealth final=4.2190, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.295, selective_acc=0.276, abstained=294, first_abstain_idx=4

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.3996, steps=5.00, coverage=0.007, abstain_n=414, final_wealth=5.0000
    - fixed_k: F1=0.2788, steps=4.73, coverage=1.000, abstain_n=0, final_wealth=3.9855
    - raise_budget: F1=0.2768, steps=4.06, coverage=1.000, abstain_n=0, final_wealth=4.3184

