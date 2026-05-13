# Stage3 E-value 报告：2wiki

- Probe checkpoint: `/home/x12dpg/hjx/Pandora-RAG/artifacts/probe/2wiki/probe_mlp_alias_metric_aligned.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1902, ECE: 0.0264, Pos rate: 0.53

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.5699 | 0.4030 | 1.77 | — |
| 0.1 | Probe+E-value | 0.5820 | 0.3920 | 1.86 | 0.260 |
| 0.1 | Probe+CP | 0.5379 | 0.4360 | 4.32 | 0.805 |

  E-wealth final=2.0318, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.713, selective_acc=0.596, abstained=287, first_abstain_idx=11

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.5667, steps=1.90, coverage=0.010, abstain_n=990, final_wealth=10.0000
    - fixed_k: F1=0.5728, steps=2.78, coverage=1.000, abstain_n=0, final_wealth=5.3573
    - raise_budget: F1=0.5881, steps=2.18, coverage=1.000, abstain_n=0, final_wealth=5.5642

| 0.2 | Probe | 0.5699 | 0.4030 | 1.77 | — |
| 0.2 | Probe+E-value | 0.5858 | 0.3880 | 1.98 | 0.372 |
| 0.2 | Probe+CP | 0.5492 | 0.4280 | 3.88 | 0.733 |

  E-wealth final=0.0006, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.813, selective_acc=0.608, abstained=187, first_abstain_idx=9

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.4583, steps=2.12, coverage=0.008, abstain_n=992, final_wealth=5.0000
    - fixed_k: F1=0.5763, steps=2.54, coverage=1.000, abstain_n=0, final_wealth=0.0008
    - raise_budget: F1=0.5902, steps=2.14, coverage=1.000, abstain_n=0, final_wealth=0.0006

