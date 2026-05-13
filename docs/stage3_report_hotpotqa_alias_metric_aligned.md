# Stage3 E-value 报告：hotpotqa

- Probe checkpoint: `/home/x12dpg/hjx/Pandora-RAG/artifacts/probe/hotpotqa/probe_mlp_alias_metric_aligned.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1947, ECE: 0.1044, Pos rate: 0.63

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.6594 | 0.2990 | 1.72 | — |
| 0.1 | Probe+E-value | 0.6661 | 0.2910 | 1.81 | 0.315 |
| 0.1 | Probe+CP | 0.6690 | 0.2990 | 4.14 | 0.755 |

  E-wealth final=0.0136, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.926, selective_acc=0.704, abstained=74, first_abstain_idx=5

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.6111, steps=2.00, coverage=0.004, abstain_n=996, final_wealth=10.0000
    - fixed_k: F1=0.6646, steps=2.08, coverage=1.000, abstain_n=0, final_wealth=0.0140
    - raise_budget: F1=0.6698, steps=1.91, coverage=1.000, abstain_n=0, final_wealth=0.0163

| 0.2 | Probe | 0.6594 | 0.2990 | 1.72 | — |
| 0.2 | Probe+E-value | 0.6814 | 0.2760 | 1.90 | 0.399 |
| 0.2 | Probe+CP | 0.6755 | 0.2920 | 3.63 | 0.683 |

  E-wealth final=0.0000, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=1.000, selective_acc=0.724, abstained=0, first_abstain_idx=None

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.6814, steps=1.90, coverage=1.000, abstain_n=0, final_wealth=0.0000
    - fixed_k: F1=0.6814, steps=1.90, coverage=1.000, abstain_n=0, final_wealth=0.0000
    - raise_budget: F1=0.6814, steps=1.90, coverage=1.000, abstain_n=0, final_wealth=0.0000

