# Stage3 E-value 报告：musique

- Probe checkpoint: `/home/x12dpg/hjx/Pandora-RAG/artifacts/probe/musique/probe_mlp_alias_metric_aligned.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Quality bar calib method: quantile
- Shift type: none

## γ = 0.5

Quality Model — Brier: 0.1910, ECE: 0.1029, Pos rate: 0.36

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.4245 | 0.5755 | 3.56 | — |
| 0.1 | Probe+E-value | 0.4245 | 0.5755 | 3.58 | 0.286 |
| 0.1 | Probe+CP | 0.4157 | 0.5899 | 4.85 | 0.808 |

  E-wealth final=5.6112, cap=1/α=10.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.453, selective_acc=0.365, abstained=228, first_abstain_idx=3

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.2046, steps=4.00, coverage=0.005, abstain_n=415, final_wealth=10.0000
    - fixed_k: F1=0.4030, steps=4.40, coverage=1.000, abstain_n=0, final_wealth=4.7125
    - raise_budget: F1=0.4301, steps=3.76, coverage=1.000, abstain_n=0, final_wealth=5.6112

| 0.2 | Probe | 0.4245 | 0.5755 | 3.56 | — |
| 0.2 | Probe+E-value | 0.4257 | 0.5755 | 3.60 | 0.411 |
| 0.2 | Probe+CP | 0.4141 | 0.5899 | 4.65 | 0.752 |

  E-wealth final=2.3266, cap=1/α=5.0
  Selective Prediction (abstain if pre-sample wealth ≥ cap): coverage=0.504, selective_acc=0.381, abstained=207, first_abstain_idx=3

  Post-cap interventions (wealth ≥ cap before sample):
    - abstain: F1=0.2046, steps=4.00, coverage=0.005, abstain_n=415, final_wealth=5.0000
    - fixed_k: F1=0.4163, steps=4.31, coverage=1.000, abstain_n=0, final_wealth=2.1191
    - raise_budget: F1=0.4337, steps=3.76, coverage=1.000, abstain_n=0, final_wealth=2.3266

