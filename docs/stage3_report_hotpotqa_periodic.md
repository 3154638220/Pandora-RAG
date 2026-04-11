# Stage3 E-value 报告：hotpotqa

- Probe checkpoint: `artifacts/probe/hotpotqa/probe_mlp_pdopt_binary_last_token.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Shift type: periodic

## γ = 0.5

Quality Model — Brier: 0.1622, ECE: 0.0320, Pos rate: 0.45

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.5273 | 0.4410 | 2.85 | — |
| 0.1 | Probe+E-value | 0.4764 | 0.4900 | 4.98 | 0.970 |
| 0.1 | Probe+CP | 0.4882 | 0.4770 | 4.46 | 0.871 |

  E-wealth final=10.0000, cap=1/α=10.0

| 0.2 | Probe | 0.5273 | 0.4410 | 2.85 | — |
| 0.2 | Probe+E-value | 0.4764 | 0.4900 | 4.98 | 0.970 |
| 0.2 | Probe+CP | 0.4979 | 0.4670 | 4.16 | 0.812 |

  E-wealth final=5.0000, cap=1/α=5.0

