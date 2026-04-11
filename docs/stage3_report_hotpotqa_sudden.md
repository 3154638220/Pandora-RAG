# Stage3 E-value 报告：hotpotqa

- Probe checkpoint: `artifacts/probe/hotpotqa/probe_mlp_pdopt_binary_last_token.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Shift type: sudden

## γ = 0.5

Quality Model — Brier: 0.1622, ECE: 0.0320, Pos rate: 0.45

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.3628 | 0.5920 | 3.07 | — |
| 0.1 | Probe+E-value | 0.3263 | 0.6330 | 4.99 | 0.970 |
| 0.1 | Probe+CP | 0.3333 | 0.6240 | 4.56 | 0.871 |

  E-wealth final=10.0000, cap=1/α=10.0

| 0.2 | Probe | 0.3628 | 0.5920 | 3.07 | — |
| 0.2 | Probe+E-value | 0.3263 | 0.6330 | 4.99 | 0.970 |
| 0.2 | Probe+CP | 0.3383 | 0.6180 | 4.31 | 0.812 |

  E-wealth final=5.0000, cap=1/α=5.0

