# Stage3 E-value 报告：2wiki

- Probe checkpoint: `artifacts/probe/2wiki/probe_mlp_pdopt_binary_hidden_residual.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Shift type: periodic

## γ = 0.5

Quality Model — Brier: 0.1802, ECE: 0.1176, Pos rate: 0.33

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.3738 | 0.6110 | 3.37 | — |
| 0.1 | Probe+E-value | 0.3738 | 0.6110 | 3.37 | 0.000 |
| 0.1 | Probe+CP | 0.3668 | 0.6200 | 4.69 | 0.824 |

  E-wealth final=10.0000, cap=1/α=10.0

| 0.2 | Probe | 0.3738 | 0.6110 | 3.37 | — |
| 0.2 | Probe+E-value | 0.3738 | 0.6110 | 3.37 | 0.000 |
| 0.2 | Probe+CP | 0.3715 | 0.6130 | 4.45 | 0.762 |

  E-wealth final=5.0000, cap=1/α=5.0

