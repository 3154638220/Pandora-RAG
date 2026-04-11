# Stage3 E-value 报告：musique

- Probe checkpoint: `artifacts/probe/musique/probe_mlp_pdopt_binary_last_token.pt`
- Betting strategy: predictive
- Outcome aware: True
- Quality use probe prob: True
- Shift type: gradual

## γ = 0.5

Quality Model — Brier: 0.1899, ECE: 0.2392, Pos rate: 0.17

| α | 策略 | Avg F1 | Error Rate | Avg Steps | Quality Bar / CP τ |
|---|------|--------|------------|-----------|-------------------|
| 0.1 | Probe | 0.1826 | 0.7914 | 3.25 | — |
| 0.1 | Probe+E-value | 0.1328 | 0.8489 | 5.00 | 0.870 |
| 0.1 | Probe+CP | 0.1401 | 0.8417 | 4.91 | 0.786 |

  E-wealth final=10.0000, cap=1/α=10.0

| 0.2 | Probe | 0.1826 | 0.7914 | 3.25 | — |
| 0.2 | Probe+E-value | 0.1328 | 0.8489 | 5.00 | 0.870 |
| 0.2 | Probe+CP | 0.1493 | 0.8345 | 4.76 | 0.728 |

  E-wealth final=5.0000, cap=1/α=5.0

