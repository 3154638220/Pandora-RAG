# Stage3 Statistical Significance

Paired bootstrap and paired randomization results for the Stage-3 main operating point.

- Metric CI: percentile paired bootstrap over test examples.
- Delta: row strategy minus Probe, using the same test examples and ordering.
- p-value: two-sided paired permutation/randomization test on per-example F1 differences.

## Main 95% CIs

| Dataset | Strategy | N | F1 95% CI | EM 95% CI | Error 95% CI | Steps 95% CI |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | Probe | 1000 | 0.6569 [0.6309, 0.6828] | 0.5190 [0.4880, 0.5500] | 0.3030 [0.2750, 0.3310] | 1.6990 [1.6540, 1.7430] |
| hotpotqa | Probe+E-value | 1000 | 0.6654 [0.6395, 0.6911] | 0.5290 [0.4990, 0.5610] | 0.2950 [0.2670, 0.3230] | 1.8070 [1.7560, 1.8600] |
| hotpotqa | Probe+CP | 1000 | 0.6716 [0.6460, 0.6971] | 0.5310 [0.5000, 0.5630] | 0.2960 [0.2680, 0.3250] | 4.1090 [4.0290, 4.1890] |
| musique | Probe | 417 | 0.4152 [0.3734, 0.4591] | 0.3189 [0.2758, 0.3645] | 0.5803 [0.5324, 0.6259] | 3.3933 [3.2926, 3.4917] |
| musique | Probe+E-value | 417 | 0.4127 [0.3690, 0.4555] | 0.3141 [0.2686, 0.3597] | 0.5827 [0.5348, 0.6307] | 3.4101 [3.3094, 3.5108] |
| musique | Probe+CP | 417 | 0.4015 [0.3590, 0.4451] | 0.3046 [0.2614, 0.3477] | 0.5971 [0.5492, 0.6427] | 4.8537 [4.8010, 4.9017] |
| 2wiki | Probe | 1000 | 0.5639 [0.5355, 0.5924] | 0.4740 [0.4420, 0.5050] | 0.4090 [0.3790, 0.4390] | 1.8220 [1.7720, 1.8740] |
| 2wiki | Probe+E-value | 1000 | 0.5728 [0.5449, 0.6013] | 0.4810 [0.4490, 0.5120] | 0.4010 [0.3700, 0.4330] | 1.9050 [1.8540, 1.9570] |
| 2wiki | Probe+CP | 1000 | 0.5383 [0.5098, 0.5665] | 0.4340 [0.4030, 0.4650] | 0.4370 [0.4060, 0.4680] | 4.3180 [4.2450, 4.3900] |

## Paired F1 Comparisons

| Dataset | Comparison | ΔF1 95% CI | bootstrap std | paired p | interpretation |
|---|---|---:|---:|---:|---|
| hotpotqa | Probe+E-value - Probe | +0.0085 [-0.0003, +0.0176] | 0.0045 | 0.0600 | CI includes 0 |
| hotpotqa | Probe+CP - Probe | +0.0147 [-0.0068, +0.0364] | 0.0111 | 0.1886 | CI includes 0 |
| musique | Probe+E-value - Probe | -0.0026 [-0.0088, +0.0020] | 0.0029 | 0.5062 | CI includes 0 |
| musique | Probe+CP - Probe | -0.0137 [-0.0423, +0.0148] | 0.0146 | 0.3533 | CI includes 0 |
| 2wiki | Probe+E-value - Probe | +0.0089 [+0.0017, +0.0165] | 0.0038 | 0.0159 | CI excludes 0 |
| 2wiki | Probe+CP - Probe | -0.0256 [-0.0512, -0.0005] | 0.0130 | 0.0430 | CI excludes 0 |

## MuSiQue Note

MuSiQue has only 417 test examples under the repository split, so the intervals are visibly wider. Use it as a stress case: the main Probe vs E-value delta is small relative to bootstrap uncertainty, while CP is directionally worse and much more expensive.

## Run Metadata

| Dataset | checkpoint | quality_bar | CP tau | final wealth / cap | quality Brier |
|---|---|---:|---:|---:|---:|
| hotpotqa | `artifacts/probe/hotpotqa/probe_mlp_pdopt_best.pt` | 0.2829 | 0.7517 | 0.0145 / 10.0 | 0.1957 |
| musique | `artifacts/probe/musique/probe_mlp_pdopt_best.pt` | 0.2759 | 0.7704 | 4.1926 / 10.0 | 0.1973 |
| 2wiki | `artifacts/probe/2wiki/probe_mlp_pdopt_best.pt` | 0.2313 | 0.7818 | 10.0000 / 10.0 | 0.1943 |
