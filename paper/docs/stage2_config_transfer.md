# Stage2 Shared Default / Config Transfer

This report is generated from existing Stage2 artifacts. Candidate configurations are selected on dev metrics; test metrics are only used for reporting.

- Per-dataset optimal baseline: `pdopt_best`
- Shared default selected by macro dev F1: `d3_exp2_focal1`
- Strict shared-config filter: candidate suffixes whose config signature differs across datasets are excluded.
- Default excluded prefixes: `pdopt, probe_lite, p2_, smoke`
- Candidate summary CSV: `results/stage2_config_transfer_candidate_summary.csv`
- Shared-default CSV: `results/stage2_config_transfer_shared_default.csv`
- Transfer CSV: `results/stage2_config_transfer_source_transfer.csv`

## Shared Default

| Dataset | Shared F1 / steps | Per-dataset optimal F1 / steps | ΔF1 | Δsteps |
|---|---:|---:|---:|---:|
| hotpotqa | 0.5260 / 2.999 | 0.6544 / 1.732 | -0.1284 | 1.267 |
| musique | 0.1811 / 3.127 | 0.3969 / 3.305 | -0.2158 | -0.177 |
| 2wiki | 0.3582 / 3.043 | 0.5941 / 1.822 | -0.2359 | 1.221 |

Macro: shared `0.3551` F1 / `3.056` steps; per-dataset optimal `0.5485` F1 / `2.286` steps.

## Source-Selected Transfer

| Source | Selected suffix | Target | Transfer F1 / steps | Per-dataset optimal F1 / steps | ΔF1 | Δsteps |
|---|---|---|---:|---:|---:|---:|
| hotpotqa | `d3_exp2_focal1` | musique | 0.1811 / 3.127 | 0.3969 / 3.305 | -0.2158 | -0.177 |
| hotpotqa | `d3_exp2_focal1` | 2wiki | 0.3582 / 3.043 | 0.5941 / 1.822 | -0.2359 | 1.221 |
| musique | `d3_exp1_bce` | hotpotqa | 0.5235 / 2.878 | 0.6544 / 1.732 | -0.1309 | 1.146 |
| musique | `d3_exp1_bce` | 2wiki | 0.3446 / 2.471 | 0.5941 / 1.822 | -0.2495 | 0.649 |
| 2wiki | `ablate_margin_m02` | hotpotqa | 0.5167 / 2.450 | 0.6544 / 1.732 | -0.1377 | 0.718 |
| 2wiki | `ablate_margin_m02` | musique | 0.1687 / 3.038 | 0.3969 / 3.305 | -0.2282 | -0.266 |

## Candidate Dev Ranking

| Rank | Suffix | Macro dev F1 | Macro dev steps | Macro test F1 | Macro test steps |
|---:|---|---:|---:|---:|---:|
| 1 | `d3_exp2_focal1` | 0.3795 | 3.028 | 0.3551 | 3.056 |
| 2 | `ablate_margin_m05` | 0.3773 | 2.817 | 0.3524 | 2.896 |
| 3 | `d3_exp1_bce` | 0.3773 | 2.959 | 0.3486 | 2.974 |
| 4 | `ablate_margin_m02` | 0.3767 | 2.903 | 0.3562 | 2.937 |
| 5 | `d3_exp3_focal2_nols` | 0.3763 | 2.910 | 0.3523 | 3.001 |
| 6 | `ablate_compress_dim256` | 0.3721 | 2.735 | 0.3457 | 2.784 |
| 7 | `shallow` | 0.3707 | 4.073 | 0.3258 | 4.068 |
| 8 | `ablate_compress_dim512` | 0.3688 | 2.752 | 0.3372 | 2.817 |
| 9 | `(base default)` | 0.3678 | 2.613 | 0.3475 | 2.677 |
| 10 | `ablate_gw_cap105` | 0.3678 | 2.613 | 0.3475 | 2.677 |
| 11 | `ablate_gw_cap12` | 0.3678 | 2.613 | 0.3475 | 2.677 |
| 12 | `ablate_gw_cap13` | 0.3678 | 2.613 | 0.3475 | 2.677 |
| 13 | `ablate_margin_m10` | 0.3659 | 3.223 | 0.3421 | 3.444 |
| 14 | `ablate_hidden_residual` | 0.3217 | 2.810 | 0.2973 | 2.699 |
| 15 | `ablate_f1_regression` | 0.3187 | 2.723 | 0.2953 | 2.664 |
| 16 | `ablate_hidden_mean_pool` | 0.3183 | 2.744 | 0.2991 | 2.655 |
| 17 | `ablate_hidden_last_mean_blend` | 0.3096 | 2.585 | 0.2982 | 2.558 |

Interpretation: this is a robustness audit, not a replacement for the main per-dataset operating point.
