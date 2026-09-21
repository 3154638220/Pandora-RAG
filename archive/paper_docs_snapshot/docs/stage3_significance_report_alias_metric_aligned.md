# Stage3 Statistical Significance (alias_metric_aligned)

Paired bootstrap and paired randomization results for the Stage-3 main operating point.

- Metric CI: percentile paired bootstrap over test examples.
- Delta: row strategy minus Probe, using the same test examples and ordering.
- p-value: two-sided paired permutation/randomization test on per-example F1 differences.

## Main 95% CIs


| Dataset  | Strategy      | N    | F1 95% CI               | EM 95% CI               | Error 95% CI            | Steps 95% CI            |
| -------- | ------------- | ---- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| hotpotqa | Probe         | 1000 | 0.6594 [0.6336, 0.6853] | 0.5200 [0.4890, 0.5510] | 0.2990 [0.2710, 0.3270] | 1.7150 [1.6690, 1.7610] |
| hotpotqa | Probe+E-value | 1000 | 0.6661 [0.6401, 0.6918] | 0.5270 [0.4960, 0.5580] | 0.2910 [0.2620, 0.3190] | 1.8120 [1.7590, 1.8660] |
| hotpotqa | Probe+CP      | 1000 | 0.6690 [0.6436, 0.6949] | 0.5290 [0.4980, 0.5600] | 0.2990 [0.2700, 0.3280] | 4.1390 [4.0600, 4.2160] |
| musique  | Probe         | 417  | 0.4245 [0.3824, 0.4689] | 0.3381 [0.2926, 0.3837] | 0.5755 [0.5276, 0.6211] | 3.5635 [3.4652, 3.6619] |
| musique  | Probe+E-value | 417  | 0.4245 [0.3805, 0.4674] | 0.3381 [0.2926, 0.3837] | 0.5755 [0.5276, 0.6235] | 3.5755 [3.4772, 3.6739] |
| musique  | Probe+CP      | 417  | 0.4157 [0.3731, 0.4598] | 0.3333 [0.2878, 0.3789] | 0.5899 [0.5444, 0.6379] | 4.8513 [4.7938, 4.9017] |
| 2wiki    | Probe         | 1000 | 0.5699 [0.5412, 0.5984] | 0.4840 [0.4530, 0.5150] | 0.4030 [0.3730, 0.4330] | 1.7700 [1.7310, 1.8100] |
| 2wiki    | Probe+E-value | 1000 | 0.5820 [0.5542, 0.6103] | 0.4930 [0.4610, 0.5230] | 0.3920 [0.3610, 0.4230] | 1.8630 [1.8220, 1.9040] |
| 2wiki    | Probe+CP      | 1000 | 0.5379 [0.5097, 0.5660] | 0.4330 [0.4020, 0.4640] | 0.4360 [0.4050, 0.4670] | 4.3210 [4.2480, 4.3950] |


## Paired F1 Comparisons


| Dataset  | Comparison            | ΔF1 95% CI                 | bootstrap std | paired p | interpretation |
| -------- | --------------------- | -------------------------- | ------------- | -------- | -------------- |
| hotpotqa | Probe+E-value - Probe | +0.0067 [-0.0001, +0.0141] | 0.0036        | 0.0612   | CI includes 0  |
| hotpotqa | Probe+CP - Probe      | +0.0096 [-0.0119, +0.0313] | 0.0110        | 0.3860   | CI includes 0  |
| musique  | Probe+E-value - Probe | +0.0000 [+0.0000, +0.0000] | 0.0000        | 1.0000   | CI includes 0  |
| musique  | Probe+CP - Probe      | -0.0088 [-0.0379, +0.0207] | 0.0147        | 0.5490   | CI includes 0  |
| 2wiki    | Probe+E-value - Probe | +0.0120 [+0.0034, +0.0211] | 0.0045        | 0.0092   | CI excludes 0  |
| 2wiki    | Probe+CP - Probe      | -0.0321 [-0.0574, -0.0067] | 0.0129        | 0.0130   | CI excludes 0  |


## MuSiQue Note

MuSiQue has only 417 test examples under the repository split, so the intervals are visibly wider. Use it as a stress case: the main Probe vs E-value delta is small relative to bootstrap uncertainty, while CP is directionally worse and much more expensive.

## Run Metadata


| Dataset  | checkpoint                                                   | quality_bar | CP tau | final wealth / cap | quality Brier |
| -------- | ------------------------------------------------------------ | ----------- | ------ | ------------------ | ------------- |
| hotpotqa | `artifacts/probe/hotpotqa/probe_mlp_alias_metric_aligned.pt` | 0.3150      | 0.7547 | 0.0136 / 10.0      | 0.1947        |
| musique  | `artifacts/probe/musique/probe_mlp_alias_metric_aligned.pt`  | 0.2863      | 0.8079 | 5.6112 / 10.0      | 0.1910        |
| 2wiki    | `artifacts/probe/2wiki/probe_mlp_alias_metric_aligned.pt`    | 0.2601      | 0.8049 | 2.0318 / 10.0      | 0.1902        |
