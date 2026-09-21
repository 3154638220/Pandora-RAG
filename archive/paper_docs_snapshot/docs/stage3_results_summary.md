# Stage 3 结果总览

> 日期：2026-04-19
> 口径：基于当前 `artifacts/probe/*/probe_mlp_pdopt_best.pt` 重新评估。
> 收尾状态：**Stage 3 已正式关闭**；最终归档见 `docs/stage3_closeout.md`，统一 Pareto 图见 `results/stage3_final_pareto_all.png`。

---

## 1. 实验范围

Stage 3 本轮已重刷以下结果：

- 主实验：`results/stage3_evalue_{hotpotqa,musique,2wiki}.json`
- 漂移实验：`results/stage3_evalue_*_{sudden,gradual,periodic}.json`
- 多参数稳健性：`results/e2_predictive/stage3_evalue_*.json`
- betting 消融：`results/e4_fixed/stage3_evalue_*.json`
- 配套报告：`docs/stage3_report_*.md`
- 叙事整理：`docs/stage3_narrative.md`

默认重点设置为：`γ = 0.5`、`α ∈ {0.1, 0.2}`、`predictive betting`。

---

## 2. 主实验结论

### 2.1 `γ=0.5, α=0.1` 主表

| 数据集 | 策略 | Avg F1 | Error Rate | Avg Steps | 结论 |
|--------|------|--------|------------|-----------|------|
| HotpotQA | Probe | 0.6569 | 0.3030 | 1.70 | 基线 |
| HotpotQA | Probe+E-value | 0.6654 | 0.2950 | 1.81 | 小幅增步，F1/error 均改善 |
| HotpotQA | Probe+CP | 0.6716 | 0.2960 | 4.11 | F1 有升，但代价是步数翻倍以上 |
| MuSiQue | Probe | 0.4152 | 0.5803 | 3.39 | 基线 |
| MuSiQue | Probe+E-value | 0.4127 | 0.5827 | 3.41 | 成本几乎不变，质量基本持平 |
| MuSiQue | Probe+CP | 0.4015 | 0.5971 | 4.85 | 步数明显更高且质量下降 |
| 2wiki | Probe | 0.5639 | 0.4090 | 1.82 | 基线 |
| 2wiki | Probe+E-value | 0.5728 | 0.4010 | 1.91 | 小幅增步，F1/error 同步改善 |
| 2wiki | Probe+CP | 0.5383 | 0.4370 | 4.32 | 大幅增步但质量退化 |

### 2.2 总结

- `Probe+E-value` 相比 `Probe` 的额外步数约为 `+0.5% ~ +4.6%`，HotpotQA 与 2wiki 上 F1 和错误率都有改善。
- `Probe+CP` 不再是“普遍更安全”的替代物：HotpotQA 上确实能继续抬 F1，但代价是步数从 `1.70` 提到 `4.11`；MuSiQue 与 2wiki 上则是又贵又差。
- 当前更稳妥的表述是：`E-value` 更适合作为**低开销监控层**；`CP` 更像一个高成本、数据集敏感的保守门控器。

---

## 3. E-wealth 与风险检测

### 3.1 无漂移主实验下的最终 E-wealth

| 数据集 | α=0.1 | α=0.2 | 说明 |
|--------|-------|-------|------|
| HotpotQA | 0.014 / 10 | ~0 / 5 | 主实验错误率仍高于 α，但 predictive betting 在当前门控下未积累到告警阈值 |
| MuSiQue | 4.193 / 10 | 1.469 / 5 | 风险证据能持续积累，但样本量小、未触 cap |
| 2wiki | 10.000 / 10 | 0.020 / 5 | `α=0.1` 下成功触 cap；`α=0.2` 下最终未维持在 cap |

### 3.2 解释

- `E-value` 的作用仍是在线检测“错误率是否显著高于 α”，而不是把错误率压到 α 以下。
- 本轮 `α=0.1` 下，2wiki 明确触 cap，说明其错误率高到足以累积出强风险证据。
- HotpotQA 当前主实验的 wealth 很低，说明在新的 probe 与 predictive 门控配合下，系统更像是在“用少量额外步数换取质量提升”，而不是持续积累风险告警。

---

## 4. 分布漂移结果

Stage 3 继续补充了三类 shift：`sudden`、`gradual`、`periodic`。

### 4.1 核心现象

- `sudden shift` 下，三数据集在 `α=0.1` 都会触 cap；`α=0.2` 下 HotpotQA 最终未维持在 cap，但 MuSiQue 与 2wiki 仍会触发。
- `gradual shift` 下，MuSiQue 与 2wiki 在 `α=0.1/0.2` 都能触 cap；HotpotQA 只在 `α=0.1` 触发。
- `periodic shift` 下，MuSiQue 与 2wiki 在 `α=0.1/0.2` 都能触 cap；HotpotQA 依旧主要在 `α=0.1` 有明显响应。

### 4.2 结论

- `E-value` 对漂移仍然是可感知的，但本轮结果不再支持“所有 shift、所有 α、所有数据集都触 cap”的强说法。
- 更准确的结论是：在明显恶化的测试流上，`E-value` 通常能快速放大风险证据，尤其是 MuSiQue 与 2wiki；HotpotQA 的响应则更依赖 `α` 和 shift 形态。
- 这也更符合部署直觉：风险监控层的灵敏度会同时受底层 probe 质量、错误率与下注路径影响。

---

## 5. 稳健性与消融

### 5.1 多参数稳健性

基于 `results/e2_predictive/` 的 `4γ × 4α × 3数据集 = 48` 组结果：

- `V2`：步数 `≤ Probe × 1.3`，`48/48` 通过
- `V3`：F1 `≥ Probe × 0.97`，`48/48` 通过
- `quality_bar` 仍落在合理区间，无明显异常值

按 `scripts/stage3_export_tables.py` 的新汇总：

- HotpotQA：`α=0.1` 时平均步数仅增 `6.28%`，平均 F1 增 `1.23%`
- MuSiQue：步数增幅始终很小，但 `α=0.1/0.2` 下平均 F1 略降
- 2wiki：`α=0.1` 时平均步数增 `4.69%`，平均 F1 增 `1.57%`

### 5.2 Predictive vs Fixed betting

基于 `results/e4_fixed/` 与 `results/e2_predictive/` 的对比：

- HotpotQA：predictive 与 fixed 的步数/F1 相同，但 final wealth 分别约为 `0.014` vs `0.000`（`α=0.1`），信号更强一些。
- MuSiQue：predictive 的 wealth 高于 fixed（`4.193` vs `3.437` at `α=0.1`；`1.469` vs `0.937` at `α=0.2`）。
- 2wiki：`α=0.1` 下两者都触 cap；`α=0.2` 下 predictive 仍保留少量 wealth，而 fixed 已接近 0。

因此，这轮结果下 `predictive betting` 的优势比旧版叙事更温和，但整体仍优于或不弱于 fixed。

---

## 6. Selective Prediction 结果

在 `Probe+E-value` 基础上，如果样本进入前 `wealth ≥ 1/α` 则拒答，可得到以下 coverage-accuracy 折中：

| 数据集 | α | Coverage | Selective Accuracy |
|--------|---|----------|--------------------|
| HotpotQA | 0.1 | 0.931 | 0.705 |
| HotpotQA | 0.2 | 1.000 | 0.725 |
| MuSiQue | 0.1 | 0.444 | 0.373 |
| MuSiQue | 0.2 | 0.494 | 0.379 |
| 2wiki | 0.1 | 0.706 | 0.581 |
| 2wiki | 0.2 | 0.853 | 0.600 |

这说明当前 Stage 3 的拒答规则在 HotpotQA 上已经不再是“高告警、低覆盖”，而更接近温和的后备保护；MuSiQue 则依旧体现出高风险流下 coverage 被明显压缩的特征。

---

## 7. 最终结论

Stage 3 这轮重评估后，更准确的三句话是：

1. `Probe+E-value` 仍然保持了低成本特征，相比 `Probe` 只增加少量步数，并在 HotpotQA 与 2wiki 上带来小幅质量增益。
2. `Probe+CP` 不是稳定的强基线：它在 HotpotQA 上能以巨大成本换来更高 F1，但在 MuSiQue 和 2wiki 上明显退化。
3. `E-value` 的核心价值依然是在线风险监控与漂移感知，只是新 probe 下的财富轨迹更细腻，不能再用旧版“普遍强触发”的表述。

---

## 8. 推荐引用顺序

如果后续继续写论文或整理答辩材料，建议按以下顺序引用：

1. `docs/stage3_results_summary.md`
2. `docs/stage3_narrative.md`
3. `docs/stage3_report_{hotpotqa,musique,2wiki}.md`
4. `docs/stage3_report_*_{sudden,gradual,periodic}.md`
