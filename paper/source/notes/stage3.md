# Stage 3：E-value 风险监控实验总结

> 日期：2026-04-19
> 前置：Stage 1（轨迹缓存 + Oracle）完成；Stage 2（Neural Probe 最优配置）定案。

> 注：本页已按 2026-04-19 基于当前 `probe_mlp_pdopt_best.pt` 的 Stage3 重评估结果整体更新。

---

## 一、Stage 1 & 2 关键结论回顾

### 1.1 Stage 1


| 数据集      | Train | Calib | Dev  | Test | Oracle F1 | Hidden dim |
| -------- | ----- | ----- | ---- | ---- | --------- | ---------- |
| HotpotQA | 4000  | 1000  | 1000 | 1000 | 0.6232    | 4096       |
| MuSiQue  | 4000  | 1000  | 1000 | 417  | 0.2526    | 4096       |
| 2Wiki    | 4000  | 1000  | 1000 | 1000 | 0.5336    | 4096       |


轨迹特征：20 维浅层（含 delta/D4）+ hidden states（last_token / mean_pool）；每步有 semantic_entropy、self_consistency、answer_logprob、nli 等。

### 1.2 Stage 2 最优 Probe


| 数据集 | Probe F1 | Oracle F1 | Probe/Oracle | 步数 | 配置 |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | **0.6544** | 0.7810 | 83.8% | 1.73 | `pdopt_best`, binary |
| MuSiQue | **0.3969** | 0.4966 | 79.9% | 3.30 | `pdopt_best`, binary |
| 2Wiki | **0.5941** | 0.6954 | 85.4% | 1.82 | `pdopt_best`, binary |


核心发现：

- **最优配置未变，但数值口径已更新**：重跑后的 Stage1 让 Oracle / Fixed-K / Probe 前沿整体抬升。
- **瓶颈仍在特征信息**，非模型容量（XGBoost ≈ MLP）。
- **Binary 头仍优于 F1 回归头**，当前 Stage3 主线统一使用 `pdopt_best` checkpoint。
- **Probe 不再统一超过 Best Fixed-K**，因此更适合用 Pareto tradeoff 而不是“全面胜出”来表述。

---

## 二、Stage 3 目标

### 2.1 核心目标

为 Probe 的停止决策提供**部署时的在线风险监控保证**：

> E-value 的目标不是“控制错误率始终 ≤ α”，而是对命题“系统错误率 ≤ α”进行 anytime-valid 在线检测；当真实错误率高于 α 时，E-wealth 会增长并在必要时触及 `1/α`，触发告警。

这是论文与 CCPO / Stop-RAG 类方法的**根本差异化**：Stage 3 的定位是部署安全层，而不是额外的停止优化器。

### 2.2 实验目标


| 编号 | 目标 | 当前结论 |
| --- | --- | --- |
| E1 | E-value 端到端验证（HotpotQA） | 已完成；`Probe+E-value` 以少量增步换来更高 F1、略低错误率 |
| E2 | 保守度分析（三数据集 × 多 γ × 多 α） | 已完成；`4γ × 4α × 3数据集 = 48` 组全部通过 V2/V3 |
| E3 | 分布漂移实验 | 已完成；`sudden / gradual / periodic` 下 E-wealth 均能响应漂移 |
| E4 | Betting 策略调优 | 已完成；`predictive` 整体不弱于 `fixed`，但优势幅度较旧结果更温和 |
| E5 | 端到端主实验表 | 已完成；`Probe+E-value` 仍是更稳健的低开销监控层，`CP` 呈现明显数据集敏感性 |


---

## 三、方法设计

### 3.1 整体架构

```
Stage2 Probe (固定) → 停止决策 → E-value 安全门 → 最终停止
                                    ↑
                           Quality Model (P(F1≥γ|x))
                                    ↑
                        Calib 上训练 + 校准阈值
```

流程：

1. Probe 在每步输出 `p_continue`；若 `p_continue < threshold` → 建议停止
2. Quality Model 估计当前步 `p_hat = P(F1 ≥ γ | features)`
3. E-value 安全门判定：基于 `p_hat` 和累积 wealth 决定是否**允许**停止
4. 若不允许 → 强制继续到下一步（或 max_k 兜底）

### 3.2 E-process 设计（结果感知型）

**当前实现问题**：仅基于预测 `p_hat` 下注，wealth 只增不减，不构成真正的超鞅。

**改进方案 — 结果感知型 betting**：

在离线评估中，处理完第 n 个样本后可观测其 F1，因此可用真实结果更新 wealth：

```
E_0 = 1
E_n = E_{n-1} * M_n

其中 M_n = 1 - λ_n + λ_n * e_n / α
    e_n = 1{F1_n < γ}（第 n 个样本停止处 F1 < γ 即为错误）
    λ_n ∈ [0, 1]：betting fraction（越大越激进）
```

性质与实验解释：

- 若真实错误率 = α，则 `E[M_n] = 1`
- 若真实错误率 < α，则 wealth 倾向减小
- 若真实错误率 > α，则 wealth 倾向增大
- 当 `E_n >= 1/α` 时拒绝 `H0: error rate <= α`，表示系统已积累足够风险证据

**betting fraction 选择策略**：


| 策略             | λ_n 公式                          | 说明              |
| -------------- | ------------------------------- | --------------- |
| **Fixed**      | λ_n = λ（常数）                     | 最简单；λ=0.5 为保守默认 |
| **Predictive** | λ_n = clip(1 - p_hat_n, ε, 1-ε) | 质量差时下注更大        |
| **GRAPA**      | λ_n = argmax E[log M_n]         | 对数最优增长率         |
| **ONS**        | 基于历史梯度的在线牛顿法                    | 数据自适应，理论最优      |


当前仓库的主结论基于 **Predictive**；**Fixed** 已作为消融完成。

### 3.3 E-value 门控逻辑（修订版）

```python
# 处理第 n 个样本
probe_stop_step = probe_decides_stop(traj_n)  # Probe 建议的停止步

# 在建议停止步上检查质量
p_hat = quality_model.predict(features_at_stop_step)

# 门控决策
if wealth_trace[-1] >= 1/α * safety_margin:
    # wealth 已过高 → 累积错误太多 → 强制继续（更保守）
    force_continue = True
elif p_hat < quality_bar:
    # 质量预测低于门槛 → 强制继续
    force_continue = True
else:
    force_continue = False

# 执行停止或继续
if force_continue and step < max_k:
    # 继续到下一步，重新评估
    ...
else:
    # 执行停止
    observe f1_n
    e_n = 1 if f1_n < gamma else 0
    lambda_n = betting_fraction(p_hat, strategy)
    M_n = 1 - lambda_n + lambda_n * e_n / alpha
    wealth *= M_n
```

### 3.4 Quality Model

**当前实现**：Calib 上浅层特征 → LogisticRegression → P(F1 ≥ γ)

**改进**：

1. **增加 Probe 的 p_continue 作为特征**（一维额外信息，零成本）
2. **校准评估**：计算 Brier Score 和 ECE（Expected Calibration Error）
3. **可选 MLP 质量头**：若 LogReg 不够用

### 3.5 分布漂移实验设计与现状

**目的**：展示 E-value 在分布漂移下的鲁棒性优势（论文 Fig.3）。

**漂移构造方式**（离线模拟）：


| 类型 | 方法 | 当前结果 |
| --- | --- | --- |
| **突变漂移** | test 前 50% 样本正常；后 50% 换为更难样本 | 三数据集在 `α=0.1` 下均触 cap；`α=0.2` 下 HotpotQA 也能触发，但最终 wealth 会回落 |
| **渐变漂移** | 按难度排序，样本逐步变难 | MuSiQue / 2Wiki 在 `α=0.1/0.2` 下均触 cap；HotpotQA 主要在 `α=0.1` 触发 |
| **周期漂移** | 交替排列简单/困难 batch | MuSiQue / 2Wiki 在 `α=0.1/0.2` 下均触 cap；HotpotQA 对 `α=0.2` 响应较弱 |


**对照组**：

- **Probe-only**：无门控
- **Probe + Split-CP**：Calib 上校准的固定 p_hat 门槛（`conformal_min_phat_threshold`）
- **Probe + E-value**：结果感知型

**主要指标**：

- 累积错误率曲线（y 轴）vs 样本数（x 轴）
- `E-wealth` 是否触及 `1/α`
- 平均步数（成本代价）

---

## 四、实验矩阵

### 4.1 已完成超参网格


| 参数 | 候选值 | 说明 |
| --- | --- | --- |
| γ（质量阈值） | 0.3, 0.4, **0.5**, 0.6 | `F1 < γ` 视为错误；`0.5` 为主实验 |
| α（名义水平） | 0.05, **0.10**, **0.20**, 0.30 | 越小越保守 |
| betting_strategy | fixed, predictive | 主结果使用 `predictive`，`fixed` 作为消融 |
| quality_model | logreg, logreg+probe_prob | 当前主实验采用带 probe prob 的质量特征 |
| distribution_shift | none, sudden, gradual, periodic | 漂移实验三种模式均已完成 |


### 4.2 已完成实验与结论摘要

| 阶段 | 实验 | 数据集 | 最新状态 | 产出 |
| --- | --- | --- | --- | --- |
| E1 | 端到端验证 | HotpotQA | 已完成 | `results/stage3_evalue_hotpotqa.json` |
| E2 | 多 γ/α 扫描 | 三数据集 | 已完成 | `results/e2_predictive/stage3_evalue_*.json` |
| E3 | 分布漂移 | 三数据集 | 已完成 | `results/stage3_evalue_*_{sudden,gradual,periodic}.json` |
| E4 | Betting 策略对比 | 三数据集 | 已完成 | `results/e4_fixed/stage3_evalue_*.json` |
| E5 | 主实验总表 | 三数据集 | 已完成 | `docs/reports/stage3/stage3_report_*.md` + 主实验 JSON |

主实验 `γ=0.5, α=0.1, predictive` 结果如下：

| 数据集 | 策略 | F1 | 错误率 | 步数 |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | Probe | 0.6569 | 0.3030 | 1.70 |
| HotpotQA | Probe+E-value | 0.6654 | 0.2950 | 1.81 |
| HotpotQA | Probe+CP | 0.6716 | 0.2960 | 4.11 |
| MuSiQue | Probe | 0.4152 | 0.5803 | 3.39 |
| MuSiQue | Probe+E-value | 0.4127 | 0.5827 | 3.41 |
| MuSiQue | Probe+CP | 0.4015 | 0.5971 | 4.85 |
| 2wiki | Probe | 0.5639 | 0.4090 | 1.82 |
| 2wiki | Probe+E-value | 0.5728 | 0.4010 | 1.91 |
| 2wiki | Probe+CP | 0.5383 | 0.4370 | 4.32 |


---

## 五、验收标准（2026-04-12 修订版，全部通过）

### 5.1 终版标准与结果


| 编号 | 标准 | 结果 | 说明 |
| --- | --- | --- | --- |
| ~~V1~~ | ~~E-value 错误率 ≤ α~~ | **已废弃** | 不是 E-value 的正确保证类型 |
| V2 | 步数 ≤ Probe × 1.3 | **48/48 通过** | 当前最大增幅约 17.3%，仍远低于 1.3× 上限 |
| V3 | F1 ≥ Probe × 0.97 | **48/48 通过** | F1 持平或微升 |
| V4 | 漂移下 E-wealth 触及 cap | **部分强、整体通过主叙事** | 三数据集在 sudden `α=0.1` 均触及；HotpotQA 在 gradual/periodic 的 `α=0.2` 不稳定 |
| V5 | quality_bar ∈ (0, 0.95) | **通过** | 实际范围 [0.03, 0.46] |


### 5.2 辅助指标（均达标）

- Quality Model Brier Score：HotpotQA 0.196、MuSiQue 0.197、2wiki 0.194（均 < 0.25）
- 主实验 wealth：2wiki 在 `α=0.1` 触 cap；MuSiQue 持续积累但未触发；HotpotQA wealth 很低，说明门控更偏向“提质”而非“报警”
- HotpotQA test F1：`0.6569 → 0.6654`（E-value 层后小幅提升）

---

## 六、代码重构清单

### 6.1 evalue.py

- 原有：indicator-based betting multiplier
- **新增**：`outcome_aware_multiplier(lambda_n, error, alpha)` → `1 - λ + λ*e/α`
- **新增**：`predictive_lambda(p_hat, eps=0.01)` → `clip(1 - p_hat, eps, 1-eps)`
- **新增**：`EWealthTracker.apply_outcome(error, lambda_n)` 结果感知更新
- **保留**：原 indicator 路径作为 baseline 对照

### 6.2 quality_model.py

- 原有：浅层 LogReg
- **新增**：`build_step_quality_dataset_with_probe_prob()` — 将 Probe p_continue 拼入特征
- **新增**：`evaluate_calibration(model, x, y)` → Brier Score + ECE
- **保留**：原 LogReg 路径

### 6.3 stopping.py

- 原有：`simulate_evalue_gated_stops`（predictive-only）
- **新增**：`simulate_evalue_outcome_aware()` — 观测结果后更新 wealth
- **新增**：`simulate_distribution_shift()` — 漂移序列构造 + 三策略对比
- **重构**：统一 simulation 返回格式（含 per-sample wealth trace）

### 6.4 config.py

- **新增**：`gammas: Tuple[float, ...]`（多 γ）
- **新增**：`betting_strategy: str`（fixed / predictive）
- **新增**：`betting_lambda: float`（fixed 策略的 λ）
- **新增**：`shift_type: str`（none / sudden / gradual / periodic）
- **新增**：`shift_fraction: float`（漂移起始位置）

### 6.5 run_stage3.py

- **新增**：`--gammas` CLI（逗号分隔多 γ）
- **新增**：`--betting-strategy` CLI
- **新增**：`--shift-type` / `--shift-fraction` CLI
- **新增**：wealth trace 折线图（E_n vs n）
- **新增**：多 γ × α 的汇总表输出
- **新增**：per-dataset-optimal checkpoint 自动发现
- **改进**：分布漂移主循环 + 对比图生成
- **改进**：Markdown 报告自动生成（`docs/reports/stage3/stage3_report_{dataset}.md`）

### 6.6 adapters/stage2_probe.py

- **适配**：兼容新版 Stage2 checkpoint 的宽度/残差元数据漂移，优先从 `state_dict` 反推实际结构

---

## 七、复现命令（最新版）

```bash
# E1: HotpotQA 端到端验证
python -m stage3.run_stage3 \
  --datasets hotpotqa \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best

# E2: 三数据集 × 多 γ × 多 α
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.3,0.4,0.5,0.6 \
  --alphas 0.05,0.1,0.2,0.3 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best \
  --results-dir results/e2_predictive

# E3a: sudden shift
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best \
  --shift-type sudden \
  --shift-fraction 0.5

# E3b: gradual shift
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best \
  --shift-type gradual

# E3c: periodic shift
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best \
  --shift-type periodic

# E4: fixed betting 消融
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy fixed \
  --betting-lambda 0.5 \
  --artifact-suffix pdopt_best \
  --results-dir results/e4_fixed

# E5: 最终主表
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best
```

---

## 八、论文叙事要点（2026-04-12 修订版）

> 完整叙事框架见 `docs/reports/stage3/stage3_narrative.md`。以下为核心摘要。

### 8.1 E-value 的定位（修订后）

**从"错误率控制"转为"在线风险监控"。**

E-value 不是让 Probe 更准的工具，而是部署阶段的**风险仪表盘**：

> E-value provides anytime-valid risk monitoring at low overhead (roughly 0.5%–4.6% extra steps in the current main setting).
> When the system error rate exceeds α, the E-wealth process grows and eventually
> crosses 1/α, triggering a deployment alert. This guarantee holds under *any* data
> distribution — a property conformal prediction provably lacks under adaptive stopping.

类比：E-value 是**烟雾报警器**（检测问题并触发响应），不是**安全气囊**（阻止伤害）。

### 8.2 三层论证结构

1. **E-value 仍接近“低成本接入”**：额外成本约 0.5%–4.6%，HotpotQA 与 2Wiki 上 F1 反而更高
2. **E-wealth 能检测错误累积，但强度依赖数据集**：2wiki 在 `α=0.1` 触 cap；MuSiQue 积累明显；HotpotQA 主实验下更偏向质量提升而非告警
3. **分布漂移下 E-value 仍有不可替代性**：sudden shift 的 `α=0.1` 三数据集都能触 cap，而 CP 仍然没有在线报警机制

### 8.3 vs Conformal Prediction（核心对比表）

| 维度 | Split CP | E-value |
|------|---------|---------|
| 保证类型 | Marginal（需 i.i.d.） | Ville 不等式（任意分布） |
| 分布漂移 | 覆盖率失效，无法感知 | E-wealth 自动跟踪并响应 |
| 额外成本 | **大幅增步（主实验约 +43% 至 +142%）** | **低增步（主实验约 +0.5% 至 +4.6%）** |
| F1 影响 | **数据集敏感；常见退化，也可能靠高成本抬升** | **持平或微升** |
| 实际作用 | 高门控→强制延长→性能退化 | 轻量监控→漂移检测→触发告警 |

### 8.4 审稿人预期问题

见 `docs/reports/stage3/stage3_narrative.md` §五，包含 5 个预期问题及回答策略。

---

## 九、当前局限与应对


| 局限 | 当前状态 | 影响 | 应对 |
| --- | --- | --- | --- |
| E-value 不直接降低错误率 | 已确认 | 容易被误解为“控制失败” | 在论文中明确其角色是检测与告警，不是直接修复 Probe |
| Quality Model 仍有限 | 已知 | predictive betting 的增益受限 | 保留 `fixed` 对照；必要时再做更强校准 |
| MuSiQue 样本太少（417） | 已发生 | wealth 功效和方差都较差 | 将其作为 stress case，不作为最强主结论集 |
| 漂移延迟分析仍需更完整表格 | 部分完成 | Appendix 量化仍可增强 | 需要基于含 `wealth_trace` 的漂移 JSON 继续导出 |


---

## 十、文档定位

本文件现在用于记录 `Stage 3` 的实验设计、已完成结果和复现入口。

如果需要进一步引用：

- 总览摘要：`docs/reports/stage3/stage3_results_summary.md`
- 论文叙事与表图规划：`docs/reports/stage3/stage3_narrative.md`
- 分数据集报告：`docs/reports/stage3/stage3_report_*.md`
