# Stage 3：E-value 风险控制实验计划

> 日期：2026-04-11
> 前置：Stage 1（轨迹缓存 + Oracle）完成；Stage 2（Neural Probe 最优配置）定案。

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


| 数据集      | Probe F1   | Oracle F1 | Probe/Oracle | 步数   | 配置                                                    |
| -------- | ---------- | --------- | ------------ | ---- | ----------------------------------------------------- |
| HotpotQA | **0.5273** | 0.6232    | 84.6%        | 2.85 | compress_dim=256, binary                              |
| MuSiQue  | **0.1826** | 0.2526    | 72.3%        | 3.25 | compress_dim=256, binary                              |
| 2Wiki    | **0.3738** | 0.5336    | 70.1%        | 3.37 | compress_dim=64, margin=0.02, binary, hidden_residual |


核心发现：

- **瓶颈在特征信息**，非模型容量（XGBoost ≈ MLP）
- **Binary 头 > F1 回归头**（全数据集一致）
- **各数据集行为异构**（per-dataset 配置必要）
- 序列建模（GRU / seqhist）未带来稳定收益

---

## 二、Stage 3 目标

### 2.1 核心目标

为 Probe 的停止决策提供**部署时数学保证**：

> 在任意时刻 n，累积错误率 P(F1 < γ 于停止时) ≤ α，且该保证**无条件成立**（Ville 不等式）。

这是论文与 CCPO / Stop-RAG 的**根本差异化**。

### 2.2 实验目标


| 编号  | 目标                      | 核心产出                                           |
| --- | ----------------------- | ---------------------------------------------- |
| E1  | E-value 端到端验证（HotpotQA） | E-wealth 曲线 + 保守度量化                            |
| E2  | 保守度分析（三数据集 × 多 γ × 多 α） | 步数增加量 vs 错误率控制表                                |
| E3  | 分布漂移实验                  | E-process vs 固定阈值 CP 的鲁棒性对比图（论文 Fig.3）         |
| E4  | Betting 策略调优            | 最优策略选型 + 超参                                    |
| E5  | 端到端主实验表                 | Table 1 的 Probe / Probe+E-value / baselines 全行 |


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

### 3.2 E-process 设计（结果感知型 — 核心改进）

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

性质：

- 若真实错误率 = α，则 E[M_n] = 1（鞅）
- 若真实错误率 < α，则 E[M_n] < 1（wealth 倾向减小 — 正确停止）
- 若真实错误率 > α，则 E[M_n] > 1（wealth 倾向增大 — 检测到问题）
- 当 E_n > 1/α 时拒绝 H₀ → 需要干预

**betting fraction 选择策略**：


| 策略             | λ_n 公式                          | 说明              |
| -------------- | ------------------------------- | --------------- |
| **Fixed**      | λ_n = λ（常数）                     | 最简单；λ=0.5 为保守默认 |
| **Predictive** | λ_n = clip(1 - p_hat_n, ε, 1-ε) | 质量差时下注更大        |
| **GRAPA**      | λ_n = argmax E[log M_n]         | 对数最优增长率         |
| **ONS**        | 基于历史梯度的在线牛顿法                    | 数据自适应，理论最优      |


推荐先实现 **Fixed + Predictive** 两种，论文中对比。

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

### 3.5 分布漂移实验设计

**目的**：展示 E-value 在分布漂移下的鲁棒性优势（论文 Fig.3）。

**漂移构造方式**（离线模拟）：


| 类型       | 方法                                  | 预期效果                        |
| -------- | ----------------------------------- | --------------------------- |
| **突变漂移** | test 前 50% 样本正常排列；后 50% 全换为低 F1（难题） | E-wealth 在后半段快速上升 → 触发更保守停止 |
| **渐变漂移** | 按难度排序 test 样本（F1 从高到低）              | E-wealth 逐渐上升，CP 固定阈值无法感知   |
| **周期漂移** | 交替排列简单/困难 batch                     | 展示 E-value 的动态跟踪能力          |


**对照组**：

- **Probe-only**：无门控
- **Probe + Split-CP**：Calib 上校准的固定 p_hat 门槛（`conformal_min_phat_threshold`）
- **Probe + E-value**：结果感知型

**指标**：

- 累积错误率曲线（y 轴）vs 样本数（x 轴）
- α 线是否被突破
- 平均步数（成本代价）

---

## 四、实验矩阵

### 4.1 超参网格


| 参数                 | 候选值                            | 说明                                  |
| ------------------ | ------------------------------ | ----------------------------------- |
| γ（质量阈值）            | 0.3, 0.4, **0.5**, 0.6         | F1 < γ 为错误；0.5 为主实验                 |
| α（名义水平）            | 0.05, **0.10**, **0.20**, 0.30 | 越小越保守                               |
| betting_strategy   | fixed, predictive              | Fixed: λ=0.5; Predictive: λ=1-p_hat |
| quality_model      | logreg, logreg+probe_prob      | 是否加入 Probe p_continue 作为特征          |
| distribution_shift | none, sudden, gradual          | 仅 E3 实验                             |


### 4.2 实验执行顺序


| 阶段  | 实验           | 数据集           | 预计时间  | 产出                           |
| --- | ------------ | ------------- | ----- | ---------------------------- |
| E1  | 端到端验证        | HotpotQA      | 0.5 天 | wealth 曲线、error rate 曲线、JSON |
| E2a | 多 γ/α 扫描     | HotpotQA      | 0.5 天 | γ×α 网格表                      |
| E2b | 三数据集全量       | 全部            | 0.5 天 | 保守度总结表                       |
| E3  | 分布漂移         | HotpotQA → 全部 | 1 天   | Fig.3 数据                     |
| E4  | Betting 策略对比 | 全部            | 0.5 天 | 策略对比表                        |
| E5  | 端到端主表        | 全部            | 0.5 天 | Table 1 全行                   |


---

## 五、验收标准

### 5.1 必须达标（Stage 3 完成条件）


| 编号  | 标准                                            | 说明                       |
| --- | --------------------------------------------- | ------------------------ |
| V1  | E-value 累积错误率 ≤ α（三数据集、α=0.1 和 0.2）           | Ville 不等式数学保证；若代码正确则必然成立 |
| V2  | 保守度可接受：Probe+E-value 步数 ≤ Probe-only 步数 × 1.5 | 步数增加不超过 50%              |
| V3  | Probe+E-value F1 ≥ Probe-only F1 × 0.95       | F1 下降不超过 5%              |
| V4  | 分布漂移下 E-value 曲线不突破 α 线，而 CP 突破               | 论文核心论点                   |


### 5.2 辅助指标

- Quality Model 在 Calib 上 Brier Score < 0.25
- E-wealth 最终值在合理范围（不接近 0 也不接近 1/α）
- Probe+E-value 在 HotpotQA 上 test F1 ≥ 0.50（绝对值）

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
- **新增**：`shift_type: str`（none / sudden / gradual）
- **新增**：`shift_fraction: float`（漂移起始位置）

### 6.5 run_stage3.py

- **新增**：`--gammas` CLI（逗号分隔多 γ）
- **新增**：`--betting-strategy` CLI
- **新增**：`--shift-type` / `--shift-fraction` CLI
- **新增**：wealth trace 折线图（E_n vs n）
- **新增**：多 γ × α 的汇总表输出
- **新增**：per-dataset-optimal checkpoint 自动发现
- **改进**：分布漂移主循环 + 对比图生成
- **改进**：Markdown 报告自动生成（`docs/stage3_report_{dataset}.md`）

### 6.6 adapters/stage2_probe.py

- **适配**：per-dataset-optimal checkpoint 路径自动拼接（`pdopt_best` 后缀）

---

## 七、复现命令（预期）

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
  --artifact-suffix pdopt_best

# E3: 分布漂移实验
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --shift-type sudden \
  --shift-fraction 0.5 \
  --artifact-suffix pdopt_best

# E5: 最终主表
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive \
  --artifact-suffix pdopt_best
```

---

## 八、论文叙事要点

### 8.1 E-value 的定位

E-value 提供**部署时安全保障层**，而非"让 Probe 更准"：

> When the Probe's stopping decision is suboptimal, the E-value safety valve
> intervenes to prevent cumulative error rate violations, providing deployment-time
> guarantees that no existing adaptive RAG method offers.

### 8.2 Probe 不完美 → E-value 必要

Stage 2 Probe 达到 Oracle 的 70-85%，剩余 gap 正是 E-value 存在价值的论据：

- Probe 错误不可避免（无法观测真实 F1）
- E-value 在 Probe 犯错时介入，将累积风险控制在 α 以内
- 数学保证（Ville 不等式）无条件成立

### 8.3 vs Conformal Prediction


| 属性   | Split CP         | E-value          |
| ---- | ---------------- | ---------------- |
| 保证类型 | Marginal（iid 假设） | Anytime（无分布假设）   |
| 分布漂移 | 失效               | 自适应（wealth 自然跟踪） |
| 数据效率 | 需独立 calib set    | 在线更新             |
| 保守度  | 固定               | 数据驱动             |


E-value 的核心优势在**分布漂移**下：当测试分布偏离 calib 时，Split CP 的覆盖保证失效，但 E-value 的 Ville 不等式仍成立。

---

## 九、风险与应对


| 风险                       | 概率  | 影响                    | 应对                                             |
| ------------------------ | --- | --------------------- | ---------------------------------------------- |
| E-value 过于保守（步数 +50% 以上） | 中   | 论文实用性受质疑              | 调 betting fraction；加 quality_bar 松弛；展示 γ/α 敏感度 |
| Quality Model 校准差        | 中低  | predictive betting 失效 | 回退 fixed betting；加 Platt scaling               |
| 分布漂移实验差异不显著              | 中低  | 论文亮点不够                | 加大漂移强度；多种漂移模式                                  |
| MuSiQue 样本太少（417）        | 已发生 | CI 宽                  | 报告 bootstrap CI；聚焦 HotpotQA + 2Wiki            |


---

## 十、时间线


| 日程       | 任务                                      | 产出                          |
| -------- | --------------------------------------- | --------------------------- |
| Day 1 上午 | 代码重构（evalue / quality_model / stopping） | 更新后的 Stage3 模块              |
| Day 1 下午 | E1 端到端验证（HotpotQA）                      | 首张 E-wealth + error rate 曲线 |
| Day 2    | E2 多 γ/α + E4 Betting 策略                | 超参网格表                       |
| Day 3    | E3 分布漂移实验                               | Fig.3 数据 + 对比图              |
| Day 4    | E5 三数据集全量 + 报告                          | Table 1 全行 + stage3_report  |


