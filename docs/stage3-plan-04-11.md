# Stage3 修订计划：2026-04-11

> 状态：Stage3 初版实验（E1-E5）已完成，暴露出两类结构性问题，本文档记录诊断、修复方案与后续执行计划。

---

## 一、已完成实验与核心产出

### 1.1 执行摘要


| 实验  | 命令                                                                                     | 产出                                               |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| E1  | `--datasets hotpotqa --gammas 0.5 --alphas 0.1,0.2`                                    | `stage3_evalue_hotpotqa.json`                    |
| E2  | `--datasets hotpotqa,musique,2wiki --gammas 0.3,0.4,0.5,0.6 --alphas 0.05,0.1,0.2,0.3` | *(被 E5 覆盖，需重跑)*                                  |
| E3  | sudden / gradual / periodic shift，三数据集                                                 | `stage3_evalue_*_{sudden,gradual,periodic}.json` |
| E5  | `--datasets hotpotqa,musique,2wiki --gammas 0.5 --alphas 0.1,0.2`                      | 三数据集 JSON + Markdown 报告                          |


### 1.2 主结果（γ=0.5，α=0.1，predictive betting）


| 数据集      | Probe F1 | Probe 错误率 | E-value F1 | E-value 错误率 | E-value 步数 | quality_bar | 最终 wealth |
| -------- | -------- | --------- | ---------- | ----------- | ---------- | ----------- | --------- |
| HotpotQA | 0.5273   | 0.441     | 0.4764     | **0.490**   | 4.98       | **0.970**   | 8.46      |
| MuSiQue  | 0.1826   | 0.791     | 0.1328     | **0.849**   | 5.00       | 0.870       | 4.40      |
| 2wiki    | 0.3738   | 0.611     | 0.3738     | 0.611       | 3.37       | **0.000**   | 10.00     |


---

## 二、问题诊断

### 2.1 问题一：quality_bar 校准两极化

`tune_quality_bar_on_calib` 的目标是找到最大的 `bar`，使 p_hat ≥ bar 的 calib 子集满足 `经验错误率 ≤ α`。

**为什么失败：**

- HotpotQA：Probe 整体错误率 44%，只有极高置信度（p_hat ≥ 0.97）时才能将子集错误率压到 ≤ 10% → `quality_bar = 0.97`，几乎所有停止被阻断
- 2wiki：Probe 整体错误率 61%，即使 p_hat = 1.0 也无法使子集错误率 ≤ 10% → `quality_bar = 0.0`，门控完全失效
- 两种情形下均不是"适度门控"，而是"过度限制"或"完全放行"

```
Probe 错误率 >> α  →  tune_quality_bar_on_calib 失效
                       ↙                    ↘
       bar 过高（HotpotQA）          bar = 0（2wiki）
       ↓                             ↓
  几乎全部停止被阻断              E-value = Probe（无门控）
  → 强制到 max_k                 → 无任何改善
  → F1 更差，错误率更高
```

### 2.2 问题二：干预动作与 Probe 特性不兼容

当前的"干预"逻辑：`p_hat < quality_bar` 或 `wealth ≥ cap` → `continue`（跳过当前步，循环到下一步）

**根本矛盾：**

Probe 的 Phase C 阈值已经在 dev 上选到了 Pareto 最优点（F1-步数折衷最优）。强制继续到更多步数不会改善 F1，因为：

1. 更多步 = 更多召回文档 = 不一定更高质量
2. Probe 本身在多步后的停止概率也很高（大多数轨迹在 max_k 处有兜底步）
3. 最终 max_k 步的 F1 有时低于早期步

**数据支持：**

HotpotQA Probe 平均步数 2.85，F1=0.527；被迫走到 4.98 步后 F1=0.476（下降 0.051）。

### 2.3 问题三：E-wealth 已达 cap，但无实际干预机制

E-wealth 达到 1/α 理论上意味着"检测到系统错误率超过 α"，但在当前模拟框架里：

- 达到 cap 后 `exceeded_cap = True` → 后续所有步被强制继续
- 这并不能回溯纠正已发生的错误
- E-value 提供的是**检测**保证，不是**控制**保证

---

## 三、修复方案

### 3.1 修复 quality_bar 校准（P0 — 必须修）

**方案 A（推荐）：分位数校准**

用 calib 上 p_hat 的 α 分位数作为 bar，与 Conformal Prediction 思路对齐：

```python
def tune_quality_bar_on_calib(
    p_hat_stop: np.ndarray,
    error_stop: np.ndarray,
    *,
    target_error: float,
) -> float:
    ph = np.asarray(p_hat_stop, dtype=np.float64).reshape(-1)
    if ph.size == 0:
        return 0.0
    # 用 α 分位数：允许通过 (1-α) 比例的停止，阻断最低质量的 α 比例
    return float(np.quantile(ph, float(target_error)))
```

**与原方案对比：**


| 方案                 | HotpotQA bar | 2wiki bar   | 行为          |
| ------------------ | ------------ | ----------- | ----------- |
| 原版（最大子集 error ≤ α） | 0.970        | 0.000       | 过紧/完全无效     |
| 分位数（α 分位）          | 预期 ~0.3-0.5  | 预期 ~0.3-0.5 | 适度过滤底部 α 停止 |


**方案 B（备选）：在 error-based 搜索上加下界约束**

要求 bar 对应的样本数量 ≥ n * (1 - α)，防止 bar 过高导致子集过小：

```python
for bar in grid:
    mask = ph >= bar - 1e-12
    if np.sum(mask) < len(ph) * (1.0 - target_error) * 0.5:  # 至少保留 50% 样本
        continue
    ...
```

### 3.2 修复干预动作（P1 — 重要）

当前：被阻断 → 强制继续（循环到下一步直到 max_k）

**方案：最优历史步兜底**

当所有步都被阻断（或 wealth 超 cap）时，选 p_hat 最高的历史步，而不是最后的 max_k 步：

```python
# 修改 simulate_evalue_outcome_aware 中的逻辑
best_step = None
best_phat = -1.0

for step in steps:
    k = int(step.get("step", 0))
    ...
    phat = predict_success_prob(quality_model, z)
    
    if phat > best_phat:
        best_phat = phat
        best_step = step
    
    should_gate = (phat < quality_bar) or tracker.exceeded_cap
    if should_gate:
        continue  # 继续寻找更好的步
    
    chosen = step  # 允许停止
    break

# 若所有步都被阻断，退回到 best p_hat 步（而非 max_k 步）
if chosen is steps[-1] and best_step is not None:
    chosen = best_step
```

### 3.3 叙事层面调整（P2 — 论文）

**原叙事（V1 标准中声称的）：**

> 累积错误率 ≤ α（三数据集均成立）

**修订后的准确叙事：**

> E-value 提供**在线检测保证**（Ville 不等式）：若系统真实错误率 > α，E-wealth 以概率 1 最终超过 1/α；若真实错误率 ≤ α，E-wealth 超过 1/α 的概率不超过 α。在分布漂移下，该保证无需 i.i.d. 假设，Split-CP 固定阈值的 marginal 保证则不成立。

**核心卖点从"错误率控制"转为"分布无关的在线风险监控"：**


| 属性   | Split CP（当前对照）   | E-value（修订定位） |
| ---- | ---------------- | ------------- |
| 保证类型 | Marginal（i.i.d.） | Anytime（任意分布） |
| 分布漂移 | 覆盖率失效            | E-wealth 自动响应 |
| 实际作用 | 过滤低质量停止          | 检测系统性违规       |
| 主要指标 | 错误率              | E-wealth 曲线   |


---

## 四、修复后预期行为

### 4.1 HotpotQA 预期

分位数校准后（quality_bar ≈ 0.3-0.5）：

- 约 α=10% 的 Probe 停止被阻断（低质量停止）
- 被阻断时退回到 best p_hat 步（而不是 max_k）
- 预期结果：步数轻微增加（2.85 → ~3.0-3.2），F1 持平或微升（错误停止被过滤）
- quality_bar 调整后，E-value 与 Probe 差距缩小

### 4.2 2wiki 预期

分位数校准后（quality_bar ≈ 0.3-0.5）：

- 当前完全无效（bar=0）的问题消除
- 约 10% 低置信度停止被阻断
- E-value ≠ Probe（有实质差异）

### 4.3 验收标准调整


| 编号    | 原标准                      | 修订标准                                |
| ----- | ------------------------ | ----------------------------------- |
| V1    | E-value 错误率 ≤ α          | ~~已取消~~（不是 E-value 的正确保证类型）         |
| V2    | 步数 ≤ Probe × 1.5         | 步数 ≤ Probe × 1.3（更严格；分位数校准后步数增加应可控） |
| V3    | F1 ≥ Probe × 0.95        | F1 ≥ Probe × 0.97（修复后不应有大幅 F1 下降）   |
| V4    | 分布漂移下不突破 α 线             | 分布漂移下 E-wealth 响应速度快于 CP（主要论点）      |
| V5（新） | quality_bar ∈ [0.1, 0.9] | quality_bar 合理，不出现 0 或 >0.95 的极端情形  |


---

## 五、后续执行步骤

### Step 1：修复 quality_model.py（0.5 天）

- 将 `tune_quality_bar_on_calib` 改为分位数校准
- 同时保留原方案作为 `tune_quality_bar_error_based`（对照用）
- 新增 `calib_method: Literal["quantile", "error_rate"]` 参数

### Step 2：修复 stopping.py（0.5 天）

- `simulate_evalue_outcome_aware` 中加入 best_step 兜底逻辑
- 同时跟踪 `best_phat_step` 和 `max_k_step`，gate 失败时选前者

### Step 3：重跑 E5 验证修复（0.5 天）

```bash
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 \
  --alphas 0.1,0.2 \
  --betting-strategy predictive
```

验证 quality_bar 是否在合理范围（0.1-0.9），E-value 步数是否在可接受区间。

### Step 4：重跑 E2（多 γ × 多 α 扫描）（0.5 天）

E2 的多 γ 结果被 E5 覆盖，需重跑：

```bash
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.3,0.4,0.5,0.6 \
  --alphas 0.05,0.1,0.2,0.3 \
  --betting-strategy predictive
```

### Step 5：重跑 E3（分布漂移）（0.5 天）

修复后验证分布漂移实验是否仍有差异：

```bash
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 --alphas 0.1,0.2 \
  --shift-type sudden --shift-fraction 0.5
```

关注：sudden shift 下 E-wealth 是否比 none-shift 上升更快（预期是）。

### Step 6：补充 fixed betting 对照（0.5 天）

E4（betting 策略对比）尚未单独跑，可与 Step 4 合并：

```bash
python -m stage3.run_stage3 \
  --datasets hotpotqa,musique,2wiki \
  --gammas 0.5 --alphas 0.1,0.2 \
  --betting-strategy fixed --betting-lambda 0.5
```

---

## 六、风险与应对


| 风险                       | 概率  | 影响      | 应对                                                |
| ------------------------ | --- | ------- | ------------------------------------------------- |
| 分位数 bar 校准后 E-value 仍无改善 | 中   | 修复无效    | 改试方案 B（加样本数下界约束）；或调低 γ 使 Probe 错误率 < α            |
| best_step 兜底后步数增加超 1.3 倍 | 中低  | V2 不达标  | 限制兜底只在 p_hat 差异 > ε 时生效                           |
| 分布漂移实验修复后无显著差异           | 中低  | E3 论点弱化 | 加大漂移强度（shift_fraction → 0.3）；增加 gradual shift 对比  |
| MuSiQue 样本少（417）导致结论不稳定  | 已知  | CI 宽    | 聚焦 HotpotQA（1000 样本）+ 2wiki（1000 样本）；MuSiQue 仅作参考 |


---

## 七、当前已有产出（不需重跑的部分）

- **E3 分布漂移数据（sudden/gradual/periodic）**：即使修复后 E-value 行为变化，漂移数据仍可用于展示 E-wealth trace 的响应特性，只是对比基线 Probe 变了
- **Quality Model Brier Score / ECE**（HotpotQA Brier=0.162，ECE=0.032）：质量模型本身校准良好，不是问题所在
- **Stage2 探针 checkpoint**：不需要重训，Stage3 修复只在校准层面
- **wealth trace 图、累积错误率图**：格式正确，修复后自动重新生成

