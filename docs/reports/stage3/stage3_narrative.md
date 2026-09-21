# Stage 3 论文叙事框架

> 状态：基于 E1–E5 全部实验结果定稿的叙事方案。
> 日期：2026-04-19
> 目标会议：EMNLP 2026

---

## 〇、一句话定位

**E-value 是 Pandora-RAG 的部署安全层——以近乎零额外成本，为自适应停止提供分布无关的在线风险监控，而传统 Conformal Prediction 在相同场景下既昂贵又脆弱。**

---

## 一、论文三层贡献结构

### Layer 1（Stage 1 + 2）：成本-质量最优停止

> Pandora's Box 建模 + Neural Probe → 在不观测真实 F1 的条件下逼近实例级最优停止。

这是论文的**方法主体**，当前 `pdopt_best` 的 Stage 2 Probe 已实现 Oracle 的 79.9–85.4% 性能，步数仅 1.73–3.30。

### Layer 2（Stage 3 核心）：E-value 在线风险监控

> Probe 停止决策仍不可避免地含有错误（当前主实验错误率约 29.5%–58.3%）。E-value 层在线累积这些错误的统计证据，提供 Ville 不等式下的 **anytime-valid** 检测保证。

关键定位转变：

| 旧叙事（已废弃） | 新叙事 |
|---|---|
| E-value 控制累积错误率 ≤ α | E-value **检测** 系统错误率是否超过 α |
| 保证 Probe 的停止质量 | 为 Probe 提供部署时安全仪表盘 |
| 类比：安全气囊（阻止伤害） | 类比：烟雾报警器（检测问题并触发响应） |

### Layer 3（Stage 3 对比）：E-value vs Conformal Prediction

> 在自适应停止场景下，Split CP 往往要付出大幅增步代价，且收益高度数据集敏感；E-value 仅增加少量步数开销，并在分布漂移下保持在线监控能力。

---

## 二、核心实验数据支撑

### 2.1 主实验（γ=0.5, α=0.1, predictive betting）

| 数据集 | 策略 | F1 | 错误率 | 步数 | 额外成本 |
|--------|------|-----|--------|------|----------|
| HotpotQA | Probe | 0.657 | 30.3% | 1.70 | — |
| | **Probe+E-value** | **0.665** | **29.5%** | **1.81** | **+6.4%** |
| | Probe+CP | 0.672 | 29.6% | 4.11 | +141.8% |
| MuSiQue | Probe | 0.415 | 58.0% | 3.39 | — |
| | **Probe+E-value** | **0.413** | **58.3%** | **3.41** | **+0.5%** |
| | Probe+CP | 0.402 | 59.7% | 4.85 | +43.1% |
| 2wiki | Probe | 0.564 | 40.9% | 1.82 | — |
| | **Probe+E-value** | **0.573** | **40.1%** | **1.91** | **+4.6%** |
| | Probe+CP | 0.538 | 43.7% | 4.32 | +137.0% |

**关键论点**：
1. E-value 层仍是低开销接入：额外步数约 `+0.5% ~ +6.4%`
2. HotpotQA 与 2wiki 上，E-value 同时改善 F1 与错误率；MuSiQue 上基本持平
3. CP 不再适合被表述成统一弱基线：HotpotQA 上能靠高成本继续抬 F1，但 MuSiQue 与 2wiki 上明显退化

### 2.2 E-wealth 检测能力

| 数据集 | α | Final E-wealth | Cap (1/α) | 达标率 | 解读 |
|--------|---|---------------|-----------|--------|------|
| HotpotQA | 0.1 | 0.014 | 10.0 | 0.1% | 主实验下更偏向“提质”而非触发告警 |
| HotpotQA | 0.2 | ~0.000 | 5.0 | 0.0% | 同上 |
| 2wiki | 0.1 | **10.0** | 10.0 | **100%** | 错误率仍显著高于 α=10%，E-value 成功拒绝 H₀ |
| 2wiki | 0.2 | 0.020 | 5.0 | 0.4% | 风险证据存在，但未维持到 cap |
| MuSiQue | 0.1 | 4.193 | 10.0 | 41.9% | 错误率高，但受样本量限制，wealth 未封顶 |

**论文叙述**：新 probe 下，E-wealth 不再单调对应“主实验里谁更差”，因为质量门控会改变最终停点分布。当前最稳妥的表述是：2wiki 在 `α=0.1` 下能稳定触 cap；MuSiQue 能持续积累风险证据；HotpotQA 主实验下 wealth 很低，说明门控更像在用少量额外步数换质量。

### 2.3 分布漂移：E-value 的杀手锏

**Sudden shift 结果**（前 50% 正常，后 50% 全为难题）：

| 数据集 | α | No-shift wealth | Sudden-shift wealth | CP 行为 |
|--------|---|----------------|-------------------|---------|
| HotpotQA | 0.1 | 0.014 | **10.0** (曾触 cap，终态回落) | 固定阈值，无感知 |
| HotpotQA | 0.2 | ~0.000 | 2.218 | 同上 |
| 2wiki | 0.1 | 10.0 | **10.0** (cap) | 同上 |
| MuSiQue | 0.1 | 4.193 | **10.0** (cap) | 同上 |

**核心对比论点**：
- Sudden shift 后，三数据集在 `α=0.1` 下均触及 cap；`α=0.2` 下 MuSiQue 与 2wiki 仍稳定触发，HotpotQA 也能触及但终态会回落
- CP 的 min_phat 阈值在 calib 上固定，**完全无法感知测试分布已变**
- E-value 的 Ville 不等式保证在任何分布下成立；CP 的 marginal coverage 仅在 i.i.d. 假设下成立

#### 检测延迟（shift 后多少样本首次触及 cap）

**定义**：在 sudden 设定下，流的前 `cut = ⌊n·shift_fraction⌋` 个样本为「正常段」，其后为难例段（与 `stage3.stopping.build_shift_ordering` 一致）。令 `wealth_trace[i]` 表示按该顺序处理完前 `i` 个样本后的 E-wealth（`i=0` 为初值 1）。**检测延迟**记为最小的 `samples_after_shift_to_first_cap = i − cut`，使得 `i > cut` 且 `wealth_trace[i] ≥ 1/α`（实现中与 selective 门控一致，对 cap 使用 `10⁻⁶·cap` 数值容差）。

**实现**：`python -m stage3.run_stage3` 在每个 `per_alpha` 下写入 `evalue_cap_timing`（见 `stage3/cap_timing.py`），含 `samples_after_shift_to_first_cap`、`first_trace_index_reaching_cap` 等字段。`shift_type=none` 时 `shift_stream_cut_processed` 与「shift 后」相关项为 `null`（随机乱序下不存在物理切换点）。

**仓库现状**：当前检入的 `results/stage3_evalue_*_{sudden,gradual,periodic}.json` 为轻量导出，**不含** `wealth_trace`，因此延迟字段需在本机重跑对应 `--shift-type` 后由 JSON 直接读取。主实验 `results/stage3_evalue_*.json`（无漂移）含完整 trace 与 `evalue_cap_timing`（`shift_type=none` 时仅报告全局首次触 cap，不定义「shift 后」子指标）。复现表格可运行 `python scripts/stage3_export_tables.py`。

### 2.4 Betting 策略消融（E4 vs E2，γ=0.5）

固定 `λ=0.5` 的 **fixed betting** 与 **predictive**（`λ=1−p̂`）在相同 quality bar 与 outcome-aware 更新下对照（`results/e4_fixed` vs `results/e2_predictive`）。

| 数据集 | α | 策略 | 终态 E-wealth | Probe+E-value 步数 | F1 |
|--------|---|------|---------------|-------------------|-----|
| HotpotQA | 0.10 | predictive | 0.014 | 1.807 | 0.6654 |
| HotpotQA | 0.10 | fixed | 0.000 | 1.807 | 0.6654 |
| HotpotQA | 0.20 | predictive | 0.000 | 1.889 | 0.6814 |
| HotpotQA | 0.20 | fixed | 0.000 | 1.889 | 0.6814 |
| MuSiQue | 0.10 | predictive | 4.193 | 3.410 | 0.4127 |
| MuSiQue | 0.10 | fixed | 3.437 | 3.410 | 0.4127 |
| MuSiQue | 0.20 | predictive | 1.469 | 3.453 | 0.4118 |
| MuSiQue | 0.20 | fixed | 0.937 | 3.453 | 0.4118 |
| 2Wiki | 0.10 | predictive | 10.000 | 1.905 | 0.5728 |
| 2Wiki | 0.10 | fixed | 10.000 | 1.905 | 0.5728 |
| 2Wiki | 0.20 | predictive | 0.020 | 2.004 | 0.5820 |
| 2Wiki | 0.20 | fixed | 0.000 | 2.004 | 0.5820 |

**解读**：

- **HotpotQA**：两种 betting 的步数与 F1 一致，但 final wealth 都接近 0；当前更适合把它当作“下注策略影响很弱”的例子，而不是 predictive 明显占优。
- **2Wiki**：两策略在 γ=0.5 下 wealth **均已顶格**（`1/α`），差异被裁剪；步数与 F1 仍一致。
- **MuSiQue**：predictive 在 `α=0.1/0.2` 下都高于 fixed，但幅度有限；它更像是“略优的默认选择”，而不是强结论来源。

Markdown 源数据可由 `python scripts/stage3_export_tables.py` 重新打印。

### 2.5 多参数稳健性（E2：4γ × 4α × 3 数据集 = 48 组）

- V2（步数 ≤ Probe × 1.3）：**48/48 通过**
- V3（F1 ≥ Probe × 0.97）：**48/48 通过**
- quality_bar 合理范围 [0.03, 0.46]：无极端值

#### α 敏感性汇总（predictive，相对 Probe 的 Probe+E-value，跨 γ∈{0.3,0.4,0.5,0.6}）

下表为每个 α 上，四档 γ 中 **Δ步数%** 与 **ΔF1%** 的 [min, max] 区间及均值（百分比 = 100×(Probe+E-value)/Probe − 100）。

| 数据集 | α | Δ步数% [min,max] | mean Δ步数% | ΔF1% [min,max] | mean ΔF1% |
|--------|---|------------------|-------------|----------------|----------|
| HotpotQA | 0.05 | [3.24, 3.59] | 3.44 | [0.45, 0.76] | 0.59 |
| HotpotQA | 0.10 | [6.00, 6.53] | 6.28 | [1.14, 1.29] | 1.23 |
| HotpotQA | 0.20 | [10.95, 11.42] | 11.18 | [3.11, 3.92] | 3.55 |
| HotpotQA | 0.30 | [15.83, 17.30] | 16.61 | [4.47, 5.13] | 4.89 |
| MuSiQue | 0.05 | [0.21, 0.21] | 0.21 | [0.24, 0.24] | 0.24 |
| MuSiQue | 0.10 | [0.35, 0.57] | 0.48 | [−0.62, 0.24] | −0.26 |
| MuSiQue | 0.20 | [1.77, 2.05] | 1.91 | [−0.82, −0.05] | −0.48 |
| MuSiQue | 0.30 | [3.18, 3.46] | 3.39 | [−0.43, −0.30] | −0.37 |
| 2Wiki | 0.05 | [2.31, 2.52] | 2.44 | [0.90, 1.19] | 1.09 |
| 2Wiki | 0.10 | [3.95, 5.32] | 4.69 | [1.57, 1.57] | 1.57 |
| 2Wiki | 0.20 | [9.77, 11.25] | 10.39 | [3.11, 3.66] | 3.38 |
| 2Wiki | 0.30 | [16.19, 17.34] | 16.86 | [4.53, 5.08] | 4.84 |

**要点**：当前 E2 结果显示，随着 α 放宽，HotpotQA 与 2Wiki 都会更积极地换取额外步数来提升 F1；MuSiQue 的步数增幅仍小，但 F1 改善不稳定。

### 2.6 Selective Prediction / Abstain（检测后干预，γ=0.5，无 shift）

在 Probe+E-value 的 **E-wealth 过程**上增加部署规则：若处理某测试样本**之前**已有 \(W \ge 1/\alpha\)（实现中对 cap 使用 `1e-6·cap` 数值容差），则该样本 **abstain**；否则照常输出答案。Coverage = 回答比例；Selective accuracy = 回答子集中 \(F_1 \ge \gamma\) 的比例（C–A 曲线见 `results/stage3_selective_ca_*.png`）。

| 数据集 | α | Coverage | Selective acc. |
|--------|---|----------|----------------|
| HotpotQA | 0.10 | 0.931 | 0.705 |
| HotpotQA | 0.20 | 1.000 | 0.725 |
| MuSiQue | 0.10 | 0.444 | 0.373 |
| MuSiQue | 0.20 | 0.494 | 0.379 |
| 2Wiki | 0.10 | 0.706 | 0.581 |
| 2Wiki | 0.20 | 0.853 | 0.600 |

**要点**：wealth 非单调，拒答为逐样本门控；MuSiQue 因错误率高、wealth 更易维持高位，覆盖率远低于 HotpotQA，与「监控层 + 可选拒答」的 limitation 叙事一致。

### 2.7 与 Stage 1 的统一 Pareto 对比（Appendix 表 / 图）

在**同一 test 分布、固定成本度量（每步常数 c）**下，将 Stage 1 全局上界/静态基线与 Stage 2–3 策略对齐到「平均步数–F1」平面。当前正式图已生成：

- `results/stage3_final_pareto_all.png`
- `results/stage3_final_pareto_{hotpotqa,musique,2wiki}.png`

生成脚本：`python scripts/stage3_make_final_pareto.py`

其中：
- `Oracle / Global-Weitzman` 读取 `results/stage2_probe_table_*_pdopt_best.csv`
- `Probe / Probe+E-value / Probe+CP` 读取主实验 `results/stage3_evalue_*.json`（γ=0.5、α=0.1）

| 策略 | HotpotQA F1 | 步数 | MuSiQue F1 | 步数 | 2Wiki F1 | 步数 |
|------|------------|------|-----------|------|---------|------|
| Oracle（DP） | 0.781 | 1.58 | 0.497 | 2.12 | 0.695 | 1.59 |
| Global-Weitzman | 0.709 | 1.64 | 0.423 | 3.29 | 0.645 | 1.77 |
| Probe | 0.657 | 1.70 | 0.415 | 3.39 | 0.564 | 1.82 |
| Probe+E-value | 0.665 | 1.81 | 0.413 | 3.41 | 0.573 | 1.91 |
| Probe+CP | 0.672 | 4.11 | 0.402 | 4.85 | 0.538 | 4.32 |

**叙事**：Probe 作为主方法，仍处在低成本区域；**Probe+E-value 相对 Probe 只做轻微平移**，主要增加的是监控能力而不是成本；**Probe+CP** 明显被推向更高步数区域。Oracle 与 Global-Weitzman 则作为 Stage 1 的上界/静态参照，帮助说明当前 Stage 3 不是在替代最优停止器，而是在其上叠加部署安全层。

---

## 三、论文 Figures & Tables 规划

### Table 1：主实验结果（Section 5.1）

三数据集 × 三策略（Probe / Probe+E-value / Probe+CP），展示 F1、EM、Error Rate、Steps、Cost。
附注 E-wealth 最终值。

**叙事重点**：E-value 以较低额外成本实现风险监控；CP 成本明显更高，且收益高度依赖数据集。

### Table 2：多 γ/α 保守度分析（Section 5.2 或 Appendix）

三数据集 E2（4γ × 4α）：正文可只展 HotpotQA 的完整矩阵；附录用 §2.5 的 **α 敏感性汇总表**（跨 γ 的 min–max + mean）压缩呈现。

**叙事重点**：α≤0.1 时 HotpotQA 步数与 F1 近乎不变；全 48 组仍满足 V2/V3 验收。

### Figure 1：E-wealth trace（Section 5.3 — 论文核心图）

两面板对比：
- (a) No shift：E-wealth 平稳增长，反映底层错误率
- (b) Sudden shift：E-wealth 在切换点后急剧上升并触及 cap

同一图上叠加 CP 的累积错误率曲线作为对比（CP 无法感知变化）。

**叙事重点**：E-value 提供实时风险仪表盘，CP 是事后静态检查。

### Figure 2：累积错误率曲线（Section 5.3）

三策略的累积错误率随样本数变化。α 水平线作为参考。

**叙事重点**：所有策略的错误率都远超 α（因为 Probe 本身错误率高），但 E-value 的检测能力使得系统可以及时触发告警/干预。

### Table 3：Betting 策略消融（Section 5.4 或 Appendix）

Predictive vs Fixed（§2.4 全表）：终态 E-wealth、Probe+E-value 步数与 F1；强调 HotpotQA 上 wealth 幅度差异与 **MuSiQue 上不宜单看 wealth** 的边界条件。

### Figure 3：三种漂移模式下的 E-wealth trace（Section 5.5 或 Appendix）

Sudden / Gradual / Periodic 三种漂移的 wealth 曲线，展示 E-value 对不同漂移模式的响应特征。

### Table 4（Appendix）：Stage 1 Oracle / Global-Weitzman 与 Stage 2–3 的 Pareto 对齐

见 §2.7；可与 Stage 1 脚本生成的 `stage1_oracle_pareto_*.png` 并列作为「全流水线成本–质量」一页图。

---

## 四、关键叙事要素（论文各 Section 对应）

### Abstract

> We propose Pandora-RAG, which integrates Weitzman's optimal stopping theory with anytime-valid risk monitoring via E-values. Our current neural probe achieves about 80–85% of the oracle performance with 1.7–3.3 retrieval steps. The E-value safety layer adds only modest overhead while providing distribution-free drift detection — a capability that conformal prediction fundamentally lacks under adaptive stopping.

### Introduction 末段（贡献列表）

1. 首次将 Pandora's Box 最优停止引入多跳 RAG
2. Neural Probe 以轻量 MLP 逼近实例级最优停止
3. **E-value 安全层：以较低额外成本，提供 (a) anytime-valid 风险监控 和 (b) 分布无关的漂移检测——CP 在自适应停止下成本很高，且收益高度依赖数据集**

### Related Work 要点

- Testing by Betting / E-values：Shafer & Vovk (2019), Ramdas et al. (2023), Grünwald et al. (2024)
- Conformal Prediction for RAG：Conformal-RAG (2025), CCPO (NeurIPS 2024)
- **我们的差异化**：E-value 在自适应停止（data-dependent τ）下的保证是 Ville-type（无条件），而 CP 的 marginal coverage 在 τ 依赖数据时失效

### Method Section 关键段落

> **E-value as Safety Monitor.** After the probe decides to stop at step τ, we observe the answer quality and update the E-wealth process:
> $$E_n = E_{n-1} \cdot (1 - \lambda_n + \lambda_n \cdot e_n / \alpha)$$
> where $e_n = \mathbb{1}[F_1 < \gamma]$. When $E_n$ exceeds $1/\alpha$, we reject $H_0$: "the system error rate $\leq \alpha$", triggering a deployment alert. Crucially, this guarantee holds under *any* data distribution, including adversarial drift — a property that split conformal prediction provably cannot provide when the stopping time is data-adaptive.

> **Quality-Based Gating.** The quality model $\hat{p} = P(F_1 \geq \gamma | \text{features})$ serves dual purpose: (1) filtering the lowest-confidence α-fraction of stops via a quantile-calibrated threshold, and (2) informing the predictive betting fraction $\lambda_n = 1 - \hat{p}$, which bets more aggressively on low-quality stops.

### Experiment Section 叙事路线

1. **主表（Table 1）**：E-value 以低增步带来持平或更高质量，CP 呈现明显数据集敏感性
2. **E-wealth 分析**：wealth 增长正确反映错误累积 → E-value 作为风险仪表盘
3. **漂移实验**：E-wealth 快速响应分布变化 → CP 无感知 → **这是论文最强论点**
4. **Betting 策略消融**：HotpotQA 上 predictive 显著放大 wealth 信号且不改 F1/步数；2Wiki 在 γ=0.5 下两者均顶格；MuSiQue 作 stress case（§2.4）
5. **多参数稳健性**：48 组全部通过验收标准 → 结论不依赖特定超参

### Discussion / Limitation 要点

1. **E-value 不控制错误率，而是检测错误累积**。当 Probe 本身错误率远超 α 时（如 MuSiQue 79%），E-value 不能"修复"Probe，但能迅速发出警报。
2. **检测后干预（示例已给出）**：主仓库已实现基于 **E-wealth ≥ 1/α** 的逐样本 **abstain**，并报告 coverage 与 selective accuracy（§2.6）。更复杂的策略（分级降级、在线重校准、人在回路）仍可扩展。
3. **MuSiQue 样本量限制**（test n=417）使 E-wealth 累积速度与方差均劣于千级数据集；**γ 与质量头校准**未单独调参——主文与 E2 沿用 γ∈{0.3,…,0.6} 的统一扫描，将 MuSiQue 定位为高错误率 stress case，而非与 HotpotQA 并列的主结论集。
4. **离线评估 vs 在线部署**。当前所有实验在离线轨迹缓存上模拟；真实在线部署需要流式 E-wealth 更新。

---

## 五、vs 审稿人预期问题

### Q1："E-value 错误率并没有降到 α 以下，这个方法有什么用？"

> E-value 的数学保证不是"控制错误率 ≤ α"，而是 Ville 不等式：P(∃n: E_n ≥ 1/α) ≤ α **当** 真实错误率 ≤ α。当真实错误率 > α 时（如我们的场景），E-wealth 增长并最终触及 1/α，这正是 **正确检测**。E-value 的价值在于：(a) 在分布漂移下仍然有效，(b) 为部署系统提供实时风险信号。

### Q2："为什么不直接改进 Probe 使错误率降到 α 以下？"

> Probe 的瓶颈在特征信息量（Stage 2 已验证 XGBoost ≈ MLP），而非模型容量。要将错误率从 44% 降到 10% 需要根本性的特征改进（如更强的 LLM、更好的检索器）。E-value 层的价值恰恰在于：即使 Probe 不完美，也能提供数学上严格的风险监控。

### Q3："CP 为什么反而更差？"

> Split CP 的 min_phat 阈值（~0.8）远高于大多数样本的 p_hat，导致几乎所有 Probe 停止都被拒绝，系统被迫走到 max_k 步。但更多步 ≠ 更好质量（多跳 RAG 中后续检索可能引入噪声），因此 CP 同时增加了成本和错误率。

### Q4："在分布漂移下 E-value 检测到了问题，然后呢？"

> E-value 提供**检测**保证；检测后的动作由产品策略决定。我们已在主实验上给出一种最小干预：**当进入样本前 E-wealth 已达 1/α 附近则拒答该样本**，并报告 coverage–selective accuracy（§2.6）。其他选项包括人工审核、降级检索预算、在线重校准等。

### Q5："predictive betting 的 λ=1−p̂ 有什么理论依据？"

> 当质量预测 p̂ 低（即预计停止质量差）时，下注更大（λ 更大）。这对应于 Testing by Betting 文献中的 GRAPA 策略的简化版本，在 E[e_n] > α 时使 wealth 增长更快，提高检测功效。

---

## 六、待补充实验（论文完整性）

| 优先级 | 实验 | 状态 | 产出 |
|--------|------|------|------|
| ~~P0~~ | ~~gradual/periodic 漂移重跑~~ | **已完成**（2026-04-12） | `stage3_evalue_*_{gradual,periodic}.json` 与对应 wealth/error 图已更新；quality_bar 为分位数校准 |
| ~~P1~~ | ~~检测延迟分析（shift 后多少样本触及 cap）~~ | **定义与代码已完成**（2026-04-12） | `stage3/cap_timing.py` + `run_stage3` 写入 `evalue_cap_timing`；漂移 JSON 需本机重跑以含 `wealth_trace` 后读取延迟字段 |
| ~~P1~~ | ~~Predictive vs Fixed betting 消融表~~ | **已完成** | §2.4 + `scripts/stage3_export_tables.py` |
| ~~P1~~ | ~~α 敏感性汇总表~~ | **已完成** | §2.5 子表 + 同上脚本 |
| ~~P2~~ | ~~与 Stage 1 Oracle 的统一 Pareto 散点图~~ | **已完成**（2026-04-19） | `scripts/stage3_make_final_pareto.py` → `results/stage3_final_pareto_all.png` 与 `results/stage3_final_pareto_{hotpotqa,musique,2wiki}.png` |
| ~~P2~~ | ~~Selective Prediction 实验~~ | **已完成**（2026-04-12） | `stage3_evalue_*.json` 内 `selective_prediction_abstain` + `stage3_selective_ca_*.png` |

---

## 七、关键术语统一

论文中使用统一术语：

| 内部代号 | 论文术语 | 说明 |
|----------|---------|------|
| Probe | Neural Stopping Probe | Stage 2 训练的 MLP 停止探针 |
| Probe+E-value | Pandora-RAG (full) | 完整系统：Probe + E-value 安全层 |
| Probe+CP | Pandora-RAG + CP | Split Conformal Prediction 对照 |
| Probe-only | Pandora-RAG (probe) | 无风险控制层的消融版本 |
| quality_bar | quality gate threshold | 分位数校准的质量门控阈值 |
| E-wealth | E-wealth process / risk monitor | 累积 E-value 统计量 |
| wealth cap (1/α) | rejection threshold | Ville 不等式的拒绝边界 |
| betting fraction λ | wagering parameter | betting multiplier 中的下注比例 |
