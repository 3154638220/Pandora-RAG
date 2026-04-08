**项目名称：** Pandora's RAG: Optimal Stopping for Multi-Hop Retrieval with Anytime-Valid Risk Control
（潘多拉 RAG：基于任意时刻有效风险控制的多跳检索最优停止）
**目标会议：** emnlp 2026

**说明文档**（除本 README 外，仓库内 Markdown 均集中在 [`docs/`](docs/)：`plan.md`、`experiments.md`、Stage 报告与特征诊断等。）

---

## 一、 项目背景与核心动机

在多跳复杂问答（Multi-hop QA）中，迭代检索增强生成（Iterative RAG, 如 IRCoT）已成为主流范式。然而，迭代检索面临一个根本性的**最优停止问题（Optimal Stopping Problem）**：
1. **继续检索**可能带来信息增益（提高回答质量），但必然产生延迟和 API 成本；
2. **提前停止**可以节省成本，但可能因信息不足导致幻觉或错误。

现有工作（如 Stop-RAG 或启发式规则）缺乏理论最优性保障，且在数据自适应停止（Data-adaptive stopping）的情况下，传统的共形预测（Conformal Prediction）会失效，无法保证严格的错误率控制。

本项目**首次**将经济学中经典的 **Weitzman's Pandora's Box** 最优停止理论引入 NLP 领域，提出 **Pandora-RAG** 框架，通过神经网络近似（Neural Reservation Value）和 E-value 机制，实现兼具理论最优解与任意时刻严格错误率控制的智能 RAG 停止策略。

---

## 二、 问题建模与数学公式 (Problem Formulation & Math)

我们将多跳 RAG 的检索过程建模为 **Sequential Pandora's Box with Fixed Order（固定顺序的序贯潘多拉魔盒）** 变体。

### 1. 环境与状态定义
*   **输入查询**：$q$
*   **检索步数**：$k \in \{1, 2, ..., K\}$，最大允许检索轮数为 $K$。
*   **状态空间**：在第 $k$ 步，累积上下文状态为 $s_k = [q, d_1, d_2, ..., d_k]$。
*   **探测成本**：每执行一次检索，产生已知成本 $c_k > 0$（代表时间/计算开销）。
*   **信息增益**：第 $k$ 步检索带来的奖励（质量提升）为随机变量 $G_k \sim F_k(\cdot | s_{k-1})$，分布函数依赖于当前状态。
*   **回答质量**：在状态 $s_\tau$ 下停止并生成答案的最终质量得分（如 F1 score）为 $Q(s_\tau)$。

### 2. 目标函数 (Objective)
策略 $\pi$ 需要决定一个停止时刻 $\tau \in \{0, 1, ..., K\}$，使得期望净收益最大化，同时满足用户定义的错误率上限 $\alpha$：
$$
\max_\tau \mathbb{E} \left[ Q(s_\tau) - \sum_{k=1}^{\tau} c_k \right] \quad \text{s.t.} \quad P(\text{Error at } \tau) \leq \alpha 
$$

### 3. Weitzman 最优停止策略与保留值 (Reservation Value)
在 $G_k$ 条件独立假设下，Weitzman 定理指出最优策略具有阈值结构。对于每一步检索，存在一个**保留值 $r_k^*$**，它是使“继续检索的期望边际收益”等于“检索成本”的临界点：
$$
c_k = \int_{r_k^*}^{\infty} (x - r_k^*) dF_k(x) 
$$
**全局（静态）Weitzman 基线**：在训练集上估计每步增益的经验分布，解出**与 query 无关**的全局阈值 $\{r_k^*\}$，在测试时一律用 $Q(s_k)$ 与 $r_{k+1}^*$ 比较。这是分布已知但**非上下文**的近似，用于对照说明 contextual 停止的必要性。

**实例级 Oracle（Stage 1 主线上界）**：对**单条轨迹**在已知全程真实质量 $Q(s_k)$（如 F1）下，用后向归纳定义最优价值；每步成本 $c_k$ 可与理论形式 $\,Q(s_\tau) - \sum_{j=1}^{\tau} c_j\,$ 对齐——默认取常数 $c_k \equiv c$，也可按轨迹缓存的 `token_count` 或 `latency_ms` 归一化为随步递增的成本（见 `stage1.run_stage1 --oracle-cost-metric`）：
$$
V_K^i = Q(s_K^i), \quad V_k^i = \max\bigl(Q(s_k^i),\; V_{k+1}^i - c_{k+1}\bigr).
$$
在**最小**的 $k$ 满足 $Q(s_k^i) \geq V_{k+1}^i - c_{k+1}$ 时停止（否则在第 $K$ 步停止）。这对应上帝视角下**完美的上下文保留值**（未来收益被精确预知），作为 Pareto 前沿的理论天花板。**缺失步**（轨迹未观测到的 $k$）上 $Q$ 沿用上一观测 F1，成本记为基准 $c$，表示“空转一步”仍消耗资源。

写入 `artifacts/oracle/{dataset}/test_oracle_labels.jsonl` 的 `step_targets[k]` 为字典，除 `expected_continue_val`（即 $V_{k+1}^i - c_{k+1}$）外，还提供 `margin`（等于 `expected_continue_val` 与当前 $Q_k$ 之差）与 `action_label`（1=Continue，0=Stop）供 Phase 2 做回归或二分类。

### 4. 核心创新 1：Neural Reservation Value (神经保留值)
由于实际中真实分布 $F_k$ 未知，我们提出用 LLM 的内部隐藏状态（Hidden States）通过一个轻量级探针（Probe）直接**逼近实例级继续价值或边际**（由 Stage 1 的 DP Oracle 写入 `test_oracle_labels.jsonl` 的 `step_targets[k]`：`expected_continue_val` $= V_{k+1}^i - c_{k+1}$，`margin` $= (V_{k+1}^i - c_{k+1}) - Q_k$）：
$$
\hat{y}_k = f_\theta(h_k) \approx V_{k+1}^i - c_{k+1}
\quad\text{或}\quad
\widehat{\Delta r}_k \approx \mathrm{margin}_k
$$
其中 $h_k$ 是 LLM 处理当前状态 $s_k$ 时的最后一层隐藏状态或 Semantic Entropy（语义熵）。停止规则将 $\hat{y}_k$ 与当前 $Q(s_k)$ 比较（或与 `margin` 符号一致的二分类头），与 Oracle 的阈值结构同型，但阈值随上下文变化。

### 5. 核心创新 2：E-value Anytime-Valid 风险控制
由于停止时刻 $\tau$ 是依赖于数据的随机变量，标准共形预测的覆盖率保障会失效。我们使用基于 E-value 的序列测试（Testing by Betting）：
*   对于每个样本 $t$，定义 Betting Score：$e_t = \frac{1}{1-\alpha} \mathbb{1}\{Q(s_{\tau_t}) \geq \text{threshold}\}$
*   构造 E-process：$E_n = \prod_{t=1}^n e_t$
*   **Ville 不等式保障**：$P(\exists n \in \mathbb{N}: E_n \geq 1/\alpha) \leq \alpha$
当探针输出 $\hat{r}_k$ 触发停止时，只有当 E-process $E_n$ 满足条件时，才允许实际停止，从而在任何时候都提供数学上严密的 $1-\alpha$ 准确率下界保障。

---

## 三、 实验搭建与步骤 (Experimental Setup & Steps)

### 1. 数据集准备
选取多跳推理需要明确迭代检索的标准 Benchmark：
*   **HotpotQA** (2-hop)
*   **MuSiQue** (2 to 4-hop)
*   **2WikiMultiHopQA** (多实体多跳)

### 2. 评估指标 (Metrics)
*   **质量指标**：F1 Score, Exact Match (EM)
*   **效率指标**：Average Retrieval Steps (平均检索轮数), Latency (延迟)
*   **安全指标**：Empirical Error Rate (必须严格 $\leq \alpha$)

### 3. 实验落地四大阶段 (Implementation Steps)

#### Phase 1: 数据收集与 Oracle 验证 (可行性验证)
*   **操作**：在训练集上，强制执行完整的 $K$ 步检索（设 $K=5$）。记录每一步的文档 $d_k$、答案质量 $Q(s_k)$ 以及对应的 LLM 隐藏状态 $h_k$。
*   **Oracle 计算（主线上界）**：对每条轨迹用已知 $Q(s_k)$ 与步级成本 $c_k$ 做**后向归纳 DP**，得到实例最优停止步与逐步标签 `step_targets`（含 $V_{k+1}-c_{k+1}$、`margin`、`action_label`），并写入 `artifacts/oracle/{dataset}/test_oracle_labels.jsonl`。CLI：`--oracle-cost-metric {fixed,token,latency}`；非 `fixed` 时 Stage1 Pareto 横轴为平均累计归一化成本。
*   **全局 Weitzman 基线**：仍在训练集上估计每步增益分布并解全局 $r_k^*$，在 Pareto 图中以 **Global-Weitzman** 点与 **Oracle（DP）** 对比，体现「静态阈值 vs 上下文 Oracle」的差距。
*   **实验**：以 DP Oracle 为天花板绘制 Stage1 Pareto；可选分析 Global-Weitzman 作为非 Oracle 的对照。

#### Phase 2: Neural Probe 训练 (核心算法实现)
*   **网络结构**：基于 $(h_k, y_k)$ 数据对训练小型 MLP $f_\theta$，其中 $y_k$ 来自 Phase 1 的 `step_targets`（优先 `expected_continue_val` 或 `margin`），而非全局常数 $r_{k+1}^*$；回归困难时可改用 `action_label` 做二分类。
*   **损失函数**：最小化 $\mathcal{L}(\theta) = \mathbb{E}[(\hat{y}_k - y_k)^2]$（或与停止决策一致的替代损失）。
*   **测试**：分析探针的预测误差 $\epsilon$，并在验证集上测试仅依赖探针的停止策略效率。

#### Phase 3: E-value 风险控制集成
*   **操作**：将 Gauthier et al. 的 E-value 实现逻辑（Python 自定义函数）集成到推理 pipeline 中。
*   **测试**：设置目标错误率 $\alpha = 0.1, 0.2$，绘制 Empirical Error Rate 随样本数量 $n$ 变化的曲线，证明我们的方法不会突破 $\alpha$ 红线（Anytime-valid）。

#### Phase 4: PPO 联合微调 (Optional/Correction)
*   **操作**：当 Probe 精度受限时，使用 PPO 算法，将“是否停止”作为 action，将 $R = Q(s_\tau) - \lambda \cdot \tau$ 作为 reward 进行微调策略。
*   **测试**：进行完整的端到端测试。

---

## 四、 对比的 SOTA 和 Baseline

为了全面证明 Pandora-RAG 的优越性，实验将设立以下几类对比基线：

### 1. 静态/启发式基线 (Static / Heuristic)
*   **Single-RAG (Standard)**：一次性检索 Top-K 文档直接生成。
*   **IRCoT (Fixed-K)** *(ICLR 2023)*：固定迭代 $K$ 次检索，不动态停止。
*   **Threshold-based RAG**：基于大模型输出概率置信度低于阈值时才检索（无理论保障）。

### 2. 动态停止的强 SOTA (Dynamic Stopping)
*   **Stop-RAG** *(NAACL 2024)*：基于 Q-learning 的动态停止 RAG。（**最直接的竞争对手**，但无错误率控制保障）。
*   **ITER-RETGEN** *(EMNLP 2023)*：基于大模型自我评估的迭代生成。

### 3. 风险控制与共形预测最新 SOTA (Risk-Control)
*   **CCPO (Conformal Calibration for Policy Optimization)** *(NeurIPS 2024)*：使用 RL+标准共形预测控制 RAG 行为。（我们将证明在 $T \geq 5$ 长序列或自适应停止时，其覆盖率会崩溃，而我们的 E-value 依然稳健）。
*   **Conformal-RAG** *(arXiv 2025.06)*：虽用于检索后过滤（Post-retrieval filtering），但可作为基于共形预测的质量控制基准对比。

### 4. 消融实验基线 (Ablations)
*   **Oracle-Pandora-RAG**：使用 DP 给出的实例级继续价值 / 最优停止（展示性能天花板）。
*   **Global-Weitzman**：使用训练集估计的全局 $r_k^*$ 阈值（静态策略，非 Oracle）。
*   **Pandora-RAG w/o E-value**：移除 E-value 校准，退化为纯启发式停止。
*   **Pandora-RAG w/ Standard CP**：替换 E-value 为标准的共形预测（展示标准 CP 在自适应停止下的失效）。

---

## 五、 预期理论贡献与可能遇到的风险应对

**预期理论贡献：**
1. 给出 Sequential Pandora's Box with Fixed Order 变体在 NLP 信息获取中的首个正式定义及阈值最优性证明。
2. 证明 Neural Reservation Value 逼近误差 $\epsilon$ 对总预期成本的严格误差界。

**风险防控 (Risk Mitigation)：**
*   **信息增益非单调性（有时后续检索会突然找到金文档）**：Weitzman 保留值的积分公式 $\int_{r^*}^\infty (x-r^*)dF(x)$ 本质上考虑了长尾高收益分布，能够在理论上自然包容这种“突破性”增益，我们将在论文讨论部分明确论证这一点。
*   **Probe 训练难度**：如果 Hidden State 回归 $r^*$ 不准，我们将引入 Semantic Entropy (语义熵) 或 Self-consistency 比例作为 Probe 的补充输入特征。

---

**拟定时间表：** 
- 第1-2周：完成环境搭建与 Phase 1 (Oracle 计算与基线跑通)
- 第3-4周：完成 Phase 2 (Neural Probe 训练) 与理论公式推导
- 第5-6周：集成 Phase 3 (E-values) 并开展在三大数据集上的主实验与消融实验
- 第7-8周：撰写论文并打磨图表。