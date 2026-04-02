**项目名称：** Pandora's RAG: Optimal Stopping for Multi-Hop Retrieval with Anytime-Valid Risk Control
（潘多拉 RAG：基于任意时刻有效风险控制的多跳检索最优停止）
**目标会议：** NeurIPS 2026 / ICLR 2027

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
**Oracle 最优停止规则**：如果当前生成答案的预期质量 $Q(s_k) \geq r_{k+1}^*$，则**停止检索**；否则**继续检索**。

### 4. 核心创新 1：Neural Reservation Value (神经保留值)
由于实际中真实分布 $F_k$ 未知，我们提出用 LLM 的内部隐藏状态（Hidden States）通过一个轻量级探针（Probe）直接预测保留值：
$$
\hat{r}_{k+1} = f_\theta(h_k)
$$
其中 $h_k$ 是 LLM 处理当前状态 $s_k$ 时的最后一层隐藏状态或 Semantic Entropy（语义熵）。

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
*   **Oracle 计算**：根据训练集统计的真实后验信息增益分布，反向算出真实的 Weitzman 保留值 $r_k^*$。
*   **实验**：使用 $r_k^*$ 作为停止规则，评估 Oracle Pandora-RAG 的表现（验证 Pandora's Box 框架的理论上界）。

#### Phase 2: Neural Probe 训练 (核心算法实现)
*   **网络结构**：基于收集到的 $(h_k, r_k^*)$ 数据对，训练一个小型 MLP $f_\theta$ 作为探针。
*   **损失函数**：最小化逼近误差 $\mathcal{L}(\theta) = \mathbb{E}[(\hat{r}_k - r_k^*)^2]$。
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
*   **Oracle-Pandora-RAG**：使用真实保留值（展示性能天花板）。
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