**项目名称：** Pandora-RAG: Adaptive Stopping for Multi-Hop Retrieval with Anytime-Valid Risk Control
（Pandora-RAG：基于任意时刻有效风险控制的多跳检索自适应停止）
**目标会议：** WWW 2027

**入口说明：** 当前实验代码保留在根目录；有效协议和项目状态见 `[docs/active/](docs/active/)`，阶段报告见 `[docs/reports/](docs/reports/)`，论文工作区见 `[paper/](paper/)`，审稿材料见 `[reviews/](reviews/)`。

---

## 一、 项目背景与核心动机

在多跳复杂问答（Multi-hop QA）中，迭代检索增强生成（Iterative RAG, 如 IRCoT）已成为主流范式。然而，迭代检索面临一个根本性的**最优停止问题（Optimal Stopping Problem）**：

1. **继续检索**可能带来信息增益（提高回答质量），但必然产生延迟和 API 成本；
2. **提前停止**可以节省成本，但可能因信息不足导致幻觉或错误。

现有工作（如 Stop-RAG 或启发式规则）通常缺少对 fixed-order iterative retrieval 停止结构的显式刻画；同时，在数据自适应停止（data-adaptive stopping）的部署场景下，传统共形预测（Conformal Prediction, CP）的 marginal coverage 口径也难以直接提供与在线服务流匹配的风险监控。

本项目将 **Weitzman's Pandora's Box** 与序贯决策 / 动态规划视角引入多跳 RAG 停止问题，提出 **Pandora-RAG** 框架：用 Oracle 与 Bellman 结构刻画停止上界，用可学习的停止信号逼近继续价值，并用 E-value 机制提供 anytime-valid 的在线风险监控。

---

## 二、 问题建模与数学公式 (Problem Formulation & Math)

我们将多跳 RAG 的检索过程建模为 **fixed-order sequential information acquisition** 问题，并以 **Sequential Pandora's Box with Fixed Order（固定顺序的序贯潘多拉魔盒）** 作为结构化视角。

### 1. 环境与状态定义

- **输入查询**：$q$
- **检索步数**：$k \in 1, 2, ..., K$，最大允许检索轮数为 $K$。
- **状态空间**：在第 $k$ 步，累积上下文状态为 $s_k = [q, d_1, d_2, ..., d_k]$。
- **探测成本**：每执行一次检索，产生已知成本 $c_k > 0$（代表时间/计算开销）。
- **信息增益**：第 $k$ 步检索带来的奖励（质量提升）为随机变量 $G_k \sim F_k(\cdot | s_{k-1})$，分布函数依赖于当前状态。
- **回答质量**：在状态 $s_\tau$ 下停止并生成答案的最终质量得分（如 F1 score）为 $Q(s_\tau)$。

### 2. 两层目标：效用优化与风险监控 (Objective)

**Primary objective.** 策略 $\pi$ 需要决定停止时刻 $\tau_\pi \in 0, 1, ..., K$，以最大化期望净收益：
$$
\pi^* = \arg\max_\pi \mathbb{E}*\pi \left[ Q(s*{\tau_\pi}) - \sum_{k=1}^{\tau_\pi} c_k \right]
$$

**Safety monitor.** 在部署流上，我们额外用 E-process 监控错误事件是否持续超出用户给定的风险水平 $\alpha$。其保证是：若真实错误率不超过 $\alpha$，则错误触发超阈值告警的概率受控，而不是“系统输出自动满足错误率 $\le \alpha$”。

### 3. Weitzman 最优停止策略与保留值 (Reservation Value)

在 $G_k$ 条件独立假设下，Weitzman 定理指出最优策略具有阈值结构。对于每一步检索，存在一个**保留值 $r_k^*$**，它是使“继续检索的期望边际收益”等于“检索成本”的临界点：
$$
c_k = \int_{r_k^*}^{\infty} (x - r_k^*) dF_k(x) 
$$
**全局（静态）Weitzman 基线**：在训练集上估计每步增益的经验分布，解出**与 query 无关**的全局阈值 $r_k^*$，在测试时一律用 $Q(s_k)$ 与 $r_{k+1}^*$ 比较。这是结构化参考基线，而非最终部署算法，用于说明上下文自适应停止的必要性。

**实例级 Oracle（Stage 1 主线上界）**：对**单条轨迹**在已知全程真实质量 $Q(s_k)$（如 F1）下，用后向归纳定义最优价值；每步成本 $c_k$ 可与理论形式 $Q(s_\tau) - \sum_{j=1}^{\tau} c_j$ 对齐——默认取常数 $c_k \equiv c$，也可按轨迹缓存的 `token_count` 或 `latency_ms` 归一化为随步递增的成本（见 `stage1.run_stage1 --oracle-cost-metric`）：
$$
V_K^i = Q(s_K^i), \quad V_k^i = \max\bigl(Q(s_k^i), V_{k+1}^i - c_{k+1}\bigr).
$$
在**最小**的 $k$ 满足 $Q(s_k^i) \geq V_{k+1}^i - c_{k+1}$ 时停止（否则在第 $K$ 步停止）。这对应上帝视角下的**上下文最优继续价值**，作为 Pareto 前沿的理论天花板。**缺失步**（轨迹未观测到的 $k$）上 $Q$ 沿用上一观测 F1，成本记为基准 $c$，表示“空转一步”仍消耗资源。

写入 `artifacts/oracle/{dataset}/test_oracle_labels.jsonl` 的 `step_targets[k]` 为字典，除 `expected_continue_val`（即 $V_{k+1}^i - c_{k+1}$）外，还提供 `margin`（等于 `expected_continue_val` 与当前 $Q_k$ 之差）与 `action_label`（1=Continue，0=Stop）供 Phase 2 做回归或二分类。

### 4. 核心创新 1：Learned Stopping Signal / Continuation-Value Estimator

由于实际中真实分布 $F_k$ 未知，我们提出用轻量级探针（Probe）基于 LLM 隐藏状态及可观测辅助特征来**逼近实例级继续价值或停止边际**（由 Stage 1 的 DP Oracle 写入 `test_oracle_labels.jsonl` 的 `step_targets[k]`：`expected_continue_val` $= V_{k+1}^i - c_{k+1}$，`margin` $= (V_{k+1}^i - c_{k+1}) - Q_k$）：
$$
\hat{y}*k = f*\theta(h_k, \phi_k) \approx V_{k+1}^i - c_{k+1}
\quad\text{或}\quad
\widehat{\Delta}_k \approx \mathrm{margin}_k
$$
其中 $h_k$ 是 LLM 处理当前状态 $s_k$ 时的内部表示，$\phi_k$ 表示语义熵、自一致性等可观测辅助特征。停止规则将 $\hat{y}_k$ 与当前 $Q(s_k)$ 比较（或与 `margin` 符号一致的二分类头），用可学习的 stopping signal 近似 Oracle 的阈值结构，但不直接宣称当前实现已精确恢复 reservation value。

### 5. 核心创新 2：E-value Anytime-Valid 风险监控

由于停止时刻 $\tau$ 依赖于数据与模型自身输出，标准共形预测的 marginal coverage 口径在这一部署场景下存在适用性边界。我们使用基于 E-value 的序列测试（Testing by Betting）构建**在线风险监控层**：

- 对于第 $n$ 个样本，定义错误指标 $e_n = \mathbb{1}[F_1(s_{\tau_n}) < \gamma]$
- 结果感知型 E-wealth 更新：$E_n = E_{n-1} \cdot (1 - \lambda_n + \lambda_n \cdot e_n / \alpha)$
- 其中 $\lambda_n = \text{clip}(1 - \hat{p}_n, \epsilon, 1-\epsilon)$ 为自适应 betting fraction（质量差时下注更大）
- **Ville 不等式保障**：$P(\exists n \in \mathbb{N}: E_n \geq 1/\alpha) \leq \alpha$（当真实错误率 $\leq \alpha$ 时）

E-value 的核心定位是**部署安全层 / 风险仪表盘**，而非“自动把系统错误率压到 $\alpha$ 以下”的控制器：

- 当真实错误率 $> \alpha$ 时，E-wealth 倾向于增长并最终触及 $1/\alpha$，触发部署告警
- 在分布漂移下（sudden/gradual/periodic），E-wealth 可跟踪风险积累并快速响应，而固定阈值式 CP 不提供同类监控语义
- 额外成本极低：步数增加 < 2%，F1 持平（vs CP 增加 40–56% 步数且 F1 反降）

---

## 三、 实验搭建与步骤 (Experimental Setup & Steps)

### 1. 数据集准备

选取多跳推理需要明确迭代检索的标准 Benchmark：

- **HotpotQA** (2-hop)
- **MuSiQue** (2 to 4-hop)
- **2WikiMultiHopQA** (多实体多跳)

### 2. 评估指标 (Metrics)

- **质量指标**：F1 Score, Exact Match (EM)
- **效率指标**：Average Retrieval Steps (平均检索轮数), Latency (延迟)
- **风险监控**：E-wealth 轨迹、触达 `1/α` 的告警语义（Ville 型 anytime-valid 解释见 Phase 3）
- **Selective Prediction（可选）**：当样本处理前 E-wealth 已达 `1/α`（带数值容差）时拒答该样本；报告 **Coverage**（回答比例）与 **Selective accuracy**（回答子集中 F_1 \ge \gamma 的比例），见 `results/stage3_selective_ca_*.png` 与 `docs/reports/stage3/stage3_narrative.md` §2.6

### 3. 实验落地四大阶段 (Implementation Steps)

#### Phase 1: 数据收集与 Oracle 验证 (可行性验证)

- **操作**：在训练集上，强制执行完整的 $K$ 步检索（设 $K=5$）。记录每一步的文档 $d_k$、答案质量 $Q(s_k)$ 以及对应的 LLM 隐藏状态 $h_k$。
- **检索器**：Stage1 支持 `--retriever-backend bm25`（默认）或 `**contriever_bge`**（`facebook/contriever-msmarco` + `BAAI/bge-reranker-v2-m3`）。切换后端须清空 `cache/trajectories` 与 `cache/features` 后重跑；详见 `docs/active/experiments.md` **B1** 与根目录 `.env.example`。
- **vLLM（Linux）**：若遇 `libstdc++.so.6` / `CXXABI_1.3.15` 或健康检查脚本里 `echo` 与状态码之间须有空格等注意事项，见 `docs/active/experiments.md` **B2**（`vLLM（Linux）与 libstdc++ / LD_LIBRARY_PATH`）及 `.env.example` 中对应注释。
- **HF 缓存**：运行 Stage1 / 下载脚本时默认 `**HF_HOME=<仓库>/.hf_cache`**，避免沿用损坏的旧路径；覆盖方式见 `docs/active/STORAGE_LAYOUT.md`。
- **Oracle 计算（主线上界）**：对每条轨迹用已知 $Q(s_k)$ 与步级成本 $c_k$ 做**后向归纳 DP**，得到实例最优停止步与逐步标签 `step_targets`（含 $V_{k+1}-c_{k+1}$、`margin`、`action_label`），并写入 `artifacts/oracle/{dataset}/test_oracle_labels.jsonl`。CLI：`--oracle-cost-metric {fixed,token,latency}`；非 `fixed` 时 Stage1 Pareto 横轴为平均累计归一化成本。
- **全局 Weitzman 基线**：仍在训练集上估计每步增益分布并解全局 $r_k^*$，在 Pareto 图中以 **Global-Weitzman** 点与 **Oracle（DP）** 对比，体现「静态阈值 vs 上下文 Oracle」的差距。
- **实验**：以 DP Oracle 为天花板绘制 Stage1 Pareto；可选分析 Global-Weitzman 作为非 Oracle 的对照。

#### Phase 2: Neural Probe 训练 (核心算法实现)

- **网络结构**：基于 $(h_k, y_k)$ 数据对训练小型 MLP $f_\theta$，其中 $y_k$ 来自 Phase 1 的 `step_targets`（优先 `expected_continue_val` 或 `margin`），而非全局常数 $r_{k+1}^*$；回归困难时可改用 `action_label` 做二分类。
- **损失函数**：最小化 $\mathcal{L}(\theta) = \mathbb{E}[(\hat{y}_k - y_k)^2]$（或与停止决策一致的替代损失）。
- **Phase C 阈值选择**：先在 dev 上做全局阈值 sweep，按 `GW(dev)` 步数上界与 `F1 - λ·normalized_cost` 选主工作点；随后可做保守的 `per-step threshold refinement`，但**只有**在 utility 改善、matched-budget F1 改善或 Pareto frontier 外扩时才采纳。
- **测试**：分析探针的预测误差 $\epsilon$，并在验证集上测试仅依赖探针的停止策略效率。

#### Phase 3: E-value 在线风险监控（Anytime-Valid Risk Monitoring）

- **定位**：E-value 是 Pandora-RAG 的**部署安全层**——以近乎零额外成本（步数增加 < 2%），为自适应停止提供在线风险监控。
- **核心机制**：结果感知型 E-process（Testing by Betting）。每处理一个样本后观测真实 F1，用 outcome-aware multiplier $M_n = 1 - \lambda_n + \lambda_n \cdot e_n / \alpha$ 更新 E-wealth。当 E-wealth 超过 $1/\alpha$ 时，若原假设 $H_0$: "系统错误率 $\leq \alpha$" 成立，则错误触发超阈值的概率由 Ville 不等式控制在 $\alpha$ 以内。
- **vs Conformal Prediction**：Split CP 在当前自适应停止场景下增加 40–56% 步数且 F1 反降 1–4pp；E-value 步数增加 < 2% 且 F1 基本持平。这里的重点是：CP 更像离线 marginal coverage 工具，而 E-value 更适合 optional-stopping-safe 的部署期监控。
- **分布漂移检测**：E-wealth trace 在 sudden/gradual/periodic shift 下均快速响应（所有数据集触及 cap），而 CP 的固定阈值完全无法感知分布变化。
- **检测后干预（拒答）**：`python -m stage3.run_stage3` 在输出 `stage3_evalue_*.json` 时同步写入 `wealth_trace` 与 `selective_prediction_abstain`，用于 coverage–accuracy 分析（与主实验 γ、α 网格一致重跑即可刷新）。
- **详细叙事**：见 `[docs/reports/stage3/stage3_narrative.md](docs/reports/stage3/stage3_narrative.md)`，包含论文 Figure/Table 规划和审稿人 Q&A 预案。

#### Phase 4: PPO 联合微调 (Optional/Correction)

- **操作**：当 Probe 精度受限时，使用 PPO 算法，将“是否停止”作为 action，将 $R = Q(s_\tau) - \lambda \cdot \tau$ 作为 reward 进行微调策略。
- **测试**：进行完整的端到端测试。

---

## 四、 对比的 SOTA 和 Baseline

为了全面证明 Pandora-RAG 的优越性，实验将设立以下几类对比基线：

### 1. 静态/启发式基线 (Static / Heuristic)

- **Single-RAG (Standard)**：一次性检索 Top-K 文档直接生成。
- **IRCoT (Fixed-K)** *(ICLR 2023)*：固定迭代 $K$ 次检索，不动态停止。
- **Threshold-based RAG**：基于大模型输出概率置信度低于阈值时才检索（无理论保障）。

### 2. 动态停止的强 SOTA (Dynamic Stopping)

- **Stop-RAG** *(NAACL 2024)*：基于 Q-learning 的动态停止 RAG。（**最直接的竞争对手**，但无错误率控制保障）。与主实验 **同 id、同切分** 的复现见 `[paper/baselines/Stop-RAG/README.md](paper/baselines/Stop-RAG/README.md)`（Pandora 对齐路径；勿使用上游 `download.sh` 子采样划分做 head-to-head）。
- **ITER-RETGEN** *(EMNLP 2023)*：基于大模型自我评估的迭代生成。

#### Stop-RAG 对比口径

- Pandora 的主设定保持 Stage2 Phase C 的 `**F1 - λ·cost` + `GW(dev)` 步数上界约束**，不把纯 `max-F1` 阈值作为主结果。
- 与 Stop-RAG 的公平 head-to-head 必须使用 **同切分、同样本 id**，且 Stop-RAG 只能采用 `[stop_rag_test.sh](paper/baselines/Stop-RAG/README.md)` 的**在线早停**结果。
- 主预算指标统一为 `**avg_steps`**；若写成成本，可等价记为 `cost = c * t`，其中 `t` 为在线检索轮数，但主文更建议直接报步数。
- 主表应报 `**F1/EM @ matched avg_steps**` 或“达到同 F1 所需的 `avg_steps`”；主图应画两边 threshold sweep 的 `**F1 vs avg_steps` Pareto curve**。
- `best-F1 vs best-F1` 只建议放 appendix，作为 quality-first 补充，不承载主 claim。

### 3. 风险控制与共形预测最新 SOTA (Risk-Control)

- **CCPO (Conformal Calibration for Policy Optimization)** *(NeurIPS 2024)*：使用 RL+标准共形预测控制 RAG 行为。（作为与自适应停止场景适用性边界相关的参考对照。）
- **Conformal-RAG** *(arXiv 2025.06)*：虽用于检索后过滤（Post-retrieval filtering），但可作为基于共形预测的质量控制基准对比。

### 4. 消融实验基线 (Ablations)

- **Oracle-Pandora-RAG**：使用 DP 给出的实例级继续价值 / 最优停止（展示性能天花板）。
- **Global-Weitzman**：使用训练集估计的全局 $r_k^*$ 阈值（静态策略，非 Oracle）。
- **Pandora-RAG w/o E-value**：移除 E-value 校准，退化为纯启发式停止。
- **Pandora-RAG w/ Standard CP**：替换 E-value 为标准的共形预测（展示标准 CP 在该自适应停止设定中的局限与额外成本）。

---

## 五、 预期理论贡献与可能遇到的风险应对

**预期理论贡献：**

1. 给出多跳 RAG 停止问题的 fixed-order sequential information acquisition 形式化，并建立对应的 Oracle / DP / 阈值结构视角。
2. 分析 learned stopping signal / continuation-value estimator 的逼近误差对总体效用的影响，并据此指导 Probe 设计。

**风险防控 (Risk Mitigation)：**

- **信息增益非单调性（有时后续检索会突然找到金文档）**：Weitzman 保留值的积分公式 $\int_{r^*}^\infty (x-r^*)dF(x)$ 本质上考虑了长尾高收益分布，能够在理论上自然包容这种“突破性”增益，我们将在论文讨论部分明确论证这一点。
- **Probe 训练难度**：如果隐藏状态对继续价值的表征不足，我们将引入 Semantic Entropy（语义熵）或 Self-consistency 比例作为 Probe 的补充输入特征，并在论文中明确区分特征成本与部署口径。

---

**拟定时间表：** 

- 第1-2周：完成环境搭建与 Phase 1 (Oracle 计算与基线跑通)
- 第3-4周：完成 Phase 2 (Neural Probe 训练) 与理论公式推导
- 第5-6周：集成 Phase 3 (E-values) 并开展在三大数据集上的主实验与消融实验
- 第7-8周：撰写论文并打磨图表。
