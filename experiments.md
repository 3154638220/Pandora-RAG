# Pandora-RAG: 完整实验与推进方案

**基座模型设定**: Llama-3.1-8B-Instruct (全白盒开源设定，确保完全可复现性与内部状态可访问性)
**部署与推理框架**: vLLM (用于高速生成) + HuggingFace Transformers (用于特征提取)

## 第一阶段：数据全量化与 Oracle 基线锚定 (Week 1-2)

*目标：从预实验的小样本扩展到三大标准数据集的全量数据，提取真正有区分度的深层特征。*

**1. 轨迹收集与状态缓存 (Trajectory Caching)**

- **数据集扩充**：HotpotQA (2-hop), MuSiQue (2-4 hop), 2WikiMultiHopQA。每个数据集准备 Train: 4000条、Calib(校准集): 1000条、Dev: 1000条、Test: 1000条；共形预测与 E-value 阈値校准严禁在 Test 上拟合。
- **特征提取（Two-Pass 核心架构改进）**：
  - **Pass 1: 高速生成与采样 (vLLM)**：利用 vLLM 极高的吞吐量，执行多跳文档检索，并使用 `temperature=0.7`, `n=10` 进行高效的局部采样，计算 **Semantic Entropy (语义熵)** 和 **Self-Consistency (自一致性)**。
  - **Pass 2: 状态提取 (HuggingFace)**：关闭 KV-Cache 重新加载生成的历史轨迹，执行一次纯前向传播 (Forward-pass)，提取大模型生成最终答案时的最后一层 **Hidden States** (取最后一个 token，采用 float16 压缩存储)。
  - **Context-Question Overlap / NLI**：引入轻量级 `cross-encoder/nli-deberta-v3-small` 评估检索文档与历史信息的包含关系，避免使用大模型导致算力浪费。

### 第一阶段执行清单（可直接落地）

> 目标验收线：拿到 `3 数据集 x 7000 条样本` 的完整轨迹缓存，并产出 Oracle 帕累托前沿。

**A. 数据准备（D1-D3）**

- **A1. 数据下载与统一格式**
  - 范围：HotpotQA、MuSiQue、2WikiMultiHopQA。
  - **统一样本级 jsonl 字段**：`id, dataset, split, question, answer, gt_hop_count, supporting_facts(optional)`。
  - `**gt_hop_count` 提取**：优先从 `supporting_facts.title` 去重计数；MuSiQue 回退 `question_decomposition` 或样本 id 前缀（如 `2hop__`）；无法确定时记为 `-1`。
  - 输出：`data/processed/{dataset}/{train,calib,dev,test}.jsonl`。
  - 验收：每条样本字段完整；可被统一 loader 无报错读取。
- **A2. 严格四切分 (Train/Calib/Dev/Test)**
  - 配额：Train=4000、Calib=1000、Dev=1000、Test=1000（每数据集）；Calib 专用于 E-value / CP 等阈値与 Betting 相关校准，**严禁在 Test 上拟合**。
  - 规则：固定随机种子 `seed=42`；各划分样本 id **绝对正交、互不交叉**。
  - **Manifest**：`data/splits/{dataset}_seed42_manifest.json`（保存各 split 的样本 `id` 列表，便于复现与审计）。
  - 验收：重复运行切分脚本，`id` 列表完全一致；Calib 与 Test 无重叠。

**B. 轨迹收集与深层特征缓存（D4-D8）**

- **B1. 统一轨迹协议**
  - 最大检索步数 `K_max=5`。每步在轨迹 jsonl 中缓存：检索文档与分数、当前中间答案、该步成本（如 `cost.token_count`、`cost.latency_ms`、检索调用次数），用于 Oracle DP 的动态步成本 $c_k$（实现上可通过 `run_stage1 --oracle-cost-metric token|latency` 选择度量），与 payoff $Q(s_\tau)-\sum_j c_j$ 的理论定义一致。
  - 输出：`cache/trajectories/{dataset}/{split}/*.jsonl`。
  - 验收：随机抽样 100 条轨迹，步级字段完整率 100%。
- **B2. 深层信号落地（Two-Pass 与落盘）**
  - **Pass 1 (vLLM)**：`temperature=0.7`，每步采样 `n=10`；将 **Semantic Entropy / Self-Consistency** 指标写入轨迹步级字段（如 `semantic_entropy`, `self_consistency` 或等价命名），验收要求在 Dev 上分布非全零、非常数。
  - **Pass 2 (HuggingFace)**：对已定稿的生成轨迹做纯前向，提取最终答案处最后一层 hidden（`last_token`，**float16**），输出 `cache/features/{dataset}/{split}/hidden_states/*.npz`；验收：维度一致、坏文件率 0、可反序列化。
  - **NLI / 重叠**：步级字段如 `ctx_overlap`、`nli_entail`、`nli_contra`（或 `cross-encoder/nli-deberta-v3-small` 的 entailment score）；验收：随步数有统计变化，避免特征失效。

**C. Oracle 锚定与天花板估计（D9-D11）**

- **C1. 计算真实保留值 $r_k^*$ 与实例 Oracle**
  - **Global-Weitzman (静态基线)**：在 Train 集上统计每步经验增益分布，求解出一组全局静态保留值 $r_k^*$。
  - **实例 Oracle (DP 上界)**：对每条轨迹在已知逐步 $Q_k$（如 F1 Score）下做后向归纳。写入 `artifacts/oracle/{dataset}/test_oracle_labels.jsonl`，关键标签包含 `action_label` (1=Continue, 0=Stop) 以及 `margin` (决策收益差距)；可含 `expected_continue_val` 等辅助字段便于调试。
  - 验收：可回放复现 Oracle 决策，决策路径无非法状态。
- **C2. 绘制 Oracle Pareto Frontier**：绘制横轴(Cost) - 纵轴(F1) 的帕累托前沿包络图，验证 Oracle 和 Global-Weitzman 的性能差距（证明引入 Neural Probe 的理论价值）。
  - 输出示例：`results/stage1_oracle_pareto_{dataset}.png`。
  - 验收：Oracle 前沿包络须不低于 Global-Weitzman（在相同成本度量下）。

**D. 质量门禁（Go/No-Go）**
进入第二阶段（Neural Probe）前须同时满足：

- 三数据集轨迹与特征缓存完成率 $\geq 98$；
- 关键特征（hidden states / 语义熵与自一致性 / NLI）步级缺失率 $\leq 1$；
- Oracle 相对**最佳固定步长策略**在 F1–Cost 平面上存在**可复现的显著优势**（至少一个数据集上明显提升）。

若不满足：优先修复缓存链路、Two-Pass 一致性与特征稳定性，再进入 Probe 训练。

## 第二阶段：Neural Probe 攻坚战（核心难点） (Week 3-5)

*目标：训练出能够精准模拟 Weitzman 最优停止逻辑的神经网络探针。*

**1. 目标函数重构 (Target Reformulation) [⚠️核心优化]**

直接回归绝对 `margin` 会因 F1 噪声引发模型崩溃。我们将采用 **Margin-weighted Classification (边界加权二分类)**：

- **主任务**：预测 Oracle DP 产出的二值标签 `action_label` $\in 0, 1$。
- **Loss 设计 (Focal-Margin Loss)**：
$$ \mathcal{L}(\theta) = \sum_{k} | \text{margin}_k | \cdot \text{BCELoss}(\hat{p}_k, \text{actionlabel}_k) $$
通过绝对值 $|\text{margin}_k|$ 作为样本权重，模型被允许在无关紧要的步数（继续和停止收益差不多）上犯错，但被**严厉惩罚**在具有巨大经济价值差异的关键步上做出错误决策。这完美契合了经济学中最优停止的本质。

**2. 模型架构设计**

- **输入层**：Llama-3.1-8B-Instruct 提取的 4096 维 Hidden States 拼接浅层特征（熵、NLI、Cost）。
- **网络结构**：3 层 MLP + Dropout + 最终的 Sigmoid 分类头 $\hat{p}_k$。
- **评估标准**：不纯看 Accuracy，将 Probe 接入 RAG Pipeline，观察其在 Dev 集的 F1-Cost 坐标点是否显著逼近 Oracle 的帕累托前沿。

## 第三阶段：E-value 风险控制集成与验证 (Week 6-7)

*目标：实现论文的第二个核心贡献——任何时间序列上的任意时刻有效错误率控制。*

**1. 误差定义的严谨化**
在生成任务中，为了使用 E-value，必须定义什么是“Error”。我们引入可接受的质量下限 $\gamma$（如 $\gamma=0.5$）：

- 样本 $t$ 发生 Error $\iff \text{F1}(s_{\tau_t}) < \gamma$

**2. E-value 机制工程落地**

- **校准阶段 (On Calib Set)**：训练一个极轻量的质量预测器预测当前状态的 $P(\text{F1} \geq \gamma)$，据此构造 Betting Score $e_t$。
- **序列乘积与 Ville 不等式**：在用户请求流（Time/Sample index $t$）上维护累积乘积 $E_n = \prod_{t=1}^n e_t$。
- **动态安全阀 (Dynamic Safety Valve)**：
当当前步骤 $k$ 的 Neural Probe 发出 `Stop` 信号时：
  - 如果执行 Stop 后会导致系统的 $E_n$ 突破安全边界（即有破坏 $1-\alpha$ 错误率的风险），系统将**否决探针**，强制继续检索，直至找到足以恢复 Betting Score 信用的高价值文档。

**3. 核心测试：应对长尾数据分布漂移 (Distribution Shift)**

- 设置测试序列 $n$，前半段为普通问题，后半段注入高难度/混淆问题。
- **预期证明**：展示标准共形预测 (Standard CP) 在面临后续连续困难样本时，其 Empirical Error Rate 会突破 $\alpha$ 红线；而 Pandora-RAG 的 E-process 会自动变得更加保守（强制拉长检索步数），死死守住 $\alpha$ 底线。

## 第四阶段：主实验与 SOTA 对比 (Week 8-9)

*目标：在三大数据集上跑通端到端全流程，生成核心实验表格。*

**1. 主实验对比矩阵**


| 类别               | 策略                          | Avg Steps | F1    | EM    | 经验错误率 (目标 $\le \alpha=0.1$) | 推理成本  |
| ---------------- | --------------------------- | --------- | ----- | ----- | --------------------------- | ----- |
| **Static**       | Single-RAG (K=1)            | 1.0       | 最低    | -     | 破防 (>0.1)                   | 最低    |
|                  | IRCoT (Fixed-K=5)           | 5.0       | 高     | -     | -                           | 最高    |
| **Dynamic**      | ITER-RETGEN (EMNLP 23)      | 动态        | 中     | -     | 破防 (>0.1)                   | 中等    |
|                  | Adaptive-RAG (NAACL 24)     | 动态        | 高     | -     | 破防 (>0.1)                   | 中等    |
|                  | Stop-RAG (NAACL 24)         | 动态        | 高     | -     | 破防 (>0.1)                   | 中等    |
| **Risk-Control** | CCPO (NeurIPS 24 - Std CP)  | 动态        | 中     | -     | **自适应长序列下破防**               | 中等    |
| **Upper/Base**   | Global-Weitzman (Ours Base) | 动态        | 中     | -     | -                           | 低     |
|                  | Oracle-Pandora (理论上界)       | *最优*      | *最高*  | *最高*  | -                           | -     |
| **Ours**         | **Pandora-RAG (E-value)**   | **逼近最优**  | **高** | **高** | **严格达标 ($\le 0.1$)**        | **低** |


**2. 泛化性测试 (Zero-Shot OOD Transfer)**

- 使用在 HotpotQA 上训练的 Probe，直接 Zero-shot 迁移到 MuSiQue 上进行测试。
- **核心论点**：即便 Probe 因为领域不同发生退化，**E-value 安全阀机制依然能够保证错误率不被突破**，展现系统极强的实际工程部署价值。

## 第五阶段：消融实验、机制分析与作图 (Week 10-11)

**1. 可视化作图 (Visualizations)**

- **图1：Teaser Figure (首页图)**：左侧展示固定检索与过早停止的困境，右侧展示 Pandora-RAG 如何利用 E-process 和保留值在二维（成本 vs 准确率）与三维（时间轴上的风险红线）实现完美平衡。
- **图2：Pareto Frontier 图**：Oracle vs Weitzman vs Ours，证明接近最优解。
- **图3：E-process 动态防御轨迹图**：横轴为时间线上的测试用户数 $n$，纵轴为累积错误率。明确展示当遭遇到连续的困难 query 攻击时，CP 失效，而 Pandora-RAG 动态增加检索步数压制错误率。

**2. 消融实验 (Ablation Study)**

- **w/o Deep Features**：仅用浅层特征（文本长度、分数）训练 Probe，论证 Llama-3 内部隐式推理状态（Hidden States）和 Semantic Entropy 的不可替代性。
- **w/o Margin-weighting**：使用普通的 BCE 训练，论证经济学边际价值引入 Loss 设计对系统收敛的必要性。

## 第六阶段：论文撰写与理论审核 (Week 12)

- **Storyline 定调**：核心叙事不仅是“降本增效”，而是**“在大语言模型的自适应迭代信息获取中，首次实现了兼顾理论最优成本边界与任意时刻严格风险控制的工程范式”**。
- **理论 Check**：仔细审查 Appendix 中的证明，重点核对 Fixed Order Pandora's Box 放宽条件假设的合理性，以及 Ville Inequality 在连续 RAG 对话流（Query stream）上的数学适配。

---

## 💡 风险预案 (Contingency Plan)

1. **算力限制导致 vLLM / HF 调度过慢**
  *应对方案*：若 `Llama-3.1-8B-Instruct` 在提取特征时过于耗时，将轻量级基线模型替换为 `Qwen-2.5-3B-Instruct` 进行对比跑分，在关键的主数据集上再使用 8B 模型。
2. **E-value 过于保守，导致模型退化成全部检索 5 步**
  *应对方案*：这通常是因为初始信用太低。调整 Betting Score 函数，允许模型在初期积累一定的“信用盈余 (Credit surplus)”，或引入适度的 Betting 赌注折现因子。
3. **Oracle 效果不明显（多跳失效）**
  *应对方案*：如果提供所有的正确文档 F1 依然提不上去，说明遇到了 LLM 的基础阅读理解瓶颈。此时调整质量函数，从单纯的 F1 放宽为“是否包含了正确答案的实体（Recall）”，改变上限评估维度。