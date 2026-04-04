# Pandora-RAG: 完整实验与推进方案

llm: Llama-3.1-8B-Instruct

## 第一阶段：数据全量化与 Oracle 基线锚定 (Week 1-2)

*目标：从预实验的小样本扩展到三大标准数据集的全量数据，提取真正有区分度的深层特征。*

**1. 轨迹收集与状态缓存 (Trajectory Caching)**

- **数据集扩充**：HotpotQA (2-hop), MuSiQue (2-4 hop), 2WikiMultiHopQA。每个数据集至少准备 Train: 5000条, Dev: 1000条, Test: 1000条。
- **特征提取（关键改进）**：为了解决预实验中 Probe $R^2$ 过低的问题，除了保留原有的浅层特征外，**必须**在生成轨迹时缓存以下深层信号：
  - **LLM Hidden States**: 大模型生成最终答案时的最后一层输出特征（取最后 token 或平均 pooling）。
  - **Semantic Entropy (语义熵) / Self-Consistency (自一致性)**：对当前检索到的上下文，用较高 temperature 采样 3-5 次，计算答案的多样性/熵。一致性越高，质量大概率越高。
  - **Context-Question Overlap / NLI**：计算新检索到的文档与已存在文档的信息重叠度。
- **Oracle 锚定**：在三大数据集上计算真实的 Weitzman 保留值 $r_k^*$，绘制 Oracle 的 F1 vs. Cost 帕累托前沿图（Pareto Frontier），确立我们的性能天花板。

### 第一阶段执行清单（可直接落地）

> 目标验收线：拿到 `3 数据集 x 7000 条样本` 的完整轨迹缓存，并产出 Oracle 帕累托前沿与阶段报告。

**A. 数据准备（D1-D3）**

- **A1. 数据下载与统一格式**
  - 范围：HotpotQA、MuSiQue、2WikiMultiHopQA。
  - 统一字段：`id, dataset, split, question, answer, gt_hop_count, supporting_facts(optional)`。
  - `gt_hop_count` 提取策略：优先从 `supporting_facts.title` 去重计数；MuSiQue 回退 `question_decomposition` 或样本 id 前缀（如 `2hop__`）；无法确定时记为 `-1`。
  - 输出：`data/processed/{dataset}/{train,dev,test}.jsonl`。
  - 验收：每条样本字段完整；可被统一 loader 无报错读取。
- **A2. 固定样本规模与切分**
  - 配额：Train=5000、Dev=1000、Test=1000（每数据集）。
  - 规则：固定随机种子 `seed=42`；不足时保留全量并在报告注明。
  - 输出：`data/splits/{dataset}_seed42_manifest.json`（保存样本 id 列表）。
  - 验收：重复运行切分脚本，id 列表完全一致（可复现）。

**B. 轨迹收集与深层特征缓存（D4-D8）**

- **B1. 统一轨迹协议**
  - 设定最大检索步数 `K_max=5`，每步缓存：
    - 检索文档与分数；
    - 当前中间答案；
    - 该步成本（`cost.token_count`、`cost.latency_ms`、检索次数）——供 Oracle DP 使用动态步成本 $c_k$（`run_stage1 --oracle-cost-metric token|latency`），与 payoff $Q(s_\tau)-\sum_j c_j$ 的理论定义一致。
  - 输出：`cache/trajectories/{dataset}/{split}/*.jsonl`。
  - 验收：随机抽样 100 条轨迹，步级字段完整率 100%。
- **B2. Hidden States 缓存（关键）**
  - 方案：保存最终回答时最后一层 hidden states（`last_token` + `mean_pool` 两种）。
  - 压缩：`float16` 存储，减少 I/O 与磁盘压力。
  - 输出：`cache/features/{dataset}/{split}/hidden_states/*.npz`。
  - 验收：特征维度一致；坏文件率 0；单样本特征可成功反序列化。
- **B3. Semantic Entropy / Self-Consistency**
  - 配置：`temperature=0.7`，每步采样 `n=10`。
  - 指标：答案聚类数、Shannon entropy、一致性比例（majority ratio）。
  - 输出：写入轨迹步级字段 `semantic_entropy`, `self_consistency`。
  - 验收：在 Dev 集上统计分布合理（非全 0 / 非常数）。
- **B4. Context Overlap / NLI 特征**
  - Overlap：新增文档与历史文档的 token/Jaccard overlap。
  - NLI：轻量模型打分 `entail/neutral/contradict` 概率（或 entailment score）。
  - 输出：步级字段 `ctx_overlap`, `nli_entail`, `nli_contra`。
  - 验收：与步数存在统计变化（例如前后步均值不同），避免特征失效。

**C. Oracle 锚定与天花板估计（D9-D11）**

- **C1. 计算真实保留值 $r_k^*$ 与实例 Oracle**
  - 定义：全局 $r_k^*$ 仍由训练集增益分布 + 固定 $c$ 解 Weitzman；**实例 Oracle** 对每条轨迹在已知逐步 $Q_k$（F1）下做后向归纳，步成本为 $c_k$（常数或按 token/延迟缩放）；缺失步上 $Q$ 冻结、成本取基准 $c$。
  - 输出：`artifacts/oracle/{dataset}/test_oracle_labels.jsonl`（含 `oracle_steps_used`、`step_targets`：`expected_continue_val`、`margin`、`action_label`，以及 `gt_hop_count` 对齐分析）。
  - 验收：可回放复现 Oracle 决策，且决策路径无非法状态。
- **C2. 绘制 Oracle Pareto Frontier**
  - 横轴：`--oracle-cost-metric fixed` 时为平均检索步数；`token`/`latency` 时为平均累计归一化成本（与动态 $c_k$ 一致）。纵轴：F1。
  - 对比点：`K=1..5` 固定步策略 + Oracle 动态策略。
  - 输出：`results/stage1_oracle_pareto_{dataset}.png`。
  - 验收：Oracle 点严格位于或优于固定步策略包络线。

**D. 阶段交付（D12-D14）**

- **D1. 阶段报告**
  - 文件：`results/stage1_report.md`。
  - 必含内容：数据统计、缓存完整性、特征分布、Oracle 前沿图、Hop-Alignment（`oracle_steps_used` vs `gt_hop_count`）、失败样例。
  - 验收：报告可独立阅读复现阶段结论。
- **D2. 质量门禁（Go/No-Go）**
  - 进入第二阶段前必须满足：
    - 三数据集缓存完成率 >= 98%；
    - 关键特征（hidden/entropy/NLI）缺失率 <= 1%；
    - Oracle 相比最佳固定步策略在 F1-Cost 上存在显著优势（至少一个数据集明显提升）。
  - 若不满足：优先修复缓存链路与特征稳定性，再进入 Probe 训练。

**第一阶段建议里程碑**

- **Week 1**：完成 A + B1/B2（数据与核心缓存链路打通）。
- **Week 2 前半**：完成 B3/B4 + C（深层特征与 Oracle 锚定）。
- **Week 2 后半**：完成 D（报告与 Go/No-Go 评审）。

## 第二阶段：Neural Probe 攻坚战（核心难点） (Week 3-5)

*目标：训练出能够精准逼近 $r_k^*$ 或有效决策停止的神经网络探针。*

**1. 目标函数重构 (Target Reformulation)**

- 预实验中预测绝对 `F1` 效果差。Phase 1 已在 `step_targets[k]` 中写出 `expected_continue_val`（$V_{k+1}-c_{k+1}$）与 `**margin*`*（继续净优势相对当前 $Q_k$）；Probe 优先回归 `margin` 或 `expected_continue_val`，与停止规则同型。
- **分类建模 (Alternative)**：如果回归任务依然困难（$R^2 < 0.3$），直接使用 Oracle 写好的 `**action_label`**（1=Continue，0=Stop）做二分类，使用 Focal Loss 缓解类别不平衡。

**2. 模型架构设计**

- 基于缓存的 Hidden States 和 浅层特征拼接，训练一个表现更强的 Probe。
  - **方案 A (MLP)**：在 LLM Hidden States 上外接 3 层 MLP。
  - **方案 B (Lightweight Cross-Encoder)**：使用 DeBERTa-v3-small 将“Question + 当前生成的中间答案”作为输入，直接输出预期质量得分。
- **评估标准**：不仅看 Probe 本身的准确率，直接看 Probe 介入 RAG 后的 **Avg Steps 和 F1**，目标是达到 Oracle 表现的 90%（即 F1 显著高于 K=2，步数控制在 2.5 左右）。

## 第三阶段：E-value 风险控制集成与验证 (Week 6-7)

*目标：实现论文的第二个核心贡献——任意时刻有效的错误率控制。*

**1. E-value 机制工程落地**

- **Betting Score 设计**：对于每个测试样本，定义目标错误率 $\alpha$ (如 0.1)。当 Probe 发出“停止”信号时，计算当前的 E-value $e_t$。
- **序列乘积与 Ville 不等式**：维护一个序列 $E_n = \prod e_t$。
- **安全阀机制 (Safety Valve)**：只有当 Probe 输出 `Stop` **且** $E_n < 1/\alpha$ （或相应的安全阈值配置）时，才真正执行停止；否则强制继续检索。

**2. 核心实验：风险控制对比**

- 对比 Standard Conformal Prediction (标准共形预测，如 CCPO)。
- **预期图表**：横轴为测试样本数量 $n$（或不同领域数据的 Shift），纵轴为 Empirical Error Rate。
- **必须证明的现象**：在自适应停止（Adaptive Stopping）场景下，标准 CP 的错误率会逐渐失控（突破 $\alpha$ 红线），而基于 E-value 的 Pandora-RAG 始终严丝合缝地压在 $\alpha$ 以下。

## 第四阶段：主实验与 SOTA 对比 (Week 8-9)

*目标：在三大数据集上跑通端到端全流程，生成核心实验表格。*

**1. 主实验对比矩阵**


| 策略                     | 平均检索步数    | F1    | EM    | 经验错误率 (是否 $\le \alpha$) | 推理耗时  |
| ---------------------- | --------- | ----- | ----- | ----------------------- | ----- |
| Single-RAG (K=1)       | 1.0       | -     | -     | 无法控制                    | 低     |
| IRCoT (K=5)            | 5.0       | -     | -     | 无法控制                    | 高     |
| Stop-RAG               | 动态        | -     | -     | 破防                      | 中     |
| CCPO (Standard CP)     | 动态        | -     | -     | 破防                      | 中     |
| **Pandora-RAG (Ours)** | **动态(低)** | **高** | **高** | **严格达标**                | **低** |


**2. 泛化性测试 (OOD Test)**

- 使用在一个数据集（如 HotpotQA）上训练的 Probe 和计算的保留值，直接 Zero-shot 迁移到另一个数据集（如 MuSiQue），测试 E-value 机制是否依然能守住错误率底线。这将是顶会 reviewer 非常喜欢看到的鲁棒性证明。

## 第五阶段：消融实验、机制分析与作图 (Week 10-11)

*目标：回答 Reviewer 可能提出的“Why and How”问题。*

**1. 消融实验 (Ablation Study)**

- w/o E-value control: 展示没有安全阀时，模型在某些困难 query 上会提前停止导致灾难性幻觉。
- w/o Deep Features: 回退到预实验的浅层特征，展示深层特征（Hidden States/Entropy）对估算保留值的必要性。

**2. 可视化作图 (Visualizations)**

- **图1：概念图/Teaser Figure**（放在论文首页）：展示继续检索和提前停止的 Trade-off，以及 Pandora 盒子的阈值概念。
- **图2：Pareto Frontier 图**：横坐标 Avg Cost，纵坐标 F1。画出各种 Baseline 的点，连出 Oracle 的前沿线，证明 Ours 最接近 Oracle。
- **图3：E-process 轨迹图**：展示随时间推移，错误率被稳稳压制在红线以下的动态过程。
- **图4：Case Study**：挑选一个具体的多跳问题，展示模型在前两步检索无果时继续，在第 3 步找到关键信息后，Probe 值瞬间跃升超过阈值，完美触发停止。

## 第六阶段：论文撰写与打磨 (Week 12)

- **Storyline 定调**：不要把重点只放在“省钱”上，预实验证明了正确停止能**兼顾降本和提效（避免幻觉/过检索干扰）**，这是极大的卖点。引入经济学理论使得文章具备理论深度。
- **Math Review**：请具有理论基础的合作者仔细 double-check 基于 Fixed Order Pandora's Box 和 Ville Inequality 的数学推导证明（放到 Appendix）。

---

## 💡 风险预案 (Contingency Plan)

1. **风险：Probe 怎么训都不准 (回归 $R^2 < 0.2$)**
  - *应对方案*：放弃连续值回归，改用 Phase 1 中逐步的 `**action_label`**（`1=Continue`，`0=Stop`，与 `margin` 符号一致）训练二分类器；输入仍为隐藏状态与浅层特征。可通过 “Neural Approximation” 叙事与 Oracle DP 对齐。
2. **风险：E-value 过于保守，导致模型退化成全部检索 5 步**
  - *应对方案*：调整 Betting Score 的奖励函数，或者稍微放宽置信度参数。检查 E-value 的初始值设置，避免初期乘积衰减过快。
3. **风险：计算 Hidden states 导致推理过慢**
  - *应对方案*：只缓存大模型生成前几个 Token 时的 Hidden states，或者改用轻量级外置评判模型（如 Llama-3-8B 专用于 RAG 评分）。