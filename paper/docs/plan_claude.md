# Pandora-RAG 项目现状评估报告（终版）

> 评估日期：2026-04-09
> 评估范围：Stage 1（已完成）+ Stage 2（Phase D5 已完成）+ 整体 NeurIPS 投稿准备度
> 说明：本报告经三轮自审迭代，已修正初版中对 E-value 风险的高估、对 Deployable-GW 基线强度的乐观偏差、以及若干遗漏的实验方向。

---

## 一、总体判断

**Stage 2 已触及特征信息论天花板。建议用 2-3 天补充一个高杠杆特征（answer_logprob）并完成几项低成本实验，随后果断转入 Stage 3。**

Probe 在公平基线（Deployable-GW）上三数据集全胜，D1-D5 系统性诊断已确认瓶颈在于特征集无法观测真实 F1。继续在训练技巧或损失函数层面投入的边际收益极低。Stage 3（E-value 风险控制）是论文与 CCPO/Stop-RAG 等竞品的根本差异化所在，应优先验证。

---

## 二、Stage 2 详细诊断

### 2.1 工程完成度盘点

6 天内完成了从架构升级到公平基线建立的完整闭环：


| 里程碑                     | 状态  | 评价                                          |
| ----------------------- | --- | ------------------------------------------- |
| ProbeMLP_v2 双分支架构       | ✅   | 设计合理，解决了模态尺度不平衡                             |
| Delta Features (6维)     | ✅   | 统计有效但 F1 增益不稳定，符合预期                         |
| Phase A 训练稳定化           | ✅   | 三数据集 F1 均有小幅改善（+0.002~+0.010）               |
| Phase B Focal+Smoothing | ✅   | **负面结果**：高噪标签下 Focal 适得其反，dev accuracy 暴降   |
| Phase C Pareto-aware 阈值 | ✅   | 步数控制达标，musique F1 +0.023                    |
| D1 特征诊断                 | ✅   | 确认浅层特征 AUROC 偏弱                             |
| D2 XGBoost 基线           | ✅   | XGB ≈ MLP → 瓶颈在特征而非模型                       |
| D3 损失消融                 | ✅   | 四种配置无一在三数据集同时占优，排除损失函数作为突破口                 |
| D4 答案质量代理特征             | ✅   | 三个零成本特征已加入（answer_changed, streak, novelty） |
| D5 Deployable-GW 公平基线   | ✅   | 建立了正确的对比框架                                  |


### 2.2 当前最佳 Probe 成绩

**Probe vs Deployable-GW（公平口径）：**


| 数据集      | Probe F1   | D-GW F1 | 领先幅度             | Probe 步数 | D-GW 步数 |
| -------- | ---------- | ------- | ---------------- | -------- | ------- |
| HotpotQA | **0.5244** | 0.3698  | +0.1546 (+41.8%) | 2.74     | 1.01    |
| MuSiQue  | **0.1659** | 0.0907  | +0.0752 (+82.9%) | 3.13     | 1.05    |
| 2Wiki    | **0.3606** | 0.2223  | +0.1383 (+62.2%) | 2.71     | 1.01    |


**Probe vs Original-GW（半 Oracle 参照，非公平对比）vs Oracle（理论上界）：**


| 数据集      | Probe F1 | Orig-GW F1 | Oracle F1 | Probe/Oracle | Orig-GW/Oracle |
| -------- | -------- | ---------- | --------- | ------------ | -------------- |
| HotpotQA | 0.5244   | 0.6089     | 0.6283    | 83.5%        | 96.9%          |
| MuSiQue  | 0.1659   | 0.2235     | 0.2430    | 68.3%        | 92.0%          |
| 2Wiki    | 0.3606   | 0.5121     | 0.5264    | 68.5%        | 97.3%          |


Probe 达到 Oracle 的 68-84%。Probe 与 Original-GW 之间约 15-25 个百分点的差距，本质上是"能否观测真实 F1"这一信息差带来的，属于信息论硬约束。

### 2.3 瓶颈确认：信息论硬墙

**根因链条：**

1. Oracle DP 和 Original-GW 都使用真实 F1 做停止决策 → 属于"F1 可观测"策略族
2. Probe 只能用间接特征（熵、NLI、self-consistency、hidden states）→ 属于"F1 不可观测"策略族
3. 两个策略族的理论上界不同 → 让 Probe 追平 Original-GW 等价于要求特征完美预测 F1
4. self_consistency 与真实 F1 的相关性在多跳 RAG 中很弱（Deployable-GW 几乎在第 1 步就停止，佐证了这一点）

**证据矩阵：**


| 排查项                 | 结果                   | 结论                  |
| ------------------- | -------------------- | ------------------- |
| XGBoost vs MLP（D2）  | XGB ≈ MLP（差距 < 0.01） | 不是模型能力问题            |
| 四种损失配置消融（D3）        | 无一在三数据集同时占优          | 不是损失函数问题            |
| Delta 特征（优先级 4）     | 增益 ≤ 0.004           | 步间趋势信号不够            |
| D4 三个答案质量代理         | 未带来稳定增益              | 零成本特征已用尽            |
| Hidden states 收益    | +0.018~+0.050        | 信息存在但压缩过激进（4096→64） |
| Focal Loss（Phase B） | 全面负向                 | 高噪标签放大噪声            |


---

## 三、需要关注的隐患

### 隐患 1：Deployable-GW 作为"公平基线"过于弱

Deployable-GW 平均步数 1.01~1.05，本质上等价于 Single-RAG (K=1)。"Probe 碾压一个几乎等于 K=1 的基线"说服力有限，审稿人很可能追问：你的公平对比对象为什么这么弱？有没有介于 K=1 和 Oracle-GW 之间的自适应停止基线？

**根本原因**：self_consistency 在多跳 RAG 中不是 answer quality 的好代理——模型可以在第一步就高度自洽但完全答错。这本身是一个值得在论文里讨论的发现，但也意味着论文需要更多**中间强度的基线**。

**建议补充的中间基线：**

- 基于 semantic_entropy 阈值的停止策略（entropy < τ 则停止）
- 基于 answer_changed 的简单规则策略（连续 N 步答案不变则停止）
- Fixed-K=1,2,3,4,5 的完整 Pareto 前沿数据（从轨迹缓存直接读取，零成本）

### 隐患 2：Probe 相对 Best Fixed-K 的提升幅度可能偏小

改进记录中提到"Shallow-Only 几乎等于 best Fixed-K（gain ≈ 0）"，Shallow-Only 在 HotpotQA 上 F1 约 0.48。Probe（0.5244）相对 Best Fixed-K 的提升约 +0.044 F1。如果审稿人发现 Neural Probe 相比简单固定步数策略只提升了 4 个点，可能会质疑方法的实用价值。

**应对**：论文的核心论点不应是"某个固定步数下 F1 提升多少"，而是 Probe 能在**更少步数下达到接近最优 F1**——强调 Pareto 前沿上的 cost-efficiency 优势。为此必须准备完整的 Fixed-K=1..5 F1 数据来画 Pareto 图。

### 隐患 3：MuSiQue 测试集仅约 417 条

HotpotQA 和 2Wiki 各有 1000 条测试集，但 MuSiQue 因 validation 偏小，test 只有约 417 条。在 417 条上 F1 差异 0.07（Probe vs D-GW）的统计显著性可能不足。

**建议**：提前准备 bootstrap confidence interval 或 paired permutation test，论文中报告置信区间。

### 隐患 4：Probe 在难数据集上可能停得过早

仔细看步数数据：


| 数据集      | Probe 步数 | Original-GW 步数 | 差值          |
| -------- | -------- | -------------- | ----------- |
| HotpotQA | 2.74     | 2.82           | -0.08（几乎相同） |
| MuSiQue  | 3.13     | 4.09           | **-0.96**   |
| 2Wiki    | 2.71     | 3.47           | **-0.76**   |


在 MuSiQue 和 2Wiki 上，Probe 比 Original-GW 少了将近 1 步。Phase C 的步数约束（`gw_steps_cap_mult=1.05`）可能过于紧了。在这些难数据集上多花半步可能换来显著 F1 提升。

**建议**：尝试放宽 `gw_steps_cap_mult` 至 1.2 或 1.3，零成本实验（只重跑阈值选择）。

---

## 四、Stage 2 收尾建议（转入 Stage 3 前）

### 高优先级（建议执行）


| 任务                    | 预计时间   | 理由                                                                                                                                                                                                                                 |
| --------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **answer_logprob 特征** | 1.5-2天 | Token 级 log-probability 是模型置信度最直接的信号，与 F1 相关性通常远高于 self_consistency 和 semantic_entropy。vLLM 原生支持 `logprobs=True`。需要在 Stage 1 的 Pass 1 中补录并重跑 Stage 2。这可能是当前特征集里投入产出比最高的单一改进，而且它还能直接改善 E-value 的 Betting Score 质量，降低 Stage 3 的保守程度。 |
| **放宽步数约束消融**          | 0.5天   | `gw_steps_cap_mult` 从 1.05 调至 1.2/1.3，看 musique/2wiki F1 是否显著提升。零成本。                                                                                                                                                               |
| **硬 margin 过滤实验**     | 0.5天   | 直接过滤                                                                                                                                                                                                                               |
| **补充中间强度基线**          | 0.5天   | entropy-threshold 停止、answer-stability 停止、Fixed-K=1..5 完整数据。填补 D-GW 和 O-GW 之间的空白。                                                                                                                                                   |


### 中优先级（时间允许则执行）


| 任务                       | 预计时间 | 理由                                                                                                                                                                                                                     |
| ------------------------ | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hidden states 压缩维度提升** | 0.5天 | 4096→256 或 512，看收益是否从 +0.02~+0.05 提升到 +0.08+。对论文 "w/o Deep Features" 消融的说服力有帮助。                                                                                                                                        |
| **F1 回归替代方案**            | 0.5天 | 训练 Probe 直接回归预测当前步 F1（或 P(F1 ≥ γ)），用"预测 F1 ≥ 阈值"做停止。好处：(a) 回归目标比二分类更细腻，保留 margin 连续信息；(b) 预测 P(F1 ≥ γ) 可直接复用为 E-value 的 Betting Score，打通 Stage 2 和 Stage 3；(c) 避开 ~80% Stop 标签的类别不平衡。实现成本低（loss 改 MSE/Huber，标签改 f1 值）。 |


### 低优先级（不阻塞进度）


| 任务                  | 说明                                                                               |
| ------------------- | -------------------------------------------------------------------------------- |
| Hidden state 提取策略改进 | 当前只取 last token，对多跳推理可能不够。可尝试 mean pooling 混合、关键位置提取等。牵涉面大，建议作为 camera-ready 增强。 |
| D6 简化模型基线           | 逻辑回归 / 线性 Probe，用于论文消融。可在论文写作阶段补充。                                               |


---

## 五、Stage 3 E-value 评估与建议

### 5.1 风险重新评估

E-value 的可行性风险**低于**直觉预期。原因是 Ville 不等式保证 E-process 的 type-I error 控制是**数学上无条件成立**的，与 Betting Score 预测器好不好无关。预测器差只会导致 E-value 反应慢（保守），不会导致它失效。具体来说：

- E-value **永远不会**违反错误率保证——这是硬数学保证，不是经验结果
- 保守的后果是多检索几步，质量不降，只是成本增加——属于优雅退化
- 对论文来说，E-value 提供的是**数学保证层**，Probe 提供的是**效率层**，两层职责分离

真正需要验证的不是"E-value 是否有效"（数学保证已回答），而是：

1. **E-value 的保守程度是否可接受？** 如果 E-value 导致平均步数从 2.7 增加到 4.5，成本代价太大
2. **分布漂移场景下 E-value 的反应速度**是否足以做出 Fig.3 那样的对比图

### 5.2 Probe 与 E-value 的架构复用

Probe 预测"是否继续检索"，E-value 需要预测"P(F1 ≥ γ)"。两者面对同样的输入特征。建议提前规划是否共享 backbone：

**方案 A（推荐）：** 如果采用 F1 回归替代方案，Probe 和 E-value 可以共用同一个模型——Probe 用"预测 F1 ≥ 停止阈值"做停止决策，E-value 用同一个预测概率构造 Betting Score。系统更简洁，维护成本低。

**方案 B：** 保持二分类 Probe + 独立的质量预测器。两个模型，但可以共享特征提取层。

建议在进入 Stage 3 之前确定方案，避免返工。

### 5.3 推进路线


| 步骤  | 任务                                     | 预计时间    |
| --- | -------------------------------------- | ------- |
| 1   | 在 HotpotQA 上搭建 E-value 端到端原型           | 2-3天    |
| 2   | 验证正常序列下 E-value 的保守程度（avg_steps 增加多少）  | 包含在步骤 1 |
| 3   | 构造分布漂移测试序列，对比 E-process vs Standard CP | 1-2天    |
| 4   | 调优 Betting Strategy（如需要）               | 1天      |
| 5   | 扩展到 MuSiQue 和 2Wiki                    | 1-2天    |


---

## 六、Stage 4-6 及论文策略

### 6.1 基线与实验

**基线优先级：**


| 基线                                      | 实现成本    | 重要性 | 建议                 |
| --------------------------------------- | ------- | --- | ------------------ |
| Fixed-K=1..5                            | 零（从缓存读） | 必须  | Pareto 图核心数据       |
| Single-RAG (K=1)                        | 零       | 必须  | 主表下界               |
| IRCoT Fixed-K=5                         | 零       | 必须  | 主表上界               |
| Global-Weitzman (Original + Deployable) | 已有      | 必须  | 公平基线 + 半 Oracle 参照 |
| Best Fixed-K                            | 低       | 必须  | Pareto 图关键锚点       |
| Entropy-threshold 停止                    | 低       | 高   | 填补中间基线空白           |
| Answer-stability 停止                     | 低       | 高   | 填补中间基线空白           |
| CCPO                                    | 高       | 中   | 时间紧可只引用论文数字        |
| Stop-RAG                                | 高       | 中   | 时间紧可只引用论文数字        |
| Adaptive-RAG / ITER-RETGEN              | 高       | 低   | 视时间决定              |


### 6.2 论文叙事框架

论文的核心叙事不应是"我们的 Probe 多强"，而应是：

> **在不可观测答案质量的部署约束下，首次实现理论最优停止与任意时刻严格风险控制的统一。**

四层递进结构：

1. **理论层（Pandora's Box）**：经济学最优停止框架适配多跳 RAG → Sequential Fixed Order 变体给出阈值策略最优性证明
2. **算法层（Neural Reservation Values）**：在 F1 不可观测约束下逼近 Weitzman 最优解 → Probe 达到 Oracle 的 68-84%，远超朴素代理（Deployable-GW）
3. **安全层（E-value）**：任意时刻错误率控制 → 对比 Standard CP 在分布漂移下的鲁棒性
4. **实用层（端到端系统）**：在多跳 QA 基准上 F1-Cost Pareto 最优

**关于 Probe vs Oracle 差距的叙述策略（关键）：**

不回避差距，而是将其转化为 E-value 必要性的论证：

> Neural Reservation Values achieve 68-84% of the Oracle upper bound without access to ground-truth answer quality. The remaining gap reflects the fundamental information asymmetry between observable features and true F1. This is precisely where E-value risk control becomes essential: when the Probe's stopping decision is suboptimal, the safety valve intervenes to prevent error rate violations, providing deployment-time guarantees that no existing method offers.

Probe 的不完美反而是 E-value 存在价值的最佳论据。

**关于 Deployable-GW 几乎第一步就停的叙述策略：**

这同样是论点而非弱点：

> Deployable-GW applies Weitzman's rule with self-consistency as the quality proxy, yet it stops at step 1 in over 95% of cases. This reveals that self-consistency — the most accessible quality signal — fails to capture answer correctness in multi-hop settings. Neural Reservation Values trained on richer feature sets (hidden states, semantic entropy, NLI scores) overcome this limitation, demonstrating that learning-based stopping rules substantially outperform proxy-based analytic rules.

### 6.3 消融实验矩阵

大部分消融数据已经有了，可以直接复用：


| 消融项                                   | 数据来源                     | 状态          |
| ------------------------------------- | ------------------------ | ----------- |
| w/o Deep Features                     | Stage 2 Shallow-Only     | ✅ 已有        |
| w/o Margin-weighting                  | D3 实验 1（γ=0, ε=0）        | ✅ 已有        |
| w/o Delta Features                    | 优先级 3 vs 优先级 4 对比        | ✅ 已有        |
| XGBoost vs MLP                        | D2                       | ✅ 已有        |
| Deployable-GW vs Original-GW vs Probe | D5                       | ✅ 已有        |
| Focal Loss 消融                         | D3 实验 2/3                | ✅ 已有        |
| w/o E-value Safety Valve              | Stage 3 对比               | ⏳ 待 Stage 3 |
| E-value vs Standard CP                | Stage 3 分布漂移实验           | ⏳ 待 Stage 3 |
| Zero-Shot OOD Transfer                | HotpotQA Probe → MuSiQue | ⏳ 待 Stage 4 |


---

## 七、风险矩阵（终版）


| 风险                                    | 严重度 | 概率  | 应对                                                                                        |
| ------------------------------------- | --- | --- | ----------------------------------------------------------------------------------------- |
| E-value 保守程度不可接受（avg_steps 增加 > 50%）  | 高   | 中低  | 调 Betting Score 函数、信用盈余、折现因子。Ville 不等式保证有效性不受影响。                                          |
| 分布漂移实验差异不够显著                          | 高   | 中低  | 加大漂移强度（如后半段全换高难题）、换漂移模式（渐变 vs 突变）。                                                        |
| 审稿人要求与 CCPO/Stop-RAG 直接公平对比           | 中   | 高   | 优先尝试复现；如不可行，在论文中详细说明差异（不同模型、不同数据集、不同问题设定），并定性分析优劣。                                        |
| Probe 相对 Best Fixed-K 提升幅度小，审稿人质疑实用价值 | 中   | 中   | 用完整 Pareto 前沿图论证 cost-efficiency 优势（同 F1 下步数更少，或同步数下 F1 更高）。                              |
| Deployable-GW 过弱，审稿人不认可公平基线           | 中   | 中高  | 补充 entropy-threshold / answer-stability 等中间基线；在论文中分析 D-GW 过早停止的原因（self_consistency ≠ F1）。 |
| MuSiQue 整体 F1 过低 + 测试集仅 417 条         | 中   | 已发生 | 说明 LLM 阅读理解瓶颈（4-hop 太难），不影响相对优势；报告 bootstrap CI。                                          |
| answer_logprob 补录后 Probe F1 未显著提升     | 低   | 中   | 仍可作为消融行写入论文；E-value 的 Betting Score 也能受益。                                                 |
| 竞品更新（新的 RAG stopping 论文出现）            | 中   | 中   | 提交前再做一轮 arXiv 文献检索。                                                                       |


---

## 八、推荐时间线


| 阶段             | 周次        | 任务                                       | 产出                  |
| -------------- | --------- | ---------------------------------------- | ------------------- |
| **Stage 2 收尾** | Week 1 前半 | answer_logprob 特征补录 + Stage 1 部分重跑       | vLLM logprob 数据落盘   |
|                | Week 1 后半 | Stage 2 重训 + 放宽步数约束 + 硬 margin 过滤 + 中间基线 | 更新后的 Probe 成绩 + 基线表 |
| **Stage 3**    | Week 2    | E-value 端到端原型（HotpotQA）                  | E-value pipeline 可跑 |
|                | Week 3 前半 | 分布漂移实验 + 调优                              | Fig.3 数据            |
|                | Week 3 后半 | 三数据集全量                                   | E-value 全量结果        |
| **Stage 4**    | Week 4    | 主实验 + 基线对比                               | Table 1 + Pareto 图  |
| **Stage 5**    | Week 4-5  | 消融 + 可视化                                 | 消融表、Fig.1/2/3       |
| **Stage 6**    | Week 5-6  | 论文撰写 + 理论审查                              | 完整论文                |
| **Buffer**     | Week 6+   | 补实验 + 竞品检索 + 论文打磨                        | 最终版                 |


---

## 九、总结

### 做得好的地方

- **系统性诊断方法论**非常扎实：D1-D5 逐步排查、用 XGBoost 分离特征 vs 模型瓶颈、用 Deployable-GW 建立公平对比框架，这些都是高质量的实验方法论
- **实验记录详尽**：每个 Phase 都有完整的对照表和验收标准，消融数据丰富，为论文写作打下了好基础
- **负面结果同样有价值**：Focal Loss 的失败、Delta 特征的不稳定、XGB ≈ MLP，这些都是论文消融实验的有力素材

### 核心行动项

1. **answer_logprob 是当前最高杠杆的单一改进**——同时服务于 Stage 2（特征增强）和 Stage 3（Betting Score 质量），优先实现
2. **不要陷入"追平 Original-GW"的执念**——这在信息论上不可能，论文叙事应将差距转化为 E-value 必要性的论据
3. **Stage 3 是论文成败的关键**——E-value 的数学保证是无条件的（好消息），但保守程度和反应速度需要实验验证
4. **补充中间强度基线**——Deployable-GW ≈ K=1 太弱，需要 entropy-threshold、answer-stability 等中间基线来丰富对比
5. **提前想清楚 Probe 与 E-value 的架构关系**——F1 回归方案可能同时简化两个阶段

### 一句话结论

**Stage 2 的信息论天花板已确认；用 2-3 天补上 answer_logprob 和几项低成本实验后，全力推进 Stage 3——E-value 的数学保证是这篇论文最独特的卖点，也是当前最大的未验证假设。**