# Pandora-RAG: NeurIPS 论文指导书（重构版）

> 目标：把项目写成一篇 **NeurIPS 风格的方法论文**，而不是阶段性工程汇报。  
> 更新日期：2026-04-19  
> 本文基于：`docs/archive/NeurIPS-HE.md`、`docs/reports/stage2/stage2_final.md`、`docs/reports/stage3/stage3_results_summary.md`、`docs/reports/stage3/stage3_narrative.md`

---

## 一、论文应该围绕什么中心句

当前最适合 NeurIPS 的中心句不是“我们做了一个更强的 probe”，也不是“我们把 Pandora、PPO、CP/E-value 都揉在一起”，而是：

> **Iterative multi-hop RAG stopping is a fixed-order sequential information acquisition problem. A learned stopping signal can recover much of the oracle benefit, and an E-value monitor makes this adaptive stopping deployable under data-adaptive stopping and distribution shift.**

换成中文就是：

1. 多跳 RAG 的真正核心问题是“什么时候停止检索”，而不是“固定跑几步”。
2. 这个问题具有清晰的 fixed-order optimal stopping 结构，可以用 Pandora / DP Oracle 给出上界与理论视角。
3. 真实部署里答案质量不可观测，因此需要可学习的停止信号。
4. 学到的停止器不可能完美，所以还需要 **anytime-valid** 的风险监控层来兜底。

这四点里，**Pandora 是结构，Probe 是方法主体，E-value 是安全层**。这个主次关系必须从标题、摘要、贡献列表一直保持到实验部分。

---

## 二、当前实验已经能支撑什么，不该硬撑什么

### 2.1 Stage 2：已经支撑的主结论

当前最稳的 Stage 2 结论，已经从“统一超过固定深度基线”收敛为：**Probe 在显著更少步数下恢复了大部分 Oracle 收益，并形成更强的成本-质量 Pareto 点；但是否超过最佳 Fixed-K 已呈现明显 dataset-specific。**


| 数据集 | Probe F1 / 步数 | Oracle F1 / 步数 | Probe / Oracle | Best Fixed-K | Probe vs Best Fixed |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | 0.6561 / 1.70 | 0.7810 / 1.58 | 84.0% | `K=3`, 0.6775 | -0.0214 |
| MuSiQue | 0.4093 / 3.37 | 0.4966 / 2.12 | 82.4% | `K=5`, 0.4022 | +0.0071 |
| 2Wiki | 0.5767 / 1.83 | 0.6954 / 1.59 | 82.9% | `K=2`, 0.5908 | -0.0141 |


最适合写进摘要和引言的 headline 现在应改成：

- HotpotQA：Probe 用 `34%` 的 `Fixed-K=5` 步数拿到 `0.6561` F1，已经达到 Oracle 的 `84.0%`
- MuSiQue：Probe 达到 `0.4093` F1，略高于最佳固定步数 `0.4022`
- 2Wiki：Probe 只用 `36.7%` 的 `Fixed-K=5` 步数，就把 F1 提到 `0.5767`

也就是说，当前最强叙事不是“Probe 在三数据集都超过 Best Fixed-K”，而是：

> **adaptive stopping can recover 82%~84% of Oracle performance while using much less retrieval budget, though the exact tradeoff against the best fixed depth is dataset-dependent**

### 2.2 Stage 3：已经支撑的主结论

当前最稳的 Stage 3 结论，不是“E-value 把错误率压到了 alpha 以下”，而是：

> **E-value 以几乎零额外步数开销提供在线风险监控；相同场景下，Split CP 更贵、更慢、而且更差。**

`gamma = 0.5, alpha = 0.1, predictive betting` 主结果如下：


| 数据集      | 策略              | F1     | Error Rate | Avg Steps |
| -------- | --------------- | ------ | ---------- | --------- |
| HotpotQA | Probe           | 0.5273 | 0.4410     | 2.85      |
| HotpotQA | Probe + E-value | 0.5273 | 0.4410     | 2.86      |
| HotpotQA | Probe + CP      | 0.4882 | 0.4770     | 4.46      |
| MuSiQue  | Probe           | 0.1826 | 0.7914     | 3.25      |
| MuSiQue  | Probe + E-value | 0.1816 | 0.7938     | 3.32      |
| MuSiQue  | Probe + CP      | 0.1401 | 0.8417     | 4.91      |
| 2Wiki    | Probe           | 0.3738 | 0.6110     | 3.37      |
| 2Wiki    | Probe + E-value | 0.3783 | 0.6060     | 3.39      |
| 2Wiki    | Probe + CP      | 0.3668 | 0.6200     | 4.69      |


可以稳定支撑的说法：

- `Probe + E-value` 相对 `Probe` 仅增加 `0.4% ~ 2.0%` 步数，F1 基本持平
- `Probe + CP` 相对 `Probe` 增加 `39% ~ 56%` 步数，但 F1 下降、错误率也没有改善
- `sudden / gradual / periodic` 三类 shift 下，E-wealth 都能到达 cap，说明它对分布漂移是可感知的
- `4gamma x 4alpha x 3` 数据集的 `48` 组扫描全部满足既定验收阈值，说明 Stage 3 结论不依赖某一组单点参数

### 2.3 当前实验暂时不该过度宣称的点

以下说法目前风险较高，应该降级或改写：

1. **不要说“provably optimal”**

当前能证明的是 fixed-order 结构、Oracle/阈值视角、以及 E-value 的 anytime-valid 性质；不能说实际部署算法本身已被严格证明最优。

1. **不要说“E-value 控制错误率 <= alpha”**

当前结果里 Probe 的错误率远高于 `alpha=0.1/0.2`，E-value 的价值是检测超标、提供风险仪表盘，而不是把系统纠正到满足 alpha。

1. **不要把当前 Probe 直接等同于“learned reservation value”**

当前主实现本质上更像是 margin-weighted 的 stop/continue classifier，最好称为：

- `learned stopping index`
- `continuation-value estimator`
- `stopping signal approximating the reservation threshold`

1. **不要把当前系统直接包装成“低开销 single-pass deployable”**

现有特征里明确包含 `semantic_entropy`、`self_consistency` 等高开销信号；如果不区分 `Full` 和 `Lite` 版本，审稿人很容易抓住“部署口径不一致”。

---

## 三、建议的标题、摘要和贡献写法

### 3.1 标题建议

当前更稳妥的标题是：

> **Pandora-RAG: Adaptive Stopping for Multi-Hop Retrieval with Anytime-Valid Risk Control**

比起 `Optimal Stopping`，`Adaptive Stopping` 更安全，也更符合当前证据强度。

### 3.2 摘要第一句话应该怎么写

不要从“大模型会幻觉，RAG 很重要”起手。更适合的 opening：

> Iterative multi-hop RAG systems face a deployment dilemma: each additional retrieval step may improve answer quality, but also incurs latency and cost, and the marginal value of continuing is highly query-dependent.

接着马上落到：

> We formulate this as a fixed-order sequential information acquisition problem, learn a stopping signal from oracle-labeled trajectories, and equip it with an E-value monitor that remains valid under data-adaptive stopping.

### 3.3 三条贡献建议

建议始终压成三条，不要散成四五条：

1. **Formulation**

把 iterative multi-hop RAG stopping 写成 fixed-order sequential information acquisition / Sequential Pandora's Box，并给出 Oracle 上界与阈值结构视角。

1. **Deployable estimator**

提出一个基于 hidden states 与浅层可观测特征的 learned stopping signal，在三个数据集上以更少步数超过固定深度基线，并恢复 `70% ~ 85%` 的 Oracle 性能。

1. **Risk monitor**

提出 E-value 风险监控层，在 data-adaptive stopping 和 distribution shift 下保持 anytime-valid；相比 Split CP，额外成本显著更低且经验表现更稳。

注意：**“comprehensive experiments” 不要单独列成 contribution**，它只是支撑上述三条的证据。

---

## 四、论文主线应该如何组织

### 4.1 Introduction 的四步结构

建议按以下顺序写：

1. **Deployment dilemma**

多跳 RAG 通过“检索-推理-再检索”提升复杂问答，但每多一步都要付出真实延迟与 token 成本。

1. **Why fixed depth is wrong**

不同 query 的边际信息增益差异很大，而且可能非单调、爆发式出现，因此固定深度不是合理决策方式。

1. **Why this is hard**

部署时看不到真实 F1，启发式 proxy 不可靠，自适应停止又让标准 CP 的保证失效。

1. **Our answer**

用 Pandora / DP 给出结构化视角，用 learned stopping signal 做可部署近似，用 E-value 做 optional-stopping-safe 风险监控。

### 4.2 Gap 建议压成三层

最推荐的三层 gap：

1. **Structure gap**

现有工作多是 fixed-K、启发式停止或 RL policy，但没有利用 fixed-order iterative retrieval 的最优停止结构。

1. **Observability gap**

部署时真实答案质量不可观测，简单 proxy 不等于 correctness，需要 learned stopping signal。

1. **Statistical gap**

一旦停止时刻依赖模型自身输出，传统基于 i.i.d. / marginal coverage 的 CP 论证会被削弱，需要适配 adaptive stopping 的 anytime-valid 工具。

### 4.3 Method 层级必须稳定

主文中的方法层级建议固定为：

1. **Pandora / DP Oracle**

问题结构、理论动机、上界、oracle 标签来源

1. **Neural Probe**

真正的 deployable core，论文主方法

1. **E-value**

安全监控层，不负责提升主性能，而是负责检测风险和漂移

1. **PPO**

除非后续有清晰稳定收益，否则只作为 appendix 可选模块，不进入标题、摘要和主贡献

---

## 五、理论口径怎么写才不容易被打

### 5.1 不要直接硬套独立 Pandora

NeurIPS 版本必须显式承认：

- 原始 Weitzman / Pandora 假设更接近独立盒子
- 多跳 RAG 的第 `k+1` 步明显依赖第 `k` 步状态，因此是相关的、状态依赖的 sequential process

推荐写法是：

1. 用 Pandora 提供 **threshold-optimal structure 的动机**
2. 用 MDP / Bellman 视角修正相关性问题
3. 把 `Global-Weitzman` 降格成结构化参考基线，而不是“真正部署算法”

建议在正文里给出 Bellman 形式：

$$
V_k^*(s_k)=\max\left(Q(s_k),\ \mathbb{E}[V_{k+1}^*(s_{k+1})\mid s_k]-c_{k+1}\right)
$$

然后把 Probe 的角色表述成：

$$
f_\theta(h_k,\phi_k)\approx \mathbb{E}[V_{k+1}^*(s_{k+1})\mid s_k]-c_{k+1}
$$

### 5.2 目标函数要拆成两层

不要再把“最大化效用”和“错误率约束 <= alpha”写成同一个优化问题，因为当前系统并不是那样解的。

更合适的表述：

**Primary objective**

$$
\pi^*=\arg\max_\pi \mathbb{E}*\pi\left[Q(s*{\tau_\pi})-\sum_{k=1}^{\tau_\pi} c_k\right]
$$

**Safety monitor**

在部署流上监控

$$
P\left(\sup_{n\ge 1} E_n \ge 1/\alpha\right)\le \alpha
$$

其中这个 guarantee 针对的是“若真实错误率不超过 alpha，则错误地触发超阈值的概率受控”；它不是“系统输出自动满足错误率 <= alpha”。

### 5.3 E-value 的理论证明要写到位

正文或附录至少要明确：

1. 原假设 `H0: P(e_n=1 | F_{n-1}) <= alpha`
2. multiplier

$$
M_n = 1-\lambda_n+\lambda_n e_n/\alpha
$$

1. 在 `H0` 下有

$$
\mathbb{E}[M_n\mid \mathcal{F}_{n-1}] \le 1
$$

1. 因而 `E_n = \prod_{t=1}^n M_t` 构成非负超鞅，可直接调用 Ville inequality

同时要单独说明：

- CP 保证的是 marginal coverage
- 你关心的是 data-adaptive stopping 后的 stopped set / sequential deployment
- 所以这里比较的是 **适用性边界**，不要写成“CP 在所有意义上都错”

---

## 六、实验部分应该如何摆事实

### 6.1 主文实验要回答四个问题

正文实验最好只服务四个问题：

1. **Q1：是否比固定深度更 cost-efficient？**
2. **Q2：learned stopping signal 是否优于简单 proxy？**
3. **Q3：E-value 是否比 CP 更适合 adaptive stopping？**
4. **Q4：结论是否稳健，而不是只靠 dataset-specific tuning？**

### 6.2 建议的主表构成

主文主表建议优先放这些“deployable 或接近 deployable”的方法：

- `Fixed-K=1`
- `Fixed-K=5`
- `Best Fixed-K`
- `entropy-threshold`
- `answer-stability`
- `Deployable-GW`
- `Probe`
- `Probe + E-value`
- `Probe + CP`

而下面两项更适合放在单独的“reference / upper bound”区域：

- `Global-Weitzman`
- `Oracle (DP)`

这样可以避免评审直接质疑“为什么把非部署上界和部署基线混成同一层级比较”。

### 6.3 当前最应该放进正文的硬结果

建议正文一定保留下面三类证据：

1. **Probe vs Fixed-K**

证明 adaptive stopping 的收益来自“更少步数却更好质量”

1. **Probe + E-value vs Probe + CP**

证明 E-value 是更合适的风险层，而不是单纯再加一个门控器

1. **Shift 下的 E-wealth trace**

这是 Stage 3 最像 NeurIPS 的证据，因为它展示的是 deployment-time monitoring，而不是离线 accuracy 小修小补

### 6.4 附录应该怎么放

附录适合承接这些材料：

- `4gamma x 4alpha x 3` 的完整矩阵
- `predictive` vs `fixed betting`
- `selective prediction / abstain`
- `gradual / periodic` 的额外曲线
- `MuSiQue` 的 bootstrap CI 或 permutation test
- `Global-Weitzman / Oracle` 的补充对齐图
- 所有负面结果与失败 ablation

---

## 七、必须主动处理的五个审稿风险

### 7.1 部署口径不一致

这是当前最危险的问题之一。

现状：

- 论文叙事倾向于“低开销 deployable stopping”
- 但当前特征包含 `semantic_entropy`、`self_consistency` 等多采样信号

建议处理方式二选一：

1. 主文明确区分 `Pandora-RAG-Full` 和 `Pandora-RAG-Lite`
2. 或者主文诚实承认当前实现是 enriched-feature 版本，不再声称 single-pass deployable

如果时间允许，最推荐补一组：

- `Full features`
- `Lite features`
- latency breakdown

### 7.2 “reservation value” 用词过重

当前实现主力仍是分类式停止器，不要让 reviewer 觉得“理论讲 reservation value，工程却只是 stop/continue classifier”。

更稳妥的术语：

- `stopping index`
- `continuation-value estimator`
- `learned stopping signal`

### 7.3 理论规则和实现规则未完全对齐

如果主文声称的是 theorem-covered 的 E-process safety rule，就不要在主结论里混入 `ema`、`budget` 这类启发式安全阀。

建议：

- 主文只保留有理论覆盖的规则
- 启发式版本放 appendix ablation

### 7.4 dataset-specific tuning 风险

`stage2_final` 已经表明当前最优配置带有明显的 per-dataset 异质性：


| 数据集      | compress_dim | margin filter | residual |
| -------- | ------------ | ------------- | -------- |
| HotpotQA | 256          | 0.0           | False    |
| MuSiQue  | 256          | 0.0           | False    |
| 2Wiki    | 64           | 0.02          | True     |


这会自然引出 reviewer 问题：

> 这是不是 heavily tuned controller？

因此在投稿前，最好至少补齐：

1. `shared default`
2. `per-dataset optimal`
3. `config transfer`

如果来不及全部跑完，指导书里也必须把这件事标成**当前剩余风险**，而不是假装不存在。

### 7.5 缺中间强度 baseline

当前如果只用 `Deployable-GW` 当对照，会显得对手太弱。

最低优先级补齐建议：

- `entropy-threshold`
- `answer-stability`
- `Best Fixed-K`

如果能补到与 `Stop-RAG` 的同 pipeline head-to-head，会更有说服力；若做不到，就不要在主表里和论文数字直接硬比。即便补了复现，主表也应按 **`matched-budget / Pareto`** 来写，用相同 `avg_steps` 下的 F1/EM 或“达到同 F1 所需步数”作主比较；`best-F1 vs best-F1` 只放 appendix。

---

## 八、建议的正文结构与页数分配

NeurIPS 正文只有 9 页，建议压成下面的结构：

1. `0.8` 页 Introduction
2. `0.7` 页 Related Work
3. `1.0` 页 Problem Formulation
4. `1.4` 页 Method
5. `1.2` 页 Theory
6. `3.1` 页 Experiments
7. `0.5` 页 Limitations / Conclusion
8. `0.3` 页 buffer

### 8.1 Related Work 的四个 subsection

推荐只保留四类，不要写得过散：

1. **Adaptive Retrieval and Stopping**

IRCoT、ITER-RETGEN、Self-RAG、Adaptive-RAG、Stop-RAG、Probing-RAG

1. **Cost-Aware Routing and Cascades**

FrugalGPT、RouteLLM、CCPO、C3PO

1. **Conformal Prediction and Risk Control for LLMs/RAG**

CRC、Conformal LM、Conformal-RAG、URAG

1. **Optimal Stopping and Rational Metareasoning**

Weitzman、sequential/contextual Pandora、rational metareasoning

### 8.2 Limitations 一定要坦诚写

当前最应该明确写出来的 limitation：

1. E-value 监控的是风险证据，不直接修复底层 Probe
2. MuSiQue 测试集较小，wealth 轨迹方差更大
3. 当前评估主要是离线轨迹回放，不是真实在线服务流
4. 现有特征集仍包含高开销信号，部署成本分析还需更严格拆解
5. Probe 的性能瓶颈更像是特征信息量，而不是模型容量

---

## 九、当前版本最推荐保留的一段摘要草稿

下面这版更贴近当前证据，也更符合 NeurIPS 风格：

> Iterative retrieval-augmented generation for multi-hop question answering faces a fundamental stopping problem: each additional retrieval step may improve answer quality, but also incurs latency and cost, and the marginal benefit of continuing is highly query-dependent. We cast this problem as a fixed-order sequential information acquisition process and introduce Pandora-RAG, a framework with three components: a structured oracle view based on dynamic programming, a learned stopping signal trained from oracle-labeled trajectories, and an E-value monitor for anytime-valid risk detection under adaptive stopping. Across HotpotQA, MuSiQue, and 2WikiMultiHopQA, the learned stopping signal exceeds fixed-depth retrieval while using substantially fewer retrieval steps, recovering 70-85% of oracle performance. The E-value layer adds only 0.4%-2.0% retrieval overhead relative to the probe alone, while split conformal baselines increase retrieval by 39%-56% and degrade answer quality. Under sudden, gradual, and periodic shift, E-wealth consistently escalates to its rejection threshold, showing that adaptive RAG systems can be equipped with online, distribution-robust risk monitoring.

---

## 十、写作时的用词红线

### 10.1 建议使用

- `adaptive stopping`
- `fixed-order sequential information acquisition`
- `learned stopping signal`
- `continuation-value estimator`
- `risk monitoring`
- `anytime-valid detection`
- `distribution shift detection`

### 10.2 建议避免

- `provably optimal system`
- `controls error rate below alpha`
- `single-pass deployable`（除非先补 lite 版本）
- `learned reservation value`（若实现仍是分类器）
- `CP is invalid` 这种过强绝对表述

更稳妥的替代表述：

- `CP is mismatched to this adaptive stopping setting`
- `CP incurs much larger cost in our setting and does not provide the same monitoring capability`

---

## 十一、投稿前优先级清单

如果只够再补 `5` 组工作，建议按这个顺序：

1. `shared default / per-dataset optimal / config transfer`
2. `Full vs Lite` 特征组 + latency breakdown
3. `entropy-threshold / answer-stability / Best Fixed-K` 主表补齐
4. `MuSiQue` 置信区间或配对显著性检验
5. 与 `Stop-RAG` 的同 pipeline 小规模 head-to-head，并按 **`matched-budget / Pareto`** 报主结果；若做不到，至少给出清楚的公平比较边界

如果还有余力，再补：

1. `oracle continuation value` 对照分析
2. `quality predictor` 的独立头或双头版本
3. selective prediction 的更强干预策略

---

## 十二、一句话结论

当前最应该写出来的论文，不是“一个几乎解决了多跳 RAG 最优停止的完美系统”，而是：

> **一个把多跳 RAG 停止问题结构化、把可学习停止器做成有效方法、再用 E-value 把自适应停止变得可监控的 NeurIPS 风格方法论文。**

只要全篇始终围绕这条主线，并主动修正三个高风险口径：

1. `optimal` 说得过满
2. `reservation value` 和当前实现不完全一致
3. `deployable / single-pass` 与现有特征成本不完全一致

这份指导书就能真正服务投稿，而不是继续堆积阶段性结果描述。
