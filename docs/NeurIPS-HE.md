# Pandora-RAG NeurIPS 写作指南

> 目标：把 Pandora-RAG 写成一篇方法论文，而不是 Stage 1/2/3 工程汇报。  
> 口径：吸收旧版 `NeurIPS-HE.md` 中合理的写作建议，并按 2026-04-19 已完成的 Stage 1/2/3 与 Stop-RAG 对比结果重写。  
> 注意：本文不更新会议官网政策；匿名、页数、checklist 等格式要求以投稿时官方说明为准。

---

## 1. 最终中心句

论文应该围绕这一句写：

> **Iterative multi-hop RAG stopping is a fixed-order sequential information acquisition problem. Pandora-RAG learns a deployable stopping signal from oracle-labeled trajectories and equips it with an E-value monitor that provides anytime-valid risk evidence under adaptive stopping.**

这比“我们做了一个更强的 probe”更像 NeurIPS 方法论文，也比“我们把 Pandora、PPO、CP/E-value 都拼起来”更清楚。整篇文章的主次关系固定为：

1. **Bellman/Pandora structure**：给出问题形式化、oracle 上界与监督标签。
2. **Neural Probe**：真正的可部署停止器，是方法主体。
3. **E-value Monitor**：部署安全层，提供在线风险证据与漂移感知。
4. **Stop-RAG comparison**：外部在线早停基线，用于证明方法不是内部 ablation。

PPO 不应进入标题、摘要或贡献列表；若后续保留，只适合放 appendix 或 future work。

---

## 2. 当前结果能支撑什么

### 2.1 Stage 2: Probe 是低成本 Pareto 点

当前 `pdopt_best` 的稳定结论是：

> Probe recovers 79.9% to 85.4% of the oracle F1 while operating at much lower retrieval depth than fixed full-depth retrieval; however, it does not uniformly beat the best fixed depth on every dataset.


| 数据集      | Probe F1 / steps | Oracle F1 / steps | Probe / Oracle | Best Fixed-K  | Probe vs Best Fixed |
| -------- | ---------------- | ----------------- | -------------- | ------------- | ------------------- |
| HotpotQA | 0.6544 / 1.73    | 0.7810 / 1.58     | 83.8%          | `K=3`, 0.6775 | -0.0231             |
| MuSiQue  | 0.3969 / 3.31    | 0.4966 / 2.12     | 79.9%          | `K=5`, 0.4022 | -0.0053             |
| 2Wiki    | 0.5941 / 1.82    | 0.6954 / 1.59     | 85.4%          | `K=2`, 0.5908 | +0.0033             |


最适合 abstract / introduction 的说法：

- Probe 不是“全面超过 Best Fixed-K”，而是“在显著更少或相近步数下恢复大部分 Oracle 收益”。
- 2Wiki 已小幅超过最佳 fixed depth；HotpotQA 和 MuSiQue 则体现 cost-quality tradeoff。
- 这恰好说明 stopping paper 应看 Pareto frontier，而不是只看单点 F1。

### 2.2 Stage 3: E-value 是风险监控，不是错误率硬控制

主实验 \gamma=0.5,\alpha=0.1，predictive betting：


| 数据集      | 策略            | F1     | Error Rate | Avg Steps |
| -------- | ------------- | ------ | ---------- | --------- |
| HotpotQA | Probe         | 0.6569 | 0.3030     | 1.70      |
| HotpotQA | Probe+E-value | 0.6654 | 0.2950     | 1.81      |
| HotpotQA | Probe+CP      | 0.6716 | 0.2960     | 4.11      |
| MuSiQue  | Probe         | 0.4152 | 0.5803     | 3.39      |
| MuSiQue  | Probe+E-value | 0.4127 | 0.5827     | 3.41      |
| MuSiQue  | Probe+CP      | 0.4015 | 0.5971     | 4.85      |
| 2Wiki    | Probe         | 0.5639 | 0.4090     | 1.82      |
| 2Wiki    | Probe+E-value | 0.5728 | 0.4010     | 1.91      |
| 2Wiki    | Probe+CP      | 0.5383 | 0.4370     | 4.32      |


可支撑的强结论：

1. `Probe+E-value` 相对 `Probe` 的额外步数很小：约 `+0.5%` 到 `+6.4%`。
2. HotpotQA 与 2Wiki 上，E-value 小幅改善 F1 和 error；MuSiQue 上基本持平，是高错误率 stress case。
3. `Probe+CP` 不是稳定强基线：HotpotQA 上以巨大步数成本换取一点 F1，MuSiQue 和 2Wiki 上则又贵又差。
4. E-value 的价值是 online risk evidence 和 drift sensitivity，不是把经验错误率压到 \alpha 以下。

E-wealth 主表述：


| 数据集      | Final E-wealth at α=0.1 | Cap | 解读                 |
| -------- | ----------------------- | --- | ------------------ |
| HotpotQA | 0.014                   | 10  | 主实验下更像低成本提质门控      |
| MuSiQue  | 4.193                   | 10  | 风险证据持续积累但未封顶       |
| 2Wiki    | 10.000                  | 10  | 触及告警边界，强烈拒绝低错误率原假设 |


### 2.3 Stop-RAG 对齐已能进入主文

同 split、同 sample id、真实在线早停：


| 数据集      | 方法                    | F1     | EM     | Avg Steps |
| -------- | --------------------- | ------ | ------ | --------- |
| HotpotQA | Stop-RAG              | 0.5963 | 0.4640 | 5.000     |
| HotpotQA | Pandora Probe+E-value | 0.6654 | 0.5290 | 1.807     |
| MuSiQue  | Stop-RAG              | 0.2654 | 0.1942 | 4.643     |
| MuSiQue  | Pandora Probe+E-value | 0.4127 | 0.3141 | 3.410     |
| 2Wiki    | Stop-RAG              | 0.5076 | 0.4150 | 4.477     |
| 2Wiki    | Pandora Probe+E-value | 0.5728 | 0.4810 | 1.905     |


宏平均：


| 方法                    | Macro F1 | Macro EM | Macro Steps |
| --------------------- | -------- | -------- | ----------- |
| Stop-RAG              | 0.4564   | 0.3577   | 4.707       |
| Pandora Probe+E-value | 0.5503   | 0.4414   | 2.374       |


推荐写法：

> Under the aligned online evaluation, Stop-RAG tends to exhaust the retrieval budget, while Pandora achieves higher macro F1 with roughly half the retrieval steps.

边界：这是 single-point online comparison，不是完整 threshold-sweep Pareto frontier。主文可以用，但图注和正文要诚实说明。

---

## 3. 贡献列表

建议压成三条主贡献：

**Contribution 1: Fixed-order stopping formulation.**  
We formulate iterative multi-hop RAG stopping as a fixed-order finite-horizon optimal stopping problem and derive a Bellman oracle that provides both a structural upper bound and training labels.

**Contribution 2: Deployable stopping signal.**  
We train a lightweight neural stopping probe from LLM hidden states and low-cost observable features. The probe recovers 79.9% to 85.4% of oracle F1 while using much less retrieval budget than fixed full-depth retrieval.

**Contribution 3: Anytime-valid monitoring.**  
We attach an E-value risk monitor to adaptive stopping. The monitor supplies online risk evidence and distribution-shift sensitivity with modest retrieval overhead, while static CP gates are costly and dataset-sensitive in this setting.

Stop-RAG 对齐结果放在实验贡献里：

> We further reproduce Stop-RAG under an aligned online protocol and show that Pandora improves macro F1 while reducing the average retrieval depth by roughly half.

不要单独列 “comprehensive experiments” 作为贡献；它是证据，不是方法贡献。

---

## 4. Introduction 应该怎么写

### 4.1 开头不要太泛

不要从“LLMs hallucinate, RAG helps”起手。更适合：

> Iterative multi-hop RAG systems face a deployment dilemma: each additional retrieval step can reveal bridge evidence and improve reasoning, but it also incurs latency and computation cost. The marginal value of another retrieval is highly query-dependent and state-dependent.

### 4.2 三层 gap

**Structure gap.**  
现有多跳 RAG 多使用 fixed depth、启发式触发或 RL policy，但较少显式利用 fixed-order iterative retrieval 的 optimal stopping 结构。

**Observability gap.**  
部署时真实 F1 不可观测，简单代理信号如 self-consistency 不足以刻画多跳 correctness。当前 Deployable-GW 退化到一步停止，正好证明 learned signal 的必要性。

**Statistical gap.**  
停止时刻依赖模型自身输出后，传统静态 calibration / CP 门控无法提供跨样本的在线风险轨迹。E-values 可以给出 optional-stopping-safe 的风险证据过程。

### 4.3 Introduction 末段结构

推荐顺序：

1. We formulate stopping as fixed-order optimal stopping.
2. We derive an oracle DP and train a neural probe from oracle-labeled trajectories.
3. We attach an E-value monitor for anytime-valid risk evidence.
4. Experiments show Pareto efficiency, low-overhead monitoring, and aligned Stop-RAG gains.

---

## 5. Method 部分层级

主文方法层级建议固定为：

1. **Problem formulation**
  - s_k=(q,d_1,\ldots,d_k)
  - Q(s_k), c_k, \tau
  - primary objective Q(s_\tau)-\sum c_k
2. **Bellman oracle**
  - V_K=Q_K
  - V_k=\max(Q_k,V_{k+1}-c_{k+1})
  - `margin` and `action_label`
  - Oracle is upper bound and supervision, not deployable
3. **Neural stopping probe**
  - hidden state + shallow features
  - binary Continue classifier
  - focal BCE + label smoothing
  - dev Pareto threshold selection + per-step refinement
4. **E-value monitor**
  - quality model \hat p=P(F_1\ge\gamma)
  - e_n=\mathbf{1}F_1<\gamma
  - E_n=\prod_t(1-\lambda_t+\lambda_t e_t/\alpha)
  - Ville guarantee
  - quality bar / wealth-aware gate as implementation
5. **Baselines**
  - Fixed-K, Probe, Probe+CP, Stop-RAG
  - Global-Weitzman and Oracle separated as semi-oracle / oracle

---

## 6. 理论 claim 边界

可以证明：

1. Fixed-order stopping admits Bellman optimality.
2. A margin estimator recovers oracle actions outside low-margin and high-error regions.
3. The E-process is a nonnegative supermartingale under H_0:\mathbb{E}[e_n|\mathcal{G}_{n-1}]\le\alpha, giving Ville-style anytime-valid alerts.

不能证明：

1. The deployed Probe is optimal.
2. The Probe exactly learns Weitzman reservation values.
3. E-value forces empirical error rate below \alpha.
4. CP is universally invalid under all adaptive procedures.

推荐术语：

- `adaptive stopping`
- `fixed-order optimal stopping`
- `Bellman oracle`
- `learned stopping signal`
- `continuation-decision estimator`
- `E-value risk monitor`
- `anytime-valid risk evidence`

避免术语：

- `provably optimal deployed system`
- `exact neural reservation value`
- `guaranteed error-rate control`
- `CP fails by theory`

---

## 7. Experiments 组织

主文实验应该回答四个问题。

### Q1. Adaptive stopping 是否有 Pareto 价值？

主图或主表：

- Fixed-K=1..5
- Oracle DP
- Global-Weitzman
- Probe

叙事：

> Probe is not uniformly better than the best fixed depth, but it recovers most oracle benefit at a much lower or comparable retrieval budget.

### Q2. Learned signal 是否比简单 proxy 好？

放：

- Deployable-GW 退化到接近 K=1
- Shallow-only / XGBoost / hidden-state ablations
- `p2_full31` 负结果可放 appendix

叙事：

> Simple analytic proxies do not expose multi-hop correctness; hidden states and learned stopping labels are necessary.

### Q3. E-value 是否是低开销风险监控？

主表：

- Probe
- Probe+E-value
- Probe+CP

主图：

- E-wealth trace
- cumulative error curve
- sudden / gradual / periodic shift trace

叙事：

> E-value is not a magic error reducer; it provides online evidence and low-overhead gating.

### Q4. 与 Stop-RAG 的关系是什么？

主表可放 aligned online single-point comparison。推荐写清：

- 同 split、同 sample id
- Stop-RAG 使用真实在线早停文件
- 当前不是完整 Pareto sweep

叙事：

> Stop-RAG runs close to max budget in this aligned setting; Pandora obtains higher F1 with fewer steps.

---

## 8. Figures & Tables 规划

**Table 1: Main Stopping Results.**  
三数据集 × Fixed-K / Oracle / Global-Weitzman / Probe。突出 avg steps 与 F1。

**Table 2: Risk Monitor Comparison.**  
Probe / Probe+E-value / Probe+CP，报告 F1、EM、error rate、avg steps、final wealth。

**Table 3: Stop-RAG Aligned Online Comparison.**  
三数据集 + macro average。

**Figure 1: Method Overview.**  
Query -> iterative retrieval states -> Bellman oracle labels -> neural probe -> E-value monitor。

**Figure 2: Pareto Frontier.**  
Avg steps vs F1，标出 Oracle、Global-Weitzman、Probe、Probe+E-value、Probe+CP。

**Figure 3: E-wealth Under Shift.**  
No-shift vs sudden/gradual/periodic，展示 wealth 如何积累风险证据。

**Appendix Tables.**

- Full Fixed-K=1..5
- per-dataset optimal hyperparameters
- p2_full31 negative result
- predictive vs fixed betting
- \gamma,\alpha sensitivity
- selective prediction coverage-accuracy
- Stop-RAG stopping distribution

---

## 9. Related Work 组织

建议四小节：

**Adaptive Retrieval and Stopping.**  
IRCoT、ITER-RETGEN、Self-RAG、Adaptive-RAG、FLARE、DRAGIN、Stop-RAG、Probing-RAG。定位：多数工作关注是否/何时检索，但较少提供 fixed-order optimal stopping 结构与在线风险证据。

**Optimal Stopping and Pandora's Box.**  
Weitzman、contextual/correlated Pandora、rational metareasoning。定位：本文不是直接套独立 Pandora，而是用 fixed-order Bellman 递推修正多跳 RAG 的状态依赖性。

**Risk Control for LLMs and RAG.**  
Conformal risk control、Conformal-RAG、CCPO、selective prediction。定位：现有 CP 风格方法多是静态或 marginal guarantee；本文关注 adaptive stopping stream 上的 online risk evidence。

**Cost-aware LLM Routing and Cascades.**  
FrugalGPT、RouteLLM、LLM cascades、C3PO/CCPO。定位：routing/cascade 与 iterative retrieval depth control 相近但不相同；本文的动作是是否继续检索同一问题。

---

## 10. Reviewer 风险与回应

### R1: “E-value 没把 error rate 降到 α 以下，有什么用？”

回应：

> E-values are used as an anytime-valid monitor, not a hard controller. The guarantee is about false alarms under a low-error null. When the error rate is high, wealth accumulation is evidence that the null should be rejected and deployment intervention is needed.

### R2: “Probe 只是分类器，不是 reservation value。”

回应：

> Correct. We describe it as a learned stopping signal / continuation-decision estimator. The Bellman oracle supplies action labels and margins; the deployed model learns the decision boundary rather than exact values.

### R3: “Pandora 假设独立盒子，多跳 RAG 不是。”

回应：

> We use Pandora as motivation for threshold-style information acquisition, but the formal result is a finite-horizon fixed-order Bellman recursion that permits state dependence.

### R4: “CP 为什么比较差，是不是实现不公平？”

回应：

> CP is implemented as a split quantile gate over the same quality model and same probe stops. It is a static gate and tends to force more retrieval. We report it as a natural calibration baseline, while emphasizing that CP and E-values provide different statistical objects.

### R5: “Stop-RAG 对比是不是只挑了一个点？”

回应：

> We report an aligned online single-point evaluation using Stop-RAG's selected checkpoint and threshold. We explicitly mark it as a single-point comparison; a full threshold sweep is optional future/appendix work, not the basis of the main theoretical claim.

### R6: “绝对 F1 不是 SOTA。”

回应：

> The paper studies stopping efficiency under a fixed RAG backbone, not maximizing QA backbone accuracy. For internal policies we compare under the same cached trajectories; for Stop-RAG we use the same split and sample ids under a true online stopping protocol.

---

## 11. Abstract 草稿

可从下面这版开始改：

> Iterative multi-hop retrieval can improve complex question answering, but each additional retrieval step incurs latency and may introduce distracting evidence. We study when to stop retrieval. We formulate iterative RAG stopping as a fixed-order finite-horizon optimal stopping problem and derive a Bellman oracle that provides both an upper bound and supervision for stopping decisions. Since oracle answer quality is unobservable at deployment time, Pandora-RAG trains a lightweight neural stopping probe from LLM hidden states and low-cost observable features. To make adaptive stopping monitorable in deployment, we attach an E-value process that supplies anytime-valid risk evidence under data-adaptive stopping. Across HotpotQA, MuSiQue, and 2Wiki, the probe recovers 79.9% to 85.4% of oracle F1 while using far fewer retrieval steps than fixed full-depth retrieval. The E-value monitor adds only modest retrieval overhead relative to the probe and provides online drift-sensitive risk traces. Under an aligned online evaluation, Pandora also improves macro F1 over Stop-RAG while reducing average retrieval depth by roughly half.

注意：如果摘要字数紧张，Stop-RAG 句子可移到 introduction 贡献后半段。

---

## 12. 最终写作策略

主文不要写成：

> Stage 1 did oracle, Stage 2 trained probe, Stage 3 added E-value.

要写成：

> We identify stopping as the missing decision problem in iterative RAG, derive its oracle structure, learn a deployable approximation, and add online risk evidence so that the approximation can be monitored under adaptive deployment.

最强的论文叙事是：

1. Fixed-K wastes budget because information gain is state-dependent.
2. Bellman DP gives an oracle stopping structure and labels.
3. A learned probe recovers much of the oracle benefit from observable signals.
4. E-value makes the learned adaptive stopper monitorable under optional stopping and shift.
5. Stop-RAG alignment shows the method beats a direct dynamic stopping baseline in the current online setting.

这条线既吸收了旧版文档里最有价值的建议，也匹配现在已经跑完的真实结果。