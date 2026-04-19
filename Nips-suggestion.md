# Pandora-RAG NeurIPS 投稿评估与补强路线

> 日期：2026-04-20  
> 范围：重新审视当前仓库的 README、methods、Stage 1/2/3 报告、Stop-RAG 对齐结果、结果表与主实现后，对 NeurIPS 投稿胜算、主线叙事、风险边界和补强优先级做一次统一修订。  
> 结论先行：Pandora-RAG 已经具备一篇 NeurIPS-style 方法论文的骨架，但当前还不是“高胜算成稿”。它最适合被包装成 **adaptive stopping for iterative multi-hop RAG + anytime-valid risk monitoring**，而不是 QA SOTA 或“已证明最优的 RAG 系统”。

---

## 1. 总体判断

当前项目已经从单纯阶段性实验推进到完整论文胚子：

- **Stage 1** 已有 fixed-order stopping / Bellman oracle / Global-Weitzman 结构化参照。
- **Stage 2** 已有可部署 Probe、`pdopt_best` 最终 checkpoint、per-step threshold、消融与负结果。
- **Stage 3** 已完成 E-value 主实验、CP 对比、`4gamma x 4alpha x 3` 稳健性扫描、shift 实验、selective prediction 与统一 Pareto 图。
- **Stop-RAG** 已完成同 split、同样本 id、真实在线早停的 aligned comparison。

因此它不是“还差一个想法”的项目，而是“需要收准主张并补齐若干强审稿点”的项目。

粗略投稿判断：

| 状态 | 投稿判断 |
| --- | --- |
| 当前直接写 NeurIPS 主会 | 可以冲，但胜算中等偏低；主要风险是 incremental、best fixed-depth 竞争力强、deployability 成本口径不够干净 |
| 补 Stop-RAG Pareto + Lite Probe + cost accounting | 有真实竞争力；主张会从“有趣组合”变成“完整方法链” |
| 再加 config transfer / 跨 backbone 小验证 / 统计显著性 | 可以认真作为 NeurIPS 主会稿打磨 |
| 时间不足 | EMNLP / ACL Findings / COLM / NeurIPS workshop 更稳 |

最推荐的定位是：

> **Pandora-RAG studies when to stop iterative multi-hop retrieval. It derives a Bellman oracle for the fixed-order stopping structure, learns a deployable stopping signal from oracle-labeled trajectories, and attaches an E-value monitor for anytime-valid online risk evidence.**

中文理解：

> **Pandora 是结构，Probe 是部署停止器，E-value 是风险仪表盘。**

---

## 2. 当前项目最强证据

### 2.1 方法结构已经闭环

Pandora-RAG 的三层结构比较适合方法论文：

| 层 | 当前项目对应 | 论文作用 |
| --- | --- | --- |
| Fixed-order stopping / Bellman oracle | `stage1/run_stage1.py`、Oracle labels、Global-Weitzman | 给出问题结构、上界和训练标签 |
| Neural stopping probe | `stage2/run_stage2.py`、`ProbeMLP_v2`、`pdopt_best` | 主方法，可部署近似停止器 |
| E-value monitor | `stage3/run_stage3.py`、`stage3/stopping.py`、`quality_model.py` | 在线风险证据、漂移感知、低开销安全层 |
| External baseline | `baselines/Stop-RAG/` aligned online results | 证明不是只赢内部 baseline |

这条链比“训练一个 stop classifier”更强，因为它解释了：

1. 停止问题为什么有结构。
2. Oracle 为什么不可部署但能提供监督。
3. Learned probe 为什么是必要近似。
4. Learned probe 不完美时为什么需要在线风险监控。

### 2.2 Stage 2: Probe 是低成本 Pareto 点

当前 `pdopt_best` 的稳定结论是：

> Probe recovers 79.9% to 85.4% of the DP oracle F1 while using much less retrieval budget than fixed full-depth retrieval. It does **not** uniformly beat the best fixed depth on all datasets.

| 数据集 | Probe F1 / steps | Oracle F1 / steps | Probe / Oracle | Best Fixed-K | Probe vs Best Fixed | Probe vs Fixed-K=5 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| HotpotQA | 0.6544 / 1.73 | 0.7810 / 1.58 | 83.8% | `K=3`, 0.6775 | -0.0231 | -0.0124 |
| MuSiQue | 0.3969 / 3.31 | 0.4966 / 2.12 | 79.9% | `K=5`, 0.4022 | -0.0053 | -0.0053 |
| 2Wiki | 0.5941 / 1.82 | 0.6954 / 1.59 | 85.4% | `K=2`, 0.5908 | +0.0033 | +0.0653 |

推荐写法：

> The learned probe recovers most of the oracle benefit at a substantially lower retrieval depth than full-depth retrieval, while the exact comparison to the best fixed depth remains dataset-dependent.

不要写：

> The probe outperforms fixed-depth retrieval on all datasets.

### 2.3 Stage 3: E-value 是低开销监控层

主实验口径：`gamma=0.5, alpha=0.1, predictive betting`。

| 数据集 | 策略 | F1 | EM | Error Rate | Avg Steps |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | Probe | 0.6569 | 0.5190 | 0.3030 | 1.70 |
| HotpotQA | Probe+E-value | 0.6654 | 0.5290 | 0.2950 | 1.81 |
| HotpotQA | Probe+CP | 0.6716 | 0.5310 | 0.2960 | 4.11 |
| MuSiQue | Probe | 0.4152 | 0.3189 | 0.5803 | 3.39 |
| MuSiQue | Probe+E-value | 0.4127 | 0.3141 | 0.5827 | 3.41 |
| MuSiQue | Probe+CP | 0.4015 | 0.3046 | 0.5971 | 4.85 |
| 2Wiki | Probe | 0.5639 | 0.4740 | 0.4090 | 1.82 |
| 2Wiki | Probe+E-value | 0.5728 | 0.4810 | 0.4010 | 1.91 |
| 2Wiki | Probe+CP | 0.5383 | 0.4340 | 0.4370 | 4.32 |

稳妥结论：

- `Probe+E-value` 相比 `Probe` 只增加少量步数：HotpotQA `+6.4%`，MuSiQue `+0.5%`，2Wiki `+4.6%`。
- HotpotQA 和 2Wiki 上 F1 / error 同步小幅改善；MuSiQue 基本持平，是高错误率 stress case。
- `Probe+CP` 不是稳定强基线：HotpotQA 上能以巨大步数成本换一点 F1，MuSiQue 和 2Wiki 上又贵又差。
- E-value 的价值是 **online risk evidence / drift sensitivity**，不是把经验错误率压到 `alpha` 以下。

E-wealth 主现象：

| 数据集 | Final E-wealth at alpha=0.1 | Cap | 解读 |
| --- | ---: | ---: | --- |
| HotpotQA | 0.014 | 10 | 主实验下更像低成本提质门控，而非告警 |
| MuSiQue | 4.193 | 10 | 风险证据持续积累但未封顶 |
| 2Wiki | 10.000 | 10 | 触及 cap，强烈拒绝低错误率原假设 |

### 2.4 Stop-RAG aligned online comparison 是重要外部证据

同 split、同 sample id、真实在线早停：

| 数据集 | 方法 | N | F1 | EM | Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | Stop-RAG | 1000 | 0.5963 | 0.4640 | 5.000 |
| HotpotQA | Pandora Probe+E-value | 1000 | 0.6654 | 0.5290 | 1.807 |
| MuSiQue | Stop-RAG | 417 | 0.2654 | 0.1942 | 4.643 |
| MuSiQue | Pandora Probe+E-value | 417 | 0.4127 | 0.3141 | 3.410 |
| 2Wiki | Stop-RAG | 1000 | 0.5076 | 0.4150 | 4.477 |
| 2Wiki | Pandora Probe+E-value | 1000 | 0.5728 | 0.4810 | 1.905 |

宏平均：

| 方法 | Macro F1 | Macro EM | Macro Steps |
| --- | ---: | ---: | ---: |
| Stop-RAG | 0.4564 | 0.3577 | 4.707 |
| Pandora Probe+E-value | 0.5503 | 0.4414 | 2.374 |

推荐写法：

> Under the aligned online evaluation, Stop-RAG tends to exhaust the retrieval budget, while Pandora achieves higher macro F1 with roughly half the retrieval steps.

边界必须写清楚：这目前是 **single-point online comparison**，不是 Stop-RAG 的完整 threshold-sweep Pareto frontier。主文可以用，但如果要画主 Pareto 对比图，最好补多阈值在线重测。

### 2.5 负结果也很有用

当前项目已经有不少能帮助审稿的负结果：

- `p2_full31` 完整 31 维扩展特征没有提升主工作点，2Wiki 还明显退化。
- F1 regression 头一致差于 binary Continue 头。
- GRU / seq-history / mean pooling / compress_dim=512 等没有稳定收益。
- XGBoost 与 MLP 表明瓶颈更像是特征信息量，而不是模型容量。

这些负结果不应堆进正文，但适合放 appendix，用来回应：

> Is this heavily tuned? Did you try simpler / larger / sequence models?

---

## 3. 最推荐的 NeurIPS 主线

论文不要写成：

> We build a stronger RAG system.

也不要写成：

> We solve optimal stopping for RAG.

建议中心句：

> **Iterative multi-hop RAG stopping is a fixed-order sequential information acquisition problem. Pandora-RAG uses a Bellman oracle to expose the stopping structure, learns a deployable stopping signal from observable states, and attaches an E-value monitor that provides anytime-valid risk evidence under adaptive stopping and distribution shift.**

三条贡献建议：

1. **Fixed-order stopping formulation.**  
   Formulate iterative multi-hop RAG stopping as a fixed-order finite-horizon optimal stopping problem and derive a Bellman oracle that provides both an upper bound and supervision.

2. **Deployable stopping signal.**  
   Train a neural stopping probe from LLM hidden states and observable features. Across HotpotQA, MuSiQue, and 2Wiki, the probe recovers 79.9% to 85.4% of oracle F1 while using much less retrieval budget than fixed full-depth retrieval.

3. **Anytime-valid risk monitoring.**  
   Attach an E-value monitor to adaptive stopping. The monitor supplies online risk evidence and shift sensitivity with modest retrieval overhead, while static CP gates are costly and dataset-sensitive in this setting.

Stop-RAG 不建议单独列为方法贡献，可以放在实验贡献里：

> We further reproduce Stop-RAG under an aligned online protocol and show that Pandora improves macro F1 while reducing the average retrieval depth by roughly half.

---

## 4. 当前最主要审稿风险

### R1. Probe 没有统一超过 best Fixed-K

这是最直接的审稿风险。HotpotQA 和 MuSiQue 上，Probe 主工作点仍低于最佳 fixed depth。

应对策略：

- 主文强调 **Pareto / budgeted stopping**，不要单点吹 F1。
- 把 `Fixed-K=1..5` 全部画在 `avg_steps-F1` 平面上。
- 使用 `Fixed-K=5` 作为 full-depth cost baseline，用 `Best Fixed-K` 作为 quality baseline，两个概念分开。
- Appendix 放 max-F1 operating point，说明质量优先时 Probe 还能换取更高 F1，但主文坚持 cost-aware 口径。

### R2. E-value 容易被误解成错误率控制器

当前 empirical error rate 远高于 `alpha=0.1`，所以绝不能写：

> E-value controls the error rate below alpha.

正确说法：

> E-value provides anytime-valid evidence against a low-error-rate null and serves as an online monitor. It detects risk accumulation; it does not automatically repair the base stopper.

正文或附录必须把原假设写清楚：

$$
H_0:\mathbb{E}[e_n|\mathcal{G}_{n-1}]\le \alpha
$$

以及：

$$
E_n=\prod_{t=1}^{n}\left(1-\lambda_t+\lambda_t e_t/\alpha\right),\qquad
P_{H_0}\left(\sup_n E_n\ge 1/\delta\right)\le \delta.
$$

### R3. Pandora / Weitzman 假设与多跳 RAG 有张力

原始 Pandora's Box 假设更接近独立盒子；多跳 RAG 的下一步检索显然依赖当前状态。

应对策略：

- 用 Pandora 解释 threshold-style information acquisition 的动机。
- 用 finite-horizon Bellman recursion 作为正式方法：

$$
V_k^*(s_k)=\max\left(Q(s_k),\mathbb{E}[V_{k+1}^*(s_{k+1})|s_k]-c_{k+1}\right).
$$

- 把 Global-Weitzman 降格成 structured reference baseline，不说它是部署算法。

### R4. 部署成本口径还不够干净

当前 Full Probe 包含一些可能较贵的 uncertainty / consistency 特征。如果主文说 “lightweight / low-cost deployable”，审稿人会问 stopping overhead 是否抵消检索节省。

应对策略：

- 主文明确区分 `Full Probe` 和 `Lite Probe`。
- Lite 版本只用 hidden state、retrieval score、step index、cheap lexical / score features。
- 成本表至少拆成：retrieval steps、generation calls、feature overhead、probe overhead、total normalized cost。

### R5. Per-dataset tuning 风险

当前最佳配置带有明显 dataset-specific：

| 数据集 | compress_dim | margin filter | residual |
| --- | ---: | ---: | --- |
| HotpotQA | 256 | 0.0 | False |
| MuSiQue | 256 | 0.0 | False |
| 2Wiki | 64 | 0.02 | True |

应对策略：

- 补 `shared default` vs `per-dataset optimal`。
- 若时间允许，补 config transfer：在一个数据集选配置，迁移到另两个数据集。
- 来不及补时，把 per-dataset optimal 放 appendix，正文突出主方法而非“统一超参最优”。

### R6. Stop-RAG 目前只是 single-point

当前 aligned result 很强，但 Stop-RAG 审稿人可能会说阈值没调好。

应对策略：

- 最好补 Stop-RAG 多 threshold 在线 sweep。
- 主图画 `F1 vs avg_steps` Pareto frontier。
- 若不补，只能写成 aligned online single-point comparison，不承载完整 Pareto claim。

### R7. 绝对 QA F1 不是 SOTA

这个项目不应该和强 QA/RAG backbone 比绝对分数。

应对策略：

- Scope 固定为 stopping efficiency under a fixed RAG backbone。
- 主比较都使用同 backbone / 同轨迹 / 同 split。
- 可补一组更强 retriever 或更强 generator 的小规模验证，用来证明 stopping 结论可迁移，而不是为了刷榜。

---

## 5. 投稿前补强优先级

### P0: Stop-RAG threshold sweep / Pareto frontier

优先级最高。

要补：

- 对 Stop-RAG 多个 threshold 做真实在线早停测试。
- 画 `F1 vs avg_steps` 和 `EM vs avg_steps`。
- 报 matched-budget F1 或达到同 F1 所需步数。

收益：

- 直接化解 “Stop-RAG threshold 没调好” 的质疑。
- 让外部 baseline 从 single-point 变成真正 Pareto 对手。

### P0: Lite Probe + cost accounting

优先级最高。

建议做：

| 版本 | 特征 | 目的 |
| --- | --- | --- |
| Full Probe | 当前完整特征 | 主性能上限 |
| Lite Probe | hidden state + retrieval scores + step index + cheap features | 部署口径 |
| Shallow / proxy | self-consistency、entropy、answer stability 等简单规则 | 简单 baseline |

成本表建议：

- avg retrieval steps
- generation calls
- feature computation overhead
- probe forward cost
- total normalized cost

### P1: Shared default / config transfer

要回答：

> Is the method robust, or did it require dataset-specific tuning?

最低可做：

- 一个 shared default 配置。
- 与 `per-dataset optimal` 对比。
- 简短说明性能损失与稳定性。

更强版本：

- HotpotQA 选超参，转 MuSiQue / 2Wiki。
- MuSiQue 选超参，转 HotpotQA / 2Wiki。

### P1: Shift detection timing

E-value 的杀手锏是 online monitoring。

建议在主文或 appendix 报：

- sudden / gradual / periodic shift 下 first cap timing。
- `samples_after_shift_to_first_cap`。
- no-shift vs shift 的 wealth trace。
- CP 固定阈值没有跨样本 risk trajectory。

### P1: 统计显著性和 MuSiQue 稳健性

MuSiQue test 只有 417 条，且错误率高，容易被质疑。

建议补：

- paired bootstrap CI for F1。
- 对 `Probe` vs `Probe+E-value` 和 `Probe+CP` 的 paired comparison。
- 至少对主表加 95% CI 或 bootstrap std。

### P2: 跨 backbone / retriever 小规模验证

目标不是刷 SOTA，而是证明 stopping 结论不是 BM25 + Llama-3.1-8B 的偶然现象。

可选：

- dense retriever + reranker 小规模。
- 更强 generator 小规模。
- 只在 HotpotQA 或 2Wiki 上跑一组 sanity check。

### P2: 检测后干预策略

Selective prediction 已有，但目前更像最小示例。

可补：

- wealth 达 cap 后拒答。
- wealth 达 cap 后强制更高 retrieval budget。
- wealth 达 cap 后切换到 conservative fixed-K。

主文可以只放一个最小干预，强调 E-value 是 monitor，intervention 是 policy layer。

---

## 6. 主文实验组织

建议正文实验只回答四个问题。

### Q1. Adaptive stopping 是否有 Pareto 价值？

主表 / 主图：

- Fixed-K=1..5
- Oracle DP
- Global-Weitzman
- Probe

结论：

> Probe is a low-cost Pareto point that recovers most oracle benefit, although best fixed depth remains competitive on some datasets.

### Q2. Learned stopping signal 是否必要？

放：

- Deployable-GW 退化到接近 `K=1` 的结果。
- Shallow-only / XGBoost / hidden ablations。
- `p2_full31` 作为负结果。

结论：

> Simple proxy thresholds are insufficient; learned signals from hidden representations and oracle labels are necessary.

### Q3. E-value 是否比 CP 更适合 adaptive stopping？

主表：

- Probe
- Probe+E-value
- Probe+CP

主图：

- E-wealth trace。
- cumulative error curve。
- sudden / gradual / periodic shift trace。

结论：

> E-value is a low-overhead online monitor; CP acts as a static gate and is costly / dataset-sensitive in this setting.

### Q4. 外部 dynamic stopping baseline 怎么样？

主表：

- Stop-RAG aligned online single point。
- Pandora Probe+E-value。

若补 sweep：

- Stop-RAG frontier vs Pandora frontier。

结论：

> In the aligned online setting, Pandora improves macro F1 while reducing retrieval depth by roughly half.

---

## 7. Figures & Tables 建议

### 主文

| 编号 | 内容 | 作用 |
| --- | --- | --- |
| Figure 1 | Method overview: Bellman oracle -> Probe -> E-value monitor | 一图讲清方法链 |
| Table 1 | Fixed-K / Oracle / Global-Weitzman / Probe | 证明 adaptive stopping 的 Pareto 价值 |
| Figure 2 | Avg steps vs F1 Pareto plot | 避免陷入单点 F1 |
| Table 2 | Probe / Probe+E-value / Probe+CP | 证明 E-value 低开销，CP 高成本且不稳 |
| Figure 3 | E-wealth trace under no-shift and shift | 展示 E-value 的 NeurIPS 味道 |
| Table 3 | Stop-RAG aligned online comparison | 外部 baseline 支撑 |

### Appendix

- Full Fixed-K=1..5。
- Per-dataset optimal hyperparameters。
- Shared default / config transfer。
- p2_full31 negative result。
- F1 regression / binary / GRU / seq-history / mean pooling ablations。
- Predictive vs fixed betting。
- `gamma, alpha` sensitivity。
- Selective prediction coverage-accuracy。
- Stop-RAG stopping distribution。

---

## 8. 写作红线

不要写：

- `provably optimal deployed RAG system`
- `E-value controls the error rate below alpha`
- `Probe uniformly outperforms fixed-depth retrieval`
- `CP is invalid in all adaptive settings`
- `single-pass low-cost deployable`，除非补了 Lite Probe 与成本表
- `learned exact reservation value`
- `state-of-the-art multi-hop QA`

可以写：

- `fixed-order optimal stopping formulation`
- `Bellman oracle upper bound`
- `oracle-labeled adaptive stopping`
- `learned stopping signal`
- `continuation-decision estimator`
- `cost-quality Pareto tradeoff`
- `anytime-valid online risk evidence`
- `distribution-shift sensitivity`
- `aligned online Stop-RAG comparison`

---

## 9. Abstract 草稿

可以从这版开始改：

> Iterative multi-hop retrieval can improve complex question answering, but each additional retrieval step incurs latency and may introduce distracting evidence. We study when to stop retrieval. We formulate iterative RAG stopping as a fixed-order finite-horizon optimal stopping problem and derive a Bellman oracle that provides both an upper bound and supervision for stopping decisions. Since oracle answer quality is unobservable at deployment time, Pandora-RAG trains a neural stopping probe from LLM hidden states and observable retrieval features. To make adaptive stopping monitorable in deployment, we attach an E-value process that supplies anytime-valid risk evidence under data-adaptive stopping and distribution shift. Across HotpotQA, MuSiQue, and 2Wiki, the probe recovers 79.9% to 85.4% of oracle F1 while using far fewer retrieval steps than fixed full-depth retrieval. The E-value monitor adds only modest retrieval overhead relative to the probe, outperforms static CP gates in cost-quality tradeoff on two of three datasets, and provides online drift-sensitive risk traces. Under an aligned online evaluation, Pandora also improves macro F1 over Stop-RAG while reducing average retrieval depth by roughly half.

如果摘要字数紧张，Stop-RAG 句子移到 Introduction 贡献段。

---

## 10. Introduction 结构建议

1. **Deployment dilemma**  
   Iterative multi-hop RAG 每多检索一步可能找到 bridge evidence，也可能增加延迟、成本和噪声。

2. **Why fixed depth is wrong**  
   不同 query 的边际收益高度异质；固定 K 会浪费简单样本预算，也会在难样本上缺乏风险感知。

3. **Why stopping is hard**  
   部署时看不到真实 F1；简单 proxy 不可靠；自适应停止让静态 calibration / CP 难以提供跨样本在线风险轨迹。

4. **Our answer**  
   Bellman oracle 暴露结构，Probe 学可部署近似，E-value 提供 anytime-valid risk evidence。

5. **Evidence**  
   三数据集 Pareto 结果、E-value vs CP、shift trace、Stop-RAG aligned online comparison。

---

## 11. Related Work 组织

建议四小节：

1. **Adaptive Retrieval and Stopping**  
   IRCoT、ITER-RETGEN、Self-RAG、Adaptive-RAG、FLARE、DRAGIN、Stop-RAG、Probing-RAG。

2. **Optimal Stopping and Pandora's Box**  
   Weitzman、contextual / correlated Pandora、rational metareasoning。强调本文不是直接套独立 Pandora，而是 fixed-order Bellman 视角。

3. **Risk Control for LLMs and RAG**  
   Conformal risk control、Conformal-RAG、CCPO、selective prediction。强调本文关注 adaptive stopping stream 上的 online risk evidence。

4. **Cost-aware LLM Routing and Cascades**  
   FrugalGPT、RouteLLM、LLM cascades。强调 routing / cascade 和 iterative retrieval depth control 的相似与差异。

---

## 12. 最终建议

当前项目可以冲 NeurIPS，但最危险的写法是：

> 我们提出一个几乎解决多跳 RAG 最优停止的强系统。

最有胜算的写法是：

> 我们把 iterative multi-hop RAG 的停止问题结构化为 fixed-order sequential information acquisition，用 Bellman oracle 生成监督信号，训练一个可部署 adaptive stopper，并用 E-value 给这个自适应停止器加上 online risk monitoring。

如果投稿前只能再做三件事，建议顺序是：

1. Stop-RAG threshold sweep / Pareto frontier。
2. Lite Probe + cost accounting。
3. Shared default / config transfer 或 MuSiQue bootstrap CI。

如果这三件补齐，Pandora-RAG 的主线会从“有趣但可能 incremental 的组合”提升为“问题定义清楚、方法链完整、实验口径可信”的投稿状态。
