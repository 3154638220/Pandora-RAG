# Pandora-RAG Methods 与论文叙事

> 用途：辅助论文写作，统一项目方法、实验口径和叙事主线。  
> 最终口径：以 2026-04-19 的 `docs/reports/stage2/stage2_final.md`、`docs/reports/stage3/stage3_results_summary.md`、`docs/reports/baselines/stop_rag_alignment_closeout.md` 为准。

---

## 1. 论文中心句

Pandora-RAG 的论文中心句应当是：

> Iterative multi-hop RAG stopping is a fixed-order sequential information acquisition problem. Pandora-RAG uses a Bellman/Pandora-style oracle to expose the stopping structure, learns a deployable stopping signal from observable states, and attaches an E-value monitor for anytime-valid risk evidence under adaptive stopping.

中文展开：

1. 多跳 RAG 的关键不是“是否做检索”，而是“在每个问题上检索到第几步才值得停”。
2. 固定步数策略忽略了 query-level 信息增益的异质性：有些问题一步足够，有些问题必须等 bridge evidence 出现。
3. 真实部署中看不到当前答案 F1，因此需要从 hidden state 和低成本特征中学习 stop/continue 信号。
4. 学到的停止器不可能完美，因此再用 E-value 作为在线风险监控层，而不是把它包装成错误率硬控制器。

一句话版本：

> **Pandora 是结构，Probe 是部署停止器，E-value 是安全仪表盘。**

---

## 2. 方法总览

Pandora-RAG 分三层：

| 层 | 作用 | 当前实现 | 论文定位 |
| --- | --- | --- | --- |
| Stage 1 | 构造完整轨迹与 Oracle 上界 | 强制跑到 `K=5`，缓存每步答案、F1/EM、hidden states、浅层特征；用后向 DP 生成 oracle stop labels | Problem structure / oracle upper bound |
| Stage 2 | 学习可部署停止器 | `ProbeMLP_v2` 融合 hidden state 与浅层特征，预测 Continue/Stop；dev 上选 per-step thresholds | Main deployable method |
| Stage 3 | 在线风险监控 | 质量模型估计 \(P(F_1\ge\gamma)\)，E-process 累积错误证据；对比 CP-quantile | Safety monitor / drift detector |

Stop-RAG 是额外对齐的外部 baseline，不属于方法本体，但现在已可进入论文主对比。

---

## 3. Experimental Backbone

### 3.1 数据与切分

三组多跳 QA benchmark：

| 数据集 | Train | Calib | Dev | Test |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 4000 | 1000 | 1000 | 1000 |
| MuSiQue | 4000 | 1000 | 1000 | 417 |
| 2WikiMultiHopQA | 4000 | 1000 | 1000 | 1000 |

`calib` split 专门服务 Stage 3 的质量模型、quality bar 与 CP 阈值校准；`dev` split 用于 Stage 2 阈值选择；`test` 只用于最终评估。

### 3.2 RAG 轨迹

每条样本最多执行 \(K=5\) 步迭代检索。每一步记录：

- 当前检索文档与检索分数
- 当前答案与 `F1 / EM`
- Llama-3.1-8B-Instruct 的 hidden state，主口径使用 last-token hidden state，维度 4096
- 浅层特征，包括 retrieval score、answer logprob、NLI entail/contra、上下文重叠、历史变化特征、token/latency 近似成本等

当前主 checkpoint 的浅层输入为 20 维基础特征；后来补跑的 `p2_full31` 完整 31 维扩展没有带来收益，因此不进入主结果。

---

## 4. Stage 1: Oracle 与结构化上界

### 4.1 Fixed-order optimal stopping

对第 \(i\) 条轨迹，记第 \(k\) 步答案质量为 \(Q_k^i\)，每步成本为 \(c_k^i\)。轨迹级 DP Oracle 用后向归纳计算：

$$
V_K^i=Q_K^i,\qquad
V_k^i=\max\{Q_k^i,V_{k+1}^i-c_{k+1}^i\}.
$$

Oracle 在最小满足

$$
Q_k^i\ge V_{k+1}^i-c_{k+1}^i
$$

的 \(k\) 停止，否则停在 \(K\)。同时写出：

- `expected_continue_val = V_{k+1} - c_{k+1}`
- `margin = expected_continue_val - Q_k`
- `action_label = 1[margin > 0]`

这些标签是 Stage 2 Probe 的监督来源。

### 4.2 Pandora/Weitzman 的角色

Weitzman reservation value 的积分公式提供“继续探测是否值得”的动机，但原始 Pandora's Box 假设独立盒子；多跳 RAG 的后续检索分布显然依赖当前状态。因此正文中建议：

- 用 Pandora 解释 threshold-style stopping intuition
- 用 Bellman/DP 作为真正可证明的 fixed-order 主线
- 把 `Global-Weitzman` 放在 oracle/semi-oracle baselines 中，而不是部署方法

### 4.3 Stage 1 结果定位

Stage 1 的主要产物不是单点 SOTA，而是 Pareto 上界和 oracle labels。最终 Stage 2 表中，Oracle 上界为：

| 数据集 | Oracle F1 | Avg Steps |
| --- | ---: | ---: |
| HotpotQA | 0.7810 | 1.58 |
| MuSiQue | 0.4966 | 2.12 |
| 2Wiki | 0.6954 | 1.59 |

---

## 5. Stage 2: Neural Stopping Probe

### 5.1 模型

主模型是 `ProbeMLP_v2`：

1. Hidden branch：对 4096 维 last-token hidden state 做 LayerNorm，再线性压缩到 `compress_dim`。
2. Shallow branch：对浅层特征做 MLP 编码。
3. Fusion head：拼接两路表示，输出 Continue logit。

最终 checkpoint 的有效结构为：

| 数据集 | Hidden 压缩 | 浅层输入 | 残差 hidden branch | 监督目标 |
| --- | ---: | ---: | --- | --- |
| HotpotQA | 256 | 20 | 否 | binary Continue |
| MuSiQue | 256 | 20 | 否 | binary Continue |
| 2Wiki | 64 | 20 | 是 | binary Continue |

训练使用 focal BCE、label smoothing、dropout、weight decay 和 early stopping。二分类头在三数据集上均优于 F1 regression，因此主文应称为 `learned stopping signal` 或 `continuation-decision estimator`，不要说当前实现精确回归 reservation value。

### 5.2 阈值选择

Probe 输出 \(p_\theta(\text{Continue}\mid s_k)\)。部署规则为：

$$
\textsc{Continue}\quad\text{if}\quad p_\theta(k)\ge\eta_k,
$$

否则停止。阈值在 dev split 上选择：

1. 候选点必须满足平均步数不超过 `Global-Weitzman(dev) × 1.05`，若无可行点则回退。
2. 在可行点上最大化 \(F1-\lambda\cdot\text{normalized cost}\)。
3. 进一步做保守的 per-step threshold refinement，只有 utility 或 Pareto 改善时采纳。

最终 per-step thresholds：

| 数据集 | Per-step thresholds |
| --- | --- |
| HotpotQA | `[0.73, 0.69, 0.81, 0.73, 0.73]` |
| MuSiQue | `[0.61, 0.63, 0.63, 0.69, 0.63]` |
| 2Wiki | `[0.61, 0.79, 0.73, 0.77, 0.67]` |

### 5.3 Stage 2 主结果

Stage 2 的主结论应写成：

> Probe recovers most of the oracle stopping benefit at a much lower retrieval budget, but the exact comparison to the best fixed depth is dataset-specific.

最终 `pdopt_best` 测试结果：

| 策略 | HotpotQA F1 / steps | MuSiQue F1 / steps | 2Wiki F1 / steps |
| --- | --- | --- | --- |
| Fixed-K=1 | 0.4340 / 1.00 | 0.0797 / 1.00 | 0.2846 / 1.00 |
| Fixed-K=2 | 0.6773 / 2.00 | 0.1830 / 2.00 | 0.5908 / 2.00 |
| Fixed-K=3 | 0.6775 / 3.00 | 0.3210 / 3.00 | 0.5583 / 3.00 |
| Fixed-K=4 | 0.6747 / 4.00 | 0.3816 / 4.00 | 0.5382 / 4.00 |
| Fixed-K=5 | 0.6668 / 4.99 | 0.4022 / 5.00 | 0.5288 / 5.00 |
| Global-Weitzman | 0.7092 / 1.64 | 0.4229 / 3.29 | 0.6449 / 1.77 |
| Oracle | 0.7810 / 1.58 | 0.4966 / 2.12 | 0.6954 / 1.59 |
| Probe | 0.6544 / 1.73 | 0.3969 / 3.31 | 0.5941 / 1.82 |

相对 Oracle：

| 数据集 | Probe / Oracle F1 | Probe vs Best Fixed-K | Probe vs Fixed-K=5 |
| --- | ---: | ---: | ---: |
| HotpotQA | 83.8% | -0.0231 | -0.0124 |
| MuSiQue | 79.9% | -0.0053 | -0.0053 |
| 2Wiki | 85.4% | +0.0033 | +0.0653 |

主文不要说“Probe 三数据集统一超过最佳 Fixed-K”；更准确的是“Probe 是接近 Oracle 的低成本 Pareto 点”。

---

## 6. Stage 3: E-value Risk Monitor

### 6.1 质量模型

Stage 3 在 calib split 上训练轻量质量模型：

$$
\hat p_k = P(F_1(s_k)\ge\gamma\mid \phi_k, p_\theta(k)).
$$

当前实现使用 StandardScaler + Logistic Regression，`class_weight=balanced`。主实验设置：

- \(\gamma=0.5\)
- \(\alpha\in\{0.1,0.2\}\)，主表用 \(\alpha=0.1\)
- `betting_strategy = predictive`
- `outcome_aware = true`
- `calib_method = quantile`

### 6.2 E-process

对第 \(n\) 个样本停止后的错误事件

$$
e_n=\mathbf{1}\{F_1(s_{\tau_n})<\gamma\},
$$

E-value 更新为

$$
E_n=E_{n-1}\left(1-\lambda_n+\lambda_n\frac{e_n}{\alpha}\right),
$$

其中 predictive betting 取

$$
\lambda_n=\mathrm{clip}(1-\hat p_n,\epsilon,1-\epsilon).
$$

在 \(H_0:\mathbb{E}[e_n\mid\mathcal{G}_{n-1}]\le\alpha\) 下，\((E_n)\) 是非负超鞅，因此

$$
P_{H_0}\left(\sup_n E_n\ge 1/\delta\right)\le\delta.
$$

论文中必须强调：这是检测保证，不是错误率硬控制。

### 6.3 E-value 门控

当 Probe 建议停止时：

1. 计算质量模型 \(\hat p\)。
2. 若 \(\hat p <\) quality bar，则强制继续。
3. 若当前 wealth 已接近 \(1/\alpha\)，下一次停止前更保守。
4. 最终停止后观测 F1，更新 E-wealth。

因此 `Probe+E-value` 的定位是“低开销监控 + 轻量门控”，不是一个替代 Probe 的新停止优化器。

### 6.4 与 CP 的区别

CP-quantile baseline 在 calib 上对 Probe 自然停止处的 \(\hat p\) 取 \((1-\alpha)\) 分位，得到固定 `min_phat`。测试时，Probe 想停还必须满足 \(\hat p\ge\) `min_phat`。它没有跨样本 wealth，也不会根据测试流中的错误累积产生告警。

在当前实验中，CP-quantile 往往变成高成本静态保守门控；E-value 则保持低增步，并额外提供在线风险轨迹。

---

## 7. Stage 3 主结果

主实验：\(\gamma=0.5,\alpha=0.1\)，predictive betting。

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

推荐写法：

- E-value 相比 Probe 仅增加少量步数：HotpotQA +6.4%、MuSiQue +0.5%、2Wiki +4.6%。
- HotpotQA 和 2Wiki 上 F1/error 同时改善；MuSiQue 基本持平，体现高错误率 stress case。
- CP 在 HotpotQA 上能靠大幅增步换取略高 F1，但在 MuSiQue 和 2Wiki 上又贵又差。

E-wealth 主现象：

| 数据集 | Final E-wealth at α=0.1 | Cap | 解读 |
| --- | ---: | ---: | --- |
| HotpotQA | 0.014 | 10 | 主实验下 wealth 低，门控更像提质而非报警 |
| MuSiQue | 4.193 | 10 | 风险证据持续积累但未封顶 |
| 2Wiki | 10.000 | 10 | 明确触及 cap，拒绝低错误率原假设 |

---

## 8. Distribution Shift 与 Selective Prediction

Stage 3 已完成 sudden、gradual、periodic 三类 drift：

- sudden：测试流前半正常、后半更难
- gradual：难度逐步升高
- periodic：简单/困难样本周期性交替

稳妥结论：

1. E-value 对明显恶化的测试流可感知，尤其 MuSiQue 与 2Wiki 响应强。
2. HotpotQA 的响应依赖 \(\alpha\) 和 shift 形态，不能写成“所有设置都强触发”。
3. CP 固定阈值无法提供同样的跨样本风险轨迹。

Selective Prediction 已实现：若处理样本前 wealth 已达 \(1/\alpha\)，则拒答该样本。主实验 \(\gamma=0.5,\alpha=0.1\)：

| 数据集 | Coverage | Selective Accuracy |
| --- | ---: | ---: |
| HotpotQA | 0.931 | 0.705 |
| MuSiQue | 0.444 | 0.373 |
| 2Wiki | 0.706 | 0.581 |

这可以作为检测后干预的最小示例，但不是主性能优化目标。

---

## 9. Stop-RAG 对齐结果

Stop-RAG 是最直接的外部 stopping baseline。本项目已完成同 split、同 sample id、真实在线早停口径的对齐评估。

| 数据集 | 方法 | N | F1 | EM | Avg Steps |
| --- | --- | ---: | ---: | ---: | ---: |
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

推荐主文写法：

> Under aligned online evaluation, Stop-RAG tends to exhaust the retrieval budget, while Pandora achieves higher macro F1 with roughly half the retrieval steps.

边界说明：

- 当前 Stop-RAG 是 single-point threshold，不是完整 Pareto sweep。
- 若主文强调 matched-budget frontier，仍需补 Stop-RAG 多 threshold 在线 sweep；否则主表应明确这是 aligned online single-point comparison。

---

## 10. 建议的 Methods Section 结构

### 10.1 Problem Formulation

写 fixed-order sequential information acquisition：

- 状态 \(s_k=(q,d_1,\ldots,d_k)\)
- 停止时刻 \(\tau\)
- 质量 \(Q(s_\tau)\)
- 成本 \(\sum c_k\)
- 主目标 \(Q(s_\tau)-\sum c_k\)

### 10.2 Oracle Stopping

写 Bellman 递推和轨迹级 DP labels：

- \(V_k=\max(Q_k,V_{k+1}-c_{k+1})\)
- margin 与 action label
- Oracle 是不可部署上界和监督来源

### 10.3 Neural Stopping Probe

写：

- hidden state + shallow features
- binary Continue classifier
- focal BCE + label smoothing
- dev Pareto threshold selection
- per-step threshold refinement

避免写：

- exact reservation value regression
- provably optimal deployed policy

### 10.4 E-value Monitor

写：

- 质量模型 \(\hat p=P(F_1\ge\gamma)\)
- predictive betting
- \(E_n=\prod(1-\lambda+\lambda e/\alpha)\)
- Ville guarantee
- quality bar / wealth-aware gate 是部署实现

重点句：

> The E-value layer is a monitor rather than an optimizer: it provides anytime-valid evidence against a low-error-rate null while adding only a small retrieval overhead.

### 10.5 Baselines

主文基线建议分层：

- Deployable: Fixed-K, Probe, Probe+E-value, Probe+CP, Stop-RAG
- Semi-oracle: Global-Weitzman
- Oracle upper bound: DP Oracle
- Appendix: Deployable-GW, XGBoost, shallow-only, feature/target ablations, p2_full31 negative result

---

## 11. 可直接放进论文的贡献列表

1. **Formulation.** We formulate iterative multi-hop retrieval stopping as a fixed-order finite-horizon optimal stopping problem and derive an oracle Bellman recursion that provides both an upper bound and supervision for stopping decisions.

2. **Deployable stopping signal.** We train a lightweight neural stopping probe from LLM hidden states and low-cost observable features. Across HotpotQA, MuSiQue, and 2Wiki, the probe recovers about 80% to 85% of oracle F1 while using far fewer retrieval steps than fixed full-depth retrieval.

3. **Anytime-valid risk monitoring.** We attach an E-value monitor to adaptive stopping. Unlike static CP gates, the monitor accumulates online risk evidence under data-adaptive stopping and distribution shift, with modest additional retrieval cost.

4. **Aligned external comparison.** We reproduce Stop-RAG under the same split and online stopping protocol; Pandora achieves higher macro F1 while reducing average retrieval steps by roughly half.

如果正文贡献只能放三条，把第 4 条合并到实验贡献里，不要单独作为方法贡献。

---

## 12. 红线与推荐表述

### 可以写

- `adaptive stopping for iterative multi-hop RAG`
- `fixed-order optimal stopping`
- `Bellman oracle`
- `learned stopping signal`
- `continuation-decision estimator`
- `anytime-valid risk monitoring`
- `online risk evidence`
- `distribution-shift sensitivity`

### 不要写

- `provably optimal deployed system`
- `we learn the exact reservation value`
- `E-value controls the error rate below alpha`
- `CP is invalid in all adaptive settings`
- `Probe uniformly beats the best fixed depth`

### 最稳 abstract 结果句

> On three multi-hop QA benchmarks, Pandora's neural probe recovers 79.9% to 85.4% of the DP oracle F1. Adding the E-value monitor changes the average retrieval depth by only 0.5% to 6.4% relative to the probe while preserving or improving F1 on two of three datasets, and it outperforms an aligned online Stop-RAG baseline in both macro F1 and retrieval cost.

---

## 13. 最终叙事

整篇论文最好按这个逻辑推进：

1. **Fixed depth is wasteful and sometimes harmful.**
2. **Oracle stopping has a clean Bellman threshold structure.**
3. **The oracle depends on unobservable F1, so we learn a stopping signal.**
4. **The learned signal is imperfect, so deployment needs online risk evidence.**
5. **E-value gives that evidence under adaptive stopping, while CP acts as a costly static gate in this setting.**
6. **Aligned Stop-RAG results show Pandora is not merely a polished internal ablation; it wins against a direct dynamic stopping baseline under the same split and online protocol.**

这条主线足够强，也足够诚实。它不会把项目写成“Stage 1/2/3 进度汇报”，而是把三阶段实验组织成一篇方法论文的自然闭环。
