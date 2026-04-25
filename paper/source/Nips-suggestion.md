# Pandora-RAG 中稿差距审计与改稿路线

> 日期：2026-04-20  
> 对象：`Pandora-RAG.tex` 当前快照  
> 目标：回答“距离中稿还差什么”，并把 NeurIPS 投稿建议从项目路线重构为可执行的论文改稿清单。

## 0. 结论先行

`Pandora-RAG.tex` 已经不是“想法草稿”，而是一篇有完整实验链的论文初稿。它已经具备：

- 明确问题：iterative multi-hop RAG 什么时候停止检索。
- 方法闭环：Bellman oracle -> neural stopping probe -> E-value risk monitor。
- 主实验：Fixed-K、Global-Weitzman、Oracle、Probe、Probe+E-value、Probe+CP。
- 外部对齐：Stop-RAG aligned online threshold sweep。
- 关键边界：承认 Probe 不统一打赢 best Fixed-K，承认 E-value 是风险证据而不是错误率控制器。

但它还没有到“中稿”。最短判断是：

> 当前主稿约等于 **65% 到 70% 的中稿**。核心实验足够，主要缺口不是再发明方法，而是补齐论文工程、引用系统、Related Work、图表叙事和 appendix 证据链。

如果把“中稿”定义为可以发给外部同学或导师严肃审读的版本，当前还差五类 P0 工作：

1. **可编译投稿包**：仓库当前没有 `neurips_2026.sty`、`.bib`、`checklist.tex`；主稿末尾也保留三条 TODO。
2. **Related Work 与引用系统**：全文当前没有 `\cite{...}` 和 bibliography，这是最大硬缺口。
3. **方法总览图和风险监控图**：已有 Pareto 图和 Stop-RAG 图，但缺一张一眼讲清方法链的 Figure 1，以及 E-wealth/no-shift vs shift 证据图。
4. **主文/附录分工**：现在 Results 很密，Ablations 只有段落式摘要；中稿需要明确哪些结果进正文，哪些进 appendix。
5. **数值口径统一**：Stage 2 表和 Stage 3 表的 Probe 数值来自不同评估入口，合理但需要一句说明，避免审稿人误以为同一行结果不一致。

推荐路线：

> 不要继续大规模刷实验。先把 `Pandora-RAG.tex` 做成可编译、可引用、可审稿的中稿。新实验只补“缺图表导出/appendix 归档”，不再改变主 claim。

---

## 1. 当前主稿体检

### 1.1 结构状态

当前 `Pandora-RAG.tex` 的结构：

| 模块 | 当前状态 | 中稿判断 |
| --- | --- | --- |
| Title / Abstract | 已有，且主张克制 | 基本可用 |
| Introduction | 问题、gap、贡献都在 | 需要加引用和更强 opening |
| Method | Bellman oracle、Probe、E-value 都写清楚 | 基本可用，需加方法图 |
| Experimental setup | 数据集、split、backbone、metrics 已有 | 需补实现细节和复现入口到 appendix |
| Results | 主表完整，Stop-RAG 对齐已进入主文 | 内容强，但密度高 |
| Ablations | 已概括多个负结果 | 中稿需要表格化和 appendix 指向 |
| Limitations | 写得诚实 | 可保留 |
| Conclusion | 稳妥 | 可保留 |
| Related Work | 缺失 | P0 |
| References | 缺失 | P0 |
| Checklist | 缺失 | P0 |
| Appendix | 缺失 | P0/P1 |

### 1.2 当前最强部分

当前稿件最强的地方是“方法链完整而且 claim 没有吹过头”：

- **Bellman oracle** 给出固定检索序列上的结构上界和监督标签。
- **Probe** 是真正部署的 stopping signal，而不是使用未来 F1 的 oracle。
- **Lite Probe + cost accounting** 回答了 deployability 和 feature overhead。
- **E-value monitor** 提供 anytime-valid online risk evidence，而不是声称控制经验错误率。
- **Stop-RAG threshold sweep** 把外部 baseline 从单点比较升级为 Pareto/frontier 比较。

这条主线应该保留：

> Pandora is the structure, Probe is the deployable stopper, E-value is the monitor.

中文写作口径：

> Pandora-RAG 把多跳 RAG 的检索深度选择从固定超参改成实例自适应停止，并把这个自适应停止流变成可在线监控的部署对象。

### 1.3 当前最大短板

最大短板不是实验，而是“论文像不像一篇 NeurIPS 中稿”：

- 没有 Related Work，导致 novelty 无法被定位。
- 没有引用，所有方法背景和 baseline 背景都悬空。
- 没有 bibliography，不能编译成正式论文形态。
- 没有 method overview figure，读者进入 Method 前缺少全局地图。
- 没有 appendix/checklist，许多补强实验虽然已完成，但没有落到论文结构里。

当前主稿末尾的 TODO 很准确：

- Add Related Work after citation audit.
- Add bibliography once citation keys are settled.
- Add NeurIPS checklist when checklist.tex is available.

中稿的第一目标就是关闭这三条 TODO。

---

## 2. 距离中稿还差什么

### P0. 可编译投稿骨架

当前仓库检查结果：

- `Pandora-RAG.tex` 使用 `\usepackage{neurips_2026}`。
- 仓库根目录没有发现 `neurips_2026.sty`。
- 没有 `.bib` 文件。
- 没有 `checklist.tex`。
- 图像引用的四个 PNG 已存在：
  - `results/stage3_final_pareto_all.png`
  - `results/stop_rag_pareto_hotpotqa.png`
  - `results/stop_rag_pareto_musique.png`
  - `results/stop_rag_pareto_2wiki.png`

中稿验收标准：

- 能执行一次 `pdflatex/bibtex/pdflatex/pdflatex` 或 `latexmk`。
- 没有 undefined references。
- 没有 missing figure。
- bibliography 能生成。
- NeurIPS checklist 能包含或明确暂缓。

建议动作：

1. 添加 `neurips_2026.sty` 或改成当前可获得的 NeurIPS style。
2. 新建 `references.bib`。
3. 在文末加入 `\bibliographystyle{plainnat}` 和 `\bibliography{references}`，或按 NeurIPS 模板使用相应样式。
4. 加入 `checklist.tex`，即使先填草稿版。
5. 编译并把 warning 归档成一个短清单。

### P0. Related Work 与 citation audit

这是从初稿到中稿最大的内容缺口。建议新增一个正文 section：

```tex
\section{Related Work}
```

放在 Introduction 后或 Method 前。四个小段即可，不要写成百科综述：

| 小节 | 要解决的问题 | 必引方向 |
| --- | --- | --- |
| Adaptive retrieval and stopping | 说明现有 RAG/iterative retrieval 如何决定是否继续 | IRCoT、ITER-RETGEN、Self-RAG、Adaptive-RAG、FLARE、DRAGIN、Stop-RAG、Probing-RAG |
| Optimal stopping and Pandora's box | 说明本文的 stopping 结构来自哪里，又为什么不是直接套独立 Pandora | Weitzman Pandora's Box、optimal stopping、metareasoning、contextual/correlated Pandora |
| Risk control and monitoring | 说明 E-value 与 CP/CRC/LLM risk control 的关系 | conformal prediction、conformal risk control、testing by betting、E-values、selective prediction |
| Cost-aware LLM systems | 说明本文和 routing/cascades 的相邻关系 | FrugalGPT、LLM cascades、RouteLLM、budgeted inference |

Related Work 的目标不是堆 citation，而是给审稿人三句话：

1. 本文不是又一个 fixed-depth RAG pipeline。
2. 本文不是声称解决独立 Pandora's Box，而是 fixed-order Bellman stopping。
3. 本文的风险层不是传统 CP 替代品，而是 adaptive stopped stream 上的 online evidence process。

### P0. Figure 1 方法总览图

当前主文第一张图是 `F1 vs average steps` Pareto 图。它很重要，但不适合作为 Figure 1。中稿需要一张方法图，让读者在 Method 前知道系统怎么流动。

建议 Figure 1 内容：

```text
Full rollout trajectories
  -> Bellman DP oracle
  -> step labels / continuation margins
  -> neural probe with hidden states + observable features
  -> dev threshold selection
  -> online stopped stream
  -> E-value monitor / alarm / optional abstention
```

图中必须标注两条边界：

- Oracle sees future realized F1 and is not deployable.
- Probe/E-value use only observable state before the outcome, except that the E-value update receives revealed evaluation outcome for monitoring.

这样可以提前化解两个审稿误解：

- “你是不是部署时用了真实 F1？”
- “E-value 是不是直接修正答案质量？”

### P0. 主结果图表重排

当前正文图表已经很多：

- Table 1: dataset splits
- Table 2: Stage 2 adaptive stopping
- Table 3: Lite cost audit
- Figure 1: final Pareto
- Table 4: Stage 3 risk monitoring
- Table 5: selective prediction
- Table 6: Stop-RAG sweep
- Figure 2: Stop-RAG frontiers

这对 NeurIPS 正文页数会偏挤。中稿先不一定要压到最终页数，但要明确主文和 appendix 的分工。

推荐主文保留：

| 编号 | 内容 | 理由 |
| --- | --- | --- |
| Figure 1 | Method overview | 一图讲清贡献链 |
| Table 1 | Stage 2 main: Fixed-K / Global-Weitzman / Oracle / Probe | 证明 adaptive stopping 的 Pareto 价值 |
| Figure 2 | F1 vs avg steps Pareto | 防止单点 F1 误读 |
| Table 2 | Probe / Probe+E-value / Probe+CP | 证明 E-value 低开销，CP 高成本且不稳 |
| Figure 3 | E-wealth no-shift + shift trace | 展示 online monitoring 的核心价值 |
| Table 3 | Stop-RAG threshold-sweep best frontier point | 外部 baseline |

建议挪到 appendix 或压缩：

- dataset split table，可放 setup 段落或 appendix。
- Lite Probe cost table，可以正文保留三行小表，也可以 appendix 主表，正文一句总结。
- selective prediction table，建议 appendix，正文只提“可作为 intervention”。
- 大段 ablation paragraphs，建议表格化后放 appendix。

### P0. 数值口径统一说明

当前主稿里 Stage 2 的 Probe 表和 Stage 3 的 Probe 表数值不同：

- Stage 2 Probe：HotpotQA `0.6544 / 1.73`，MuSiQue `0.3969 / 3.31`，2Wiki `0.5941 / 1.82`。
- Stage 3 Probe：HotpotQA `0.6569 / 1.70`，MuSiQue `0.4152 / 3.39`，2Wiki `0.5639 / 1.82`。

这不一定是错误，可能来自 Stage 3 的 calibrated/adapter evaluation path、stopped logs 或 monitor evaluation入口。但中稿必须加一句：

> Stage-2 and Stage-3 tables are produced by different evaluation entry points: Stage 2 reports the stopping-probe operating point selected on development data, while Stage 3 re-evaluates the monitor-compatible stopped stream used for E-value and CP comparison. We therefore compare methods within each table rather than treating the two Probe rows as duplicate measurements.

如果实际上它们应当完全一致，则这是 P0 bug，需要回查导出脚本。中稿前必须二选一：解释清楚，或修正对齐。

### P1. Appendix 证据链

已有很多补强实验，但还没有落成 appendix。中稿需要一个 appendix skeleton，哪怕内容先是简表。

建议 appendix 结构：

1. **Implementation Details**
   - RAG backbone、K=5、generation setting、hidden extraction、feature list。
2. **Dataset and Split Details**
   - train/calib/dev/test，MuSiQue test 为 417 的原因。
3. **Stage-2 Operating Point Selection**
   - dev threshold selection、cost-aware utility、GW budget cap。
4. **Lite Probe and Cost Accounting**
   - Full vs Lite feature list，normalized weights。
5. **Ablations**
   - F1 regression、GRU/seq-history、mean pooling、compress dim、p2_full31。
6. **Shared Configuration and Transfer**
   - shared default 负结果，per-dataset tuning limitation。
7. **E-value Robustness**
   - gamma/alpha grid，predictive vs fixed betting，cap timing。
8. **Statistical Uncertainty**
   - paired bootstrap CI and p-values。
9. **Stop-RAG Details**
   - thresholds、checkpoint、matched-budget、stopping distribution。
10. **Cross-Retriever Sanity**
   - HotpotQA bm25 vs contriever_bge oracle headroom。
11. **NeurIPS Checklist**

### P1. E-wealth / shift 图

现在主文写了 shift 现象，但没有图。E-value 的 NeurIPS 味道主要来自“在线过程”，不是一张 aggregate table。

建议主文 Figure 3：

- 左：no-shift E-wealth trace，三数据集或代表数据集。
- 中：sudden shift wealth trace。
- 右：cumulative error 或 cap timing。

如果空间紧张，正文放一个代表数据集，appendix 放全部 `sudden/gradual/periodic`。

正文推荐句：

> The E-process is useful precisely because it is a trajectory, not a static accept/reject threshold: under degraded streams, wealth accumulates online risk evidence and crosses the alarm boundary.

### P1. 统计显著性放进主文或 appendix

已有 `scripts/stage3_significance.py` 和 `docs/stage3_significance_report.md`。中稿需要至少引用一次这些结果。

主文可写短句：

> Paired bootstrap intervals show that Probe+E-value improves Probe on 2Wiki by 0.0089 F1 with 95% CI [0.0017, 0.0165], is borderline on HotpotQA, and is statistically indistinguishable on MuSiQue.

这句话的作用不是吹显著性，而是主动管住 MuSiQue 风险。

### P1. Shared default / config transfer 作为 limitation

已有 shared default 审计，结论偏负：

- strict shared default macro F1/steps：`0.3551 / 3.056`
- per-dataset operating points macro F1/steps：`0.5485 / 2.286`

这不适合当正结果，但很适合当诚实 limitation：

> The strongest operating points are selected per dataset on development data. A strict shared-configuration audit is substantially weaker, suggesting that current stopping features and thresholds remain dataset-sensitive.

这句话可以放 Limitations，也可以 appendix。

### P2. Cross-backbone / retriever sanity

已有 HotpotQA small-scale diagnostic：

| Retriever | Oracle F1 | Best Fixed-K F1 | Oracle gain |
| --- | ---: | ---: | ---: |
| bm25 | 0.7390 | 0.6133 | +0.1257 |
| contriever_bge | 0.7642 | 0.6760 | +0.0882 |

建议只放 appendix，不要扩成主 claim。它证明的是：

> stopping headroom exists beyond the default retriever.

不要写成：

> Pandora is backbone-independent.

---

## 3. 当前主 claim 应该怎样写

### 3.1 推荐中心句

> Iterative multi-hop RAG stopping is a fixed-order sequential information acquisition problem. Pandora-RAG exposes the stopping structure with a Bellman oracle, learns a deployable continuation signal from observable states, and attaches an E-value monitor that provides anytime-valid online risk evidence for the adaptively stopped stream.

中文版本：

> 多跳 RAG 的关键不是固定检索几步，而是在已观察到的证据状态下判断下一步检索是否值得。Pandora-RAG 用 Bellman oracle 暴露这个停止结构，用 Probe 学可部署的停止信号，再用 E-value 给自适应停止后的输出流提供在线风险证据。

### 3.2 三条贡献

1. **Fixed-order stopping formulation**
   - 将 iterative multi-hop RAG stopping 表述为 fixed-order finite-horizon optimal stopping。
   - 用 Bellman oracle 作为 upper bound 和 supervision。

2. **Deployable stopping signal**
   - 从 LLM hidden states 和 observable retrieval/state features 训练 continuation probe。
   - Probe 恢复 `79.9%` 到 `85.4%` oracle F1，使用远少于 full-depth 的检索步数。
   - Lite Probe 和 cost accounting 说明部署开销不会让结论失效。

3. **Anytime-valid monitoring**
   - 将 E-value process 接到 adaptively stopped stream。
   - 它提供 online risk evidence 和 shift sensitivity。
   - 它不是 empirical error-rate controller，不保证把错误率压到 alpha 以下。

Stop-RAG 对齐建议作为实验贡献，而不是方法贡献：

> We further reproduce Stop-RAG under an aligned online threshold sweep and show that Pandora-RAG attains higher F1 and EM with fewer retrieval steps on all three benchmarks.

### 3.3 不要写的 claim

不要写：

- `provably optimal deployed RAG system`
- `E-value controls the error rate below alpha`
- `Probe uniformly outperforms fixed-depth retrieval`
- `CP is invalid in adaptive settings`
- `state-of-the-art multi-hop QA`
- `learned exact reservation values`
- `feature pruning greatly reduces total inference cost`

可以写：

- `fixed-order optimal stopping`
- `Bellman oracle upper bound`
- `oracle-labeled stopping supervision`
- `deployable continuation-decision estimator`
- `cost-quality Pareto tradeoff`
- `Lite Probe with normalized cost accounting`
- `anytime-valid online risk evidence`
- `aligned online Stop-RAG threshold sweep`

---

## 4. 逐段改稿建议

### Abstract

当前摘要已经稳妥。中稿前建议压缩两处：

- “probe recovers 79.9--85.4% of oracle F1”保留。
- “E-value changes average retrieval depth by 0.5--6.4%”保留。
- Stop-RAG 句子保留，但如果摘要超字数，可移到 Introduction 贡献段。

摘要里不要额外加入“controls risk”之类强措辞。

### Introduction

当前 Introduction 逻辑顺，但中稿需要更强的 problem framing 和 citation hooks。

推荐五段结构：

1. Iterative retrieval 的 deployment dilemma。
2. Fixed depth 为什么不够。
3. Stopping 为什么难：真实 F1 不可观测，继续价值依赖状态。
4. Adaptive stopping 后为什么需要 online monitoring。
5. 本文贡献和主要实证结果。

建议新增一句：

> The stopped stream is itself a deployment object: once stopping depends on model states, a monitoring layer should reason about the sequence of stopped outcomes rather than a fixed-depth offline table alone.

### Method

Method 基本可用。建议补三点：

- 在 Section 2 开头加一段“overview and notation”，对应 Figure 1。
- 在 Bellman oracle 段明确“fixed-order, state-dependent, not classical independent-box Pandora”。
- 在 E-value 段强调 outcome is revealed only for monitoring/evaluation, not for choosing the current answer。

### Experimental Setup

当前 setup 可读，但中稿建议加一小段：

- compute/hardware 可放 appendix。
- generation deterministic setting 和 retriever/generator 版本可放 appendix。
- Stop-RAG checkpoint 和 thresholds 放 appendix。

### Results

建议重排为四个问题：

1. **Does adaptive stopping give a useful Pareto point?**
   - Fixed-K / Global-Weitzman / Oracle / Probe。
2. **Is the stopping signal deployable?**
   - Lite Probe + cost accounting，正文可短。
3. **Does E-value monitoring add low-overhead risk evidence?**
   - Probe / Probe+E-value / Probe+CP。
4. **How does Pandora compare to Stop-RAG?**
   - aligned online threshold sweep。

当前 Results 的内容都在，但第 2 和第 3 个问题之间有点挤。中稿可把 selective prediction 移 appendix。

### Ablations and Robustness

当前是 paragraph list。中稿建议至少做一个 appendix table：

| Ablation | Outcome | Interpretation |
| --- | --- | --- |
| F1 regression | worse than binary stopping labels | continuation labels are more reliable |
| p2_full31 | no main-point gain | more shallow features did not solve bottleneck |
| GRU / seq-history | no stable gain | sequence model capacity not bottleneck |
| mean pooling / blend | no stable gain | last-token hidden state is adequate for current setup |
| shared config | much weaker | per-dataset tuning remains limitation |

正文保留 2 到 3 个最关键结论即可。

### Limitations

Limitations 现在很诚实，是优点。中稿建议保留，并补一句：

> The current manuscript uses per-dataset development-selected operating points; strict shared-configuration transfer remains substantially weaker.

### Conclusion

当前可用。最终成稿时可以更短。

---

## 5. 审稿风险矩阵

| 风险 | 严重度 | 当前状态 | 中稿应对 |
| --- | --- | --- | --- |
| 没有引用和 Related Work | 高 | 未补 | P0 立即补 |
| 不能编译 | 高 | style/bib/checklist 缺失 | P0 立即补 |
| Probe 没统一超过 best Fixed-K | 高 | 已诚实处理 | 用 Pareto/budgeted framing |
| E-value 被误解成 error control | 高 | 主文已克制 | 保留 null 和 Ville inequality，避免硬控制措辞 |
| Stop-RAG 阈值没调好 | 中 | threshold sweep 已补 | 说明 fixed checkpoint + sweep，不声称重训 |
| per-dataset tuning | 中 | shared default 偏负 | 放 limitation/appendix |
| MuSiQue 样本小且高错误 | 中 | CI 已补 | 写成 stress case |
| 部署开销 | 中 | Lite + accounting 已补 | 不夸大 total cost 降幅 |
| backbone 依赖 | 中 | small retriever sanity 已补 | appendix 限定为 diagnostic |
| 绝对 QA 分数非 SOTA | 中 | 口径清楚 | 固定为 stopping efficiency under same backbone |

---

## 6. 中稿验收清单

### 必须完成

- [ ] 加入 `neurips_2026.sty` 或可用投稿 style。
- [ ] 新建 `references.bib`。
- [ ] 全文加入 citation keys。
- [ ] 新增 Related Work。
- [ ] 新增 bibliography。
- [ ] 新增或暂存 NeurIPS checklist。
- [ ] 添加 Figure 1 method overview。
- [ ] 添加 E-wealth/no-shift vs shift 图，或明确移 appendix。
- [ ] 解释 Stage 2 / Stage 3 Probe 数值口径差异。
- [ ] 至少一次完整编译。

### 强烈建议完成

- [ ] Appendix skeleton。
- [ ] Ablation table。
- [ ] Lite Probe cost accounting table 或 appendix 表。
- [ ] Shared config/transfer negative result 写入 limitation。
- [ ] Paired bootstrap CI 写入 appendix。
- [ ] Stop-RAG thresholds/checkpoint/stopping distribution 写入 appendix。

### 可选完成

- [ ] Cross-retriever sanity table。
- [ ] cap timing table。
- [ ] selective prediction table 移 appendix。
- [ ] 更强 generator 小样本验证，不建议作为中稿阻塞项。

---

## 7. 推荐改稿顺序

### Day 1: 论文骨架闭环

1. 加 NeurIPS style、bib、checklist。
2. 新建 Related Work 占位并插入 citations。
3. 跑一次编译，生成 warning 清单。

验收：

- PDF 能生成。
- 不再有“无 bibliography”的硬伤。

### Day 2: 图表与主线

1. 画 Figure 1 method overview。
2. 选择一张 E-wealth/shift 图进入正文。
3. 重排 Results，压缩 selective prediction 和部分 Lite 内容。

验收：

- 读者不看代码也能理解三层方法。
- Results 从“堆表”变成四个问题。

### Day 3: Appendix 与风险口径

1. 建 appendix skeleton。
2. 填 ablation table、significance、shared config、Stop-RAG details。
3. 检查所有 claims 是否符合写作红线。

验收：

- 审稿人可能问的“阈值、调参、显著性、外部 baseline、成本”都有落点。

### Day 4: 语言打磨

1. 压缩重复表述。
2. 统一 notation。
3. 检查 Figure/Table caption 是否自洽。
4. 重新编译。

验收：

- 可以发给外部读者审稿。

---

## 8. 当前投稿判断

当前项目的实验资产已经足够支撑一篇 NeurIPS-style 方法论文，但 `Pandora-RAG.tex` 还处在“强初稿”而非“中稿”。

最合理的投稿定位：

> A cost-aware adaptive stopping framework for iterative multi-hop RAG, with a Bellman oracle for structure and supervision, a deployable neural stopping probe, and an anytime-valid E-value monitor for the adaptively stopped stream.

最危险的投稿定位：

> A new SOTA multi-hop QA system with provably optimal stopping and guaranteed low error.

中稿前不建议继续追求大规模新实验。真正高杠杆的工作是：

1. 让论文能编译。
2. 让论文有引用和 Related Work。
3. 让方法链一图讲清。
4. 让 appendix 承接已有补强实验。
5. 让所有 claim 都落在 Pareto、monitoring、fixed-order stopping 的安全边界内。

一句话：

> 离中稿不远，但差的是论文形态，不是实验主干。把引用、Related Work、方法图、E-wealth 图、appendix 和编译链补齐后，它就能进入真正的中稿审读阶段。
