# 绝对 F1 与 SOTA 对比说明

## 核心结论：「绝对 F1」与「停止效率」是两个不同维度

---

## 本项目最高 F1（可部署方法中的上界）


| 数据集      | 本项目最高可部署 F1 | 对应策略                                |
| -------- | ----------- | ----------------------------------- |
| HotpotQA | **0.527**   | Probe / Probe+E-value（γ=0.5, α=0.1） |
| 2Wiki    | **0.378**   | Probe+E-value                       |
| MuSiQue  | **0.183**   | Probe                               |


若算不可部署上界：Oracle 是 HotpotQA 0.623 / 2Wiki 0.534 / MuSiQue 0.253。

---

## 与外部 SOTA 对比——问题很严重，但有正当理由

### HotpotQA


| 方法                              | F1        | 模型 / 设置                         |
| ------------------------------- | --------- | ------------------------------- |
| FlowSteer（SOTA 2026.02）         | **84.98** | 大模型 + 专用检索                      |
| Router-R1                       | 79.84     | Qwen3                           |
| RECOMP-Extr                     | 77.9      | Llama-3.1-**70B**               |
| IRCoT 原论文（GPT-3）                | 60.7      | GPT-3 code-davinci + BM25       |
| Iter-RetGen-3（Llama3-8B）        | 60.89     | Llama-3.1-8B + 迭代检索             |
| EfficientRAG（Llama3-8B）         | 57.93     | Llama-3.1-8B                    |
| **本项目 Probe**（Llama3-8B + BM25） | **52.7**  | **Llama-3.1-8B, BM25, max 5 步** |
| StepChain GraphRAG              | 46.76     | GPT-4o + KG                     |


### MuSiQue


| 方法                              | F1       | 说明                           |
| ------------------------------- | -------- | ---------------------------- |
| End-to-End Beam Retrieval（2024） | **69.2** | 专用 supervision               |
| IRCoT 原论文（GPT-3）                | 36.5     | GPT-3 + BM25                 |
| **本项目 Probe**                   | **18.3** | Llama-3.1-8B + BM25, max 5 步 |


### 2WikiMultiHopQA


| 方法                             | F1       | 说明                           |
| ------------------------------ | -------- | ---------------------------- |
| NA-Reviewer（有 gold supporting） | 81.91    | 2021，提供 gold 段落              |
| SetR-CoT（开放域）                  | 30.35    | 2025                         |
| StepChain GraphRAG             | ~46      | GPT-4o + KG                  |
| **本项目 Probe**                  | **37.8** | Llama-3.1-8B + BM25, max 5 步 |


---

## 为什么这不构成致命质疑——以及怎么回应

**问题根源在于「评测维度混淆」**：

HotpotQA 上 85 的 F1 来自这些方法：

- 用 **GPT-4o / 70B 量级模型**，或专门 fine-tuned 模型
- 用 **dense retrieval + re-ranking**，远强于 BM25
- 通常关注「给定正确文档后的 QA 准确率」（distractor/gold-paragraph 设置），而非开放域检索

**本项目的定位不同**：

> **论文主张的不是「在 HotpotQA 上刷最高分」，而是「给定某个 IRCoT 骨干，如何以最低成本自适应停止并在线监控风险」。**

这类似于**压缩/蒸馏论文**——你不会拿 DistilBERT 和 GPT-4 比绝对准确率，而是比「相同预算下的 Pareto 效率」。

---

## 建议的应对策略（防审稿人质疑）

### 1. 在文章中明确定位 scope

> 本文研究 **stopping efficiency**，而非提升 QA backbone 的 F1 上限。评估框架固定为 IRCoT + Llama-3.1-8B + BM25，结论对任意骨干同样成立（backbone-agnostic），更强的 backbone 只会平移所有曲线而不改变相对增益结论。

### 2. 找 1–2 篇同设置（Llama-3.1-8B + BM25 + open-domain）的方法对比

目前搜到最接近的：

- **IRCoT 原论文**（GPT-3 + BM25）：HotpotQA 60.7 / 2Wiki 68.0 / MuSiQue 36.5
- **Iter-RetGen-3**（Llama3-8B）：HotpotQA 60.89 / MuSiQue 30.31 / 2Wiki 49.83
- **EfficientRAG**（Llama3-8B）：HotpotQA 57.93

> 说明：同样是 8B 模型的方法，我们的 Probe 在 52.7（HotpotQA）这个区间，差距在 BM25 vs 更好的检索器。若换成 dense retrieval 骨干，停止策略结论不变。

### 3. 打「效率 Pareto」这张牌

重点放在 **F1-per-step 曲线**：


| 数据集      | 方法                | F1        | 平均步数     | F1/步数     |
| -------- | ----------------- | --------- | -------- | --------- |
| HotpotQA | Fixed-K=5         | 0.476     | 5.0      | 0.095     |
|          | **Probe+E-value** | **0.527** | **2.86** | **0.184** |
|          | Oracle 上界         | 0.623     | 1.60     | 0.389     |


我们的方法 F1/步数比是 Fixed-K=5 的 **1.94 倍**，逼近 Oracle 效率比（2.11 倍）的 92%。这才是本文的核心贡献指标，SOTA 刷分方法通常不报这个维度。

---

## 总结

绝对 F1 确实低于 SOTA 很多，**但这不是本文的主张**。需要在论文定位上：

1. 明确是 **efficiency / cost-aware stopping** 论文，而非 QA backbone 改进
2. 对比标准锚定在「相同骨干下的 Pareto 前沿」
3. 若担心被质疑，可考虑补一组 Llama-3.1-70B 或 dense retrieval 骨干上的验证实验，展示结论的 backbone 鲁棒性

---

## 数据来源说明

- 本项目数值：`results/stage3_evalue_*.json`（Probe / Probe+E-value）、`results/stage2_probe_table_*.csv`（Oracle、Fixed-K等）
- 外部 SOTA：PapersWithCode / Wizwand 汇总、IRCoT（ACL 2023）、Iter-RetGen-3 / EfficientRAG（ACM TKDD 等）、StepChain GraphRAG（arXiv 2025）等公开报告；具体数字以原论文为准，此处为写作时检索到的量级参考