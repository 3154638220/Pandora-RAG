# Stop-RAG 对齐实验收尾与对比口径

> 日期：2026-04-19  
> 状态：Closed / Ready for paper comparison  
> 口径：同 Pandora test split、同样本 id、Stop-RAG 在线早停结果；Pandora 主方法采用 `Probe+E-value`，`gamma=0.5`，`alpha=0.1`，predictive betting。

---

## 1. 结论摘要

Stop-RAG 对齐实验已经完成到可写论文对比的状态。本轮结果的核心结论是：

1. **Stop-RAG 的当前在线早停点明显偏满预算。**  
   HotpotQA 全部样本停在第 5 步；MuSiQue 平均 4.64 步；2Wiki 平均 4.48 步。它在当前对齐设置下更像一个 high-budget stopping baseline，而不是低成本停止器。
2. **Pandora 在三数据集上同时取得更高 F1 与更低平均步数。**  
   宏平均上，`Probe+E-value` 的 F1 为 **0.5503**，Stop-RAG 为 **0.4564**；平均步数分别为 **2.37** 与 **4.71**。
3. **主文可以使用当前 single-point 在线结果，但要说明它不是完整 Pareto sweep。**  
   当前文件来自 `stop_rag_find_best.sh` 选出的 checkpoint/threshold 再经 `stop_rag_test.sh` 在线测试；若论文主图要画 `F1 vs avg_steps` frontier，还需要补 Stop-RAG threshold sweep 的在线重测点。

一句话写法：

> Under the aligned online evaluation, Stop-RAG tends to exhaust the retrieval budget, while Pandora achieves higher answer quality with roughly half the retrieval steps on average.

---

## 2. 公平性与复现口径

本次 Stop-RAG 对齐只承认 `baselines/Stop-RAG/scripts/stop_rag_test.sh` 产出的在线测试文件，原因是该脚本在 `test_subsampled` 上逐轮运行 stop head：一旦预测 `STOP`，后续检索不会继续发生。

不作为最终 head-to-head 数字的文件：

- `compute_scores/*.jsonl`：用于 eval 上离线选 checkpoint 与 threshold。
- `src/test/stop_rag_test.py` 的事后回放文件：可分析，但不等价于真实在线早停。
- 上游 `download.sh` 重新划分的数据：与 Pandora 主线 split 不一致。

当前有效结果路径如下：

| 数据集 | Stop-RAG 在线结果 | checkpoint / threshold |
| --- | --- | --- |
| HotpotQA | `baselines/Stop-RAG/results/hotpotqa_ours_contriever/online_test/hotpotqa_test_ckpt1000_thr-0.09.jsonl` | ckpt1000 / -0.09 |
| MuSiQue | `baselines/Stop-RAG/results/musique_ours_contriever/online_test/musique_test_ckpt1200_thr-0.02.jsonl` | ckpt1200 / -0.02 |
| 2WikiMultiHopQA | `baselines/Stop-RAG/results/2wikimultihopqa_ours_contriever/online_test/2wikimultihopqa_test_ckpt2400_thr0.03.jsonl` | ckpt2400 / 0.03 |

Stop-RAG 的平均步数来自相邻的 `*.stop_log.jsonl` 中 `stop_iter`；Pandora 的平均步数来自 `results/stage3_evalue_{dataset}.json` 中 `steps_used` 汇总后的 `avg_steps`。

---

## 3. 主对比表

### 3.1 Pandora `Probe+E-value` vs Stop-RAG

| 数据集 | 方法 | N | Avg F1 | Avg EM | Avg Steps | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HotpotQA | Stop-RAG | 1000 | 0.5963 | 0.4640 | 5.000 | 全部停在第 5 步 |
| HotpotQA | Pandora `Probe+E-value` | 1000 | **0.6654** | **0.5290** | **1.807** | F1 +0.0691，少 3.19 步 |
| MuSiQue | Stop-RAG | 417 | 0.2654 | 0.1942 | 4.643 | 312/417 样本停在第 5 步 |
| MuSiQue | Pandora `Probe+E-value` | 417 | **0.4127** | **0.3141** | **3.410** | F1 +0.1473，少 1.23 步 |
| 2WikiMultiHopQA | Stop-RAG | 1000 | 0.5076 | 0.4150 | 4.477 | 707/1000 样本停在第 5 步 |
| 2WikiMultiHopQA | Pandora `Probe+E-value` | 1000 | **0.5728** | **0.4810** | **1.905** | F1 +0.0652，少 2.57 步 |

### 3.2 宏平均

| 方法 | Macro Avg F1 | Macro Avg EM | Macro Avg Steps |
| --- | ---: | ---: | ---: |
| Stop-RAG | 0.4564 | 0.3577 | 4.707 |
| Pandora `Probe+E-value` | **0.5503** | **0.4414** | **2.374** |
| Pandora - Stop-RAG | **+0.0939** | **+0.0836** | **-2.333** |

加权到样本级时，Stop-RAG 的整体 F1 为 0.5025，整体 EM 为 0.3972，整体平均步数为 4.722；Pandora 的 Stage 3 汇总目前更建议按数据集宏平均写，因为 MuSiQue test 只有 417 条，直接样本加权会削弱该数据集的贡献。

---

## 4. Stop-RAG 停止分布

| 数据集 | Avg Steps | Step=2 | Step=3 | Step=4 | Step=5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 5.000 | 0 | 0 | 0 | 1000 |
| MuSiQue | 4.643 | 6 | 32 | 67 | 312 |
| 2WikiMultiHopQA | 4.477 | 59 | 112 | 122 | 707 |

这个分布是解释 Stop-RAG 当前结果的关键：它不是因为积极早停导致质量损失，而是在多数样本接近满预算的情况下，答案 F1 仍低于 Pandora。

---

## 5. 与 Fixed-K 的关系

由于 Stop-RAG 当前接近满预算，论文里可以额外把它和 `Fixed-K=5` 放在一起解释：

| 数据集 | 方法 | Avg F1 | Avg EM | Avg Steps |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | Fixed-K=5 | 0.6668 | 0.5240 | 4.991 |
| HotpotQA | Stop-RAG | 0.5963 | 0.4640 | 5.000 |
| MuSiQue | Fixed-K=5 | 0.4022 | 0.3022 | 5.000 |
| MuSiQue | Stop-RAG | 0.2654 | 0.1942 | 4.643 |
| 2WikiMultiHopQA | Fixed-K=5 | 0.5288 | 0.4230 | 5.000 |
| 2WikiMultiHopQA | Stop-RAG | 0.5076 | 0.4150 | 4.477 |

解释边界：

- `Fixed-K=5` 是 Pandora 轨迹上的固定满预算策略；Stop-RAG 有自己的检索与生成管线，因此不是逐样本同轨迹比较。
- 这个表只用于帮助读者理解 Stop-RAG 的预算位置：当前 Stop-RAG threshold 使它落在接近 fixed-full-budget 的区域。
- 主 claim 仍应以 `Pandora vs Stop-RAG` 的同 split 在线结果为准。

---

## 6. 推荐写法

主文表述建议：

> We reproduced Stop-RAG under the Pandora-aligned split and evaluated it with true online stopping. In this aligned setting, Stop-RAG often runs close to the maximum retrieval budget. Pandora's `Probe+E-value` improves macro F1 from 0.456 to 0.550 while reducing the average number of retrieval steps from 4.71 to 2.37.

中文报告表述建议：

> 在同切分、同样本 id、真实在线早停的公平口径下，Stop-RAG 当前阈值几乎跑满检索预算；相比之下，Pandora `Probe+E-value` 在三个数据集上都以更少步数取得更高 F1。宏平均 F1 提升 0.0939，平均检索步数减少 2.33。

主表建议列：

- `Method`
- `Dataset`
- `F1`
- `EM`
- `Avg steps`
- `N`

主图建议：

- 当前可画 single-point scatter：Stop-RAG vs Pandora。
- 若要主张完整预算曲线，补 Stop-RAG 多 threshold 的在线重测，并画 `F1 vs avg_steps` Pareto frontier。

---

## 7. 剩余边界与不再阻塞项

已完成：

- Pandora split 到 Stop-RAG 数据格式的对齐。
- Stop-RAG stop head 训练、eval 选 checkpoint/threshold。
- 三数据集 `stop_rag_test.sh` 在线测试。
- 与 Pandora `Probe+E-value` 的 F1 / EM / avg_steps 对比表。

不再作为当前收尾 blocker：

- Stop-RAG 原论文上游 split 复现。
- Stop-RAG 离线 replay 数字整理。
- LLM-Stop 等额外 baseline。

可选增强：

- 对 Stop-RAG 做多 threshold 在线 sweep，补完整 Pareto 曲线。
- 统一导出一张 `Pandora / Stop-RAG / Fixed-K=5` 的论文图。

---

## 8. 数据来源

- Stop-RAG 结果：
  - `baselines/Stop-RAG/results/hotpotqa_ours_contriever/online_test/`
  - `baselines/Stop-RAG/results/musique_ours_contriever/online_test/`
  - `baselines/Stop-RAG/results/2wikimultihopqa_ours_contriever/online_test/`
- Pandora Stage 3：
  - `results/stage3_evalue_hotpotqa.json`
  - `results/stage3_evalue_musique.json`
  - `results/stage3_evalue_2wiki.json`
- Pandora Fixed-K：
  - `results/stage2_probe_table_hotpotqa_pdopt_best.csv`
  - `results/stage2_probe_table_musique_pdopt_best.csv`
  - `results/stage2_probe_table_2wiki_pdopt_best.csv`
