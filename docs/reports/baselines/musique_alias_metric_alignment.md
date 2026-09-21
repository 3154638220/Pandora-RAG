# MuSiQue Answer Alias Metric Alignment

> 日期：2026-04-27
> 主题：修复 Pandora 与 Stop-RAG 在 MuSiQue F1/EM 计算口径上的不一致。

---

## 1. 问题摘要

当前 Pandora 与 Stop-RAG 在 MuSiQue 的答案评测上使用了不同 gold 口径：

- Pandora Stage 1 只使用主 `answer` 字段计算 F1/EM。
- Stop-RAG 使用 `[answer] + answer_aliases`，并对所有 gold aliases 取最大 F1 / 任一 EM。

这会导致 MuSiQue 上 Stop-RAG 的 F1 系统性偏高，而 Pandora 的 F1 系统性偏低。HotpotQA 通常只有单 gold，二者基本等价；2Wiki 多数样本没有 aliases，因此影响较小。

---

## 2. 现有代码差异

Pandora 当前入口：

```python
# stage1/run_stage1.py
a = _normalize_text(example.get("answer") or example.get("answers") or example.get("output"))
```

该逻辑只保留一个 `answer`，没有把 MuSiQue 的 `answer_aliases` 写入 processed record，也没有在后续 F1/EM 计算中使用 aliases。

Stop-RAG 当前入口：

```python
# baselines/stop-rag-pandora/src/pipeline/utils.py
raw = [question[fields["answer"]]] + question.get(fields["answer_aliases"], [])
f1 = max(compute_f1(pred, g) for g in gold_answers)
em = int(any(compute_exact_match(pred, g) for g in gold_answers))
```

Stop-RAG 的 MuSiQue 分数因此是 multi-gold max-F1 口径。

---

## 3. 影响范围

### 3.1 MuSiQue

影响最大。MuSiQue 数据集中存在 `answer_aliases`，例如缩写、别名、等价实体名、不同表述等。若模型预测命中了 alias，但没有命中主 `answer` 字段：

- Stop-RAG 会给较高 F1，甚至 EM。
- Pandora 当前会给较低 F1，甚至 0。

这会影响 Pandora 的：

- per-step `f1` / `em`
- oracle stop step
- Pareto table / figure
- Stage 2 probe 训练目标
- Stage 2 threshold selection
- Stage 3 最终 F1 / error 汇报

### 3.2 HotpotQA

影响很小或无影响。HotpotQA 通常为单 gold，`[answer] + answer_aliases` 退化为 `[answer]`。

### 3.3 2Wiki

影响较小。当前使用的数据大多没有有效 aliases，因此 multi-gold 与 single-gold 口径通常一致。

---

## 4. 推荐修复口径

推荐将 Pandora 对齐到 Stop-RAG 的 multi-gold 口径：

```text
gold_answers = [answer] + answer_aliases
F1 = max(F1(prediction, gold) for gold in gold_answers)
EM = any(EM(prediction, gold) for gold in gold_answers)
```

不建议反过来把 Stop-RAG 改成 single-gold。原因是 MuSiQue 原始数据显式提供 `answer_aliases`，multi-gold max-F1 更符合数据集的等价答案设定，也避免把正确别名预测误判为错误。

---

## 5. 代码修改建议

### 5.1 增加共享 multi-gold metric

在 `qa_shared/metrics.py` 增加：

```python
from typing import Iterable, Tuple


def compute_metrics_multi(prediction: str, ground_truths: Iterable[str]) -> Tuple[float, bool]:
    golds = [str(g) for g in ground_truths if g is not None and str(g).strip()]
    if not golds:
        return 0.0, False
    return (
        max(compute_f1(prediction, g) for g in golds),
        any(compute_exact_match(prediction, g) for g in golds),
    )
```

### 5.2 在 Pandora processed record 中保留 aliases

在 `stage1/run_stage1.py::_normalize_record` 中，将原始 `answer` 与 `answer_aliases` 规范化为统一字段：

```python
"answer": gold_answers[0],
"answer_aliases": gold_answers[1:],
"gold_answers": gold_answers,
```

其中 `gold_answers` 应去空、去重，并保持主 `answer` 在首位。

### 5.3 Stage 1 轨迹打分改用 multi-gold

在 Stage 1 collect trajectory 时，将：

```python
f1, em = compute_metrics(current_answer, gold)
```

改为：

```python
gold_answers = row.get("gold_answers") or [row.get("answer", "")]
f1, em = compute_metrics_multi(current_answer, gold_answers)
```

轨迹文件中建议同时保留：

```python
"gold_answer": gold_answers[0],
"answer_aliases": gold_answers[1:],
"gold_answers": gold_answers,
```

这样后续离线 rescore、审计和复现实验都不需要重新回查 raw dataset。

---

## 6. 重算策略

这次修复不要求完整重跑 Pandora Stage 1 的检索、LLM 生成和 hidden state extraction。

### 6.1 必须重算

- MuSiQue `data/processed/musique/*.jsonl` 中的 `answer_aliases` / `gold_answers`
- MuSiQue `trajectories.jsonl` 中每个 `steps[*].f1` / `steps[*].em`
- MuSiQue Stage 1 oracle labels
- MuSiQue Stage 1 oracle table / Pareto figure
- Stage 2 probe 训练、阈值选择、test 汇报
- Stage 3 主实验、漂移实验、统计汇报

### 6.2 不需要重跑

- 检索轨迹
- LLM 生成答案
- 每步 context/history
- hidden states `.npz`
- NLI / self-eval / semantic entropy 等与 gold answer 无关的已缓存特征

---

## 7. 推荐执行流程

最省成本的流程：

```text
1. 修复 multi-gold metric 代码
2. 将 MuSiQue raw answer_aliases merge 回 data/processed/musique/*.jsonl
3. 离线 rescore 现有 MuSiQue trajectories.jsonl
4. 重新计算 Stage 1 oracle labels / table / Pareto
5. 重跑 Stage 2
6. 重跑 Stage 3
7. 更新所有 MuSiQue 相关报告和跨数据集 macro 表
```

如果当前 processed 数据已经丢失 `answer_aliases`，不要重跑 LLM。只需要从 raw MuSiQue 按 `id` 对齐，把 `answer_aliases` 补回 processed record，再对已有 trajectories 离线重打分。

---

## 8. 报告与论文表述

修复后，报告中应明确写入口径：

```text
For MuSiQue, answer F1/EM is computed against the maximum score over
the canonical answer and all provided answer aliases. For HotpotQA and
2Wiki, this reduces to the standard single-gold setting when no aliases
are available.
```

中文表述：

```text
MuSiQue 的答案 F1/EM 统一按主答案及其 answer_aliases 的 multi-gold 口径计算，
即 F1 对所有 gold 取最大值，EM 对所有 gold 取任一匹配。HotpotQA 与 2Wiki
在无 aliases 时退化为标准单 gold 评测。
```

---

## 9. 结论

这是一个真实的 metric alignment 问题，不是模型效果差异。修复后，Pandora 与 Stop-RAG 在 MuSiQue 上会使用同一 answer alias 口径，跨方法比较才公平。

核心原则：

```text
不要重跑昂贵生成；重算依赖 gold metric 的标签、oracle、probe 和最终报告。
```

