# Stage2 下一步效果提升分析

## 结论（2026-04 对齐当前代码）

若目标仍是提升 Stage2 Probe 在**可部署口径**下的效果，主线里「先补 `answer_logprob` / `self_eval_score`」这一档工作**已经合入**：Stage1 轨迹会写入二者，Stage2 浅层向量现为 **20 维**（含 D4 三项与步间差分）。因此「下一刀」不应再停留在「从零接入这两个标量」，而应转向：

1. **把答案置信度从「单标量平均 logprob」做成更丰富、更可靠的信号**（例如 min、跨步 `delta`，以及稳定拿到 token 级 logprob 的推理路径）。
2. **提高 `self_eval_score` 的测量质量**（prompt / 量表 / 是否与 `self_consistency` 解耦），并在端到端上验收，而不是只看单变量 AUROC。
3. 若上述仍摸顶，再评估 **hidden 分支、新代理量（如参考模型 NLL）或论文向的 D6 消融**，而不是在浅层 MLP 超参上空转。

单变量诊断已表明：`answer_logprob` 与 `self_eval_score` **没有**在三个数据集上都呈现「强单特征」级别的 AUROC（多数仍明显低于 0.60），因此它们值得继续投资，但预期应是**迭代测量与特征工程**，而非「接入即大涨」。

## 为什么这样判断

### 1. 计划里的「缺直接答案质量代理」已部分闭合，但要升级表述

- `[plan.md](../../plans/plan.md)` 仍把「答案 token 级置信度 / 自评」列为高价值方向；**实现上** Stage1 已在每步写入 `answer_logprob`（当前来自 `pretest/utils/llm_client.py` 对**首条采样答案**的平均 token logprob）与 `self_eval_score`（额外一次自评调用）。
- 与「完全缺失」时期相比，瓶颈叙事应从「**没有字段**」调整为「**字段仍偏粗、单变量区分力有限、端到端是否吃满未单独消融**」。

### 2. 浅层维度已是 20 维：D4 与答案代理在同一管线中

`stage2/run_stage2.py` 中 `SHALLOW_FEATURE_NAMES` 为 11 维基础（含 `answer_logprob`、`self_eval_score`）+ 5 维步间差分 + `cumulative_cost_ratio` + D4 三项（`answer_changed`、`answer_consistency_streak_norm`、`retrieval_marginal_novelty`），合计 20 维。

对应实现：

```48:71:/home/x12dpg/hjx/Pandora-RAG/stage2/run_stage2.py
# 11 维基础（新增 answer_logprob/self_eval_score）+ 5 维步间差分 + cumulative_cost_ratio + Phase D4 答案/检索代理（3）
SHALLOW_FEATURE_DIM = 20
SHALLOW_FEATURE_NAMES: Tuple[str, ...] = (
    "k_norm",
    "retrieval_score",
    "semantic_entropy",
    "self_consistency",
    "answer_logprob",
    "self_eval_score",
    ...
    "retrieval_marginal_novelty",
)
```

再往前堆「低成本、不重跑 Stage1」的浅层拼特征，边际通常不如**改 Stage1 观测量**或**加强 deep 分支**。

### 3. D1 结果：新特征接入后，单变量仍然普遍偏弱

`[stage2_feature_diagnostic.md](stage2_feature_diagnostic.md)` 已按当前 `SHALLOW_FEATURE_NAMES` 全量输出（20 维）。摘要如下（单特征 → 预测 Continue 的 AUROC，越高越易分，0.5 为随机）：


| 特征                | hotpotqa | musique | 2wiki |
| ----------------- | -------- | ------- | ----- |
| `answer_logprob`  | ≈0.26    | ≈0.44   | ≈0.47 |
| `self_eval_score` | ≈0.41    | ≈0.49   | ≈0.48 |


Interpretation：AUROC 低于 0.5 表示与「原始列值越大越 Continue」的单调假设相反，但**仍可能**在多层网络里与其它特征交互后有用；整体上它们**没有**单独达到「强答案质量代理」的典型强度（例如稳定 >0.6）。这与 XGBoost 与 MLP 同特征打平的现象一致——输入侧上限仍紧。

对应分析仍见 `[stage2_xgboost_baseline.md](stage2_xgboost_baseline.md)`。

### 4. 公平对比对象仍然不是 Original-GW

Probe 的公平对照是 **Deployable-GW**（测试时用可部署代理停止、再报真实 F1），而不是在 test 上用真 F1 停表的 Original-GW。主线达标后，不必以「追平 Original-GW」为主优化目标，详见 `[plan.md](../../plans/plan.md)` 中 Deployable-GW 与 Oracle 口径说明。

## 推荐的执行顺序（修订版）

### P0：变更 Stage1 字段或采样策略后，重跑 D1 + Stage2

默认 **不必**再为「补全特征名列表」单独刷诊断——当前 `[stage2_feature_diagnostic.md](stage2_feature_diagnostic.md)` 已与 20 维对齐。若你改了轨迹 JSONL 结构、logprob 来源或自评逻辑，应顺序执行：

1. `python -m stage2.run_feature_diagnostic --datasets hotpotqa,musique,2wiki`
2. `python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`

诊断脚本从 `run_stage2` 导入 `SHALLOW_FEATURE_NAMES`，与训练特征维数一致：

```24:31:/home/x12dpg/hjx/Pandora-RAG/stage2/run_feature_diagnostic.py
from stage2.run_stage2 import (
    SHALLOW_FEATURE_NAMES,
    Stage2Config,
    ...
)
```

### P1：扩展 `answer_logprob` 系列（Stage1 落盘 + Stage2 接维）

当前每步仅一个标量（均值）。建议在 Stage1 侧一次性增加并缓存：

- `answer_logprob_mean`（可与现有字段合并语义，避免重复）
- `answer_logprob_min`（捕捉「局部低置信 token」）
- 步间 `delta_answer_logprob`（与现有 `delta_*` 设计一致）

然后在 Stage2 扩展 `SHALLOW_FEATURE_NAMES` 并重训。动机：D1 已显示**单一**平均 logprob 在 HotpotQA 上单变量极弱，更丰富分解有机会被 Probe 利用。

### P2：加固 logprob 来源与自评质量

- **Logprob**：确认生产路径上是否始终走能返回 token logprob 的后端；fallback（如 `-1` / `0`）会稀释特征。见 `pretest/utils/llm_client.py` 中 `answer_logprob` 的注释与分支。
- **Self-eval**：审视 prompt 是否稳定、分数是否与人工/EM 相关；必要时与 `self_consistency` 做相关性分析，避免三四个高度共线标量占满容量。

### P3：全量对比与低成本训练侧微调

新特征接入后的标准验收链路与此前相同：诊断 → `run_stage2` → 与当前 Probe / Deployable-GW 及 `docs/reports/stage2/` 中的表对齐。

可选、**非主攻**：将默认 `focal_gamma` 从 `2.0` 调到 `1.0` 做一轮对照（D3 中部分数据集略优），见 `[stage2_report_d3_exp2_focal1.md](stage2_report_d3_exp2_focal1.md)` 与配置项：

```101:103:/home/x12dpg/hjx/Pandora-RAG/stage2/run_stage2.py
    # Phase B：Focal BCE + label smoothing（None 表示按训练集 Continue 比例自适应 focal 正类权重）
    focal_gamma: float = 2.0
```

## 现阶段不建议优先投入的方向

1. **指望同特征上 XGBoost 大幅超过 MLP**（D2 已表明基本打平）。
2. **把 D6 当作首要涨点手段**（更适合论文下界与消融叙事）。
3. **以 Original-GW test F1 为主 KPI** 做迭代（口径不公平，易带偏工程取舍）。

## 一句话概括

`**answer_logprob` / `self_eval_score` 已接入 20 维浅层；下一步应优先把 logprob 做细、做稳并改进自评读数，再全量重训验收，而不是重复「首次补字段」或只在旧特征上调模型。**
