# 放宽步数约束消融（`gw_steps_cap_mult`）— 结果摘要

> 日期：2026-04-09  
> 依据：`docs/plan_claude.md` 中「放宽步数约束消融」建议（`1.05 → 1.2 / 1.3`，零重训、仅重选 Phase C 阈值）。

---

## 1. 工程改动

- `**stage2/run_stage2.py**` 增加 `--rethreshold-only`：从 `artifacts/probe/<dataset>/probe_mlp.pt`（shallow 模式为 `probe_mlp_shallow.pt`）加载权重与 `StandardScaler`，**不重新训练**，仅用当前 `--gw-steps-cap-mult` 等在 dev 上重跑 Phase C 并评估 test。
- 输出通过 `--artifact-suffix` 区分，避免覆盖默认 `stage2_probe_table_*.csv` / `stage2_train_meta.json`。

---

## 2. 实验设置


| 项目                  | 取值                                                               |
| ------------------- | ---------------------------------------------------------------- |
| 数据集                 | hotpotqa, musique, 2wiki                                         |
| `gw_steps_cap_mult` | 1.05、1.2、1.3                                                     |
| 权重                  | 各数据集当前默认 `probe_mlp.pt`                                          |
| 输出后缀                | `ablate_gw_cap105`、`ablate_gw_cap12`（1.2）、`ablate_gw_cap13`（1.3） |


说明：`1.2` 对应后缀 `ablate_gw_cap12` 易与「12」混淆，后续复现建议改用如 `ablate_gw_cap1p2`。

---

## 3. 主结果（test）

三档 **cap_mult 下指标完全相同**：


| cap_mult         | HotpotQA F1 | MuSiQue F1 | 2Wiki F1   | Dev 选中阈值                                 |
| ---------------- | ----------- | ---------- | ---------- | ---------------------------------------- |
| 1.05 / 1.2 / 1.3 | **0.5148**  | **0.1794** | **0.3484** | hotpot & musique **0.63**，2wiki **0.65** |


与仓库内默认 `artifacts/probe/*/stage2_train_meta.json` 中 `probe_summary` 一致。

---

## 4. 分析要点

1. **Phase C 选阈逻辑**
  对每个 λ 在「Probe dev `avg_steps` ≤ `gw_dev_avg_steps × cap_mult`」的可行阈值内最大化 `F1 − λ·归一化成本`，再在四个 λ 的结果中取 **dev F1 最高**（平手则更低成本）。
2. **为何放宽 cap 无效**
  - **HotpotQA**：最优阈值已在较紧上界内；放宽后最优解不变。  
  - **MuSiQue**：`feasible_threshold_count` 随 cap 从 **21 → 22 → 50**（1.3 时 50 个候选阈值**全部可行**），但 **λ=0.1 下 Pareto 最优仍为 0.63**，最终仍选该阈值 → test 无变化。  
  - 说明在当前 Probe 与 Phase C 下，**瓶颈不是「上界拦住了更高 F1 的阈值」**，而是 **dev 上没有任何更「多步」的阈值能在该目标下击败 0.63**。
3. **与 `plan_claude.md` 主表数字**
  文中 Hotpot **0.5244**、MuSiQue **0.1659**、2Wiki **0.3606** 与当前磁盘 checkpoint 对应的 **0.5148 / 0.1794 / 0.3484** 不一致，可能来自**另一版特征或训练**；本次消融以**现网 `probe_mlp.pt`** 为准，且三档 cap **两两无差异**。
4. **若仍希望「多走一步换 F1」**
  需调整 dev 选阈目标（如 λ 网格、数据集相关权重、步数下界等）或改进 Probe 校准/特征，**单靠放大 `gw_steps_cap_mult` 在本次设置下无收益**。

---

## 5. 复现命令

```bash
cd /path/to/Pandora-RAG

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --rethreshold-only --gw-steps-cap-mult 1.05 --artifact-suffix ablate_gw_cap105

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --rethreshold-only --gw-steps-cap-mult 1.2 --artifact-suffix ablate_gw_cap1p2

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --rethreshold-only --gw-steps-cap-mult 1.3 --artifact-suffix ablate_gw_cap13
```

产物：`results/stage2_probe_table_*_<suffix>.csv`、`artifacts/probe/*/stage2_train_meta_<suffix>.json`、`docs/stage2_report_<suffix>.md`。

---

## 6. 结论（一句话）

在现有 `probe_mlp.pt` 与 Phase C 流程下，将 `gw_steps_cap_mult` 从 1.05 放宽至 1.2、1.3 **不改变**最优停止阈值与 test F1；MuSiQue 在 cap=1.3 时已枚举全部阈值候选仍选原阈值，**步数上界不是当前性能短板**。

---

# `answer_logprob` 与 20 维浅层 — 结果摘要

> 日期：2026-04-09  
> 依据：`stage2/run_stage2.py`、`stage2_feature_diagnostic.md`、`stage2_next_steps_analysis.md`、`docs/plan.md`（18 维 D3 记录）、`docs/stage2_report.md`（当前默认产物）。

---

## A. 工程范围（避免过度归因）

- **同时合入两列基础特征**：`answer_logprob`（首条采样答案的平均 token logprob，见 `pretest/utils/llm_client.py`）与 `**self_eval_score`**。浅层由 **18 维（含 D4×3）→ 20 维**。
- **若无「只删 logprob、保留 self_eval」的单列消融**，下表「相对 18 维」的 test 数字是 **两列联合 + 全量重训 + Phase C 重选阈** 的效应，**不能**单独写成「全是 answer_logprob 造成的」。

---

## B. 端到端 test：相对 18 维 D3 默认 MLP（`plan.md` 2026-04-06）


| 数据集      | 18 维 Probe F1 | 18 维平均步数 | 当前 20 维 Probe F1 | 20 维平均步数 | 20 维 Phase C 阈值（`stage2_report.md`） |
| -------- | ------------- | -------- | ---------------- | -------- | ----------------------------------- |
| HotpotQA | 0.5256        | 2.892    | **0.5148**       | 2.532    | **0.63**                            |
| MuSiQue  | 0.1785        | 3.513    | **0.1794**       | 3.089    | **0.63**                            |
| 2Wiki    | 0.3580        | 3.109    | **0.3484**       | 2.411    | **0.65**                            |


**要点**：未出现「接入高杠杆置信度特征后三数据集普涨」；HotpotQA / 2Wiki **略降**，MuSiQue **微升**；平均步数略降。20 维下 Dev 选出的停止阈值集中在 **0.63–0.65**（高于 D2 文档中为 **XGBoost** 列出的 Phase C 阈值 0.37 / 0.19 / 0.27，见 `stage2_xgboost_baseline.md`——后者**不能**直接等同于 18 维 MLP 的阈值，因 Phase C 在**不同模型分数**上枚举；本表以 F1 / 步数对比为主）。

---

## C. D1 单变量：`answer_logprob` 与 Continue 标签

摘自 `stage2_feature_diagnostic.md`（20 维全量诊断；AUROC 为单特征预测 Continue，0.5 为随机）：


| 数据集      | point-biserial r | AUROC      |
| -------- | ---------------- | ---------- |
| hotpotqa | −0.2571          | **0.2618** |
| musique  | −0.0461          | 0.4424     |
| 2wiki    | −0.0328          | 0.4659     |


**要点**：三数据集 **均低于 0.60** 的「强单特征」经验区间；HotpotQA 上 AUROC 明显偏离 0.5 一侧，与「logprob 越大越 Continue」的简单单调叙事不一致，更依赖与其它维度的交互。与 `stage2_next_steps_analysis.md` 中「接入即大涨不现实」的判断一致。

---

## D. 与 `plan_claude.md` 的叙事对齐

- 评估报告仍将 answer_logprob 列为高优先级方向；**当前仓库快照下已实现均值字段接入**，但 **Probe test F1 未获稳定提升**，风险表「补录后未显著提升」情形 **已发生**，适合作为论文中的**诚实消融/负向结果**行。
- **仍值得做的后续**（文档已写）：`answer_logprob_min` / `std`、步间 `delta_answer_logprob`、保证推理路径稳定返回 token logprob（减少 `llm_client` 中 −1 / 0 的 fallback 稀释）。见 `docs/plan-04-08.md` P0。

---

## E. 复现

```bash
cd /path/to/Pandora-RAG

python -m stage2.run_feature_diagnostic --datasets hotpotqa,musique,2wiki
python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki
```

产物：`docs/stage2_feature_diagnostic.md`、`docs/stage2_report.md`、`results/stage2_probe_table_*.csv`。

---

## F. 一句话结论

`**answer_logprob`（与同批 `self_eval_score`）接入 20 维浅层后，D1 显示其单变量区分力有限，端到端 test 相对 18 维 D3 默认无普涨、Hotpot/2Wiki 略降；符合「特征工程需继续做细，而非单列即突破」的判断。**

---

# 硬 margin 过滤（训练集 `|Oracle margin| < τ` 丢弃）— 结果摘要

> 日期：2026-04-09  
> 依据：`docs/plan_claude.md` 中「硬 margin 过滤实验」；实现为 `stage2/run_stage2.py` 的 `--train-margin-min-abs`。

---

## G. 设置说明

- **做法**：仅在**训练集**构建 `(x, y, w)` 时丢弃 `|margin| < τ` 的步；**dev / test** 仍用全部步做 Phase C 与评估。
- **其余**：三数据集 `hotpotqa,musique,2wiki`；Phase C `gw_steps_cap_mult=1.05` 等与当前默认 Stage2 一致。
- **基线**：`results/stage2_probe_table_{dataset}.csv`（τ=0，无 artifact 后缀）。

---

## H. 训练集过滤量（约）


| τ    | HotpotQA（丢弃 / 保留） | MuSiQue      | 2Wiki        |
| ---- | ----------------- | ------------ | ------------ |
| 0.02 | 13 / 15987        | 26 / 15974   | 7 / 15993    |
| 0.05 | 401 / 15599       | 578 / 15422  | 230 / 15770  |
| 0.10 | 11695 / 4305      | 12443 / 3557 | 11711 / 4289 |


τ=0.10 时约 **73%** 训练步被丢，训练集 **Continue ≈ 73%**，与 dev 上 Continue ≈20% **严重失衡**。

---

## I. Test F1（Probe）


| τ         | HotpotQA   | MuSiQue    | 2Wiki      | 相对 τ=0                      |
| --------- | ---------- | ---------- | ---------- | --------------------------- |
| **0**（基线） | 0.5148     | 0.1794     | 0.3484     | —                           |
| **0.02**  | 0.5167     | 0.1687     | **0.3833** | +0.0019 / −0.0107 / +0.0349 |
| **0.05**  | 0.5151     | **0.1793** | 0.3628     | +0.0003 / −0.0001 / +0.0144 |
| **0.10**  | **0.5196** | 0.1576     | 0.3492     | +0.0048 / −0.0218 / +0.0008 |


---

## J. Test 平均步数（Probe）


| τ    | HotpotQA | MuSiQue | 2Wiki |
| ---- | -------- | ------- | ----- |
| 0    | 2.532    | 3.089   | 2.411 |
| 0.02 | 2.450    | 3.038   | 3.322 |
| 0.05 | 2.563    | 3.424   | 2.700 |
| 0.10 | 2.905    | 3.942   | 3.486 |


---

## K. 复现命令

```bash
cd /path/to/Pandora-RAG

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --train-margin-min-abs 0.02 --artifact-suffix ablate_margin_m02
python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --train-margin-min-abs 0.05 --artifact-suffix ablate_margin_m05
python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --train-margin-min-abs 0.10 --artifact-suffix ablate_margin_m10
```

产物：`results/stage2_probe_table_*_ablate_margin_m{02,05,10}.csv`、`docs/stage2_report_ablate_margin_m*.md`、对应 Pareto 图。

---

## L. 一句话结论

在已有 `**|margin|` 加权损失** 前提下，硬过滤小 margin **无三数据集同时收益**：τ=0.02 伤 MuSiQue；τ=0.05 仅 2Wiki 明显涨、其余持平或略降；τ=0.10 虽 Hotpot F1 略升但**步数成本上升**且难集变差。适合作为论文 **中性/负面消融**（硬过滤不如保留样本并用 margin 连续信息加权）。

---

# Hidden states 压缩维度（`compress_dim`）— 结果摘要

> 日期：2026-04-09  
> 依据：`docs/plan_claude.md` 中「Hidden states 压缩维度提升」；`ProbeMLP_v2` 将 Stage1 的 4096 维 hidden 线性压到 **d**。

---

## M. 设置

- **数据集**：hotpotqa, musique, 2wiki（全量重训 + Phase C，与默认 Stage2 一致，`gw_steps_cap_mult=1.05`）。
- **对比**：`compress_dim` ∈ {**64**（默认）, **256**, **512**}；256/512 使用 `--artifact-suffix ablate_compress_dim256` / `ablate_compress_dim512`。

---

## N. Test F1


| compress_dim | HotpotQA   | MuSiQue    | 2Wiki  | 相对 d=64                     |
| ------------ | ---------- | ---------- | ------ | --------------------------- |
| **64**（默认）   | 0.5148     | 0.1794     | 0.3484 | —                           |
| **256**      | **0.5273** | **0.1826** | 0.3273 | +0.0125 / +0.0032 / −0.0211 |
| **512**      | 0.5251     | 0.1818     | 0.3047 | +0.0103 / +0.0024 / −0.0437 |


---

## O. Test 平均步数


| compress_dim | HotpotQA | MuSiQue | 2Wiki |
| ------------ | -------- | ------- | ----- |
| 64           | 2.532    | 3.089   | 2.411 |
| 256          | 2.854    | 3.252   | 2.247 |
| 512          | 2.885    | 3.590   | 1.975 |


---

## P. 相对 Shallow-Only（w/o Deep Features）

Shallow 基线 F1（`docs/stage2_report_shallow.md`）：HotpotQA 0.4829，MuSiQue 0.1394，2Wiki 0.3552。


| compress_dim | ΔF1 vs Shallow（Hotpot / MuSiQue / 2Wiki） |
| ------------ | ---------------------------------------- |
| 64           | +0.0319 / +0.0400 / −0.0068              |
| 256          | +0.0444 / +0.0432 / −0.0279              |
| 512          | +0.0422 / +0.0424 / −0.0505              |


---

## Q. 复现命令

```bash
cd /path/to/Pandora-RAG

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --compress-dim 256 --artifact-suffix ablate_compress_dim256

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --compress-dim 512 --artifact-suffix ablate_compress_dim512
```

分项报告：`docs/stage2_report_ablate_hidden_compress.md`、`docs/stage2_report_ablate_compress_dim256.md`、`docs/stage2_report_ablate_compress_dim512.md`；产物见 `results/stage2_probe_table_*_ablate_compress_dim*.csv` 与对应 Pareto 图、checkpoint。

---

## R. 一句话结论

**HotpotQA / MuSiQue** 上 **d=256** 略优于 d=64；**d=512** 相对 256 无一致收益。**2Wiki** 增大 d 后 F1 **变差**。Plan 中期望的「相对 64 再拉到 +0.08+」**未达成**；若论文只写一个「放宽瓶颈」消融点，可取 **d=256**，但 **不宜三数据集共用更大 d 而不单独调参**。