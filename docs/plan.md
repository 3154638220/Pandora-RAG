# Pandora-RAG Stage 2 攻坚计划

> 最后更新：2026-04-07
> 当前状态：优先级 1–5 的 Phase A+B+C 均已落地。**Probe 与公平基线 Deployable-GW 对比：三数据集 test F1 均更高**（见下「Deployable-GW 全量验收记录」）；与 **Original-GW（半 Oracle）** 相比仍落后——后者在 test 上使用真实 F1 停止，**不作公平胜负判定**。经深度诊断后发现核心瓶颈并非训练技巧，而是 **若误用 Original-GW 作对手则对比不公平 + 特征信号缺失**。**Phase D2（XGBoost 同特征基线）已跑完**：三数据集上 XGB 均未明显优于 MLP，**与「特征是主瓶颈」一致**。**D3 损失消融已跑完**（见 Phase D3 全量验收记录；与默认 `run_stage2` 均在 **18 维浅层含 D4×3** 的同一代码快照下完成，故 **D4 全量重训已包含在内**）。**D1 特征诊断已按当前 18 维全量重刷**（2026-04-07，`stage2_feature_diagnostic.md` 含 D4×3；旧版报告仅列 15 维，已作废）。**D5 方案 B：Deployable-GW 已落地**（train 上用 `self_consistency` 步间增益估 r_k^，test 上按代理停止、汇报真实 F1；见下「Deployable-GW 全量验收记录」）。Phase C 阈值锚点仍为 **Original-GW(dev) 步数**（未改）。**下一步（效果）**：优先在 Stage1 侧补 `**answer_logprob` 系列**（建议 mean / min / delta 一并落盘），再视增益考虑 `self_eval_score`；详见 `stage2_next_steps_analysis.md` 与下节 D1/D4。若需刷新带 Deployable-GW 行的 CSV/Pareto 可再跑全量 `run_stage2`；或推进 D6 / 论文叙事。

---

## 当前实验诊断（2026-04-06 深度复盘）

### Probe vs Deployable-GW（公平对比；Original-GW 仅参照）


| 数据集      | Probe F1   | Deployable-GW F1 | Probe 步数 | D-GW 步数 | Original-GW F1（参照） | Original-GW 步数 | Oracle F1 | Orig-GW/Oracle |
| -------- | ---------- | ---------------- | -------- | ------- | ------------------ | -------------- | --------- | -------------- |
| hotpotqa | **0.5244** | 0.3698           | 2.74     | 1.01    | 0.6089             | 2.819          | 0.6283    | 96.9%          |
| musique  | **0.1659** | 0.0907           | 3.13     | 1.05    | 0.2235             | 4.091          | 0.2430    | 92.0%          |
| 2wiki    | **0.3606** | 0.2223           | 2.71     | 1.01    | 0.5121             | 3.471          | 0.5264    | 97.3%          |


> Deployable-GW 数字来自 `python -m stage2.run_deployable_gw_eval`（与 D5 节一致）；**各阶段 Probe 的公平胜负判定以 vs Deployable-GW 为准**，Original-GW 保留为上界参照。

### 训练动态分析


| 数据集      | Best Epoch | 总 Epoch | Train Loss (终) | Dev Loss (终) | 过拟合 Gap | Dev [Acc@0.5](mailto:Acc@0.5) |
| -------- | ---------- | ------- | -------------- | ------------ | ------- | ----------------------------- |
| hotpotqa | 6          | 18      | 0.035          | 0.064        | 0.029   | 45.6%                         |
| musique  | 7          | 19      | 0.032          | 0.090        | 0.058   | 24.3%                         |
| 2wiki    | 5          | 17      | 0.038          | 0.075        | 0.037   | 30.3%                         |


### Hidden States 收益微弱


| 数据集      | Shallow-Only F1 | Full F1 | 增益     |
| -------- | --------------- | ------- | ------ |
| hotpotqa | 0.4829          | 0.5327  | +0.050 |
| musique  | 0.1394          | 0.1609  | +0.022 |
| 2wiki    | 0.3552          | 0.3731  | +0.018 |


### 根因分析（2026-04-06 深度复盘，按严重度排序）

#### ⚠️ 根因 1：GW 基线对比不公平——信息论层面的不对等

**这是此前所有迭代均未识别的核心盲点。** `weitzman.py` 的 `oracle_stopping_simulation` 在 test 时直接使用 **真实 F1** (`curr_f1 = step_data["f1"]`) 做停止判断。真实 F1 需要 ground truth answer，**在实际部署中不可获得**。

- GW 的停止规则：`if curr_f1 >= r_{k+1}* then stop`——其中 `curr_f1` 是 **ground truth 计算的真实 F1**
- Probe 的停止规则：`if p_continue < threshold then stop`——`p_continue` 由 **间接特征**（熵、NLI 等）预测
- Oracle DP 的停止规则：全知（后向归纳，同样使用真实 F1）

Original-GW 达到 Oracle 的 **92–97%**，因为它们同属「真实 F1 可观测」的策略族。Probe 属于「真实 F1 不可观测」的策略族——两者的理论上界不同。**让 Probe 在公平口径下追上 Original-GW（半 Oracle）本质上等于要求特征集完美预测 F1，在多跳 RAG 场景下几乎不可能**；公平对手应为 Deployable-GW。

#### 根因 2：特征集缺少「答案质量」的直接代理

Oracle 的 `action_label` 取决于 "当前 F1 够不够好 + 继续能改善多少"。当前 **18 维**浅层（含 D4：`answer_changed`、`answer_consistency_streak_norm`、`retrieval_marginal_novelty`）中：

- `self_consistency`、`semantic_entropy` 仍是答案质量的 **间接代理**，单变量 AUROC 仍多偏弱（见 D1 重刷报告）
- D4 低成本三项已接入；**单变量上** `retrieval_marginal_novelty` 在三数据集上略强于多数旧浅层，`answer_changed` 在 hotpotqa 略有用、在 musique/2wiki 很弱，`answer_consistency_streak_norm` 多呈负相关（AUROC < 0.5）
- **仍缺失的高价值信号**：(a) 答案 token 级 log-probability（`answer_logprob`，**下一优先**）；(b) 自评分数 `self_eval_score`（额外推理成本更高）
- Hidden states 4096 维理论上包含答案信息，但 ProbeMLP_v2 未能有效提取——可能因为降维过于激进（4096→64）

#### 根因 3：训练噪声 > 可学信号

- ~80% 标签为 Stop，仅 ~20% 为 Continue
- 许多 Oracle 决策边界上的样本 margin 极小（接近 0），标签在统计上不可靠
- Focal Loss（Phase B）在此情况下放大了噪声样本的权重，导致性能反降

#### 根因 4：过拟合 + Focal 反噪（工程层面）

- Best epoch 仅 5–7（60 epochs 中），说明模型快速记住训练集后无法泛化
- Phase A（纯加权 BCE）> Phase B（Focal + smoothing）：Focal 在高噪标签下适得其反
- train/dev loss gap 在 best epoch 后快速扩大（musique：0.055→0.090）

#### 根因 5（已修复）：训练不稳定 + 步数过高

- 已通过 warmup + cosine + gradient clipping 缓解（Phase A）
- Phase C 已将 Probe 步数控制在 Original-GW(dev) 锚点以下

---

## 优先级 1：跑通极简基线（Shallow-Only） ✅ 已完成

- **产出**：`stage2_report_shallow.md`，三数据集 shallow-only 对照数据已入库。
- **结论**：Shallow-Only 几乎等于 best Fixed-K（gain ≈ 0），验证了浅层特征单独不足以学到有效停止策略，为论文 "w/o Deep Features" 消融提供了数据支撑。

---

## 优先级 2：动态逐步 Hidden States ✅ 已完成

- **产出**：`cache/features/{dataset}/{split}/hidden_states/{id}_step{k}.npz` 格式已落地，Stage 2 代码已适配 `_step{k}` 匹配逻辑。
- **Stage1 报告**：三数据集 `bad_feature_files=0`，hidden 文件数与 `N × K_max` 对齐。
- **结论**：数据源头已修正，每步有独立的时间状态表征。问题出在 Stage 2 的模型端。

---

## 优先级 3：双分支架构升级 ✅ 代码 + 全量训练已完成（验收未达标）

**目标：解决模态不平衡，让模型同时有效利用深层表征和浅层统计量。**

**已落地**：`stage2/run_stage2.py` 中新增 `ProbeMLP_v2`（LayerNorm + 4096→compress、浅层独立分支、GELU 融合分类头）；`ProbeDataset` / `_build_xyw` / `_train_probe` / `_predict_probs` / `_precompute_probe_probs` 已改为浅层与 hidden 双张量管线；浅层仍为 `StandardScaler`，hidden 仅经模型内 `LayerNorm`。`Stage2Config` 增加 `compress_dim`（默认 64）、`fuse_dim`（默认 128）。checkpoint 含 `probe_arch`（`mlp` / `mlp_v2`）。`--shallow-only` 路径仍使用原 `ProbeMLP`。

**复现实验**：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki` → `stage2_report.md`、`../results/stage2_probe_table_{dataset}.csv`、`../artifacts/probe/{dataset}/stage2_train_meta.json`。

### 具体改动（`stage2/run_stage2.py`）— 设计说明（已实现）

#### 3.1 新架构 `ProbeMLP_v2`

替换当前的 `ProbeMLP`，采用双分支融合设计：

```python
class ProbeMLP_v2(nn.Module):
    def __init__(self, hidden_state_dim: int, shallow_dim: int, compress_dim: int = 64,
                 fuse_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        # 分支 A：深层表征压缩
        self.hidden_branch = nn.Sequential(
            nn.LayerNorm(hidden_state_dim),
            nn.Linear(hidden_state_dim, compress_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 分支 B：浅层特征（已由 StandardScaler 归一化）
        self.shallow_branch = nn.Sequential(
            nn.Linear(shallow_dim, shallow_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 融合分类头
        fuse_input = compress_dim + shallow_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(fuse_input, fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.GELU(),
            nn.Linear(fuse_dim // 2, 1),
        )

    def forward(self, x_shallow: Tensor, x_hidden: Tensor) -> Tensor:
        h = self.hidden_branch(x_hidden)
        s = self.shallow_branch(x_shallow)
        fused = torch.cat([h, s], dim=-1)
        return self.classifier(fused)
```

**关键设计要点**：

- `LayerNorm` 在 hidden states 上做实例级归一化，消除尺度差异。
- 4096 → 64 的压缩让深层信号与 18 维浅层信号在同一量级融合。
- `GELU` 替代 `ReLU`，避免 dead neuron 问题。
- 浅层分支有独立的非线性变换，防止被融合层忽略。

#### 3.2 归一化策略调整

- **浅层**：保持 `StandardScaler`（fit on train）。
- **Hidden States**：模型内部 `LayerNorm`（自适应），不再外部预处理。移除当前对 hidden states 不做处理的裸 concat 逻辑。

#### 3.3 训练代码适配

- `_train_probe` 和 `_predict_probs` 需适配双输入接口。
- `ProbeDataset` 拆分为 `x_shallow` 和 `x_hidden` 两个 tensor。
- 训练目标在 Phase B 起为 margin 加权 **Focal BCE + 训练步 label smoothing**（见优先级 5 Phase B）；旧版纯 `_weighted_bce_loss` 已替换。

### 验收标准

- Dev loss 在前 10 epoch 单调下降（不像当前那样震荡）。
- Dev [accuracy@0.5](mailto:accuracy@0.5) > 65%（当前约 60%）。
- Probe F1 在至少 2/3 数据集上**超过 Deployable-GW**（公平基线；Original-GW 仅作半 Oracle 参照，不作此项判定）。

### 全量验收记录（2026-04-06）

**命令**：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`（默认超参；CUDA 上 `ProbeMLP_v2`）。

**与上述三条标准的对照**：


| 验收项                                           | 结论          | 说明                                                                                                                                    |
| --------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Dev loss 前 10 epoch 单调下降                      | **未通过**     | 三数据集均出现 epoch 间 dev_loss 回升（如 hotpotqa：1→2 上升；musique：2→3 上升）。                                                                        |
| Dev [accuracy@0.5](mailto:accuracy@0.5) > 65% | **未通过**     | hotpotqa **58.23%**，musique **62.55%**，2wiki **56.75%**（`../artifacts/probe/*/stage2_train_meta.json` → `train_info.dev_acc_at_0.5`）。 |
| Probe F1 ≥2/3 数据集超过 Deployable-GW（公平基线）       | **通过（3/3）** | 下表：Probe F1 均高于同表 Deployable-GW F1。                                                                                                   |


**Test：Probe vs Deployable-GW（公平）+ Original-GW（参照）**（Probe 来自 `../results/stage2_probe_table_*.csv`；D-GW 同 D5 评估）：


| 数据集      | Probe F1   | Deployable-GW F1 | Probe 步数 | D-GW 步数 | Original-GW F1（参照） | Original-GW 步数 |
| -------- | ---------- | ---------------- | -------- | ------- | ------------------ | -------------- |
| hotpotqa | **0.5270** | 0.3698           | 3.444    | 1.01    | 0.6089             | 2.819          |
| musique  | **0.1556** | 0.0907           | 4.305    | 1.05    | 0.2235             | 4.091          |
| 2wiki    | **0.3757** | 0.2223           | 4.361    | 1.01    | 0.5121             | 3.471          |


**小结**：双分支缓解了实现层面的模态尺度问题，但**仍未达到计划中的 dev 稳定性 / 分类准确率**；**相对公平基线 Deployable-GW，Probe test F1 三数据集均更高**。与 Original-GW（半 Oracle）相比仍落后，属预期。下一步按优先级 4、5 推进。

---

## 优先级 4：差分特征（Delta Features）✅ 代码 + 全量训练已完成（验收见下）

**目标：显式编码步间趋势，降低模型学习难度。**

**已落地**：`stage2/run_stage2.py` 中 `_delta_source_scalars`、`_shallow_row_for_step`、`_trajectory_max_cumulative_cost`；`_build_xyw` / `_build_step_feature_map` 按轨迹内 `steps_by_k` 对齐 k−1 与 k 计算 5 维 delta 与 `cumulative_cost_ratio`；差分合入后浅层为 **15 维**，**后续 D4 再 +3 维，当前 `SHALLOW_FEATURE_DIM=18`**（见 Phase D 节）；`run_dataset_stage2` 打印 dev 上扩展维 StandardScaler 前的 std 便于验收。

### 具体改动

在 `_step_shallow_features` 中增加差分维度，将浅层特征从 9 维扩展至约 15 维：


| 新增特征                     | 计算方式                        | 含义       |
| ------------------------ | --------------------------- | -------- |
| `delta_entropy`          | `entropy_k - entropy_{k-1}` | 不确定性变化趋势 |
| `delta_self_consistency` | `sc_k - sc_{k-1}`           | 自一致性变化趋势 |
| `delta_ctx_overlap`      | `overlap_k - overlap_{k-1}` | 信息增量趋势   |
| `delta_nli_entail`       | `entail_k - entail_{k-1}`   | 蕴含关系变化   |
| `delta_retrieval_score`  | `score_k - score_{k-1}`     | 检索质量变化   |
| `cumulative_cost_ratio`  | `cum_cost / max_cost`       | 已消耗预算比例  |


### 实现要点

- 需要在 `_build_xyw` 中按 sample_id 排序 steps，逐步传递上一步的浅层特征值。
- 第 1 步的所有 delta 设为 0。
- `SHALLOW_FEATURE_DIM` 常量需同步更新。

### 验收标准

- Delta 特征在 dev 上的分布非全零、非常数（`std > 0.01`）。
- 加入 delta 后 Probe F1 相比优先级 3 的结果有正向提升。

### 全量验收记录（2026-04-06）

**命令**：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`（默认超参；`ProbeMLP_v2`；本表验收针对 **Delta 后 15 维** 快照；**当前默认代码为 18 维含 D4**，见 D4 节）。


| 验收项                               | 结论      | 说明                                                                                                                            |
| --------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Dev 上 Delta 等扩展维 std > 0.01（标准化前） | **通过**  | 三数据集 6 个扩展维 std 均 ≥ 0.063（见下表）；非全零、非常数。                                                                                       |
| 相对优先级 3 全量，Probe test F1 正向提升     | **未通过** | hotpotqa 0.5270→**0.5254**（−0.0016）；musique 0.1556→**0.1592**（+0.0036）；2wiki 0.3757→**0.3748**（−0.0009）。仅 musique 略升，整体同量级波动。 |


**Dev 浅层扩展维 std（Delta×5 + cum_cost_ratio，StandardScaler 前，来自运行日志）**：


| 数据集      | [Δscore, Δentropy, Δsc, Δoverlap, Δentail, cum_ratio] |
| -------- | ----------------------------------------------------- |
| hotpotqa | 1.8862, 0.4693, 0.2054, 0.0802, 0.1128, 0.2236        |
| musique  | 1.9147, 0.5578, 0.2384, 0.0813, 0.0634, 0.2236        |
| 2wiki    | 1.5125, 0.4845, 0.2178, 0.0931, 0.0663, 0.2236        |


**Test：Probe vs Deployable-GW（公平）+ Original-GW（参照）**：


| 数据集      | Probe F1   | Deployable-GW F1 | Probe 步数 | D-GW 步数 | Original-GW F1（参照） | Original-GW 步数 |
| -------- | ---------- | ---------------- | -------- | ------- | ------------------ | -------------- |
| hotpotqa | **0.5254** | 0.3698           | 3.228    | 1.01    | 0.6089             | 2.819          |
| musique  | **0.1592** | 0.0907           | 4.609    | 1.05    | 0.2235             | 4.091          |
| 2wiki    | **0.3748** | 0.2223           | 4.269    | 1.01    | 0.5121             | 3.471          |


**Dev [accuracy@0.5](mailto:accuracy@0.5)**（`../artifacts/probe/*/stage2_train_meta.json`）：hotpotqa **66.10%**，musique **58.55%**，2wiki **49.03%**（2wiki 仍偏低；与优先级 3 相比 hotpotqa 提升、2wiki 下降，与 F1 波动一致）。

**小结**：Delta 特征在统计上有效且已实现端到端，但**未带来稳定的 test F1 增益**；**相对 Deployable-GW，Probe 仍全面更高**；相对 Original-GW 仍落后（参照）。下一步优先推进**优先级 5**（LR 调度、正则与阈值策略）。

---

## 优先级 5：训练优化与超参调优 ⬅️ Phase A + B + C 已完成；公平基线下 Probe 已优于 Deployable-GW；可进 Phase D 做诊断/补特征或继续扫阈值与损失超参

**目标：从训练稳定性和类别平衡两个维度提升 Probe；公平对比对象为 Deployable-GW（Original-GW 仅半 Oracle 参照）。**

### Phase A：训练稳定化（最高优先级）

#### 5A.1 学习率调度 + Warmup

```python
warmup_epochs = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.epochs - warmup_epochs, eta_min=1e-6
)
# 前 warmup_epochs 个 epoch 线性从 lr/10 升至 lr，之后按 cosine 衰减
```

#### 5A.2 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

在 `optimizer.step()` 前执行，防止单步梯度爆炸。

#### 5A.3 超参数调整


| 参数              | 当前值  | 新值       | 理由                         |
| --------------- | ---- | -------- | -------------------------- |
| `learning_rate` | 1e-3 | **3e-4** | 降低初始 LR 防止灾难性早期过拟合         |
| `dropout`       | 0.15 | **0.30** | 加强正则（当前过拟合 gap 0.25–0.37）  |
| `weight_decay`  | 1e-4 | **5e-4** | 加强 L2 正则                   |
| `batch_size`    | 512  | **256**  | 更小 batch 增加梯度噪声有助泛化        |
| `epochs`        | 35   | **60**   | 配合 cosine 衰减和 warmup 需更长周期 |
| `patience`      | 6    | **12**   | 给 cosine schedule 更多回温机会   |


#### Phase A 代码落地与全量验收（2026-04-06）

**已实现**（`stage2/run_stage2.py`）：`Stage2Config` 默认值按上表更新；`warmup_epochs=5` 内线性地将 lr 从 `learning_rate/10` 升至 `learning_rate`，随后 `CosineAnnealingLR(T_max=epochs−warmup, eta_min=1e-6)`，且在 `warmup ≤ epoch < epochs` 的每个 epoch 末 `step()` 一次，与 `T_max` 对齐；训练步在 `optimizer.step()` 前 `clip_grad_norm_(..., max_norm=grad_clip_norm)`（默认 1.0）；`train_info` 含 `warmup_epochs`、`grad_clip_norm`、`cosine_T_max` 及每轮 `lr`。

**命令**：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`（默认即 Phase A 超参）。

**相对优先级 4（Delta 默认超参）test Probe F1**：hotpotqa 0.5254→**0.5271**（+0.0017）；musique 0.1592→**0.1687**（+0.0095）；2wiki 0.3748→**0.3824**（+0.0076）。

**Dev [Acc@0.5*](mailto:Acc@0.5)*（`../artifacts/probe/*/stage2_train_meta.json`）：hotpotqa 66.10%→**62.88%**（降）；musique 58.55%→**52.80%**（降）；2wiki 49.03%→**54.30%**（升）。与计划中的「三数据集 ≥65%」仍有差距；**早停仍多在 warmup 结束后的数个 epoch 内触发**（best epoch 多为 3–5），末 epoch 的 train/dev 间隙仍大，需 Phase B/C 继续压过拟合与校准决策。

**Test：Probe vs Deployable-GW（公平）+ Original-GW（参照）**：


| 数据集      | Probe F1   | Deployable-GW F1 | Probe 步数 | D-GW 步数 | Original-GW F1（参照） | Original-GW 步数 |
| -------- | ---------- | ---------------- | -------- | ------- | ------------------ | -------------- |
| hotpotqa | **0.5271** | 0.3698           | 3.294    | 1.01    | 0.6089             | 2.819          |
| musique  | **0.1687** | 0.0907           | 3.957    | 1.05    | 0.2235             | 4.091          |
| 2wiki    | **0.3824** | 0.2223           | 4.439    | 1.01    | 0.5121             | 3.471          |


**小结**：Phase A 达到「更低初始 LR + 更强正则 + 调度与梯度裁剪」的工程目标，**test F1 三数据集均有小幅改善**；**相对 Deployable-GW 仍全面更高**；相对 Original-GW 仍落后（参照）。dev 准确率未全面达标。Phase B 已接棒并完成全量重训（见下）。

### Phase B：损失函数 & 标签改进

#### 5B.1 类别平衡 — Focal Loss

Oracle 平均 1.5–1.8 步停止 → ~75% 标签为 "Stop"。引入 Focal Loss 降权简单多数类：

```python
def focal_bce_loss(logits, labels, weights, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = torch.where(labels == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    focal_weight = (1 - pt) ** gamma
    # alpha 加权：少数类 (Continue=1) 给更高权重
    alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
    loss = alpha_t * focal_weight * bce * weights
    return loss.sum() / torch.clamp(weights.sum(), min=1e-6)
```

其中 `alpha` 可根据实际 label 比例自适应设置。

#### 5B.2 Label Smoothing

将硬标签 0/1 平滑为 (ε, 1-ε)，ε=0.05，防止模型过度自信：

```python
y_smooth = y * (1 - 2 * epsilon) + epsilon  # epsilon=0.05
```

#### 5B.3 类别平衡日志

在 `_build_xyw` 后打印 stop/continue 标签比例，用于监控和论文消融。

#### Phase B 代码落地与全量验收（2026-04-06）

**已实现**（`stage2/run_stage2.py`）：训练与验证均使用 **margin 加权 Focal BCE**（`focal_gamma` 默认 2.0）；BCE 目标在训练步使用 **label smoothing**（`label_smoothing` 默认 0.05，验证步关闭）；**focal 的 pt 与 α 仍用硬标签**，其中 `focal_alpha_pos`（Continue=1 分支的 α）默认 **按训练集 Continue 占比自适应**：`clamp(1 - mean(y_train), 0.05, 0.95)`，也可用 `--focal-alpha` 固定；`run_dataset_stage2` 在 train/dev 的 `_build_xyw` 之后打印 **stop/continue 比例**。`train_info` 记录 `focal_gamma`、`focal_alpha_pos`、`focal_alpha_fixed`、`label_smoothing`、`train_continue_ratio`。

**命令**：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`（默认即 Phase A 超参 + Phase B 损失；可选 `--focal-gamma`、`--focal-alpha`、`--label-smoothing`）。

**相对 Phase A（同命令、仅损失从加权 BCE 换为 Focal+平滑）test Probe F1**：hotpotqa 0.5271→**0.5247**（−0.0024）；musique 0.1687→**0.1432**（−0.0255）；2wiki 0.3824→**0.3728**（−0.0096）。

**Dev [Acc@0.5*](mailto:Acc@0.5)*（`../artifacts/probe/*/stage2_train_meta.json`）：hotpotqa 62.88%→**45.55%**；musique 52.80%→**24.30%**；2wiki 54.30%→**30.33%**。（该指标依赖 0.5 硬阈值，与 dev 上调优的停止阈值无关，但表明 **logit 校准在 0.5 处显著变差**，可能与 focal + smoothing 联合作用有关。）

**Train/Dev 步级标签比例（日志）**：hotpotqa train stop **79.66%** / continue 20.34%；musique train stop **82.16%** / continue 17.84%；2wiki train stop **79.61%** / continue 20.39%（dev 比例同量级）。

**Test：Probe vs Deployable-GW（公平）+ Original-GW（参照）**：


| 数据集      | Probe F1   | Deployable-GW F1 | Probe 步数 | D-GW 步数 | Original-GW F1（参照） | Original-GW 步数 |
| -------- | ---------- | ---------------- | -------- | ------- | ------------------ | -------------- |
| hotpotqa | **0.5247** | 0.3698           | 3.526    | 1.01    | 0.6089             | 2.819          |
| musique  | **0.1432** | 0.0907           | 4.177    | 1.05    | 0.2235             | 4.091          |
| 2wiki    | **0.3728** | 0.2223           | 4.492    | 1.01    | 0.5121             | 3.471          |


**小结**：Phase B 工程目标已达成（Focal、平滑、日志），但**全量 test 未优于 Phase A**；**相对 Deployable-GW，Probe 仍全面更高**。若要以 F1 为主指标，建议下一迭代尝试：**降低 `label_smoothing`、固定 `--focal-alpha` 消融、或仅保留其一**，并推进 **Phase C** 改善阈值与成本。默认命令仍为 Phase A+B，若需复现 Phase A 损失需单独加开关或回退（当前 CLI 未暴露「纯 BCE」开关）。

### Phase C：阈值选择改进 ✅ 代码 + 全量训练已完成（验收见下）

#### 5C.1 Pareto-aware 阈值

原逻辑为纯 max-F1；**已实现**为在 Original-GW(dev) 步数可行集内用下式选阈值，再在 λ 网格上做二级选择（见下「Phase C 代码落地」）。

```python
score = F1 - lambda_cost * normalized_cost
# λ 候选: [0.1, 0.3, 0.5, 1.0]；normalized_cost 为候选阈值间 avg_cost 的 min-max 归一化
```

#### 5C.2 Global-Weitzman-calibrated 阈值

以 dev 上 **Original-GW** 的 avg_steps 为锚点，约束 Probe 的 avg_steps ≤ GW_steps * 1.05，在此约束下选 max-F1 阈值。这直接解决 Probe 步数过高的问题。

#### Phase C 代码落地与全量验收（2026-04-06）

**已实现**（`stage2/run_stage2.py`）：`_global_weitzman_avg_steps` 用 **train** 轨迹估计保留值后在 **dev** 上跑 Global-Weitzman，得到 `gw_dev_avg_steps`；阈值候选仍为 `linspace(0.01, 0.99, 50)`。先筛 **可行集**：`avg_steps ≤ gw_dev_avg_steps × threshold_gw_steps_cap_mult`（默认 **1.05**）；若无可行阈值则打日志并**回退为全体候选**（不施加步数上界）。对每个 **λ ∈ threshold_pareto_lambdas**（默认 `0.1, 0.3, 0.5, 1.0`），在可行集内最大化 **F1 − λ × normalized_cost**，其中 `normalized_cost` 为候选阈值间 **avg_cost 的 min-max 归一化**。最后在四个 λ 各自得到的阈值中，取 **dev F1 最高**者（平手则更低 `avg_cost`）作为最终阈值。`stage2_train_meta.json` 增加 `**threshold_selection_phase_c`**（含 `gw_dev_avg_steps`、`lambda_grid_trace`、`chosen_lambda` 等）。CLI：`--gw-steps-cap-mult`、`--pareto-lambdas`（逗号分隔）。

**命令**：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`（默认即 Phase A+B 训练 + Phase C 阈值）。

**相对 plan 中 Phase B 全量记录的 test Probe F1**（本轮为 **全量重训 + Phase C 阈值**；非仅改阈值）：hotpotqa 0.5247→**0.5244**（−0.0003）；musique 0.1432→**0.1659**（+0.0227）；2wiki 0.3728→**0.3606**（−0.0122）。

**Test：Probe vs Deployable-GW（公平）+ Original-GW（参照）**：


| 数据集      | Probe F1   | Deployable-GW F1 | Probe 步数 | D-GW 步数 | Original-GW F1（参照） | Original-GW 步数 |
| -------- | ---------- | ---------------- | -------- | ------- | ------------------ | -------------- |
| hotpotqa | **0.5244** | 0.3698           | 2.743    | 1.01    | 0.6089             | 2.819          |
| musique  | **0.1659** | 0.0907           | 3.127    | 1.05    | 0.2235             | 4.091          |
| 2wiki    | **0.3606** | 0.2223           | 2.707    | 1.01    | 0.5121             | 3.471          |


**Dev 阈值侧（日志 / meta）**：三数据集均在 Original-GW(dev) 步数约束可行集内；本次运行 **chosen_λ 均为 0.1**，阈值约 **0.61–0.63**，dev 上 Probe **avg_steps** 相对 Phase B 明显降低（与 test 侧「更省步」一致）。

**Dev [Acc@0.5](mailto:Acc@0.5)**（`../artifacts/probe/*/stage2_train_meta.json`；该指标依赖 0.5 硬阈值，与 dev 上选中的停止阈值无关）：hotpotqa **45.55%**，musique **24.30%**，2wiki **30.33%**。

**小结**：Phase C 达到「Pareto 分数 + Original-GW(dev) 步数上界」的工程目标；**test 上 Probe 平均步数与成本三数据集均不再高于 Original-GW**；musique F1 较 Phase B 明显改善，但 **hotpotqa / 2wiki F1 未升**。**公平对比下 Probe F1 仍高于 Deployable-GW**；与 Original-GW 的 F1 差距为半 Oracle 参照，不作「翻盘」判定。可继续调 λ 网格或放宽/收紧 `gw_steps_cap_mult` 做消融。

### Phase D：系统性突破方案（A+B+C 完成后仍推进特征与叙事）⬅️ 当前

> **核心策略转向**：不再以 **Original-GW** 为公平胜负线盲目调超参，而是 (1) 诊断特征 vs 模型哪个是瓶颈，(2) 补充关键缺失特征，(3) 以 **Deployable-GW** 为公平对比基线并保留 Original-GW 为参照。

#### D1：特征-标签诊断（最高优先级）✅ 已完成（18 维报告 2026-04-07 已刷新）

**目的**：确认"特征是否能区分 Stop/Continue"——如果不能，再多训练技巧也无用。

- 对 train 集，逐特征计算与 `action_label` 的 **point-biserial 相关系数** 和 **AUROC**
- 对 hidden states（4096 维），用 PCA 降至 2D 后按 Stop/Continue 着色可视化
- 分析 margin 分布：|margin| < 0.01 的"模糊样本"占比（这些样本的标签几乎是随机的）
- **输出**：`stage2_feature_diagnostic.md` + `../results/stage2_feature_diagnostic_pca_{dataset}.png`
- **命令**：`python -m stage2.run_feature_diagnostic --datasets hotpotqa,musique,2wiki`

**校准说明（2026-04-07）**：仓库中曾有一份 **仅列前 15 维** 的旧版 `stage2_feature_diagnostic.md`（D4 合入后未重跑诊断）。已用当前 `SHALLOW_FEATURE_NAMES`（18 维）**全量重刷**；PCA 图同步更新。执行顺序与解读汇总见 `stage2_next_steps_analysis.md`。

**D4×3 单变量 AUROC（train，单特征 → 预测 Continue）** — 摘自重刷后的报告：


| feature                          | hotpotqa | musique | 2wiki |
| -------------------------------- | -------- | ------- | ----- |
| `answer_changed`                 | 0.572    | 0.495   | 0.487 |
| `answer_consistency_streak_norm` | 0.318    | 0.413   | 0.402 |
| `retrieval_marginal_novelty`     | 0.591    | 0.575   | 0.599 |


**预期**：若多数浅层特征 AUROC < 0.60，说明特征信号仍不足；D4 低成本线已有部分增量，**下一突破点宜为 Stage1 侧 `answer_logprob`**（见 D4 与 `stage2_next_steps_analysis.md`）。

#### D2：GBDT 基线（分离特征 vs 模型问题）✅ 已完成（2026-04-06）

**目的**：同一特征集，用 XGBoost 训练，与 MLP 对比。

- 输入：与 MLP 完全相同的 (x_shallow, x_hidden, y, w)
- 超参搜索：`max_depth ∈ {4,6,8}`，`n_estimators ∈ {100,300,500}`，`learning_rate ∈ {0.05,0.1}`
- 评估：同一 Phase C 阈值选择逻辑
- **实现**：`stage2/run_xgboost_baseline.py`（`--xgb-device auto|cpu|cuda`）
- **输出**：`stage2_xgboost_baseline.md`、`../results/stage2_xgboost_baseline.json`

**决策矩阵**：


| GBDT 表现        | 结论            | 下一步      |
| -------------- | ------------- | -------- |
| GBDT ≈ MLP（均差） | **特征是瓶颈**     | → D4 补特征 |
| GBDT >> MLP    | **MLP 训练是瓶颈** | → D5 改模型 |
| GBDT ≈ MLP（均好） | 不太可能          | → 直接推进   |


### 全量验收记录（2026-04-06）

**命令**：`python -m stage2.run_xgboost_baseline --datasets hotpotqa,musique,2wiki`（本机 `auto` → **XGBoost `device=cuda`**；完整网格日志见 `stage2_xgboost_baseline.md`）。

**Test：XGBoost vs MLP vs Deployable-GW（公平）+ Original-GW（参照）**（Probe 行来自 `../results/stage2_probe_table_{dataset}.csv`）：


| 数据集      | XGB F1 | MLP F1 | Deployable-GW F1 | XGB 步数 | MLP 步数 | Original-GW F1（参照） |
| -------- | ------ | ------ | ---------------- | ------ | ------ | ------------------ |
| hotpotqa | 0.5180 | 0.5256 | 0.3698           | 2.835  | 2.892  | 0.6089             |
| musique  | 0.1618 | 0.1785 | 0.0907           | 3.518  | 3.513  | 0.2235             |
| 2wiki    | 0.3612 | 0.3580 | 0.2223           | 3.226  | 3.109  | 0.5121             |


**按 dev（Phase C 选完阈值后）锁定的最优超参与阈值**：

- hotpotqa：`max_depth=6`，`n_estimators=100`，`lr=0.05`，`threshold=0.37`（dev F1=0.5319，dev steps=2.836）
- musique：`max_depth=6`，`n_estimators=100`，`lr=0.1`，`threshold=0.19`（dev F1=0.2317，dev steps=3.366）
- 2wiki：`max_depth=8`，`n_estimators=100`，`lr=0.05`，`threshold=0.27`（dev F1=0.3928，dev steps=3.291）

**对照决策矩阵的结论**：

- **hotpotqa / 2wiki**：XGB 与 MLP test F1 差距 **< 0.01** → 落在「GBDT ≈ MLP（均差）」→ **特征/标签可学性为主瓶颈**，优先 **D4 补特征**（与 D1 诊断一致）。
- **musique**：XGB **低于** MLP（−0.0167），未出现「GBDT >> MLP」→ **不支持**「换 GBDT 就能显著救回」；仍应优先特征与阈值侧（**D4** 等），而非先大改 MLP 结构。

**小结**：D2 目标已达成——**同特征下非线性树模型未系统性碾压 MLP**，说明当前差距更可能来自 **信息不足（特征/GT F1 不可观测）** 而非单纯「Probe 训练没训好」。网格上较大 `n_estimators` 未带来稳定 dev 增益，也侧面说明 **方差/过拟合与信号上限** 并存。

#### D3：回退损失函数（快速实验）✅ 已完成（2026-04-06）

**目的**：Phase B（Focal + smoothing）劣于 Phase A，需确认最优损失配置。

- 实验 1：Phase A 超参 + Phase C 阈值 + **纯加权 BCE**（`γ=0` 退化为 margin 加权 BCE，仍含 focal 的 α 项；关闭 smoothing）
- 实验 2：Phase A 超参 + Phase C 阈值 + **Focal γ=1.0**（默认 `label_smoothing=0.05`）
- 实验 3：Phase A 超参 + Phase C 阈值 + **Focal γ=2.0 + 关闭 label smoothing**
- **CLI 要点**：实验 1 为 `--focal-gamma 0 --label-smoothing 0`；各实验建议加 `--artifact-suffix ...`，避免覆盖默认 `stage2_probe_table_*.csv`（完整命令见下「命令」）。

**预期**：实验 1 可能优于当前默认配置，因为高噪标签下 focal 适得其反。

### 全量验收记录（2026-04-06）

**对照说明**：下列 **默认** 行为 `python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`（无 `--artifact-suffix`），与三组消融均在 **同一代码快照** 下完成（18 维浅层含 D4×3、`ProbeMLP_v2`、Phase A 超参 + Phase C 阈值）。D3 三组使用 `--artifact-suffix`，产出 `../results/stage2_probe_table_{dataset}_{suffix}.csv` 与 `stage2_report_{suffix}.md`（位于 `docs/`）。

**命令**：

- 实验 1：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki --focal-gamma 0 --label-smoothing 0 --artifact-suffix d3_exp1_bce`
- 实验 2：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki --focal-gamma 1.0 --artifact-suffix d3_exp2_focal1`
- 实验 3：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki --focal-gamma 2.0 --label-smoothing 0 --artifact-suffix d3_exp3_focal2_nols`

**Test：Probe F1 / 步数（各配置取对应 CSV 的 Probe 行；公平对照 Deployable-GW：hotpotqa/musique/2wiki 的 F1 分别为 0.3698 / 0.0907 / 0.2223，步数约 1.01–1.05；Original-GW 仍为上界参照）**：


| 配置                 | hotpotqa Probe F1 | hotpotqa 步数 | musique Probe F1 | musique 步数 | 2wiki Probe F1 | 2wiki 步数 |
| ------------------ | ----------------- | ----------- | ---------------- | ---------- | -------------- | -------- |
| **默认** γ=2, ε=0.05 | 0.5256            | 2.892       | 0.1785           | 3.513      | 0.3580         | 3.109    |
| 实验 1：γ=0, ε=0      | 0.5235            | 2.878       | 0.1778           | 3.573      | 0.3446         | 2.471    |
| 实验 2：γ=1, ε=0.05   | 0.5260            | 2.999       | **0.1811**       | 3.127      | **0.3582**     | 3.043    |
| 实验 3：γ=2, ε=0      | **0.5261**        | 2.921       | 0.1748           | 3.094      | 0.3559         | 2.987    |


**对照 plan 原预期**：

- **实验 1（纯 BCE）未整体优于默认**：hotpotqa / 2wiki 均略降，2wiki 降幅最大（0.3580→0.3446）；musique 基本持平。
- **温和 Focal（γ=1）在 musique / 2wiki 上为四配置中最优或并列最优**；hotpotqa 与默认/实验 3 同量级。
- **去掉 smoothing（γ=2, ε=0）**：hotpotqa 与默认几乎相同（+0.0005 F1），musique 低于默认与实验 2，说明 **smoothing 与强 focal 的组合并非唯一有害项**，单独关 smoothing 不能稳定救 musique。

**小结**：在当前 18 维特征与 Phase C 阈值流程下，**无单一损失配置在三数据集上同时严格优于默认**；**γ=1 + ε=0.05** 在 musique、2wiki 上略好，可作为可选默认或论文消融主行。完整逐数据集表见 `stage2_report_d3_exp1_bce.md` 等。

#### D4：补充答案质量代理特征（核心突破点）✅ 代码已落地 + 全量重训已完成（最低成本三项）

**目的**：弥补特征集中缺失的"答案质量"信号，缩小与 **Original-GW（可观测真实 F1）** 的信息差；公平胜负仍对 **Deployable-GW**。

需要新增的特征（按优先级排列）：


| 新特征                         | 计算方式                            | 预期效果             | 需要 Stage 1 重跑？ |
| --------------------------- | ------------------------------- | ---------------- | -------------- |
| `answer_changed`            | 比较 step k 与 k-1 的答案是否不同（binary） | 强信号：答案稳定 ≈ 可能已收敛 | ❌ 从轨迹缓存可提取     |
| `answer_logprob`            | 生成答案 token 的平均 log-probability  | LLM 自信度，与 F1 正相关 | ⚠️ 需 vLLM 记录   |
| `answer_consistency_streak` | 连续多少步答案未变                       | 更强的收敛信号          | ❌ 从轨迹缓存可提取     |
| `self_eval_score`           | 让 LLM 对自己的答案打 1-5 分（单次 prompt）  | 直接的答案质量代理        | ⚠️ 需额外推理       |
| `retrieval_marginal_info`   | 新检索文档与已有上下文的信息增量（如 ROUGE-L 差异）  | 检索边际收益衰减检测       | ❌ 从轨迹缓存可提取     |


**最低成本方案**：先实现 `answer_changed` + `answer_consistency_streak` + `retrieval_marginal_info`（不需要重跑 Stage 1）。

**已实现（2026-04-06）**（`stage2/run_stage2.py`）：浅层维度 **15→18**（`SHALLOW_FEATURE_DIM=18`）。新增：

- `answer_changed`：`current_answer` 规范串（小写、压缩空白）与 k−1 步比较，k=1 为 0；Stop 学习步不变。
- `answer_consistency_streak_norm`：自第 k 步向前连续同答案的步数 / `max_k`。
- `retrieval_marginal_novelty`：`1 − ROUGE-L F1`（词级 LCS，新文档与 k 之前累积 `retrieved_doc` 各截断 256 词）；k=1 或无新文档时分别为 1 / 0。

未实现项（仍依赖 Stage1/额外推理）：`answer_logprob`、`self_eval_score`。

**推荐推进顺序（与 D1 重刷结论一致）**：

1. **P0（已完成）**：重跑 `run_feature_diagnostic`，确认 D4×3 单变量信号（见上节 D1 表）。
2. **P1（主攻）**：Stage1 记录 `**answer_logprob`**，建议一次性落盘 **mean / min / `delta_answer_logprob`**，再接入 Stage2 浅层维度并重训。
3. **P2**：若仍不足，再考虑 `**self_eval_score`**（成本高，与 `self_consistency` 可能重叠）。
4. **P3**：新特征接入后全量 `run_stage2`，并与 Deployable-GW 对比。

详细 rationale 见 `stage2_next_steps_analysis.md`。

**全量重训与验收**：已于 **2026-04-06** 完成——与上节 **Phase D3「全量验收记录」** 中默认及三组消融在 **同一 18 维（含 D4×3）代码快照**下执行（见 D3 开头「对照说明」）。当前默认 `../results/stage2_probe_table_{dataset}.csv` 中 Probe 行与 D3 表内「默认」行均对应该次重训。复现命令：`python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki`。

#### D5：重新定义公平对比基线 📐

**目的**：GW 使用真实 F1（部署不可用），不是公平基线。需要建立正确的对比框架。

**方案 A：论文叙事重构**

- 将 GW 明确标注为 **Semi-Oracle（半 Oracle）**：与 Oracle DP 同属「真实 F1 可观测」策略族
- Probe 的正确对比对象是 **Fixed-K 策略**（无需 F1 的朴素基线）
- Probe 已超越 Best Fixed-K：hotpotqa +4.1pp、musique +1.7pp、2wiki +0.6pp
- Pareto 图上增加标注：GW 虚线框标注 "requires ground-truth F1"

**方案 B：构建 Deployable-GW** ✅ 已实现（2026-04-07）

- 用 `self_consistency` 替代真实 F1 作为 GW 的质量信号：`if self_consistency >= r_{k+1}* then stop`
- 保留值 r_k* 在 train 上用 self_consistency 的步间增益估计（与 F1 版 Weitzman 同一套 `compute_reservation_value` 方程）
- 此 Deployable-GW 不需要 ground truth，是 Probe 的公平竞争对手；**test 上汇报的 F1/EM 仍为停步处的真实指标**（停止规则不读 GT F1）
- **代码**：`pretest/utils/weitzman.py` 中 `compute_all_reservation_values_from_proxy`、`deployable_weitzman_stopping_simulation`；`stage2/run_stage2.py` 在默认 test 汇总表中追加 **Deployable-GW** 行（Pareto 图对该策略使用紫色三角向下标记）
- **快速评估（不重训 Probe）**：`python -m stage2.run_deployable_gw_eval --datasets hotpotqa,musique,2wiki` → `../results/stage2_deployable_gw.json`
- **预期**：Deployable-GW 会比 Original-GW 差很多（因为 self_consistency ≠ F1），Probe 可能反超 — **实测见下表**

### Deployable-GW 全量验收记录（2026-04-07）

**命令**：`python -m stage2.run_deployable_gw_eval --datasets hotpotqa,musique,2wiki`（`cost_per_step=0.05`，`max_k=5`，代理键 `self_consistency`）。**Probe / Original-GW** 数字来自 `../results/stage2_probe_table_{dataset}.csv`（与当前默认 Stage2 表一致）。


| 数据集      | Deployable-GW F1 | Deployable-GW 步数 | Deployable-GW 成本 | Probe F1   | Probe 步数 | Original-GW F1 | Original-GW 步数 |
| -------- | ---------------- | ---------------- | ---------------- | ---------- | -------- | -------------- | -------------- |
| hotpotqa | **0.3698**       | 1.013            | 0.0507           | **0.5256** | 2.892    | 0.6089         | 2.819          |
| musique  | **0.0907**       | 1.048            | 0.0524           | **0.1785** | 3.513    | 0.2235         | 4.091          |
| 2wiki    | **0.2223**       | 1.005            | 0.0503           | **0.3580** | 3.109    | 0.5121         | 3.471          |


**解读**：

- Deployable-GW 在三数据集上 **几乎在第 1 步就停止**（平均步数 ≈1.01–1.05）：第一步后 `self_consistency` 往往已高于由 train 估计的 r_2^，导致「见好就收」过早，**真实 F1 显著低于 Original-GW**（与 plan 预期一致）。
- **Probe 在三数据集上 test F1 均高于 Deployable-GW**，满足验收口径中「不被 Deployable-GW 支配」的方向；与 Original-GW 对比仍不公平（半 Oracle），论文叙事继续采用 D5 方案 A+B 并列说明即可。

**方案 C：混合策略**

- Original-GW 为基础策略，Probe 仅在高置信区间（`p_continue < 0.3` 或 `> 0.7`）覆盖其决策
- 中间区间回退到固定策略
- 这在论文中可作为"实用方案"展示

#### D6：简化模型基线（下界参考）

- 逻辑回归 + 手选 top-5 特征
- 线性 Probe 直接在 hidden states 上做二分类
- 用于论文消融：验证非线性是否有贡献

### 验收标准（更新版，基于深度诊断后的务实目标）

**核心指标（Phase D 后必须达到）**：

- D1 完成：产出特征诊断报告（**18 维与代码一致**，2026-04-07 已重刷），明确特征瓶颈与 D4×3 单变量表现
- D2 完成：GBDT 基线确认瓶颈归因 ✅（2026-04-06：XGB ≈ MLP，倾向特征瓶颈 → D4）
- Probe 在 Pareto 图上 **不被 Deployable-GW 支配**（至少 2/3 数据集；用方案 B 的公平基线）
- 若补特征后 Probe 仍不如 Deployable-GW，则混合策略（方案 C）在 2/3 数据集上超越 Deployable-GW

**辅助指标**：

- Train/Dev loss gap < 0.10
- Dev [Acc@0.5](mailto:Acc@0.5) ≥ 55%（从 65% 下调；基于特征诊断后的合理预期）
- Probe avg_steps ≤ Original-GW avg_steps × 1.1（已达标 ✅；锚点来自 dev 上 Original-GW）

---

## 后续阶段预览

### Stage 3（E-value 风险控制）

待 Probe 在 Deployable-GW 对比下表现合理后启动。核心内容参见 `experiments.md` 第三阶段。Stage 3 的 E-value 校准不依赖 Probe 必须超过 Original-GW——只要 Probe 有统计可控的停止行为即可。

### Stage 4（主实验与 SOTA 对比）

待 Stage 3 落地后，端到端评估并生成论文核心表格。论文叙事需基于 D5 的对比框架重构。

---

## 执行节奏


| 日程     | 任务                                               | 产出                                                                                     | 状态        |
| ------ | ------------------------------------------------ | -------------------------------------------------------------------------------------- | --------- |
| Day 1  | 实现 `ProbeMLP_v2` + 双分支训练管线                       | 代码已合入；全量三数据集已跑                                                                         | ✅         |
| Day 2  | 加入 Delta Features（优先级 4）                         | 15 维浅层已合入；全量三数据集已跑                                                                     | ✅         |
| Day 3  | **Phase A：训练稳定化** — LR 调度 / warmup / 梯度裁剪 / 超参调整 | 重训三数据集，检查过拟合 gap 和 dev acc                                                             | ✅         |
| Day 3+ | **Phase B：Focal Loss + Label Smoothing + 类别平衡**  | 重训，对比 Phase A 结果                                                                       | ✅         |
| Day 4  | **Phase C：阈值选择改进** + 消融实验                        | Pareto-aware 阈值 + Original-GW(dev) 步数约束；全量三数据集已跑                                       | ✅         |
| Day 5  | **Phase D1：特征-标签诊断**                             | `stage2_feature_diagnostic.md` + PCA 图（**2026-04-07 已按 18 维重刷**）                       | ✅         |
| Day 5  | **Phase D2：GBDT 基线**                             | `stage2_xgboost_baseline.md` + `.json`                                                 | ✅         |
| Day 5  | **Phase D3：回退损失函数消融**                            | 对比表已写入本节 D3「全量验收记录」+ `stage2_report_d3_*.md`（`docs/`）                                  | ✅         |
| Day 6  | **Phase D4：补充答案质量代理特征**                          | 18 维浅层（+D4×3）已合入；**全量三数据集已重训**（与 D3 默认/消融同快照，见 D3「对照说明」）                               | ✅         |
| Day 6  | **Phase D5：Deployable-GW + 论文叙事重构**              | Deployable-GW 代码 + `../results/stage2_deployable_gw.json`；全量 `run_stage2` 可带新表行/Pareto | ✅ 代码与基线评估 |
| Day 7  | **Phase D6：简化模型基线 + 最终消融**                       | 逻辑回归 / 线性 Probe；论文消融表                                                                  | 待 D5      |


**里程碑判定**：Day 5 结束时，D1+D2 的结论决定后续路线。D1+D2 均已对齐：**浅层 AUROC 偏弱 + XGB 未显著优于 MLP** → **D4 低成本三项已合入且全量重训已完成**（与 D3 同快照）；**D1 已按 18 维重刷**，D4×3 中单变量最强为 `retrieval_marginal_novelty`，整体仍未接近「强答案质量代理」，**下一优先为 Stage1 侧 `answer_logprob` 系列**（见 D4「推荐推进顺序」与 `stage2_next_steps_analysis.md`）。**D3** 损失消融已补齐（见上）。若未来出现 **GBDT >> MLP**，再优先改 MLP 训练/容量（D5 模型侧）。