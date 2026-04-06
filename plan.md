# Pandora-RAG Stage 2 攻坚计划

> 最后更新：2026-04-06
> 当前状态：优先级 1、2 已完成，**Probe 全面输给 Global-Weitzman**，需要从架构和特征两个方向同时发力。

---

## 当前实验诊断（问题定位）

### Probe vs Global-Weitzman（核心矛盾）


| 数据集      | Probe F1 | GW F1      | Probe 步数 | GW 步数 | 结论                          |
| -------- | -------- | ---------- | -------- | ----- | --------------------------- |
| hotpotqa | 0.5327   | **0.6089** | 3.21     | 2.82  | Probe 多花 14% 成本，F1 低 7.6pp  |
| musique  | 0.1609   | **0.2235** | 4.13     | 4.09  | 成本持平，F1 低 6.3pp             |
| 2wiki    | 0.3731   | **0.5121** | 4.39     | 3.47  | Probe 多花 26% 成本，F1 低 13.9pp |


**Neural Probe 还不如不看任何实例特征的纯统计规则。** 模型未学到有效的停止决策边界。

### Hidden States 收益微弱


| 数据集      | Shallow-Only F1 | Full F1 | 增益     |
| -------- | --------------- | ------- | ------ |
| hotpotqa | 0.4829          | 0.5327  | +0.050 |
| musique  | 0.1394          | 0.1609  | +0.022 |
| 2wiki    | 0.3552          | 0.3731  | +0.018 |


4096 维信号存在但未被有效利用。

### 根因

1. **模态不平衡**：9 维浅层特征与 4096 维 hidden states 直接 `concat`，浅层信号被淹没。
2. **Hidden States 未归一化**：仅浅层做了 `StandardScaler`，4096 维裸输入，尺度差异巨大。
3. **网络瓶颈过窄**：`Linear(4105, 256)` 信息压缩过猛，且梯度被 hidden states 主导。
4. **模型退化**：threshold 极低（0.23~0.35），模型近乎"总是继续"，未学到有区分度的决策面。
5. **无 LR 调度**：35 epoch 固定 LR=1e-3，前期震荡后期过拟合。

---

## 优先级 1：跑通极简基线（Shallow-Only） ✅ 已完成

- **产出**：`results/stage2_report_shallow.md`，三数据集 shallow-only 对照数据已入库。
- **结论**：Shallow-Only 几乎等于 best Fixed-K（gain ≈ 0），验证了浅层特征单独不足以学到有效停止策略，为论文 "w/o Deep Features" 消融提供了数据支撑。

---

## 优先级 2：动态逐步 Hidden States ✅ 已完成

- **产出**：`cache/features/{dataset}/{split}/hidden_states/{id}_step{k}.npz` 格式已落地，Stage 2 代码已适配 `_step{k}` 匹配逻辑。
- **Stage1 报告**：三数据集 `bad_feature_files=0`，hidden 文件数与 `N × K_max` 对齐。
- **结论**：数据源头已修正，每步有独立的时间状态表征。问题出在 Stage 2 的模型端。

---

## 优先级 3：双分支架构升级 ⬅️ 当前焦点

**目标：解决模态不平衡，让模型同时有效利用深层表征和浅层统计量。**

### 具体改动（`stage2/run_stage2.py`）

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
- `_weighted_bce_loss` 不变。

### 验收标准

- Dev loss 在前 10 epoch 单调下降（不像当前那样震荡）。
- Dev [accuracy@0.5](mailto:accuracy@0.5) > 65%（当前约 60%）。
- Probe F1 在至少 2/3 数据集上**超过 Global-Weitzman**。

---

## 优先级 4：差分特征（Delta Features）

**目标：显式编码步间趋势，降低模型学习难度。**

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

---

## 优先级 5：训练优化与超参调优

**目标：稳定收敛，压榨最后的性能增益。**

### 5.1 学习率调度

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
```

每 epoch 末调用 `scheduler.step()`，配合 early stopping。

### 5.2 超参数调整


| 参数                  | 当前值  | 建议值      | 理由                       |
| ------------------- | ---- | -------- | ------------------------ |
| `dropout`           | 0.15 | **0.25** | 当前过拟合明显，需更强正则化           |
| `epochs`            | 35   | **60**   | 配合 cosine 衰减需要更长训练周期     |
| `patience`          | 6    | **10**   | 给 cosine schedule 更多回温机会 |
| `weight_decay`      | 1e-4 | **5e-4** | 加强 L2 正则                 |
| `hidden_dim` (fuse) | 256  | **128**  | 融合层不需要太宽                 |
| `learning_rate`     | 1e-3 | **3e-4** | 配合新架构降低初始 LR             |
| `batch_size`        | 512  | **256**  | 更小 batch 增加噪声有助泛化        |


### 5.3 阈值选择改进

当前 `_pick_best_threshold` 纯选 max F1。改为 **Pareto-aware 阈值选择**：在 (cost, F1) 平面上选 Pareto 最优点，或引入加权目标 `score = F1 - λ * cost`（λ 可在 dev 上交叉验证）。

### 验收标准

- Train loss 与 dev loss 的间距（过拟合 gap）< 0.05。
- Probe 在 Pareto 图上位于 Global-Weitzman 的右上方或左上方（被 Pareto 包络）。
- 三数据集 `oracle_gap` 均 < 0.08（当前 0.08~0.15）。

---

## 后续阶段预览

### Stage 3（E-value 风险控制）

待 Probe 在至少 2/3 数据集上超越 Global-Weitzman 后启动。核心内容参见 `experiments.md` 第三阶段。

### Stage 4（主实验与 SOTA 对比）

待 Stage 3 落地后，端到端评估并生成论文核心表格。

---

## 执行节奏


| 日程    | 任务                         | 产出                     |
| ----- | -------------------------- | ---------------------- |
| Day 1 | 实现 `ProbeMLP_v2` + 双分支训练管线 | 新架构代码、三数据集初步结果         |
| Day 2 | 加入 Delta Features（优先级 4）   | 扩展特征维度、对比实验            |
| Day 3 | 训练优化（优先级 5）+ 超参搜索          | 最终 Probe 模型 + Pareto 图 |
| Day 4 | 消融实验整理 + 结果可视化             | 论文用 Pareto 图、消融表       |


**里程碑判定**：若 Day 3 结束时 Probe F1 仍未超过 Global-Weitzman，则需重新审视 Oracle 标签质量（检查 `action_label` 的类别平衡、margin 分布）和轨迹数据本身的信噪比。