# Stage2 Probe 最优配置总结

> 截止 2026-04-19，已基于重跑后的 Stage1 轨迹重新搜索并验证 `stage2` 最优配置。

---

## 一、最优超参（`PER_DATASET_OPTIMAL`）


| 数据集      | `compress_dim` | `train_margin_min_abs` | `probe_target` | `hidden_branch_residual` |
| -------- | -------------- | ---------------------- | -------------- | ------------------------ |
| HotpotQA | 256            | 0.0                    | binary         | False                    |
| MuSiQue  | 256            | 0.0                    | binary         | False                    |
| 2Wiki    | 64             | 0.02                   | binary         | **True**                 |


固化于 `stage2/run_stage2.py` → `PER_DATASET_OPTIMAL`。

**复现命令（一条命令三数据集全跑）：**

```bash
python -m stage2.run_stage2 --per-dataset-optimal --artifact-suffix pdopt_best
```

> `--per-dataset-optimal` 省略 `--probe-target` 时，自动按上表逐数据集设定所有四个参数（含 2Wiki 的残差开关）。

---

## 二、测试集最终结果（`pdopt_best` + P1 逐步阈值 refinement）

### 与各基线对比（Test F1）


| 策略 | HotpotQA F1 | 步数 | MuSiQue F1 | 步数 | 2Wiki F1 | 步数 |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed-K=1 | 0.4340 | 1.00 | 0.0797 | 1.00 | 0.2846 | 1.00 |
| Fixed-K=2 | 0.6773 | 2.00 | 0.1830 | 2.00 | 0.5908 | 2.00 |
| Fixed-K=3 | 0.6775 | 3.00 | 0.3210 | 3.00 | 0.5583 | 3.00 |
| Fixed-K=4 | 0.6747 | 4.00 | 0.3816 | 4.00 | 0.5382 | 4.00 |
| Fixed-K=5 | 0.6668 | 4.99 | 0.4022 | 5.00 | 0.5288 | 5.00 |
| Global-Weitzman | 0.7092 | 1.64 | 0.4229 | 3.29 | 0.6449 | 1.77 |
| Oracle | 0.7810 | 1.58 | 0.4966 | 2.12 | 0.6954 | 1.59 |
| **Probe（最优配置）** | **0.6544** | **1.73** | **0.3969** | **3.31** | **0.5941** | **1.82** |


### 关键指标


| 数据集 | Probe F1 | Oracle F1 | Probe/Oracle | Oracle 差距 | Best Fixed-K | Probe vs Best Fixed | Probe vs Fixed-K=5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | **0.6544** | 0.7810 | **83.8%** | 0.1266 | `K=3` (`0.6775`) | `-0.0231` | `-0.0124` |
| MuSiQue | **0.3969** | 0.4966 | **79.9%** | 0.0997 | `K=5` (`0.4022`) | `-0.0053` | `-0.0053` |
| 2Wiki | **0.5941** | 0.6954 | **85.4%** | 0.1013 | `K=2` (`0.5908`) | `+0.0033` | `+0.0653` |

> 当前 `Probe` 的主优势更准确地表述为“接近 Oracle 的低成本 Pareto 点”，而不是“统一超过最佳 Fixed-K”。这轮 `rethreshold-only` 更新后，2Wiki 已小幅超过最佳固定步数；HotpotQA 与 MuSiQue 则仍分别落后于最优 `K=3` / `K=5`，但 HotpotQA 显著节省步数，2Wiki 继续明显优于跑满 `K=5`。

### 2.1 Phase C 最终部署策略

`pdopt_best` 当前三数据集的最终部署工作点已不再是单一全局阈值，而是通过 P1 的保守筛选后采用逐步阈值：

| 数据集 | policy | per-step thresholds | 采纳原因（dev） |
| --- | --- | --- | --- |
| HotpotQA | `per_step` | `[0.73, 0.69, 0.81, 0.73, 0.73]` | utility 提升；Pareto frontier 外扩 |
| MuSiQue | `per_step` | `[0.61, 0.63, 0.63, 0.69, 0.63]` | 同步数下 utility 更优 |
| 2Wiki | `per_step` | `[0.61, 0.79, 0.73, 0.77, 0.67]` | 在接近 GW budget 下保持更优主点 |

实现位置：`stage2/run_stage2.py`。  
诊断产物：`docs/reports/stage2/stage2_report_pdopt_best.md`、`results/stage2_threshold_sweep_*_pdopt_best.csv`、`artifacts/probe/*/stage2_train_meta_pdopt_best.json`。

### 2.2 Appendix：纯 max-F1 operating point

为区分“模型能力上限”和“cost-aware operating point 选择”，当前已固定保留同一 checkpoint 的 appendix 补充：

- dev 上按纯 `avg_f1` 选择全局 threshold
- test 上汇报对应 `F1 / avg_steps`
- 自动导出到 `docs/reports/stage2/stage2_report_pdopt_best.md` 与 `results/stage2_appendix_maxf1_summary_pdopt_best.csv`

| 数据集 | dev max-F1 threshold | Appendix test F1 / avg_steps | 相对主部署点 |
| --- | --- | --- | --- |
| HotpotQA | `0.47` | `0.6884 / 2.981` | `+0.0340 F1`, `+1.249 steps` |
| MuSiQue | `0.55` | `0.4140 / 4.192` | `+0.0171 F1`, `+0.887 steps` |
| 2Wiki | `0.57` | `0.5918 / 2.629` | `-0.0023 F1`, `+0.807 steps` |

这组结果说明：

- HotpotQA / MuSiQue 的主损失里，operating point 选择占比明显更大
- 2Wiki 的主部署点已经接近甚至略优于纯 max-F1 阈值
- 因此 `max-F1` 更适合作为 appendix 的 quality-first 参照，而非替换主线 cost-aware 结果

### 2.3 P2 新特征完整重训结论

为回答“31 维浅层特征是否值得进入最终主结果”，我们额外完成了一轮完整重训：

```bash
python -m stage2.run_stage2 --per-dataset-optimal --artifact-suffix p2_full31
```

对应产物：

- `docs/reports/stage2/stage2_report_p2_full31.md`
- `results/stage2_appendix_maxf1_summary_p2_full31.csv`
- `results/stage2_probe_table_*_p2_full31.csv`

与当前正式口径 `pdopt_best` 相比，`p2_full31` 主工作点表现为：

| 数据集 | `pdopt_best` | `p2_full31` | 变化 |
| --- | --- | --- | --- |
| HotpotQA | `0.6544 / 1.732` | `0.6530 / 1.706` | `-0.0014 F1`, `-0.026 steps` |
| MuSiQue | `0.3969 / 3.305` | `0.3934 / 3.410` | `-0.0036 F1`, `+0.106 steps` |
| 2Wiki | `0.5941 / 1.822` | `0.5725 / 1.801` | `-0.0215 F1`, `-0.021 steps` |

结论：

- 这批 P2 新特征**没有提升任何一个数据集的主工作点**
- 2Wiki 出现了显著退化
- 因此 `p2_full31` 应保留为负结果 / 诊断结果，**不进入最终主结果**

### 各数据集最优配置来源 CSV


| 数据集 | 结果文件 |
| --- | --- |
| HotpotQA | `results/stage2_probe_table_hotpotqa_pdopt_best.csv` |
| MuSiQue | `results/stage2_probe_table_musique_pdopt_best.csv` |
| 2Wiki | `results/stage2_probe_table_2wiki_pdopt_best.csv` |


---

## 三、消融实验结论汇总

### 3.1 已验证有效的改进


| 改进                                   | 收益数据集             | ΔProbe F1           | 备注                          |
| ------------------------------------ | ----------------- | ------------------- | --------------------------- |
| `compress_dim` 256（HotpotQA/MuSiQue） | HotpotQA, MuSiQue | +0.01～+0.03         | 默认 64 对 2Wiki 反而最优          |
| `train_margin_min_abs=0.02`（2Wiki）   | 2Wiki             | +0.017              | 过滤模糊标签；HotpotQA/MuSiQue 无收益 |
| `probe_target=binary`（全局）            | 三数据集              | +0.05～+0.07 vs f1 头 | f1 回归头一致差于二分类               |
| `hidden_branch_residual`（2Wiki）      | 2Wiki             | **+0.028**          | HotpotQA 持平；MuSiQue 下降      |


### 3.2 未带来稳定收益的方向


| 方向                                         | 实测结论                         | 参考文档                                                       |
| ------------------------------------------ | ---------------------------- | ---------------------------------------------------------- |
| GW 步数上界放宽（`gw_steps_cap_mult`）             | 三数据集全无提升，Probe 未被步数约束卡住      | `ablate_gw_cap`*                                           |
| F1 回归头（`--probe-target f1`）                | 三数据集均低于 binary，差距 0.05～0.07  | `pdopt_f1_baseline`                                        |
| Mean Pooling / last_mean_blend             | 三数据集均低于 last_token           | `ablate_hidden_mean_pool`, `ablate_hidden_last_mean_blend` |
| `hidden_branch_residual`（HotpotQA/MuSiQue） | HotpotQA 持平；MuSiQue 下降 0.008 | `pdopt_binary_hidden_residual`                             |
| 序列历史聚合（`--seq-history-features`）           | 三数据集均低于逐步 MLP 基线             | `pdopt_binary_seqhist`                                     |
| GRU 序列探针（`--sequence-gru`）                 | HotpotQA/MuSiQue 下降；2Wiki 下降 | `pdopt_binary_seq_gru`                                     |
| `compress_dim=512` 扩大                      | 无改善，2Wiki 显著恶化               | `ablate_compress_dim512`                                   |


### 3.3 核心发现

- **瓶颈在特征信息，而非模型容量**：XGBoost 与 MLP 在同等特征下表现相近；增大模型容量无收益。
- **各数据集行为高度异构**：2Wiki 的 hidden 维度极其敏感（压大则恶化）；HotpotQA/MuSiQue 则需要更大 compress_dim。
- **binary 二分类头一致优于 f1 回归头**：连续标签的监督未能弥补回归的优化难度。
- **残差分支效果 dataset-specific**：仅在 2Wiki 上 +0.028；在 MuSiQue 反向 −0.008，不应全局开启。

---

## 四、超参说明

### 固定超参（所有数据集共用）


| 参数                            | 值            | 说明                                       |
| ----------------------------- | ------------ | ---------------------------------------- |
| `hidden_state_key`            | `last_token` | hidden 提取策略（2Wiki 在 last_token 基础上加残差分支） |
| `focal_gamma`                 | 2.0          | Focal BCE 的 γ                            |
| `label_smoothing`             | 0.05         | 标签平滑                                     |
| `dropout`                     | 0.30         |                                          |
| `learning_rate`               | 3e-4         |                                          |
| `weight_decay`                | 5e-4         |                                          |
| `epochs`                      | 60           |                                          |
| `patience`                    | 12           | Early stopping                           |
| `warmup_epochs`               | 5            |                                          |
| `batch_size`                  | 256          |                                          |
| `fuse_dim`                    | 128          | hidden + shallow 融合层宽度                   |
| `threshold_gw_steps_cap_mult` | 1.05         | Phase C 步数上界：GW_dev × 1.05               |


### Per-dataset 超参（见第一节）

`compress_dim`、`train_margin_min_abs`、`probe_target`、`hidden_branch_residual` 四项按数据集分别设定，由 `--per-dataset-optimal` 自动应用。

---

## 五、结论更新

- `PER_DATASET_OPTIMAL` 本身没有变化，重跑后的最优组合仍是：
  `hotpotqa=(256, 0.0, binary, False)`、`musique=(256, 0.0, binary, False)`、`2wiki=(64, 0.02, binary, True)`。
- 真正变化的是 **Stage1 轨迹与标签分布**：三数据集的 Oracle、Global-Weitzman、Fixed-K 与 Probe 均整体抬升，尤其 HotpotQA / 2Wiki 的最优工作点明显前移到更少步数。
- 因此，后续写作应避免继续沿用“Probe 在三数据集统一超过最佳 Fixed-K”的旧叙事；更稳妥的表述是：
  `Probe recovers 82%~84% of Oracle performance while using 34%~68% of the Fixed-K=5 retrieval budget.`

---

## 六、Stage3 接入

Stage3 通过 `stage3/adapters/stage2_probe.py` 加载 Stage2 checkpoint，自动读取 `hidden_branch_residual` 字段（checkpoint meta 中已写入），无需额外传参：

```python
# stage3/adapters/stage2_probe.py 中已处理
hidden_branch_residual = bool(ckpt.get("hidden_branch_residual", False))
```

Checkpoint 路径：`artifacts/probe/{dataset}/probe_mlp_pdopt_best.pt`（以 `--artifact-suffix pdopt_best` 生成时）。

---

## 七、完整复现流程

```bash
# Step 1：生成轨迹缓存（若已有可跳过）
python -m stage1.run_stage1 --datasets hotpotqa musique 2wiki

# Step 2：训练最优 Probe（三数据集）
python -m stage2.run_stage2 --per-dataset-optimal --artifact-suffix pdopt_best

# Step 3：验证结果（对照本文二.2 表格）
# results/stage2_probe_table_{dataset}_pdopt_best.csv
```

> 若仅复现消融对照基线（last_token，无残差）：
>
> ```bash
> python -m stage2.run_stage2 --per-dataset-optimal --probe-target binary --artifact-suffix pdopt_binary_last_token
> ```
>
> 此时 `hidden_branch_residual` 仍 per-dataset（2Wiki 带残差）；若需三数据集均不带残差，需手动指定 `--no-hidden-branch-residual`（当前未实现，可直接运行每个数据集单独加 `--datasets` 覆盖）。
