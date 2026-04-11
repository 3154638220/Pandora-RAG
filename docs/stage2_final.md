# Stage2 Probe 最优配置总结

> 截止 2026-04-11，所有消融实验完成后的最终定案。

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

## 二、测试集最终结果

### 与各基线对比（Test F1）


| 策略              | HotpotQA F1 | 步数       | MuSiQue F1 | 步数       | 2Wiki F1   | 步数       |
| --------------- | ----------- | -------- | ---------- | -------- | ---------- | -------- |
| Fixed-K=1       | 0.3665      | 1.0      | 0.0888     | 1.0      | 0.2248     | 1.0      |
| Fixed-K=2       | 0.4325      | 2.0      | 0.1207     | 2.0      | 0.2891     | 2.0      |
| Fixed-K=3       | 0.4579      | 3.0      | 0.1328     | 3.0      | 0.3264     | 3.0      |
| Fixed-K=5       | 0.4764      | 5.0      | 0.1328     | 5.0      | 0.3582     | 5.0      |
| Global-Weitzman | 0.6060      | 2.80     | 0.2371     | 4.06     | 0.5178     | 3.44     |
| Oracle          | 0.6232      | 1.60     | 0.2526     | 1.52     | 0.5336     | 1.77     |
| **Probe（最优配置）** | **0.5273**  | **2.85** | **0.1826** | **3.25** | **0.3738** | **3.37** |


### 关键指标


| 数据集      | Probe F1   | Oracle F1 | Probe/Oracle | 差距    | vs Fixed-K=5 |
| -------- | ---------- | --------- | ------------ | ----- | ------------ |
| HotpotQA | **0.5273** | 0.6232    | **84.6%**    | 0.096 | **+0.051**   |
| MuSiQue  | **0.1826** | 0.2526    | **72.3%**    | 0.070 | **+0.050**   |
| 2Wiki    | **0.3738** | 0.5336    | **70.1%**    | 0.160 | **+0.016**   |


> Probe 步数均显著少于 Fixed-K=5（最优固定步数），以约一半成本取得了更高 F1。

### 各数据集最优配置来源 CSV


| 数据集      | 结果文件                                                                |
| -------- | ------------------------------------------------------------------- |
| HotpotQA | `results/stage2_probe_table_hotpotqa_pdopt_binary_last_token.csv`   |
| MuSiQue  | `results/stage2_probe_table_musique_pdopt_binary_last_token.csv`    |
| 2Wiki    | `results/stage2_probe_table_2wiki_pdopt_binary_hidden_residual.csv` |


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

## 五、Stage3 接入

Stage3 通过 `stage3/adapters/stage2_probe.py` 加载 Stage2 checkpoint，自动读取 `hidden_branch_residual` 字段（checkpoint meta 中已写入），无需额外传参：

```python
# stage3/adapters/stage2_probe.py 中已处理
hidden_branch_residual = bool(ckpt.get("hidden_branch_residual", False))
```

Checkpoint 路径：`artifacts/probe/{dataset}/probe_mlp_pdopt_best.pt`（以 `--artifact-suffix pdopt_best` 生成时）。

---

## 六、完整复现流程

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

