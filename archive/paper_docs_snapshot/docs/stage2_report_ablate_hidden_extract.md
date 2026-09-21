# Stage2 消融：Hidden State 提取与 Hidden 分支结构

对应 `docs/plan-04-10.md` §三.4。分两组：**§1** 统一默认超参 + **F1 回归**（与仓库默认 CLI 一致）；**§2** **per-dataset 最优超参** + **二分类**（与 Continue/Stop 标签一致，适合作为「主线分类 Probe」下的 Hidden 策略对照）。

## 实现说明

- `--hidden-state-key mean_pool`：自 Stage1 `.npz` 读 `mean_pool`。
- `--hidden-state-key last_mean_blend`：读 `last_token` 与 `mean_pool`，用 `0.5*(a+b)` 作为输入（维数仍为 4096）。
- `--hidden-branch-residual`：`ProbeMLP_v2` 在 `LayerNorm→Linear→GELU` 后做 `z + Linear(z)`，再 `Dropout`；checkpoint 字段 `hidden_branch_residual`（Stage3 加载已对齐）。

---

## 1. 统一默认（`compress_dim=64`、`train_margin_min_abs=0`）+ F1 回归

与 `python -m stage2.run_stage2` 默认（`--probe-target f1`）一致。

```bash
python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --hidden-state-key mean_pool --artifact-suffix ablate_hidden_mean_pool

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --hidden-state-key last_mean_blend --artifact-suffix ablate_hidden_last_mean_blend

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --hidden-branch-residual --artifact-suffix ablate_hidden_residual
```

### 结果（test，Probe 行）

| 配置 | HotpotQA F1 / 步数 | MuSiQue F1 / 步数 | 2Wiki F1 / 步数 |
| --- | --- | --- | --- |
| last_token（基线） | 0.5148 / 2.53 | 0.1794 / 3.09 | 0.3484 / 2.41 |
| mean_pool | 0.4485 / 2.89 | 0.1331 / 2.16 | 0.3158 / 2.92 |
| last_mean_blend | 0.4392 / 2.54 | 0.1181 / 1.66 | 0.3371 / 3.47 |
| last_token + hidden 残差 | 0.4569 / 2.82 | 0.1428 / 2.18 | 0.2924 / 3.10 |

产物：`results/stage2_probe_table_{dataset}_ablate_hidden_*.csv`。

**小结：** 三数据集均未超过该设定下的 `last_token` 基线。

---

## 2. Per-dataset 最优超参 + 二分类（推荐用于「分类 Probe」Hidden 消融）

使用 `PER_DATASET_OPTIMAL` 中的 **结构**（HotpotQA/MuSiQue：`compress_dim=256`；2Wiki：`compress_dim=64` + `train_margin_min_abs=0.02`）。默认 `--per-dataset-optimal` 已含 Hotpot/MuSiQue→binary；若需 **三数据集统一二分类**（含 2Wiki，覆盖表内 f1），须显式 **`--probe-target binary`**。

```bash
python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --per-dataset-optimal --probe-target binary --artifact-suffix pdopt_binary_last_token

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --per-dataset-optimal --probe-target binary --hidden-state-key mean_pool \
  --artifact-suffix pdopt_binary_mean_pool

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --per-dataset-optimal --probe-target binary --hidden-state-key last_mean_blend \
  --artifact-suffix pdopt_binary_last_mean_blend

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --per-dataset-optimal --probe-target binary --hidden-branch-residual \
  --artifact-suffix pdopt_binary_hidden_residual
```

### 结果（test，Probe 行）

| 配置 | HotpotQA F1 / 步数 | MuSiQue F1 / 步数 | 2Wiki F1 / 步数 |
| --- | --- | --- | --- |
| last_token | 0.5273 / 2.85 | 0.1826 / 3.25 | 0.3463 / 2.57 |
| mean_pool | 0.4856 / 2.63 | 0.1216 / 1.99 | 0.3179 / 2.37 |
| last_mean_blend | 0.5142 / 2.42 | 0.1672 / 3.38 | 0.3598 / 2.67 |
| last_token + hidden 残差 | 0.5273 / 3.00 | 0.1743 / 3.17 | **0.3738** / 3.37 |

产物：`results/stage2_probe_table_{dataset}_pdopt_binary_*.csv`。

**小结：** 在 **二分类 + 最优超参** 下，**2Wiki** 上 **`--hidden-branch-residual` 相对 `last_token` +0.0275 F1**；HotpotQA 残差与基线持平；MuSiQue 仍以 `last_token` 最优。注意：同超参下 **F1 回归头** 的 2Wiki test 仍可达约 **0.383**（见 `docs/plan-04-10.md` §一），与二分类数值不可直接横向混比。

---

## 综合结论

- **仅换 Hidden 提取**（mean / blend）在多数设定下 **不占优**；**残差 hidden 分支** 在 **2Wiki + 二分类 + per-dataset-optimal** 上 **有明确增益**。
- 报告 Hidden 消融时应 **同时写明 `probe_target` 与是否 per-dataset-optimal**，避免与 plan §一「0.3833」等 **F1 头** 数字混淆。
