# Stage 2 Phase D1：特征–标签诊断报告

> 训练集逐步样本：浅层特征与 Oracle `action_label`（Continue=1 / Stop=0）的单变量关联；
> Hidden states（4096 维）经 PCA 至 2D 着色；Oracle `margin` 小值占比（模糊标签）。

- **模糊样本定义**：|margin| < 0.01（Oracle 决策边界附近，标签噪声大）

## 数据集：`hotpotqa`

### 样本量

- 可训练步级样本：`16000`

### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）

| feature | point-biserial r | p-value | AUROC |
|---------|------------------|---------|-------|
| k_norm | -0.2037 | 0.0000 | 0.3585 |
| retrieval_score | 0.0436 | 0.0000 | 0.5416 |
| semantic_entropy | 0.2965 | 0.0000 | 0.7128 |
| self_consistency | -0.2973 | 0.0000 | 0.2887 |
| ctx_overlap | -0.1287 | 0.0000 | 0.3994 |
| nli_entail | -0.0261 | 0.0009 | 0.4734 |
| nli_contra | 0.0253 | 0.0014 | 0.5198 |
| log1p_token_count | 0.0196 | 0.0132 | 0.5100 |
| log1p_latency_ms | 0.1019 | 0.0000 | 0.5646 |
| delta_retrieval_score | 0.0350 | 0.0000 | 0.5749 |
| delta_semantic_entropy | 0.0771 | 0.0000 | 0.5462 |
| delta_self_consistency | -0.0761 | 0.0000 | 0.4584 |
| delta_ctx_overlap | 0.0238 | 0.0026 | 0.5210 |
| delta_nli_entail | -0.0013 | 0.8668 | 0.5055 |
| cumulative_cost_ratio | -0.2037 | 0.0000 | 0.3585 |

### Oracle margin 分布

- 样本数：16000
- |margin| < 0.01 占比：**0.09%** （15 / 16000）
- |margin| 绝对值均值：0.2257；中位数：0.0500

### Hidden states PCA（2D）

- 图：`results/stage2_feature_diagnostic_pca_hotpotqa.png`
- 用于 PCA 的点数：16000；全零 hidden 行数：0
- 前两主成分方差比：10.52%，5.26%

## 数据集：`musique`

### 样本量

- 可训练步级样本：`16000`

### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）

| feature | point-biserial r | p-value | AUROC |
|---------|------------------|---------|-------|
| k_norm | -0.1836 | 0.0000 | 0.3660 |
| retrieval_score | 0.0188 | 0.0177 | 0.5154 |
| semantic_entropy | 0.0797 | 0.0000 | 0.5641 |
| self_consistency | -0.0890 | 0.0000 | 0.4314 |
| ctx_overlap | -0.0671 | 0.0000 | 0.4269 |
| nli_entail | 0.0011 | 0.8888 | 0.4697 |
| nli_contra | 0.0424 | 0.0000 | 0.5340 |
| log1p_token_count | 0.0418 | 0.0000 | 0.5296 |
| log1p_latency_ms | 0.0287 | 0.0003 | 0.5274 |
| delta_retrieval_score | 0.0338 | 0.0000 | 0.5663 |
| delta_semantic_entropy | 0.0301 | 0.0001 | 0.5176 |
| delta_self_consistency | -0.0327 | 0.0000 | 0.4815 |
| delta_ctx_overlap | 0.0289 | 0.0003 | 0.5213 |
| delta_nli_entail | 0.0026 | 0.7380 | 0.5100 |
| cumulative_cost_ratio | -0.1836 | 0.0000 | 0.3660 |

### Oracle margin 分布

- 样本数：16000
- |margin| < 0.01 占比：**0.12%** （19 / 16000）
- |margin| 绝对值均值：0.1788；中位数：0.0500

### Hidden states PCA（2D）

- 图：`results/stage2_feature_diagnostic_pca_musique.png`
- 用于 PCA 的点数：16000；全零 hidden 行数：0
- 前两主成分方差比：10.81%，6.81%

## 数据集：`2wiki`

### 样本量

- 可训练步级样本：`16000`

### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）

| feature | point-biserial r | p-value | AUROC |
|---------|------------------|---------|-------|
| k_norm | -0.2252 | 0.0000 | 0.3438 |
| retrieval_score | 0.1351 | 0.0000 | 0.5967 |
| semantic_entropy | 0.0657 | 0.0000 | 0.5493 |
| self_consistency | -0.0754 | 0.0000 | 0.4450 |
| ctx_overlap | -0.1122 | 0.0000 | 0.3966 |
| nli_entail | -0.0073 | 0.3559 | 0.4815 |
| nli_contra | 0.0137 | 0.0831 | 0.5110 |
| log1p_token_count | 0.0825 | 0.0000 | 0.5599 |
| log1p_latency_ms | 0.0677 | 0.0000 | 0.5551 |
| delta_retrieval_score | 0.0322 | 0.0000 | 0.5719 |
| delta_semantic_entropy | 0.0218 | 0.0059 | 0.5128 |
| delta_self_consistency | -0.0272 | 0.0006 | 0.4832 |
| delta_ctx_overlap | 0.0128 | 0.1048 | 0.5153 |
| delta_nli_entail | 0.0030 | 0.7007 | 0.4921 |
| cumulative_cost_ratio | -0.2252 | 0.0000 | 0.3438 |

### Oracle margin 分布

- 样本数：16000
- |margin| < 0.01 占比：**0.01%** （1 / 16000）
- |margin| 绝对值均值：0.2456；中位数：0.0500

### Hidden states PCA（2D）

- 图：`results/stage2_feature_diagnostic_pca_2wiki.png`
- 用于 PCA 的点数：16000；全零 hidden 行数：0
- 前两主成分方差比：17.80%，10.58%

## 解读要点（对照 plan D1）

- 若多数浅层特征 **AUROC < 0.60**，单变量上难以区分 Stop/Continue，提示 **特征信号不足**，宜推进 D4 补特征。
- **模糊样本占比高** 时，硬标签噪声大，会拖累 Probe 与 Focal 等损失；可与 D3 回退损失联动。
- PCA 平面上两类 **严重重叠** 时，仅靠线性可分 hidden 信息不足，与 ProbeMLP_v2 难提取信号一致。

