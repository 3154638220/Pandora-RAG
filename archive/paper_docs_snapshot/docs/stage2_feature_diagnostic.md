# Stage 2 Phase D1：特征–标签诊断报告

> 训练集逐步样本：浅层特征与 Oracle `action_label`（Continue=1 / Stop=0）的单变量关联；
> Hidden states（4096 维）经 PCA 至 2D 着色；Oracle `margin` 小值占比（模糊标签）。

- **模糊样本定义**：|margin| < 0.01（Oracle 决策边界附近，标签噪声大）

## 数据集：`hotpotqa`

### 样本量

- 可训练步级样本：`16000`

### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）


| feature                        | point-biserial r | p-value | AUROC  |
| ------------------------------ | ---------------- | ------- | ------ |
| k_norm                         | -0.2067          | 0.0000  | 0.3556 |
| retrieval_score                | 0.0410           | 0.0000  | 0.5336 |
| semantic_entropy               | 0.2935           | 0.0000  | 0.7120 |
| self_consistency               | -0.2968          | 0.0000  | 0.2886 |
| answer_logprob                 | -0.2571          | 0.0000  | 0.2618 |
| self_eval_score                | -0.1644          | 0.0000  | 0.4149 |
| ctx_overlap                    | -0.1393          | 0.0000  | 0.3943 |
| nli_entail                     | -0.0254          | 0.0013  | 0.4637 |
| nli_contra                     | 0.0366           | 0.0000  | 0.5269 |
| log1p_token_count              | 0.0232           | 0.0034  | 0.5128 |
| log1p_latency_ms               | 0.0925           | 0.0000  | 0.5600 |
| delta_retrieval_score          | 0.0390           | 0.0000  | 0.5752 |
| delta_semantic_entropy         | 0.0731           | 0.0000  | 0.5478 |
| delta_self_consistency         | -0.0761          | 0.0000  | 0.4534 |
| delta_ctx_overlap              | 0.0158           | 0.0453  | 0.5149 |
| delta_nli_entail               | 0.0037           | 0.6408  | 0.4964 |
| cumulative_cost_ratio          | -0.2067          | 0.0000  | 0.3556 |
| answer_changed                 | 0.1200           | 0.0000  | 0.5686 |
| answer_consistency_streak_norm | -0.2654          | 0.0000  | 0.3189 |
| retrieval_marginal_novelty     | 0.1311           | 0.0000  | 0.5986 |


### Oracle margin 分布

- 样本数：16000
- |margin| < 0.01 占比：**0.05%** （8 / 16000）
- |margin| 绝对值均值：0.2247；中位数：0.0500

### Hidden states PCA（2D）

- 图：`../results/stage2_feature_diagnostic_pca_hotpotqa.png`
- 用于 PCA 的点数：16000；全零 hidden 行数：0
- 前两主成分方差比：10.77%，5.24%

## 数据集：`musique`

### 样本量

- 可训练步级样本：`16000`

### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）


| feature                        | point-biserial r | p-value | AUROC  |
| ------------------------------ | ---------------- | ------- | ------ |
| k_norm                         | -0.1830          | 0.0000  | 0.3654 |
| retrieval_score                | 0.0208           | 0.0085  | 0.5189 |
| semantic_entropy               | 0.0769           | 0.0000  | 0.5632 |
| self_consistency               | -0.0857          | 0.0000  | 0.4329 |
| answer_logprob                 | -0.0461          | 0.0000  | 0.4424 |
| self_eval_score                | -0.0163          | 0.0387  | 0.4892 |
| ctx_overlap                    | -0.0729          | 0.0000  | 0.4235 |
| nli_entail                     | -0.0006          | 0.9424  | 0.4686 |
| nli_contra                     | 0.0420           | 0.0000  | 0.5298 |
| log1p_token_count              | 0.0578           | 0.0000  | 0.5437 |
| log1p_latency_ms               | 0.0411           | 0.0000  | 0.5362 |
| delta_retrieval_score          | 0.0323           | 0.0000  | 0.5624 |
| delta_semantic_entropy         | 0.0400           | 0.0000  | 0.5241 |
| delta_self_consistency         | -0.0411          | 0.0000  | 0.4733 |
| delta_ctx_overlap              | 0.0196           | 0.0133  | 0.5210 |
| delta_nli_entail               | -0.0014          | 0.8555  | 0.4977 |
| cumulative_cost_ratio          | -0.1830          | 0.0000  | 0.3654 |
| answer_changed                 | -0.0258          | 0.0011  | 0.4831 |
| answer_consistency_streak_norm | -0.1274          | 0.0000  | 0.4238 |
| retrieval_marginal_novelty     | 0.0898           | 0.0000  | 0.5762 |


### Oracle margin 分布

- 样本数：16000
- |margin| < 0.01 占比：**0.09%** （15 / 16000）
- |margin| 绝对值均值：0.1777；中位数：0.0500

### Hidden states PCA（2D）

- 图：`../results/stage2_feature_diagnostic_pca_musique.png`
- 用于 PCA 的点数：16000；全零 hidden 行数：0
- 前两主成分方差比：10.93%，6.73%

## 数据集：`2wiki`

### 样本量

- 可训练步级样本：`16000`

### 浅层特征：point-biserial r 与 AUROC（单特征 → 预测 Continue）


| feature                        | point-biserial r | p-value | AUROC  |
| ------------------------------ | ---------------- | ------- | ------ |
| k_norm                         | -0.2056          | 0.0000  | 0.3571 |
| retrieval_score                | 0.1268           | 0.0000  | 0.5910 |
| semantic_entropy               | 0.0837           | 0.0000  | 0.5623 |
| self_consistency               | -0.0953          | 0.0000  | 0.4307 |
| answer_logprob                 | -0.0328          | 0.0000  | 0.4659 |
| self_eval_score                | -0.0364          | 0.0000  | 0.4779 |
| ctx_overlap                    | -0.1098          | 0.0000  | 0.4019 |
| nli_entail                     | -0.0164          | 0.0378  | 0.4745 |
| nli_contra                     | 0.0126           | 0.1113  | 0.5079 |
| log1p_token_count              | 0.0971           | 0.0000  | 0.5699 |
| log1p_latency_ms               | 0.0806           | 0.0000  | 0.5649 |
| delta_retrieval_score          | 0.0154           | 0.0513  | 0.5588 |
| delta_semantic_entropy         | 0.0277           | 0.0005  | 0.5225 |
| delta_self_consistency         | -0.0377          | 0.0000  | 0.4734 |
| delta_ctx_overlap              | 0.0120           | 0.1284  | 0.5153 |
| delta_nli_entail               | -0.0027          | 0.7320  | 0.4927 |
| cumulative_cost_ratio          | -0.2056          | 0.0000  | 0.3571 |
| answer_changed                 | -0.0098          | 0.2130  | 0.4939 |
| answer_consistency_streak_norm | -0.1610          | 0.0000  | 0.4050 |
| retrieval_marginal_novelty     | 0.1153           | 0.0000  | 0.5984 |


### Oracle margin 分布

- 样本数：16000
- |margin| < 0.01 占比：**0.03%** （4 / 16000）
- |margin| 绝对值均值：0.2448；中位数：0.0500

### Hidden states PCA（2D）

- 图：`../results/stage2_feature_diagnostic_pca_2wiki.png`
- 用于 PCA 的点数：16000；全零 hidden 行数：0
- 前两主成分方差比：17.85%，10.82%

## 解读要点（对照 plan D1）

- 若多数浅层特征 **AUROC < 0.60**，单变量上难以区分 Stop/Continue，提示 **特征信号不足**，宜推进 D4 补特征。
- **模糊样本占比高** 时，硬标签噪声大，会拖累 Probe 与 Focal 等损失；可与 D3 回退损失联动。
- PCA 平面上两类 **严重重叠** 时，仅靠线性可分 hidden 信息不足，与 ProbeMLP_v2 难提取信号一致。

