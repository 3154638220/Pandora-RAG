# Baseline 全景对比（含 vs 本文方法）

**本文方法**：Probe+E-value Predictive，γ=0.5，α=0.1。

ΔF1 / Δ步数均为**三数据集（HotpotQA / 2Wiki / MuSiQue）平均**；正值表示本文方法更好；Δ步数为正表示本文方法平均多走检索步数，为负表示**节省步数**。

**补充口径**：Pandora 主线当前定位为**低成本最优停止器**，因此与 Stop-RAG 的主对比应使用 **`matched-budget / Pareto`**，而不是两个“各自 best-F1 单点”的直接互比。预算主指标统一为 `avg_steps`；若写成成本，则等价于 `cost = c * t`。

---

## 各基线具体是什么（定义与实现）

以下与代码、数据路径一一对应，便于复现与审稿对照。

### 本文方法：Probe + E-value（Predictive）

- **停止**：与 Stage 2 相同，用 MLP 探针在每一步输出「继续检索」概率；与在 dev 上选定的阈值比较，低于阈值则在该步停止（详见 `stage2/run_stage2.py` Phase C）。
- **安全层**：在 Probe 建议停止时，用独立校准的**质量模型**估计当前步答案满足 F1≥γ 的概率（记为 `phat`，对应 `stage3/quality_model.py` 的 `predict_success_prob`）；结合 **E-process / E-wealth** 对错误事件 1[F1<γ] 做序列下注更新；可选 **quality_bar** 门控（`phat` 过低则强制继续）。下注系数 λ 采用 **predictive**：λ = clip(1−`phat`, ε, 1−ε)，质量越差下注越大，wealth 累积更快。实现见 `stage3/stopping.py` 中 `simulate_evalue_outcome_aware`，主结果见 `results/stage3_evalue_*.json`（`betting_strategy: predictive`）。

### Oracle（DP，Stage 1 上界）

- **含义**：对**每条轨迹**已知全程每步真实 F1，用后向动态规划求最优停止时刻：V_K=Q(s_K)，V_k=max(Q(s_k), V_{k+1}−c_{k+1})，在最小满足 Q(s_k)≥V_{k+1}−c_{k+1} 的 k 停止（否则 max_k）。成本 c_k 默认每步常数，也可按 token/latency（`oracle_cost_metric`）。
- **用途**：理论 Pareto 上界；并为 Probe 生成训练标签（`expected_continue_val`、`margin`、`action_label`）。实现：`pretest/utils/weitzman.py` 中 `compute_trajectory_oracle`；标签写入 `artifacts/oracle/{dataset}/test_oracle_labels.jsonl`。表中数值来自 Stage 2 汇总行 `Oracle`（`results/stage2_probe_table_*.csv`）。

### Fixed-K（K 取 1, …, max_k）

- **含义**：**不做**实例级停止决策：每条查询固定执行 K 轮检索后停止，取第 K 步（若轨迹不足则取最后一步）的 F1/EM 与累计成本。
- **用途**：最常见 naive baseline，代表「固定预算」策略。实现：`stage2/run_stage2.py` 中 `_eval_fixed_k`。

### Global-Weitzman

- **含义**：在**训练集**上按 Weitzman 保留值公式，用每步 **F1 增量**（信息增益）的经验分布解出与 query 无关的全局阈值 r_k*；测试时在每一步比较**当前真实 F1** 与 r_{k+1}*，满足则停止（`oracle_stopping_simulation`）。
- **限制**：测试时需要**观测当前步真实 F1**，实际系统不可部署，仅作「结构化静态阈值 + 真值」参照。实现：`pretest/utils/weitzman.py` 中 `compute_all_reservation_values` + `oracle_stopping_simulation`；Stage 2 表内行名 `Global-Weitzman`。

### Deployable-GW（可部署 Weitzman）

- **含义**：与 Global-Weitzman **同一套** r_k* 估计流程，但训练/测试时用于比较的「质量」不是 F1，而是轨迹里可观测的代理标量 **self_consistency**（步间增量仍按 Weitzman 式估计保留值）；汇报的 F1/EM 仍是该步真实指标，停止规则不读 GT 答案。
- **现象**：代理信号弱时，策略往往接近「一步就停」。实现：`compute_all_reservation_values_from_proxy`、`deployable_weitzman_stopping_simulation`；表内 `Deployable-GW`。亦见 `stage2/run_deployable_gw_eval.py`（若单独跑 GW 评估）。

### MLP Probe（停止策略主体，双分支）

- **含义**：用 **ProbeMLP_v2**（或序列变体）融合 **LLM 隐藏状态**（如 last_token）与 **浅层特征**（检索分数、语义熵、自一致性、NLI、成本归一等，见 `stage2/run_stage2.py` 中 `SHALLOW_FEATURE_NAMES`），预测继续/停止或回归目标；阈值在 **dev** 上选，并受 Global-Weitzman 在 dev 上的平均步数×倍数约束（Phase C）。测试时仅依赖模型输出与可观测特征。
- **用途**：本文**核心可学习停止策略**；Stage 3 中「无安全层」的 Probe 与此同一套概率与阈值（`stage3/adapters/stage2_probe.py` 中 `simulate_probe_baseline`）。表内 `Probe` / `MLP Probe` 指该策略；checkpoint 如 `artifacts/probe/{dataset}/probe_mlp*.pt`。

### Shallow-Only Probe

- **含义**：与 MLP Probe 相同的训练/阈值流程，但**不使用** hidden state，仅用浅层 MLP（`ProbeMLP`）与 StandardScaler 后的手工特征。
- **用途**：证明「深度表征」对停止的收益。结果文件：`results/stage2_probe_table_{dataset}_shallow.csv`。

### XGBoost-Probe

- **含义**：用 **XGBoost分类器**替代 MLP，输入为与历史兼容的浅层特征（不含 LLM hidden），在 dev 上网格搜索树深/棵数/学习率，阈值同样受 GW 步数约束选优。
- **用途**：非神经网络停止器对照。脚本：`stage2/run_xgboost_baseline.py`；汇总：`results/stage2_xgboost_baseline.json`。

### Stop-RAG（NAACL 2024）

- **含义**：基于 value / Q-learning 的动态停止 RAG，是本文最直接的外部 stopping baseline。
- **公平复现**：必须使用 `baselines/Stop-RAG/README.md` 中的 Pandora 对齐流程，保证 **同切分、同样本 id**，并且只使用 `stop_rag_test.sh` 产出的**在线早停**结果。
- **当前对齐结果**：三数据集均已完成同切分、同样本 id 的在线早停测试；收尾文档见 `docs/reports/baselines/stop_rag_alignment_closeout.md`。在 `Probe+E-value`（γ=0.5, α=0.1）主口径下，Pandora 宏平均 F1 **0.5503** vs Stop-RAG **0.4564**，宏平均步数 **2.374** vs **4.707**。
- **主比较口径**：Stop-RAG 的当前 single-point threshold 明显偏满预算；若写主图，仍建议使用 threshold sweep 后的在线 `F1 vs avg_steps` Pareto frontier。当前 single-point 表可报：
  - `F1/EM @ matched avg_steps`
  - `avg_steps to reach the same F1`
  - `F1 vs avg_steps` Pareto frontier
- **预算定义**：Stop-RAG 用在线日志里的 `stop_iter`，Pandora 用 `steps_used`；当两边共享同一个常数 `c` 时，`cost = c * t` 与 `avg_steps` 等价。

### Probe（Stage 3，无安全层）

- **含义**：在 Stage 3 评估脚本里，仅用上述 Probe 停止规则，**不**叠加 E-value 门控、也**不**叠加共形门控；用于衡量「加监控层」的纯增量。实现：`stage3/run_stage3.py` 调 `simulate_probe_baseline`。

### Probe + CP-quantile（Split Conformal 门控）

- **含义**：Probe 在某步想停止时，额外要求质量模型给出的 **P(F1≥γ)** 估计 `phat` 不低于阈值 **min_phat**；**min_phat** 在 **Calib** 集上对「Probe 自然停止处」的 `phat` 取 **(1−α) 分位数**（保守下界），属 split conformal 风格的单变量门控。
- **特点**：无跨样本 wealth 过程；阈值在 calib 上固定，**分布漂移时无法像 E-value 那样累积证据**。实现：`stage3/stopping.py` 中 `simulate_conformal_phat_gate`、`conformal_min_phat_threshold`；表内 `Probe+CP-quantile`。

### Probe + E-value Fixed（λ=0.5）

- **含义**：与本文 Predictive 相同的 E-value 更新与 quality_bar 逻辑，但下注系数 **固定** λ=0.5，不随 `phat` 变化。
- **用途**：消融「adaptive betting」；F1/步数与 Predictive 在多数设置下一致，但终态 E-wealth 幅度不同（如 HotpotQA 上 predictive 约为 fixed 的 2.5 倍）。结果目录：`results/e4_fixed/`（`betting_strategy: fixed`）。

### （代码内对照）E-value indicator 门控

- **含义**：较早的 **indicator betting** 版本：Probe 建议停止时，若按当前 `phat` 计算的下注乘子会使 wealth 立刻超过 cap，则强制继续；与主文采用的 **outcome-aware** 更新（`simulate_evalue_outcome_aware`）不同。
- **用途**：实现保留在 `stage3/stopping.py` 的 `simulate_evalue_gated_stops`，便于消融；主表与论文叙事以 outcome-aware + predictive 为准。

### 质量模型（Stage 3 共用）

- **含义**：在 **Calib** 上训练的浅层特征 →成功概率分类器（sklearn `Pipeline`，含 `StandardScaler`），与 Probe 共用同一套逐步 `phat`，供 E-value 的 quality_bar、CP 的 min_phat 校准使用。实现：`stage3/quality_model.py`，由 `stage3/run_stage3.py` 拟合并写入评估 JSON的 `quality_model_eval` 等字段。

---


| 层次      | Baseline 名                       | 是否可部署   | 是否观测真实 F1 | 安全监控     | vs 本文方法（ΔF1 avg / Δ步数 avg）                       |
| ------- | -------------------------------- | ------- | --------- | -------- | ------------------------------------------------ |
| Stage 1 | **Oracle（DP）**                   | 否（上界）   | 是         | 无        | F1 差 -24.2%，步数多 +1.56（理论参照，不可部署）                 |
| Stage 2 | **Fixed-K=1**                    | 是       | 否         | 无        | **F1 +68% 相对提升**，步数 +2.1（质量大幅领先，代价是多检索）          |
| Stage 2 | **Fixed-K=3**（步数相近）              | 是       | 否         | 无        | **F1 +22.6% 相对提升**，步数约持平（+0.19）                  |
| Stage 2 | **Fixed-K=5**（固定最多步）             | 是       | 否         | 无        | **F1 +17.7% 相对提升**，同时节省 1.81 步                   |
| Stage 2 | **Stop-RAG（在线）**                    | 是       | 否         | 无        | 同切分在线复现已完成：本文方法宏平均 F1 **+0.0939**，平均节省 **2.33** 步；详见 `docs/reports/baselines/stop_rag_alignment_closeout.md` |
| Stage 2 | **Global-Weitzman**              | 否（不可部署） | 是         | 无        | F1 差 -21.1%（GW 偷看真实 F1，非公平对比）                    |
| Stage 2 | **Deployable-GW**                | 是       | 否（用代理）    | 无        | **F1 +67.0% 相对提升**（GW 几乎退化为 1 步）                 |
| Stage 2 | **XGBoost-Probe**                | 是       | 否         | 无        | **F1 +6.2% 相对提升**，步数基本持平                         |
| Stage 2 | **Shallow-Only Probe**（无 hidden） | 是       | 否         | 无        | **F1 +15.3% 相对提升**，同时节省 0.88 步                   |
| Stage 2 | **MLP Probe（停止策略主体）**            | 是       | 否         | 无        | F1 持平（+0.2%），步数 +0.03（E-value 近零开销）              |
| Stage 3 | **Probe（无安全层）**                  | 是       | 否         | 无        | **F1 +0.2%**，步数仅多 0.03（监控层几乎无代价）                 |
| Stage 3 | **Probe+CP-quantile**            | 是       | 否         | 静态，不感知漂移 | **F1 +13.6% 相对提升**，节省 1.50 步                     |
| Stage 3 | **Probe+E-value Fixed（λ=0.5）**   | 是       | 否         | 有        | F1/步数与 Predictive 一致；wealth 信号弱约 2.5 倍（HotpotQA） |


## 说明

- **Oracle / Global-Weitzman**：测试时需观测真实 F1，属不可部署上界；本文方法 F1 低于它们是预期现象。
- **Fixed-K**：在步数与本文接近（K=3）时，本文 F1 平均高约 **+22.6%**；在固定满步（K=5）时，本文 F1 高约 **+17.7%** 且平均少用 **1.81** 步。
- **Deployable-GW**：公平的可部署结构化基线，本文 F1 平均高约 **+67%**。
- **Stop-RAG**：同切分、在线早停的 single-point 对齐已完成；当前 Stop-RAG 几乎跑满预算（宏平均 4.707 步），本文方法以 2.374 步取得更高宏平均 F1（0.5503 vs 0.4564）。完整 `matched-budget / Pareto` 图仍需要 Stop-RAG 多 threshold 在线 sweep。
- **Probe+CP-quantile**：主要竞争基线，本文 F1 平均高约 **+13.6%** 且平均少用 **1.50** 步，并具备分布漂移下的监控语义。
- **E-value vs 无安全层 Probe**：额外成本极小，换取 anytime-valid 风险监控与漂移响应能力。

## 数据来源

- Stage 2：`results/stage2_probe_table_{dataset}.csv`、`results/stage2_probe_table_{dataset}_shallow.csv`、`results/stage2_xgboost_baseline.json`
- Stage 3：`results/stage3_evalue_{dataset}.json`（predictive betting）、`results/e4_fixed/`（fixed λ 消融）
- Stop-RAG：`docs/reports/baselines/stop_rag_alignment_closeout.md` 与 `paper/baselines/Stop-RAG/results/*/online_test/`
