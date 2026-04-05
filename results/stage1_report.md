# Stage1 Report

## Data Statistics

- `hotpotqa`: train=4000, calib=1000, dev=1000, test=1000
- `musique`: train=4000, calib=1000, dev=1000, test=417
- `2wiki`: train=4000, calib=1000, dev=1000, test=1000

## Cache Integrity

- `hotpotqa`: trajectory_cached=7000, hidden_state_files=1000, bad_feature_files=0
- `musique`: trajectory_cached=6417, hidden_state_files=417, bad_feature_files=0
- `2wiki`: trajectory_cached=7000, hidden_state_files=1000, bad_feature_files=0

## Feature Distribution

- `hotpotqa`: entropy_mean=0.5215, self_consistency_mean=0.7790, overlap_mean=0.0869
- `musique`: entropy_mean=1.0594, self_consistency_mean=0.5727, overlap_mean=0.0666
- `2wiki`: entropy_mean=0.6196, self_consistency_mean=0.7343, overlap_mean=0.0722

## Oracle Frontier

- `hotpotqa` pareto: `results/stage1_oracle_pareto_hotpotqa.png`
- `musique` pareto: `results/stage1_oracle_pareto_musique.png`
- `2wiki` pareto: `results/stage1_oracle_pareto_2wiki.png`

## Hop Alignment (Oracle Step vs GT Hop)

- `hotpotqa`: known_gt=1000, unknown_gt=0, exact_match_over_known=14.20%; 2-hop 142/1000 (14.20%)
- `musique`: known_gt=417, unknown_gt=0, exact_match_over_known=5.52%; 2-hop 7/73 (9.59%); 3-hop 7/113 (6.19%); 4-hop 9/231 (3.90%)
- `2wiki`: known_gt=1000, unknown_gt=0, exact_match_over_known=11.80%; 2-hop 108/796 (13.57%); 4-hop 10/204 (4.90%)

## Go/No-Go Checks

- `hotpotqa`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1453, pass=True
- `musique`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.0938, pass=True
- `2wiki`: cache_ok=True, feature_missing_rate=0.0000, oracle_gain=0.1721, pass=True

## 实验完成状态（2026-04-05）

- **三数据集 Stage1 管线已跑通**：HotpotQA、MuSiQue、2Wiki 的 Train/Calib/Dev/Test 轨迹均已缓存；Test 上 **hidden state（`*.npz`）数量与 Test 条数一致**，坏文件数为 0；Oracle 帕累托图与 Go/No-Go 均已产出。
- **MuSiQue Test=417**：与 `experiments.md` 中「validation 不足以同时满额 Calib+Dev+Test」的自适应收窄一致；其余两数据集 Test 均为 1000。
- **门禁结论**：三数据集均 `pass=True`，满足进入 Stage2（Neural Probe）的预设门槛（缓存完整、步级关键特征无缺失、Oracle 相对最优固定步长策略 F1 有正增益）。

## 结果分析

1. **Oracle 相对固定步长的提升（`oracle_gain`）**  
   三数据集上 Oracle 动态停止相对「最佳 Fixed-K」的 F1 提升均为正：**2Wiki（≈0.172）> HotpotQA（≈0.145）> MuSiQue（≈0.094）**。说明在现有检索–生成设定下，**实例依赖的最优停止**相对统一步长策略存在可复现的帕累托空间；2Wiki/Hotpot 上边际更大，MuSiQue 更难（多跳与分解更复杂），但仍有明确正增益，支撑后续用 Neural Probe 去逼近 Oracle、而非仅用全局 Weitzman 静态保留值。

2. **浅层轨迹特征分布**  
   - **语义熵**：MuSiQue 最高（均值约 **1.06**），HotpotQA 最低（约 **0.52**），与「多跳/分解更难、采样答案更分散」的直觉一致。  
   - **自一致性**：HotpotQA 最高（约 **0.78**），MuSiQue 最低（约 **0.57**），可与任务难度及文档干扰程度一并解读。  
   - **Context–Question overlap**：三数据集均值均在 **0.07–0.09** 量级，步级上仍有变化（报告统计为 Test 全步聚合均值），可作为 Probe 的辅助信号。

3. **Oracle 停止步与 GT 跳数（Hop Alignment）**  
   「最优停止步」由 **F1 驱动的 Oracle DP** 决定，**不必等于**数据集中标注的 hop 数。HotpotQA 上「步数与 GT hop 完全一致」比例约 **14%**；MuSiQue 按 hop 分层后比例约 **6–10%**。该结果**不表示管线错误**，而是说明：**最小检索代价下的最优停止**与**标注的多跳结构**不是同一目标；后续 Probe 应以 **Oracle 标签/边际** 为监督信号，而非拟合 hop 数本身。

4. **风险与下一步**  
   Stage1 仅锚定 Oracle 上界与特征缓存；**Probe 在 Dev 上是否逼近 Oracle 帕累托前沿**需在 Stage2 中验证。若 MuSiQue 上增益偏小，可优先在该集做难例分析与特征消融（见总方案「第五阶段」）。
