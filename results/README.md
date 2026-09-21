# 实验结果索引

`results/` 保持现有路径不变，以兼容 Stage 1/2/3 和分析脚本。结果按文件名前缀或子目录区分：

- `stage1_*`：轨迹、Oracle 和检索器 sanity 结果。
- `stage2_*`：Probe、消融、特征诊断和 Pareto 结果。
- `stage3_*`：E-value、漂移、显著性和 selective prediction 结果。
- `stop_rag_*`：Stop-RAG 训练、在线测试和 Pareto 结果。
- `alias_metric_aligned/`、`e2_predictive/`、`e4_fixed/`、`stop_rag_sweep_backup/`：参数或策略变体。

论文使用的精选图表位于 [`../paper/figs/`](../paper/figs/) 和 [`../paper/results/`](../paper/results/)，不会替代这里的原始结果。
