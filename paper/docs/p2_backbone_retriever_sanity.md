# P2：跨检索器小规模验证（bm25 vs Contriever+BGE）

目的：在**不追求刷榜**的前提下，说明「自适应停止 / Oracle 相对 fixed-depth 的 headroom」并非仅由 **BM25 + Llama-3.1-8B** 这一组检索-生成组合偶然产生；主论文仍应以全量主设定（默认 BM25）报告。

## 当前状态

**已完成**（2026-04-20）：已在 HotpotQA 小规模配额上完成 `bm25` 与 `contriever_bge` 对照，并生成汇总表 `results/p2_backbone_retriever_sanity.csv`。

| dataset | retriever | Oracle F1 | Oracle avg steps | best Fixed-K F1 | best K | Oracle gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | bm25 | 0.7390 | 1.77 | 0.6133 | 5 | +0.1257 |
| hotpotqa | contriever_bge | 0.7642 | 1.58 | 0.6760 | 4 | +0.0882 |

结论：两种检索后端下，Oracle 相对各自 best Fixed-K 都保留正增益，说明 adaptive stopping 的 headroom 并非 BM25-only artifact。由于检索器会改变证据分布，表中绝对 F1 **不**应被写成跨检索器强弱比较；它只支持「跨 retriever 仍存在停止策略可利用空间」这一 small-scale diagnostic 结论。

## 协议

- **生成器**：与主实验一致，Stage1 默认 `meta-llama/Meta-Llama-3.1-8B-Instruct`（vLLM + 逐步 hidden）。
- **检索器 A**：`bm25`（`--retriever-backend bm25`）。
- **检索器 B**：`contriever_bge`（`facebook/contriever-msmarco` 短名单 + `BAAI/bge-reranker-v2-m3` 重排）。
- **隔离**：两种后端必须使用**不同**的 `--root-dir`，各自拥有独立的 `cache/trajectories` 与 `cache/features`，禁止与主实验目录混用。
- **小规模**：通过 `--train-quota/--calib-quota/--dev-quota/--test-quota` 缩小（脚本默认约数百～一千级样本；可按算力改环境变量）。

## 一键脚本

```bash
# 需：vLLM；Contriever+BGE 建议单独 GPU：export RETRIEVER_DEVICE=cuda:2
chmod +x scripts/run_p2_backbone_retriever_sanity.sh
./scripts/run_p2_backbone_retriever_sanity.sh
```

可选：若本仓库已准备好 `data/processed`，可加速准备阶段：

```bash
export P2_COPY_DATA_FROM=/path/to/Pandora-RAG
export P2_DATASETS=hotpotqa,2wiki
./scripts/run_p2_backbone_retriever_sanity.sh
```

## 产出

- 各 `--root-dir` 下：`results/stage1_oracle_table_{dataset}.csv`（Stage1 在计算 Oracle Pareto 时写出，含 Fixed-K、Global-Weitzman、Oracle 行）。
- 汇总：`results/p2_backbone_retriever_sanity.csv`（由 `scripts/p2_backbone_retriever_aggregate.py` 合并 bm25 与 `contriever_bge` 的表）。

## 论文中如何表述

- **可比性**：不同检索器改变证据分布，**绝对 F1 不可直接横向当作「谁更强」**；应强调在同一套 stopping 形式下，**Oracle 相对 best fixed-K 的增益（`oracle_gain_over_best_fixed`）** 在两套检索上均为正，呈现一致的「存在自适应 headroom」现象。
- **边界**：本 sanity **不**替代全量 Stage2/Stage3；若只跑小样本，需在文中标明为 **small-scale diagnostic**。

## 更强生成器（可选后续）

若需「更强 backbone」而非检索，可在独立 `root-dir` 下将环境变量/配置指向另一 vLLM 模型，并清空该 root 的 cache 后重跑 Stage1；成本显著高于本表，建议仅单数据集极小配额。
