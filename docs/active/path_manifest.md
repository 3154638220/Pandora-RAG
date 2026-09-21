# 目录迁移清单

本清单记录本次目录整理的主路径变化；所有迁移均保留在 Git 历史中。

| 原位置 | 现在位置 | 状态 |
| --- | --- | --- |
| `papar/code/` | `archive/legacy_papar/code_snapshot/` | 历史代码快照，非执行入口 |
| `paper/docs/` | `archive/paper_docs_snapshot/docs/` | 历史论文文档快照 |
| 根目录 `Pandora-RAG.tex` | `paper/source/legacy_neurips/Pandora-RAG-root.tex` | NeurIPS 历史稿 |
| `paper/source/Pandora-RAG.tex` | `paper/source/legacy_neurips/Pandora-RAG-paper-source.tex` | NeurIPS 历史稿 |
| 根目录方法/实验笔记 | `paper/source/notes/` | 论文写作参考材料 |
| `审稿意见.md`、`answer.md` | `reviews/neurips/` | NeurIPS 投稿材料 |
| `docs/experiments.md`、`STORAGE_LAYOUT.md` | `docs/active/` | 当前有效协议 |
| Stage 1/2/3 报告 | `docs/reports/stage{1,2,3}/` | 阶段结果报告 |
| Stop-RAG/检索器报告 | `docs/reports/baselines/` | 基线与对齐报告 |
| 日期型计划 | `docs/plans/` | 历史计划 |
| `Stop-RAG.pdf` | `paper/baselines/Stop-RAG/Stop-RAG.pdf` | 论文基线材料 |

当前唯一实验代码入口仍是根目录的 `stage1/`、`stage2/`、`stage3/`、
`pretest/`、`qa_shared/` 和 `scripts/`；WWW 2027 写作入口是
`paper/source/www2027/`。
