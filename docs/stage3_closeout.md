# Stage 3 正式收尾与归档

> 状态：**Closed / Archived**  
> 日期：2026-04-19  
> 口径：基于当前 `artifacts/probe/*/probe_mlp_pdopt_best.pt`、`results/stage3_evalue_*.json` 与新增统一 Pareto 图

---

## 一、正式结论

Stage 3 到此正式收尾。当前仓库内，`Pandora-RAG` 的第三阶段已经完成了方法实现、主实验、漂移实验、稳健性扫描、消融实验、Selective Prediction，以及论文叙事所需的核心图表闭环。

这一轮最终应固定的三条结论是：

1. **`Probe+E-value` 是低开销监控层，而不是错误率硬控制器。**  
   它相对 `Probe` 只增加极少量步数，在 HotpotQA 与 2Wiki 上带来小幅质量增益，在 MuSiQue 上基本持平；更准确的定位是 deployment-time **online risk monitor**。
2. **`E-value` 的核心价值是分布无关的在线检测与漂移感知。**  
   在 no-shift 主实验中，wealth 轨迹可作为风险仪表盘；在 sudden / gradual / periodic shift 下，它能及时累积风险证据并触及告警阈值，而不是像静态门控那样对测试流变化无感。
3. **`Probe+CP` 不是当前 setting 下的稳定强基线。**  
   它在 HotpotQA 上可以用极高成本换来更高 F1，但在 MuSiQue 与 2Wiki 上既更贵也更差，因此更适合作为“静态保守门控”的反例，而不是全文主对手。

---

## 二、最终产物

Stage 3 正式归档后，建议把下面这些文件视为主入口：

- 结果总览：[docs/stage3_results_summary.md](/home/x12dpg/hjx/Pandora-RAG/docs/stage3_results_summary.md:1)
- 论文叙事：[docs/stage3_narrative.md](/home/x12dpg/hjx/Pandora-RAG/docs/stage3_narrative.md:1)
- 本页收尾：[docs/stage3_closeout.md](/home/x12dpg/hjx/Pandora-RAG/docs/stage3_closeout.md:1)

本轮新增并补齐的 Pareto 图：

- 总览图：[results/stage3_final_pareto_all.png](/home/x12dpg/hjx/Pandora-RAG/results/stage3_final_pareto_all.png)
- 分数据集图：
  [results/stage3_final_pareto_hotpotqa.png](/home/x12dpg/hjx/Pandora-RAG/results/stage3_final_pareto_hotpotqa.png)
  [results/stage3_final_pareto_musique.png](/home/x12dpg/hjx/Pandora-RAG/results/stage3_final_pareto_musique.png)
  [results/stage3_final_pareto_2wiki.png](/home/x12dpg/hjx/Pandora-RAG/results/stage3_final_pareto_2wiki.png)

生成脚本：

- [scripts/stage3_make_final_pareto.py](/home/x12dpg/hjx/Pandora-RAG/scripts/stage3_make_final_pareto.py:1)

这组图把 `Oracle / Global-Weitzman / Probe / Probe+E-value / Probe+CP` 放到了同一 `avg_steps–avg_f1` 平面上，补齐了此前 Stage 3 里最后一个未闭合的附录图项。

---

## 三、待办归档

### 3.1 已关闭的 Stage 3 待办

以下事项现在全部归档，不再作为 Stage 3 blocker：

| 事项 | 当前状态 | 归档说明 |
|------|---------|---------|
| 主实验（E5） | 已完成 | `results/stage3_evalue_*.json` |
| 漂移实验（E3） | 已完成 | sudden / gradual / periodic 三类均已产出 |
| 多参数稳健性（E2） | 已完成 | `4γ × 4α × 3` 共 `48` 组 |
| betting 消融（E4） | 已完成 | predictive vs fixed |
| 检测延迟字段 | 已完成 | `stage3/cap_timing.py` + `evalue_cap_timing` |
| Selective Prediction | 已完成 | JSON + `stage3_selective_ca_*.png` |
| 与 Stage 1 的统一 Pareto 图 | **已完成** | 本轮新增 `stage3_final_pareto_*.png` |

### 3.2 从 Stage 3 移出、转入论文阶段的事项

这些不再属于 `stage3` 实验实现本身，而属于后续写作/投稿整理：

- 主文如何压缩呈现 `γ/α` 网格与漂移结果
- 是否把 `Probe+CP` 放入主表还是附录
- 是否补充更多外部 baseline（如 Stop-RAG）做预算匹配主对比
- 是否在论文附录中展示更多 wealth trace 个案图

换句话说，**Stage 3 本身已经结束；剩下的是 paper packaging，而不是 Stage 3 engineering。**

---

## 四、建议的后续工作边界

如果后面继续推进，建议把任务名称从“做 Stage 3”改成下面两类之一：

1. **论文整编**  
   目标是主文/附录结构、图表取舍、叙事压缩，不再修改 Stage 3 方法口径。
2. **跨方法主对比**  
   目标是和 Stop-RAG 或其他 baseline 做预算匹配评测，这属于 whole-project evaluation，不再是 Stage 3 子任务。

---

## 五、最终判定

**Stage 3 已正式关闭。**

从项目管理角度看，现在可以把它从“进行中”改成“已完成并归档”；从论文角度看，它已经具备进入主文/附录的稳定证据链，不需要再以“补实验”为理由继续拖延。
