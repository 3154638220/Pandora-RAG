# Pandora-RAG: 完整实验与推进方案

**基座模型设定**: Llama-3.1-8B-Instruct (全白盒开源设定，确保完全可复现性与内部状态可访问性)
**部署与推理框架**: vLLM (用于高速生成) + HuggingFace Transformers (用于特征提取)

## 第一阶段：数据全量化与 Oracle 基线锚定 (Week 1-2)

*目标：从预实验的小样本扩展到三大标准数据集的全量数据，提取真正有区分度的深层特征。*

**1. 轨迹收集与状态缓存 (Trajectory Caching)**

- **数据集扩充**：HotpotQA (2-hop), MuSiQue (2-4 hop), 2WikiMultiHopQA。目标配额：**Train 4000、Calib 1000、Dev 1000、Test 1000**（每数据集）；共形预测与 E-value 阈値校准严禁在 Test 上拟合。若某数据集在 Hub 上 **无独立 test split**、且 **validation 总条数不足以同时容纳 Calib+Dev+满额 Test**，则 `prepare_data` **自动收窄从 validation 划出的 Test**（优先保证 Calib/Dev 各 1000），日志会打印 WARNING；当前 **MuSiQue**（`dgslibisey/MuSiQue`）在默认配额下 Test 约为 **417 条**，HotpotQA / 2Wiki 仍可满 **1000** Test。
- **特征提取（Two-Pass 核心架构改进）**：
  - **Pass 1: 高速生成与采样 (vLLM)**：利用 vLLM 极高的吞吐量，执行多跳文档检索，并使用 `temperature=0.7`, `n=10` 进行高效的局部采样，计算 **Semantic Entropy (语义熵)** 和 **Self-Consistency (自一致性)**。
  - **Pass 2: 状态提取 (HuggingFace) — 方案 B1（动态逐步）**：在**每一步**检索并完成该步生成后，对该步的输入（问题 + 截至该步的累积上下文 + 该步当前答案）做一次前向（`output_hidden_states=True`，关闭 KV cache），提取最后一层 **Hidden States**（`last_token`，可选 `mean_pool`，**float16** 落盘）。多跳 RAG 的状态随检索轮次变化，**必须为每个时间步各存一份表征**，而不能只在整条轨迹结束后对「最终答案」做一次前向（否则 Stage 2 与时间步错位）。落盘文件名：`{id}_step{k}.npz`，`k=1..K_max`（见下文 **B2**）。
  - **Context-Question Overlap / NLI**：引入轻量级 `cross-encoder/nli-deberta-v3-small` 评估检索文档与历史信息的包含关系，避免使用大模型导致算力浪费。

### 第一阶段执行清单（可直接落地）

> 目标验收线：三数据集均完成轨迹与特征缓存（Train+Calib+Dev+Test 按 manifest 实际条数；Hotpot/2Wiki 合计约 7000/数据集，MuSiQue 在自适应 Test 下合计约 **6417**/数据集），**且每个样本具备 `K_max` 个逐步 hidden 文件**（`{id}_step{k}.npz`），并产出 Oracle 帕累托前沿。

**A. 数据准备（D1-D3）**

- **A1. 数据下载与统一格式**
  - 范围：HotpotQA、MuSiQue、2WikiMultiHopQA。
  - **统一样本级 jsonl 字段**：`id, dataset, split, question, answer, gt_hop_count, supporting_facts(optional)`。
  - `**gt_hop_count` 提取**：优先从 `supporting_facts.title` 去重计数；MuSiQue 回退 `question_decomposition` 或样本 id 前缀（如 `2hop__`）；无法确定时记为 `-1`。
  - 输出：`data/processed/{dataset}/{train,calib,dev,test}.jsonl`。
  - 验收：每条样本字段完整；可被统一 loader 无报错读取。
- **A2. 严格四切分 (Train/Calib/Dev/Test)**
  - **目标配额**：Train=4000、Calib=1000、Dev=1000、Test=1000（由 `stage1.run_stage1` 的 `--*-quota` 控制）。Calib 专用于 E-value / CP 等阈値与 Betting 相关校准，**严禁在 Test 上拟合**。
  - **无独立 test split 时的 Test 自适应**（实现：`stage1/run_stage1.py` 的 `prepare_data`）：HotpotQA、MuSiQue、2Wiki 当前镜像均不从 Hub 拉取可用 test gold，Test 从 **validation 尾部**切出。切分前须预留 **Calib+Dev** 共 2000 条在剩余 validation 中；若 `len(validation) < 2000 + test_quota`，则令 `test_actual = min(test_quota, len(validation) - 2000)`，避免 MuSiQue 等 **validation 偏小** 时无法划分。Train 仍从 **train** split 抽样；各 split **id 正交**不变。
  - 规则：固定随机种子 `seed=42`；各划分样本 id **绝对正交、互不交叉**。
  - **Manifest**：`data/splits/{dataset}_seed42_manifest.json` 含 `quota_requested`（命令行目标）与 `**quota`（各 split 实际条数）** 及 `selected_ids`，便于复现与审计。
  - 验收：重复运行切分脚本，`id` 列表完全一致；Calib 与 Test 无重叠；manifest 中 `quota` 与 `data/processed/.../*.jsonl` 行数一致。

**B. 轨迹收集与深层特征缓存（D4-D8）**

- **B1. 统一轨迹协议**
  - 最大检索步数 `K_max=5`。每步在轨迹 jsonl 中缓存：检索文档与分数、当前中间答案、该步成本（如 `cost.token_count`、`cost.latency_ms`、检索调用次数），用于 Oracle DP 的动态步成本 $c_k$（实现上可通过 `run_stage1 --oracle-cost-metric token|latency` 选择度量），与 payoff $Q(s_\tau)-\sum_j c_j$ 的理论定义一致。
  - 输出：`cache/trajectories/{dataset}/{split}/*.jsonl`。
  - 验收：随机抽样 100 条轨迹，步级字段完整率 100%。
- **B2. 深层信号落地（Two-Pass 与落盘）**
  - **Pass 1 (vLLM)**：`temperature=0.7`，每步采样 `n=10`；将 **Semantic Entropy / Self-Consistency** 指标写入轨迹步级字段（如 `semantic_entropy`, `self_consistency` 或等价命名），验收要求在 Dev 上分布非全零、非常数。
    - **HTTP 次数说明**：`pretest/utils/llm_client.py` 的 `generate_n` 在兼容 OpenAI 接口支持 `n` 参数时，每检索步为 **1 次 HTTP 请求**（单次请求内 `n=10` 个 completion）；若批量请求失败则回退为 **每步 10 次**逐条请求。故单条轨迹在 `K_max=5` 下理想为 **5 次 HTTP/样本**，fallback 时约 **50 次 HTTP/样本**（与「解码/采样总次数」量级分开理解：服务端仍为每步生成 10 个答案）。
  - **Pass 2 (HuggingFace) — 与 plan.md「优先级 2 / B1」一致（动态深层特征）**：
    - **落盘约定**：每样本、每检索步各 **1 个**压缩文件：`cache/features/{dataset}/{split}/hidden_states/{id}_step{k}.npz`（`k` 与轨迹中 `steps[].step` 一致，通常 `1..K_max`），内含 `last_token`（及 `mean_pool` 等），**float16**。
    - **语义要求**：第 `k` 步文件对应「第 `k` 步检索后的累积上下文 + 该步生成后的当前答案」下的表征，与 `cache/trajectories/.../trajectories.jsonl` 中同一步的浅层统计（熵、NLI 等）**时间对齐**。文件总数约为「各 split 样本数 × `K_max`」，约为旧版「每样本单文件」方案的 `**K_max` 倍**（例如 `K_max=5` 时约 5 倍）。
    - **Stage 2 读取**：`stage2/run_stage2.py` 优先匹配 `{id}_step{k}`；若仍存在历史 **无后缀** `{id}.npz`，仅作为 `step=0` 的兜底（正式论文实验应淘汰该形态）。
    - **工程命令**（`conda activate pandora-rag`，仓库根目录）：
      - 全量重跑 Pass 1+2（检索、LLM、逐步 npz）：`python -m stage1.run_stage1 --datasets hotpotqa,musique,2wiki --skip-prepare`（或去掉 `--skip-prepare` 以重切分）。
      - **仅补跑某 split 的轨迹+逐步 npz**（其余 split 沿用已有 `trajectories.jsonl`）：`--skip-prepare --collect-splits train`（示例：修复 train 条数不足）。
      - **已有完整轨迹、仅重提 hidden（不调用 vLLM）**：`--skip-prepare --reextract-hidden-only`；可用 `--reextract-splits calib,dev` 限定 split。
    - **一致性自检**：`wc -l data/processed/{ds}/{split}.jsonl` 应与 `wc -l cache/trajectories/{ds}/{split}/trajectories.jsonl` **相等**；`find cache/features/{ds}/{split}/hidden_states -name '*.npz' | wc -l` 应约为 `N_split × K_max`（缺步或中断会导致明显偏少）。
    - **验收**：维度一致、坏文件率 0、可反序列化；**禁止**在仅存在「每样本单 npz」的情况下进入 Stage 2 主实验（除非明确做消融且清楚其与逐步 Oracle 标签不对齐）。
  - **NLI / 重叠**：步级字段如 `ctx_overlap`、`nli_entail`、`nli_contra`（或 `cross-encoder/nli-deberta-v3-small` 的 entailment score）；验收：随步数有统计变化，避免特征失效。

**C. Oracle 锚定与天花板估计（D9-D11）**

- **C1. 计算真实保留值 $r_k^*$ 与实例 Oracle**
  - **Global-Weitzman (静态基线)**：在 Train 集上统计每步经验增益分布，求解出一组全局静态保留值 $r_k^*$。
  - **实例 Oracle (DP 上界)**：对每条轨迹在已知逐步 $Q_k$（如 F1 Score）下做后向归纳。写入 `../artifacts/oracle/{dataset}/test_oracle_labels.jsonl`，关键标签包含 `action_label` (1=Continue, 0=Stop) 以及 `margin` (决策收益差距)；可含 `expected_continue_val` 等辅助字段便于调试。
  - 验收：可回放复现 Oracle 决策，决策路径无非法状态。
- **C2. 绘制 Oracle Pareto Frontier**：绘制横轴(Cost) - 纵轴(F1) 的帕累托前沿包络图，验证 Oracle 和 Global-Weitzman 的性能差距（证明引入 Neural Probe 的理论价值）。
  - 输出示例：`../results/stage1_oracle_pareto_{dataset}.png`。
  - 验收：Oracle 前沿包络须不低于 Global-Weitzman（在相同成本度量下）。

**D. 质量门禁（Go/No-Go）**
进入第二阶段（Neural Probe）前须同时满足：

- 三数据集轨迹与特征缓存完成率 $\geq 98$；
- 各 split 上 `**data/processed` 行数 = `cache/trajectories/.../trajectories.jsonl` 行数**（避免只跑部分 train/test 却误以为全量）；
- **B1 动态特征**：hidden 落盘为 `{id}_step{k}.npz`，且各 split 的 `.npz` 数量与 `N × K_max` 一致（允许工程上极少数坏文件，但不得系统性缺步）；
- 关键特征（hidden states / 语义熵与自一致性 / NLI）步级缺失率 $\leq 1$；
- Oracle 相对**最佳固定步长策略**在 F1–Cost 平面上存在**可复现的显著优势**（至少一个数据集上明显提升）。

若不满足：优先修复缓存链路、**逐步 hidden 与轨迹条数**、Two-Pass 一致性与特征稳定性，再进入 Probe 训练。

## 第二阶段：Neural Probe 攻坚战（核心难点） (Week 3-5)

*目标：训练出能够精准模拟 Weitzman 最优停止逻辑的神经网络探针。*

**1. 目标函数重构 (Target Reformulation) [⚠️核心优化]**

直接回归绝对 `margin` 会因 F1 噪声引发模型崩溃。我们将采用 **Margin-weighted Classification (边界加权二分类)**：

- **主任务**：预测 Oracle DP 产出的二值标签 `action_label` $\in 0, 1$。
- **Loss 设计 (Focal-Margin Loss)**：
$$ \mathcal{L}(\theta) = \sum_{k} | \text{margin}_k | \cdot \text{BCELoss}(\hat{p}_k, \text{actionlabel}_k) $$
通过绝对值 $|\text{margin}_k|$ 作为样本权重，模型被允许在无关紧要的步数（继续和停止收益差不多）上犯错，但被**严厉惩罚**在具有巨大经济价值差异的关键步上做出错误决策。这完美契合了经济学中最优停止的本质。

**2. 模型架构设计**

- **输入层**：Llama-3.1-8B-Instruct 提取的 4096 维 Hidden States 拼接浅层特征（熵、NLI、Cost）。
- **网络结构**：3 层 MLP + Dropout + 最终的 Sigmoid 分类头 $\hat{p}_k$。
- **评估标准**：不纯看 Accuracy，将 Probe 接入 RAG Pipeline，观察其在 Dev 集的 F1-Cost 坐标点是否显著逼近 Oracle 的帕累托前沿。

## 第三阶段：E-value 风险控制集成与验证 (Week 6-7)

*目标：实现论文的第二个核心贡献——任何时间序列上的任意时刻有效错误率控制。*

**1. 误差定义的严谨化**
在生成任务中，为了使用 E-value，必须定义什么是“Error”。我们引入可接受的质量下限 $\gamma$（如 $\gamma=0.5$）：

- 样本 $t$ 发生 Error $\iff \text{F1}(s_{\tau_t}) < \gamma$

**2. E-value 机制工程落地**

- **校准阶段 (On Calib Set)**：训练一个极轻量的质量预测器预测当前状态的 $P(\text{F1} \geq \gamma)$，据此构造 Betting Score $e_t$。
- **序列乘积与 Ville 不等式**：在用户请求流（Time/Sample index $t$）上维护累积乘积 $E_n = \prod_{t=1}^n e_t$。
- **动态安全阀 (Dynamic Safety Valve)**：
当当前步骤 $k$ 的 Neural Probe 发出 `Stop` 信号时：
  - 如果执行 Stop 后会导致系统的 $E_n$ 突破安全边界（即有破坏 $1-\alpha$ 错误率的风险），系统将**否决探针**，强制继续检索，直至找到足以恢复 Betting Score 信用的高价值文档。

**3. 核心测试：应对长尾数据分布漂移 (Distribution Shift)**

- 设置测试序列 $n$，前半段为普通问题，后半段注入高难度/混淆问题。
- **预期证明**：展示标准共形预测 (Standard CP) 在面临后续连续困难样本时，其 Empirical Error Rate 会突破 $\alpha$ 红线；而 Pandora-RAG 的 E-process 会自动变得更加保守（强制拉长检索步数），死死守住 $\alpha$ 底线。

## 第四阶段：主实验与 SOTA 对比 (Week 8-9)

*目标：在三大数据集上跑通端到端全流程，生成核心实验表格。*

**1. 主实验对比矩阵**


| 类别               | 策略                          | Avg Steps | F1    | EM    | 经验错误率 (目标 $\le \alpha=0.1$) | 推理成本  |
| ---------------- | --------------------------- | --------- | ----- | ----- | --------------------------- | ----- |
| **Static**       | Single-RAG (K=1)            | 1.0       | 最低    | -     | 破防 (>0.1)                   | 最低    |
|                  | IRCoT (Fixed-K=5)           | 5.0       | 高     | -     | -                           | 最高    |
| **Dynamic**      | ITER-RETGEN (EMNLP 23)      | 动态        | 中     | -     | 破防 (>0.1)                   | 中等    |
|                  | Adaptive-RAG (NAACL 24)     | 动态        | 高     | -     | 破防 (>0.1)                   | 中等    |
|                  | Stop-RAG (NAACL 24)         | 动态        | 高     | -     | 破防 (>0.1)                   | 中等    |
| **Risk-Control** | CCPO (NeurIPS 24 - Std CP)  | 动态        | 中     | -     | **自适应长序列下破防**               | 中等    |
| **Upper/Base**   | Global-Weitzman (Ours Base) | 动态        | 中     | -     | -                           | 低     |
|                  | Oracle-Pandora (理论上界)       | *最优*      | *最高*  | *最高*  | -                           | -     |
| **Ours**         | **Pandora-RAG (E-value)**   | **逼近最优**  | **高** | **高** | **严格达标 ($\le 0.1$)**        | **低** |


**2. 泛化性测试 (Zero-Shot OOD Transfer)**

- 使用在 HotpotQA 上训练的 Probe，直接 Zero-shot 迁移到 MuSiQue 上进行测试。
- **核心论点**：即便 Probe 因为领域不同发生退化，**E-value 安全阀机制依然能够保证错误率不被突破**，展现系统极强的实际工程部署价值。

## 第五阶段：消融实验、机制分析与作图 (Week 10-11)

**1. 可视化作图 (Visualizations)**

- **图1：Teaser Figure (首页图)**：左侧展示固定检索与过早停止的困境，右侧展示 Pandora-RAG 如何利用 E-process 和保留值在二维（成本 vs 准确率）与三维（时间轴上的风险红线）实现完美平衡。
- **图2：Pareto Frontier 图**：Oracle vs Weitzman vs Ours，证明接近最优解。
- **图3：E-process 动态防御轨迹图**：横轴为时间线上的测试用户数 $n$，纵轴为累积错误率。明确展示当遭遇到连续的困难 query 攻击时，CP 失效，而 Pandora-RAG 动态增加检索步数压制错误率。

**2. 消融实验 (Ablation Study)**

- **w/o Deep Features**：仅用浅层特征（文本长度、分数）训练 Probe，论证 Llama-3 内部隐式推理状态（Hidden States）和 Semantic Entropy 的不可替代性。
- **w/o Margin-weighting**：使用普通的 BCE 训练，论证经济学边际价值引入 Loss 设计对系统收敛的必要性。

## 第六阶段：论文撰写与理论审核 (Week 12)

- **Storyline 定调**：核心叙事不仅是“降本增效”，而是**“在大语言模型的自适应迭代信息获取中，首次实现了兼顾理论最优成本边界与任意时刻严格风险控制的工程范式”**。
- **理论 Check**：仔细审查 Appendix 中的证明，重点核对 Fixed Order Pandora's Box 放宽条件假设的合理性，以及 Ville Inequality 在连续 RAG 对话流（Query stream）上的数学适配。

---

## 💡 风险预案 (Contingency Plan)

### Stage1 全量前检查（工程）

- **vLLM**：并发与耗时与后端吞吐强相关；多跳间 prompt 前缀重叠大，建议部署时开启 **Prefix Caching**（如 `--enable-prefix-caching`）。
- **NLI**：`NLICrossEncoderScorer`（`stage1/run_stage1.py`）懒加载；正式跑之前应在 `**conda activate pandora-rag`** 环境下做冒烟（见下），确认日志出现 `**NLI CrossEncoder 已加载`**；若仅见 `NLI 推理失败，回退启发式` 则说明未走真实 CrossEncoder。
- **落盘 I/O**：`cache/`（轨迹 jsonl + `hidden_states/*.npz`）会产生大量小文件，**务必放在 SSD**；可用 `python -m stage1.run_stage1 --root-dir <SSD 上的目录>` 将缓存根指到快速盘。

**NLI 冒烟（不跑完整 Stage1、不加载 8B hidden 模型）**：

```bash
conda activate pandora-rag
cd /path/to/Pandora-RAG
python -c "
import logging, importlib.util, sys
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
p = Path('stage1/run_stage1.py')
spec = importlib.util.spec_from_file_location('stage1_run', p)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m.NLICrossEncoderScorer().entail_contra('A dog runs.', 'A dog runs in the park.')
"
```

1. **算力限制导致 vLLM / HF 调度过慢**
  *应对方案*：若 `Llama-3.1-8B-Instruct` 在提取特征时过于耗时，将轻量级基线模型替换为 `Qwen-2.5-3B-Instruct` 进行对比跑分，在关键的主数据集上再使用 8B 模型。
2. **E-value 过于保守，导致模型退化成全部检索 5 步**
  *应对方案*：这通常是因为初始信用太低。调整 Betting Score 函数，允许模型在初期积累一定的“信用盈余 (Credit surplus)”，或引入适度的 Betting 赌注折现因子。
3. **Oracle 效果不明显（多跳失效）**
  *应对方案*：如果提供所有的正确文档 F1 依然提不上去，说明遇到了 LLM 的基础阅读理解瓶颈。此时调整质量函数，从单纯的 F1 放宽为“是否包含了正确答案的实体（Recall）”，改变上限评估维度。

