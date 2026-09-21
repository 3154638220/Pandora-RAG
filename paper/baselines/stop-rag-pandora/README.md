<h1 align="center"> Stop-RAG: Value-Based Retrieval Control for Iterative RAG </h1>

本目录为论文 **Stop-RAG** 的实现代码。**在 Pandora-RAG 仓库中，默认只推荐「与主实验同切分、同样本 id」的对齐流程**；上游仓库自带的 `download.sh` 会重新子采样划分，与 `data/processed/*.jsonl` **不一致**，请勿用于与 Pandora 主线的公平对比。

## 口径警告

- **公平对比的测试结果**必须来自 `./scripts/stop_rag_test.sh`，它会在 `test_subsampled` 上执行**真实在线早停**：每轮检索后立刻用训练好的 stop head 判定 `STOP/CONTINUE`，最多 5 轮。
- `./scripts/dataset.sh` 的职责是**构造训练/开发用 partial traces**。其中 `METHOD=ours` 默认使用 `--sd-provider nostop` 跑满 `max_iterations=5`，这是为了给停止头训练和阈值选择准备 `k=1..5` 的样本，**不是**最终测试口径。
- `./scripts/stop_rag_find_best.sh` 的职责是**在 eval_subsampled 上离线选 checkpoint 与 threshold**。这一步是开发集调参，可以使用 partial traces；但它输出的 `compute_scores/*.jsonl` 和 `src/test/stop_rag_test.py` 的事后回放结果，**不能**作为与 Pandora head-to-head 的最终测试数字。
- 换句话说：**允许离线的只有开发集调参，不允许在测试集上“先跑满再事后选停点”**。

## 与 Pandora 的主比较指标

- Pandora 的主设定保持 Stage2 Phase C 的 **`F1 - λ·cost` + `GW(dev)` 步数上界**，因此主结果不是 raw `best-F1`，而是**预算受约束的最优停止点**。
- Stop-RAG 与 Pandora 的统一预算主指标应使用 **`avg_steps`**：
  - Stop-RAG：在线结果中的 `stop_iter`
  - Pandora：在线结果中的 `steps_used`
- 如果只想写成成本，也只能在两边共享同一个常数 `c` 时记为 `cost = c * t`；主文更建议直接报 `avg_steps`，避免引入任意缩放常数。
- **主表**建议报：
  - `F1 @ matched avg_steps`
  - `EM @ matched avg_steps`
  - `avg_steps to reach the same F1`
- **主图**建议画两边 threshold sweep 后得到的在线 **`F1 vs avg_steps` Pareto frontier**。
- `best-F1 vs best-F1` 只适合作为 appendix 的 quality-first 补充，不建议承载 Pandora 作为“低成本最优停止器”的主 claim。

---

## 与本仓库对齐的复现路径（推荐）

### 前置条件

1. **Pandora 已产出切分**  
   仓库根目录下存在  
   `data/processed/{hotpotqa,musique,2wiki}/{train,calib,dev,test}.jsonl`  
   （由 Stage1 `prepare_data` / `run_stage1` 等流程生成。）

2. **Hugging Face 缓存**  
   与主仓库一致时，将缓存指向仓库根目录的 `.hf_cache`（见 `pretest/hf_env.py`）：

   ```bash
   export HF_HOME="/path/to/Pandora-RAG/.hf_cache"
   ```

   `scripts/prepare_pandora_datasets.py` 会设置离线相关环境变量；**首次**若缓存中缺少权重/数据，需在有网络环境下预先 `huggingface-cli download` 或让程序在线拉取一次，再离线复现。

3. **运行前建议核对权重与缓存**  

   - **HF 数据集与检索类模型**：通常在 `$HF_HOME/hub` 下可见快照（如 `datasets--hotpot_qa`、`models--facebook--contriever-msmarco`、`models--BAAI--bge-reranker-v2-m3`）。`prepare_pandora_datasets.py` 读数据集依赖此处或等价缓存。
   - **vLLM（Llama）— 与 Stage1 共用本机目录（已默认自动解析）**  
     若存在 **`Pandora-RAG/models/Meta-Llama-3.1-8B-Instruct/config.json`**（与 Stage1 相同布局，见 `stage1/run_stage1.py` 的 `_resolve_local_llama_weights_dir`），则 **`dataset.sh` / `llm_stop_test.sh`** 与 **`python -m src.pipeline.pipeline` 等入口** 会默认使用该本地目录，无需再手动 `export`。实现：`scripts/pandora_resolve_vllm.sh`（shell）与 `src/pandora_repo_defaults.py`（Python）。  
     若需覆盖，仍可使用 `export STOP_RAG_VLLM_MODEL=...`（或命令行 `--vllm-model-id`）。若本地目录不存在，则回退为 Hub id **`meta-llama/Llama-3.1-8B-Instruct`**（需 `HF_HOME` 等能解析到权重）。
   - **Stop 头训练（DeBERTa）**：`train.sh` 默认 **`microsoft/deberta-v3-large`**，一般仍依赖 HF 拉取或缓存；与 Stage1 的 Llama 目录是两套资源，需单独就绪。

### 环境

推荐使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 与 Python 3.11：

```bash
cd baselines/Stop-RAG
pip install uv
uv sync --no-dev
```

### 1. 从 Pandora 切分生成 Stop-RAG 的 `data/raw` 与 `data/corpus`

在 **Pandora 仓库根目录**执行（或显式传入路径）：

```bash
cd /path/to/Pandora-RAG
export HF_HOME="/path/to/Pandora-RAG/.hf_cache"
python baselines/Stop-RAG/scripts/prepare_pandora_datasets.py \
  --pandora-root /path/to/Pandora-RAG \
  --stop-rag-root /path/to/Pandora-RAG/baselines/Stop-RAG \
  --datasets hotpotqa,musique,2wiki
```

脚本会将 Pandora 的 **train / calib+dev+test 池 / dev / test** 写成 Stop-RAG 可读格式，并基于当前样本构建 `corpus.jsonl`。**不要**再运行上游 `download.sh` 去拉 Hotpot 全量 wiki 或替换划分。

### 2. 仅构建检索索引（Contriever 或 BM25）

对齐流程已写好语料，只需建索引。**请勿**直接跑上游 `./scripts/download.sh`，否则会按上游逻辑重新下载并子采样，破坏与 Pandora 的一致性。

在 `baselines/Stop-RAG` 下，**Contriever + FAISS**（与 `dataset.sh` 默认一致）：

```bash
cd /path/to/Pandora-RAG/baselines/Stop-RAG
export HF_HOME="/path/to/Pandora-RAG/.hf_cache"
PYTHON_BIN="${STOP_RAG_PYTHON:-python}"
CONTRIEVER_MODEL_PATH="${STOP_RAG_CONTRIEVER_MODEL:-facebook/contriever-msmarco}"

for ds in hotpotqa 2wikimultihopqa musique; do
  "${PYTHON_BIN}" src/pipeline/contriever/passage_embedder.py \
    --passages "data/corpus/${ds}/passages/corpus.jsonl" \
    --output_dir "data/corpus/${ds}/contriever_embeddings" \
    --model_name_or_path "${CONTRIEVER_MODEL_PATH}"

  "${PYTHON_BIN}" -m src.pipeline.contriever.passage_retriever \
    --passages "data/corpus/${ds}/passages/corpus.jsonl" \
    --embeddings "data/corpus/${ds}/contriever_embeddings" \
    --save_or_load_index
done
```

若使用 **BM25**，将上面循环替换为：

```bash
for ds in hotpotqa 2wikimultihopqa musique; do
  "${PYTHON_BIN}" -m src.pipeline.bm25.bm25_retriever \
    --passages "data/corpus/${ds}/passages/corpus.jsonl" \
    --index_path_dir "data/corpus/${ds}/bm25_index" \
    --save_or_load_index
done
```

### 3. 跑管线生成 `data/processed/...`（轨迹与标签）

变量含义与上游相同：

- `DATASET`：`musique` | `hotpotqa` | `2wikimultihopqa`
- `RETRIEVER`：`contriever` | `bm25`
- `METHOD`：`ours` | `corag`

```bash
cd /path/to/Pandora-RAG/baselines/Stop-RAG
export HF_HOME="/path/to/Pandora-RAG/.hf_cache"
./scripts/dataset.sh "${DATASET}" "${RETRIEVER}" "${METHOD}"
```

这里生成的是**训练/开发用 partial traces 与标签**，不是公平测试结果。为了覆盖 `k=1..5` 的停止候选，脚本会在数据构造阶段保留完整 5 轮轨迹，再切成 partial traces；这一步只服务于后续训练和阈值选择。

当前与 Pandora 主实验对齐的默认控制项还包括：

- `max_iterations=5`
- `repeat_size=1`
- `STOP_RAG_VLLM_MAX_MODEL_LEN=2048`

可通过环境变量覆盖默认模型与 vLLM 参数（见 `scripts/dataset.sh` 中的 `STOP_RAG_*`）。

### 4. 训练 Stop-RAG 停止头

本仓库内 `train.sh` 默认关闭 WandB（`WANDB_DISABLED=true`）。

```bash
./scripts/train.sh "${DATASET}" "${RETRIEVER}" "${METHOD}"
```

### 5. 校验集选阈值与 checkpoint

```bash
./scripts/stop_rag_find_best.sh "${DATASET}" "${RETRIEVER}" "${METHOD}"
```

这一步会在 `eval_subsampled` 的 partial traces 上离线打分，选择最优 `checkpoint + threshold`。这是**开发集调参**，不是最终测试。

### 6. 测试集评测

将上一步打印的 `CKPT` 与 `THRESHOLD` 代入：

```bash
./scripts/stop_rag_test.sh "${DATASET}" "${RETRIEVER}" "${METHOD}" "${CKPT}" "${THRESHOLD}"
```

也支持一次顺序跑多个数据集：

```bash
./scripts/stop_rag_test.sh \
  hotpotqa,2wikimultihopqa,musique \
  "${RETRIEVER}" \
  "${METHOD}" \
  "${CKPT_HOTPOT},${CKPT_2WIKI},${CKPT_MUSIQUE}" \
  "${THR_HOTPOT},${THR_2WIKI},${THR_MUSIQUE}"
```

若三个数据集共用同一个 checkpoint / threshold，也可以只传一个值；`DATASET` 还可写成 `all`。

这个脚本现在是**公平对比的唯一推荐测试入口**：

- 它会在 `test_subsampled` 上执行**真实在线早停**。
- 每轮只在当前已检索历史上运行 stop head；若预测 `STOP`，后续轮次不会再检索。
- 输出写到 `results/<dataset>_<method>_<retriever>/online_test/`。

若你看到的是 `compute_scores/*.jsonl`、`src/test/stop_rag_test.py`、或任何“按 `iter_cnt` 回放挑第一条过阈值记录”的结果，请把它视为**离线分析文件**，不要拿来和 Pandora 的在线步数 / F1 做最终对比。

### 7. 多阈值在线 sweep 与 Pareto 汇总

论文主图需要 Stop-RAG 的完整预算曲线时，用同一个 checkpoint 跑多个 threshold 的**真实在线测试**：

```bash
cd /path/to/Pandora-RAG/baselines/stop-rag-pandora
./scripts/stop_rag_threshold_sweep.sh \
  hotpotqa \
  contriever \
  ours \
  1000 \
  "-0.20,-0.16,-0.12,-0.09,-0.06,-0.03,0.00,0.03,0.06,0.10"
```

三个数据集可一次跑完：

```bash
./scripts/stop_rag_threshold_sweep.sh \
  all \
  contriever \
  ours \
  "1000,2400,1200" \
  "-0.20,-0.16,-0.12,-0.09,-0.06,-0.03,0.00,0.03,0.06,0.10"
```

跑完后回到 Pandora 仓库根目录汇总：

```bash
cd /path/to/Pandora-RAG
python scripts/stop_rag_make_pareto.py
```

该汇总脚本只读取 `results/*/online_test/*.metrics.json` 与相邻在线 stop log / trace，不读取 `compute_scores` 离线回放。输出包括：

- `results/stop_rag_online_threshold_sweep.csv`
- `results/stop_rag_online_pareto_frontier.csv`
- `results/stop_rag_matched_budget.csv`
- `results/stop_rag_pareto_{hotpotqa,musique,2wiki}.png`
- `docs/reports/baselines/stop_rag_threshold_sweep.md`

### （可选）LLM-Stop 基线

```bash
./scripts/llm_stop_test.sh "${DATASET}" "${RETRIEVER}" "${METHOD}"
```

---

## 上游论文原始复现流程（可选，划分与 Pandora 不一致）

以下步骤来自官方 README，适用于**独立复现 Stop-RAG 论文**，**不**保证与 Pandora `data/processed` 的 id 与切分一致：

1. `./scripts/download.sh {RETRIEVER}` — 从 CMU / Dropbox / Google Drive 等拉取上游数据并子采样。  
2. `./scripts/dataset.sh {DATASET} {RETRIEVER} {METHOD}`  
3. `./scripts/train.sh ...` → `./scripts/stop_rag_find_best.sh` → `./scripts/stop_rag_test.sh`  

官方实验多在4×H100 规模下完成；单机可通过 `STOP_RAG_VLLM_TP_SIZE` 等环境变量酌情调整。

---

## Acknowledgments

The code for Contriever is adapted from [EfficientRAG](https://github.com/NIL-zhuang/EfficientRAG-official), and the dataset download scripts are adapted from [IRCoT](https://github.com/StonyBrookNLP/ircot).
