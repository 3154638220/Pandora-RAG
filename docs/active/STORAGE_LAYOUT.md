# Pandora-RAG 存储布局说明（单仓库根目录）

更新时间：2026-04-16

## 背景

代码与**大文件默认均落在同一仓库根目录**（示例：`/home/x12dpg/hjx/Pandora-RAG`），不再依赖外置挂载盘。若系统盘空间紧张，可将个别目录改为符号链接指向其它磁盘，或仅用环境变量把模型根目录指到别处（见下文 `PANDORA_MODELS_ROOT`）。

## Conda 环境

| 项 | 说明 |
| --- | --- |
| **环境名** | `pandora-rag` |
| **Python** | 3.11 |
| **依赖** | 仓库根目录 `requirements.txt`（`pip install -r requirements.txt`） |
| **推理** | 同环境中可安装 **vLLM**（未写入 `requirements.txt`，其它机器需自行安装） |

激活：`conda activate pandora-rag`。

## 仓库内推荐目录结构

均在 `<repo>/` 下使用**真实目录**（不要用已失效路径做符号链接）：

```text
<repo>/
├── data/                 # 数据集；Stage1 写入 data/processed/（见 .gitignore）
├── models/               # 本地权重（如 Meta-Llama-3.1-8B-Instruct、NLI 等）
├── checkpoints/         # 训练/微调检查点
├── logs/                 # 运行日志（*.log 亦在 .gitignore）
├── cache/                # 轨迹与特征缓存（trajectories、features）
├── artifacts/            # Oracle 等产物
├── results/              # 图表与实验输出
└── .hf_cache/            # Hugging Face Hub 缓存（由代码默认设置 HF_HOME，见下文）
```

代码中继续使用 `data/`、`models/` 等**相对仓库根**的路径即可。

### Stage1 本地权重与 `--root-dir`

- **Hidden states** 默认使用 **Meta-Llama-3.1-8B-Instruct** 本地目录。解析顺序：`{--root-dir}/models/Meta-Llama-3.1-8B-Instruct` → **本仓库根** `models/Meta-Llama-3.1-8B-Instruct` → 环境变量 **`PANDORA_MODELS_ROOT`** 下的同名子目录。
- 若权重只放在非仓库路径，可设置：`export PANDORA_MODELS_ROOT=/你的路径/models`（该目录下仍为 `Meta-Llama-3.1-8B-Instruct/` 子文件夹）。

### 数据集与 Hugging Face `test` split

- **HotpotQA**、**MuSiQue** 在 Hub 上通常只有 `train` 与 `validation`，**没有**可用的 `test` gold。Stage1 从 `validation` 中切出 dev / test；代码不请求不存在的 `test` split。

## Hugging Face 缓存（`HF_HOME`）

**默认**：`stage1.run_stage1`、`scripts/download_stage1_assets.py`、`pretest.step1_collect_trajectories` 启动时将 **`HF_HOME` 设为 `<repo>/.hf_cache`**（自动创建）。与根目录 `.env.example` 中示例一致：

```bash
HF_HOME=/home/x12dpg/hjx/Pandora-RAG/.hf_cache
```

若需保留 shell 里已导出的其它 `HF_HOME`，设置 `export PANDORA_USE_SYSTEM_HF_HOME=1`。

可选：

```bash
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

`transformers` / `datasets` / `huggingface_hub` 会把下载内容缓存在 `HF_HOME` 下。

## 克隆到其它机器

在仓库根目录按需创建 `data/`、`models/`、`checkpoints/`、`logs/` 等；大权重用 `scripts/download_stage1_assets.py` 或自行拷贝到 `models/`。若曾使用旧环境的符号链接且目标路径已不存在，应删除断链并改成本地目录，例如：

```bash
cd /path/to/Pandora-RAG
rm -f checkpoints logs   # 若为指向不存在路径的符号链接
mkdir -p checkpoints logs
```

## 常用检查命令

```bash
df -h "$(pwd)"
ls -la
echo "$HF_HOME"
du -sh .hf_cache data models checkpoints cache 2>/dev/null
```

## Git 注意

`data/processed/`、`cache/`、`artifacts/`、`logs/`、`.hf_cache/` 等已在 `.gitignore` 中忽略；模型二进制扩展名亦被忽略。仅提交代码与小配置文件即可。
