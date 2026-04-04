# Pandora-RAG 存储布局说明（代码在 NVMe，数据在 haoge）

更新时间：2026-04-03

## 背景

项目代码位于系统 NVMe（`/home/x12dpg/hjx/Pandora-RAG`），便于高 I/O 开发与 Git 操作。大文件（数据集、模型权重、检查点、日志、Hugging Face 缓存）放在挂载于 `/home/x12dpg/haoge` 的大容量存储上，避免撑满系统盘。

## Conda 环境

| 项 | 说明 |
|----|------|
| **环境名** | `pandora-rag` |
| **Python** | 3.11 |
| **依赖** | 仓库根目录 `requirements.txt`（`pip install -r requirements.txt`） |
| **推理** | 同环境中已安装 **vLLM**（当前 `vllm==0.18.1`，随附 PyTorch CUDA；未写入 `requirements.txt`，其他机器需自行 `pip install vllm`） |

激活：`conda activate pandora-rag`。

## 大盘目录结构

实际数据根目录：

```text
/home/x12dpg/haoge/hjx_data/
├── Pandora-RAG/
│   ├── datasets/     # 项目内 ./data 指向此处
│   ├── models/       # 项目内 ./models 指向此处
│   ├── checkpoints/  # 训练/微调检查点
│   └── logs/         # 运行日志（与 .gitignore 中 logs/ 一致时，建议写此路径）
└── huggingface_cache/  # HF_HOME，见下文
```

## 项目根目录下的符号链接

在仓库根目录执行 `ls -la` 可见：

| 项目内路径     | 指向 |
|----------------|------|
| `data`         | `/home/x12dpg/haoge/hjx_data/Pandora-RAG/datasets` |
| `models`       | `/home/x12dpg/haoge/hjx_data/Pandora-RAG/models` |
| `checkpoints`  | `/home/x12dpg/haoge/hjx_data/Pandora-RAG/checkpoints` |
| `logs`         | `/home/x12dpg/haoge/hjx_data/Pandora-RAG/logs` |

代码中仍使用 `data/`、`models/` 等相对路径即可，无需改逻辑。

### Stage1 本地权重与 `--root-dir`

- **Hidden states** 默认使用 **Meta-Llama-3.1-8B-Instruct** 本地目录。解析顺序为：`{--root-dir}/models/...` → **本仓库根目录** `models/Meta-Llama-3.1-8B-Instruct`（即上表符号链接，实际在 haoge）→ 环境变量 **`PANDORA_MODELS_ROOT`**（可设为 `/home/x12dpg/haoge/hjx_data/Pandora-RAG/models`）。因此即使用 `--root-dir` 指到临时目录，只要仓库内 `models` 已正确链接到大盘，仍会加载 haoge 上的权重，无需手写绝对路径。
- 若在无 symlink 的机器上仅有大盘路径，可设置：`export PANDORA_MODELS_ROOT=/home/x12dpg/haoge/hjx_data/Pandora-RAG/models`。

### 数据集与 Hugging Face `test` split

- **HotpotQA**、**MuSiQue** 在 Hugging Face 上通常只有 `train` 与 `validation`，**没有**名为 `test` 的 split。Stage1 会从 `validation` 中切出 dev / test 两段，与官方实验划分习惯一致；代码不再对这两个数据集请求不存在的 `test` split，因此不会出现误报的加载失败日志。

## Hugging Face 缓存（`HF_HOME`）

已在 `~/.bashrc` 末尾增加：

```bash
export HF_HOME="/home/x12dpg/haoge/hjx_data/huggingface_cache"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

**新开终端**或执行 `source ~/.bashrc` 后生效。`transformers` / `datasets` / `huggingface_hub` 会把模型与数据集缓存在该目录下，而不再使用默认的 `~/.cache/huggingface`（原目录已删除以释放系统盘空间）。

### 关于 haoge 与符号链接

当前 haoge 挂载（CIFS/SMB）**不支持创建符号链接**（`ln -s` 会报 Input/output error）。Hugging Face Hub 默认在缓存里大量使用「快照目录 → blobs」的相对符号链接。

因此迁移缓存时采用了 **`rsync -aL`**（跟随符号链接、复制真实文件），而不是保留链接。后果是：

- 大盘上缓存体积会比「仅 blobs + 链接」略大（存在 blobs 与快照中的重复内容），但在约 68T 可用空间下通常可接受。
- 之后在本机使用 `HF_HOME` 指向该路径时，`huggingface_hub` 会检测到该目录不支持 symlink，新下载会走「非链接」策略，与当前磁盘能力一致。

若将来将 haoge 以支持 `mfsymlinks` 等方式重新挂载并确认可创建 symlink，可再考虑改用链接式缓存以节省空间（需自行评估与重新同步）。

## 克隆本仓库到其他机器时

目标机器上若没有相同的 `/home/x12dpg/haoge/...` 挂载，需要先创建上述目录，再在项目根目录按相同方式 `ln -s`，或改为自己环境下的数据根路径。

## 常用检查命令

```bash
df -h /home/x12dpg/hjx/Pandora-RAG /home/x12dpg/haoge
ls -la /home/x12dpg/hjx/Pandora-RAG
echo "$HF_HOME"
du -sh /home/x12dpg/haoge/hjx_data/Pandora-RAG/* /home/x12dpg/haoge/hjx_data/huggingface_cache
```

## Git 注意

`data`、`models` 现为符号链接。若仓库曾跟踪过其下文件，克隆侧或 `git status` 可能显示变更；按需将「仅本地数据」保留在大盘，并在 `.gitignore` 中排除不应版本化的路径（本仓库已忽略部分大文件扩展名与 `logs/` 等）。
