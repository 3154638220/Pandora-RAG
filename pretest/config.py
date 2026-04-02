"""全局配置。可通过环境变量或直接修改默认值来调整实验参数。"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── 数据集 ──────────────────────────────────────────────
    dataset_name: str = "hotpot_qa"
    dataset_config: str = "distractor"   # HotpotQA distractor setting（含 10 段落/题）
    train_size: int = 1000               # 用于统计 G_k 分布的训练样本数
    test_size: int = 500                 # 用于评测停止策略的测试样本数
    random_seed: int = 42

    # ── RAG 设置 ─────────────────────────────────────────────
    max_k: int = 5                       # 最大检索轮数 K

    # ── Weitzman 参数 ─────────────────────────────────────────
    cost_per_step: float = 0.05          # 每步检索的虚拟成本 c

    # ── LLM ──────────────────────────────────────────────────
    # OpenAI 兼容接口：vLLM / TGI / Ollama。默认对齐第一阶段 Llama-3.1-8B-Instruct。
    llm_backend: str = "openai"
    model_name: str = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    api_base: Optional[str] = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "EMPTY"))
    max_tokens: int = 150
    temperature: float = 0.0

    # ── 路径 ──────────────────────────────────────────────────
    data_dir: str = "data"
    results_dir: str = "results"
    trajectory_file: str = "data/trajectories.json"

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# 全局默认配置实例（各脚本直接 import 使用）
cfg = Config()
