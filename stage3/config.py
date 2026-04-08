from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Stage3Config:
    """Stage3 运行配置（路径与超参）；与 Stage2 对齐的字段委托给 ``Stage2Config``。"""

    root_dir: Path = Path(".")
    max_k: int = 5
    cost_per_step: float = 0.05
    oracle_cost_metric: str = "fixed"
    hidden_state_key: str = "last_token"
    artifact_suffix: str = ""
    # F1 低于该阈值视为「错误停止」（与 README / plan 中 γ 一致）
    gamma: float = 0.5
    # 目标名义水平；E-wealth 上界为 1/alpha
    alphas: Tuple[float, ...] = (0.1, 0.2)
    # 测试集上 E-process 跨样本顺序（可重复）
    shuffle_test_seed: int = 42
    # 质量头（逻辑回归）
    quality_max_iter: int = 300
    quality_random_state: int = 42
    # 绘图与汇总输出
    results_dir: Path = Path("results")

    @property
    def artifacts_probe_dir(self) -> Path:
        return self.root_dir / "artifacts" / "probe"
