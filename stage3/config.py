from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple


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
    # 多 γ 扫描（run_stage3 的 --gammas 覆盖此项时 gamma 被忽略）
    gammas: Tuple[float, ...] = (0.5,)
    # 目标名义水平；E-wealth 上界为 1/alpha
    alphas: Tuple[float, ...] = (0.1, 0.2)

    # ---- Betting 策略 ----
    betting_strategy: Literal["fixed", "predictive"] = "predictive"
    betting_lambda: float = 0.5
    # outcome_aware=True 使用结果感知型 E-process（推荐）
    # outcome_aware=False 保留原版 indicator betting 做对照
    outcome_aware: bool = True

    # ---- Quality Model ----
    # 是否将 Probe p_continue 加入质量模型特征
    quality_use_probe_prob: bool = True
    quality_max_iter: int = 300
    quality_random_state: int = 42

    # ---- 分布漂移实验 ----
    shift_type: Literal["none", "sudden", "gradual", "periodic"] = "none"
    # 漂移起始位置（占总样本比例）；sudden 在此处切换，gradual 从此处开始增大难度
    shift_fraction: float = 0.5
    # 漂移排序键：按此字段升序作为"从难到易"或反序
    shift_sort_key: str = "f1"

    # ---- 测试集 E-process 跨样本顺序 ----
    shuffle_test_seed: int = 42

    # ---- 输出 ----
    results_dir: Path = Path("results")

    @property
    def artifacts_probe_dir(self) -> Path:
        return self.root_dir / "artifacts" / "probe"
