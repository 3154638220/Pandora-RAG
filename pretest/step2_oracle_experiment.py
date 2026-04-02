"""
Step 2：验证 Pandora's Box Oracle 停止效果
==========================================
核心问题：若拥有"上帝视角"（真实 F1 + 真实 Weitzman 保留值），
Pandora-RAG 能否以更少的步数达到接近 Always-K5 的性能？

流程：
  1. 从 train 轨迹统计每步 k 的信息增益分布 G_k。
  2. 求解 Weitzman 保留值 r_k*（c=0.05）。
  3. 在 test 轨迹上模拟 Oracle 停止。
  4. 对比：Always-K1 ~ Always-K5，Oracle-Pandora-RAG。
  5. 输出 Pareto 表格 + 散点图。

运行：python -m pretest.step2_oracle_experiment
"""
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pretest.config import cfg
from pretest.utils.weitzman import (
    compute_all_reservation_values,
    oracle_stopping_simulation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
def load_trajectories(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fixed_k_performance(test_trajs: list, k: int) -> dict:
    """模拟 Always-K=k 策略（取第 k 步结果，不足 k 步则取最后一步）。"""
    f1s, ems = [], []
    for traj in test_trajs:
        steps = traj["steps"]
        if not steps:
            continue
        # 找到 step==k 的条目，否则取最后一步
        target = next((s for s in steps if s["step"] == k), steps[-1])
        f1s.append(target["f1"])
        ems.append(int(target["em"]))
    return {
        "strategy": f"Always-K={k}",
        "avg_steps": k,
        "avg_f1": float(np.mean(f1s)),
        "avg_em": float(np.mean(ems)),
        "f1_std": float(np.std(f1s)),
    }


def summarize_oracle(oracle_results: list) -> dict:
    f1s = [r["f1"] for r in oracle_results]
    ems = [r["em"] for r in oracle_results]
    steps = [r["steps_used"] for r in oracle_results]
    return {
        "strategy": "Oracle-Pandora-RAG",
        "avg_steps": float(np.mean(steps)),
        "avg_f1": float(np.mean(f1s)),
        "avg_em": float(np.mean(ems)),
        "f1_std": float(np.std(f1s)),
    }


# ──────────────────────────────────────────────────────────────
def plot_pareto(rows: list, out_dir: str):
    """绘制 Pareto 前沿：平均检索步数 vs 平均 F1。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, cfg.max_k))
    oracle_color = "#e84040"

    ax = axes[0]
    for i, row in enumerate(rows):
        color = oracle_color if "Oracle" in row["strategy"] else colors[i]
        marker = "*" if "Oracle" in row["strategy"] else "o"
        size = 200 if "Oracle" in row["strategy"] else 100
        ax.scatter(row["avg_steps"], row["avg_f1"], color=color, marker=marker,
                   s=size, zorder=5, label=row["strategy"])
        ax.annotate(
            f"  {row['strategy']}\n  F1={row['avg_f1']:.3f}",
            (row["avg_steps"], row["avg_f1"]),
            fontsize=8, color=color,
        )
    ax.set_xlabel("平均检索步数 (Avg Retrieval Steps)")
    ax.set_ylabel("平均 F1 分数")
    ax.set_title("Pareto 前沿：成本 vs 质量")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    # 右图：柱状图对比
    ax2 = axes[1]
    strategies = [r["strategy"].replace("Always-", "K=").replace("-RAG", "\nRAG") for r in rows]
    f1_vals = [r["avg_f1"] for r in rows]
    bar_colors = [oracle_color if "Oracle" in r["strategy"] else "#5b9bd5" for r in rows]
    bars = ax2.bar(strategies, f1_vals, color=bar_colors, alpha=0.85)
    for bar, row in zip(bars, rows):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{row['avg_f1']:.3f}\n(μ_steps={row['avg_steps']:.1f})",
            ha="center", va="bottom", fontsize=8,
        )
    ax2.set_ylabel("平均 F1 分数")
    ax2.set_title("各停止策略 F1 对比")
    ax2.set_ylim(0, min(1.0, max(f1_vals) * 1.25))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "step2_pareto.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Pareto 图已保存至 %s", out_path)


def plot_stopping_distribution(oracle_results: list, out_dir: str):
    """绘制 Oracle 停止时步数分布直方图。"""
    steps = [r["steps_used"] for r in oracle_results]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(range(1, cfg.max_k + 2))
    ax.hist(steps, bins=[b - 0.5 for b in bins + [cfg.max_k + 1]],
            color="#e84040", alpha=0.75, edgecolor="white")
    ax.set_xlabel("停止时的检索步数 k")
    ax.set_ylabel("样本数")
    ax.set_xticks(bins[:-1])
    ax.set_title("Oracle Pandora-RAG 停止步数分布")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "step2_stopping_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("停止分布图已保存至 %s", out_path)


# ──────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(cfg.trajectory_file):
        raise FileNotFoundError(
            f"轨迹文件不存在：{cfg.trajectory_file}\n请先运行 step1_collect_trajectories.py"
        )

    logger.info("加载轨迹文件...")
    all_trajs = load_trajectories(cfg.trajectory_file)

    train_trajs = [t for t in all_trajs if t["split"] == "train"]
    test_trajs = [t for t in all_trajs if t["split"] == "test"]
    logger.info("train=%d，test=%d", len(train_trajs), len(test_trajs))

    # ── 1. 计算 Weitzman 保留值 ───────────────────────────────
    logger.info("计算 Weitzman 保留值（c=%.3f）...", cfg.cost_per_step)
    reservation_values = compute_all_reservation_values(
        train_trajs, cfg.max_k, cfg.cost_per_step
    )
    logger.info("保留值：%s", {k: f"{v:.4f}" for k, v in reservation_values.items()})

    # 保存保留值供 Step3 复用
    rv_path = os.path.join(cfg.data_dir, "reservation_values.json")
    with open(rv_path, "w") as f:
        json.dump(reservation_values, f, indent=2)
    logger.info("保留值已保存至 %s", rv_path)

    # ── 2. 模拟各 Always-K 策略 ───────────────────────────────
    rows = []
    for k in range(1, cfg.max_k + 1):
        row = fixed_k_performance(test_trajs, k)
        rows.append(row)
        logger.info(
            "Always-K=%d: avg_steps=%.1f, F1=%.4f, EM=%.4f",
            k, row["avg_steps"], row["avg_f1"], row["avg_em"],
        )

    # ── 3. 模拟 Oracle Pandora-RAG ────────────────────────────
    oracle_results = oracle_stopping_simulation(test_trajs, reservation_values, cfg.max_k)
    oracle_row = summarize_oracle(oracle_results)
    rows.append(oracle_row)

    logger.info(
        "Oracle-Pandora-RAG: avg_steps=%.2f, F1=%.4f, EM=%.4f",
        oracle_row["avg_steps"], oracle_row["avg_f1"], oracle_row["avg_em"],
    )

    # 关键指标：与 Always-K=max_k 的 F1 比较
    upper_bound_f1 = rows[cfg.max_k - 1]["avg_f1"]
    if upper_bound_f1 > 0:
        efficiency = oracle_row["avg_f1"] / upper_bound_f1 * 100
        cost_saving = (1 - oracle_row["avg_steps"] / cfg.max_k) * 100
        logger.info(
            "★ Oracle 用 %.2f 步（节省 %.1f%%）达到满血 F1 的 %.1f%%",
            oracle_row["avg_steps"], cost_saving, efficiency,
        )

    # ── 4. 输出表格 ───────────────────────────────────────────
    df = pd.DataFrame(rows)
    table_path = os.path.join(cfg.results_dir, "step2_results.csv")
    df.to_csv(table_path, index=False)
    print("\n" + "=" * 60)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 60)

    # ── 5. 绘图 ───────────────────────────────────────────────
    plot_pareto(rows, cfg.results_dir)
    plot_stopping_distribution(oracle_results, cfg.results_dir)

    # 保存 oracle 结果供 Step3 对比用
    oracle_path = os.path.join(cfg.data_dir, "oracle_results.json")
    with open(oracle_path, "w") as f:
        json.dump({"summary": oracle_row, "per_sample": oracle_results}, f, indent=2)


if __name__ == "__main__":
    main()
