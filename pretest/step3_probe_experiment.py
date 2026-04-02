"""
Step 3：验证"无需重度微调"——轻量 ML Probe 也能 Work
====================================================
核心问题：能否用 LLM 输出的浅层特征（logprob、熵等）预测当前答案质量 Q(s_k)，
并以此替代"上帝视角"驱动停止决策？

流程：
  1. 从轨迹中提取特征 (mean_logprob, entropy, token_count, bm25_score, step_k) 与标签 Q(s_k)。
  2. 在 train 集上训练 XGBoost 回归器预测 Q(s_k)。
  3. 在 test 集上：用 predicted_Q(s_k) 与 Oracle r_{k+1}* 做比较来决定是否停止。
  4. 对比三类策略：
       - Oracle-Pandora-RAG（Step2 结果，性能天花板）
       - Probe-Pandora-RAG（本步骤，预测质量 + Oracle 保留值）
       - Fixed-Threshold-RAG（固定阈值 τ ∈ {0.3, 0.5, 0.7}）
  5. 绘制特征重要性图 + 性能对比图。

运行：python -m pretest.step3_probe_experiment
"""
import json
import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from pretest.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_NAMES = ["mean_logprob", "entropy", "token_count", "bm25_score", "step"]


# ──────────────────────────────────────────────────────────────
def load_data() -> Tuple[List[dict], List[dict], Dict[int, float]]:
    for path, desc in [
        (cfg.trajectory_file, "轨迹文件"),
        (os.path.join(cfg.data_dir, "reservation_values.json"), "保留值文件"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{desc}不存在：{path}\n请先运行 step1 和 step2。")

    with open(cfg.trajectory_file, encoding="utf-8") as f:
        all_trajs = json.load(f)
    with open(os.path.join(cfg.data_dir, "reservation_values.json")) as f:
        # JSON 的 key 是字符串，转为 int
        rv = {int(k): v for k, v in json.load(f).items()}

    train_trajs = [t for t in all_trajs if t["split"] == "train"]
    test_trajs = [t for t in all_trajs if t["split"] == "test"]
    return train_trajs, test_trajs, rv


def extract_features_labels(trajs: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """从轨迹中提取 (X: features, y: F1_score) 对。每步 k 产生一个样本。"""
    X_rows, y_vals = [], []
    for traj in trajs:
        for step_data in traj["steps"]:
            feat = step_data["features"]
            row = [feat.get(name, 0.0) for name in FEATURE_NAMES]
            X_rows.append(row)
            y_vals.append(step_data["f1"])
    return np.array(X_rows, dtype=float), np.array(y_vals, dtype=float)


# ──────────────────────────────────────────────────────────────
def train_probe(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.random_seed,
        verbosity=0,
    )
    # 5 折交叉验证评估泛化能力
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    logger.info("5折CV R²: mean=%.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────────────────────
def probe_stopping_simulation(
    test_trajs: List[dict],
    model: XGBRegressor,
    reservation_values: Dict[int, float],
) -> List[dict]:
    """
    用 Probe 预测的 Q̂(s_k) 与 Oracle r_{k+1}* 比较来做停止决策。
    停止规则：若 Q̂(s_k) >= r_{k+1}*，停止；否则继续。
    """
    results = []
    for traj in test_trajs:
        steps = traj["steps"]
        final_f1, final_em, steps_used = 0.0, False, 0

        for step_data in steps:
            k = step_data["step"]
            steps_used = k
            feat = step_data["features"]
            x = np.array([[feat.get(name, 0.0) for name in FEATURE_NAMES]])
            predicted_q = float(model.predict(x)[0])
            predicted_q = max(0.0, min(1.0, predicted_q))  # clip to [0,1]

            next_r = reservation_values.get(k + 1, -999.0) if k < cfg.max_k else -999.0
            if predicted_q >= next_r or k == cfg.max_k:
                final_f1 = step_data["f1"]
                final_em = step_data["em"]
                break

        results.append({"f1": final_f1, "em": final_em, "steps_used": steps_used})
    return results


def fixed_threshold_simulation(
    test_trajs: List[dict],
    threshold: float,
) -> List[dict]:
    """
    基于"预测质量 > 固定阈值"的停止策略。
    使用真实 F1 作为质量代理（模拟理想特征），验证阈值策略的上限。
    注：实际部署时用 predicted F1，此处用 true F1 给固定阈值最好的条件，做对比更公平。
    """
    results = []
    for traj in test_trajs:
        steps = traj["steps"]
        final_f1, final_em, steps_used = 0.0, False, 0
        for step_data in steps:
            k = step_data["step"]
            steps_used = k
            if step_data["f1"] >= threshold or k == cfg.max_k:
                final_f1 = step_data["f1"]
                final_em = step_data["em"]
                break
        results.append({"f1": final_f1, "em": final_em, "steps_used": steps_used})
    return results


def summarize(results: list, strategy: str) -> dict:
    f1s = [r["f1"] for r in results]
    ems = [r["em"] for r in results]
    steps = [r["steps_used"] for r in results]
    return {
        "strategy": strategy,
        "avg_steps": float(np.mean(steps)),
        "avg_f1": float(np.mean(f1s)),
        "avg_em": float(np.mean(ems)),
        "f1_std": float(np.std(f1s)),
    }


# ──────────────────────────────────────────────────────────────
def plot_feature_importance(model: XGBRegressor, out_dir: str):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sorted_names[::-1], sorted_vals[::-1], color="#5b9bd5", alpha=0.85)
    for bar, val in zip(bars, sorted_vals[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("特征重要性（XGBoost gain）")
    ax.set_title("Probe 特征重要性：哪些信号最能预测答案质量？")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "step3_feature_importance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("特征重要性图已保存至 %s", out_path)


def plot_pred_vs_true(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: XGBRegressor,
    out_dir: str,
):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color="#e84040")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="理想预测线")
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ax.set_xlabel("真实 F1 分数 Q(s_k)")
    ax.set_ylabel("预测 F1 分数 Q̂(s_k)")
    ax.set_title(f"Probe 预测 vs 真实质量\nMAE={mae:.4f}, R²={r2:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "step3_pred_vs_true.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("预测 vs 真实图已保存至 %s", out_path)


def plot_strategy_comparison(rows: list, upper_bound_f1: float, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    strategies = [r["strategy"] for r in rows]
    f1_vals = [r["avg_f1"] for r in rows]
    step_vals = [r["avg_steps"] for r in rows]

    colors = []
    for r in rows:
        if "Oracle" in r["strategy"]:
            colors.append("#e84040")
        elif "Probe" in r["strategy"]:
            colors.append("#2ca02c")
        else:
            colors.append("#aec7e8")

    ax1 = axes[0]
    bars = ax1.bar(range(len(rows)), f1_vals, color=colors, alpha=0.85)
    ax1.axhline(upper_bound_f1, color="#e84040", linestyle="--", lw=1.5,
                label=f"Always-K={cfg.max_k} 上界 ({upper_bound_f1:.3f})")
    for i, (bar, row) in enumerate(zip(bars, rows)):
        ax1.text(i, bar.get_height() + 0.005,
                 f"F1={row['avg_f1']:.3f}", ha="center", fontsize=8)
    ax1.set_xticks(range(len(rows)))
    ax1.set_xticklabels(strategies, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("平均 F1 分数")
    ax1.set_title("平均 F1 对比")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    bars2 = ax2.bar(range(len(rows)), step_vals, color=colors, alpha=0.85)
    for i, (bar, row) in enumerate(zip(bars2, rows)):
        ax2.text(i, bar.get_height() + 0.02,
                 f"{row['avg_steps']:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(range(len(rows)))
    ax2.set_xticklabels(strategies, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("平均检索步数")
    ax2.set_title("平均步数对比（越少越省）")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "step3_strategy_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("策略对比图已保存至 %s", out_path)


# ──────────────────────────────────────────────────────────────
def main():
    train_trajs, test_trajs, reservation_values = load_data()
    logger.info("train=%d，test=%d", len(train_trajs), len(test_trajs))

    # ── 1. 提取特征 ───────────────────────────────────────────
    X_train, y_train = extract_features_labels(train_trajs)
    X_test, y_test = extract_features_labels(test_trajs)
    logger.info("特征维度：train=%s，test=%s", X_train.shape, X_test.shape)

    # ── 2. 训练 XGBoost Probe ────────────────────────────────
    logger.info("训练 XGBoost Probe 预测 Q(s_k)...")
    model = train_probe(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    logger.info("测试集预测：MAE=%.4f，R²=%.4f", mae, r2)

    # ── 3. 模拟停止策略 ───────────────────────────────────────
    # Probe-Pandora-RAG
    probe_results = probe_stopping_simulation(test_trajs, model, reservation_values)
    probe_row = summarize(probe_results, "Probe-Pandora-RAG")
    logger.info(
        "Probe-Pandora-RAG: avg_steps=%.2f, F1=%.4f",
        probe_row["avg_steps"], probe_row["avg_f1"],
    )

    # Fixed-Threshold 系列
    rows = []
    for tau in [0.3, 0.5, 0.7]:
        ft_results = fixed_threshold_simulation(test_trajs, tau)
        ft_row = summarize(ft_results, f"Fixed-τ={tau}")
        rows.append(ft_row)
        logger.info(
            "Fixed-Threshold(τ=%.1f): avg_steps=%.2f, F1=%.4f",
            tau, ft_row["avg_steps"], ft_row["avg_f1"],
        )

    # Always-K5 上界（从轨迹计算）
    ak5_f1s = [
        t["steps"][-1]["f1"] for t in test_trajs if t["steps"]
    ]
    upper_bound_f1 = float(np.mean(ak5_f1s))

    # 加载 Oracle 结果（Step2 产物）
    oracle_path = os.path.join(cfg.data_dir, "oracle_results.json")
    oracle_row = None
    if os.path.exists(oracle_path):
        with open(oracle_path) as f:
            oracle_row = json.load(f)["summary"]
            oracle_row["strategy"] = "Oracle-Pandora-RAG"

    all_rows = rows + [probe_row]
    if oracle_row:
        all_rows.append(oracle_row)

    # ── 4. 输出对比表 ─────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    table_path = os.path.join(cfg.results_dir, "step3_results.csv")
    df.to_csv(table_path, index=False)
    print("\n" + "=" * 70)
    print(df.to_string(index=False, float_format="%.4f"))
    print("=" * 70)

    # 关键结论日志
    if oracle_row:
        oracle_f1 = oracle_row["avg_f1"]
        probe_f1 = probe_row["avg_f1"]
        best_ft_f1 = max(r["avg_f1"] for r in rows)
        logger.info(
            "★ Probe vs Fixed-Threshold 最佳：F1 差值 = %.4f（+%.1f%%）",
            probe_f1 - best_ft_f1,
            (probe_f1 - best_ft_f1) / (best_ft_f1 + 1e-8) * 100,
        )
        logger.info(
            "★ Probe vs Oracle：F1 差值 = %.4f（Probe 达到 Oracle 的 %.1f%%）",
            probe_f1 - oracle_f1,
            probe_f1 / (oracle_f1 + 1e-8) * 100,
        )

    # ── 5. 绘图 ───────────────────────────────────────────────
    plot_feature_importance(model, cfg.results_dir)
    plot_pred_vs_true(X_test, y_test, model, cfg.results_dir)
    plot_strategy_comparison(all_rows, upper_bound_f1, cfg.results_dir)

    # 保存模型
    import pickle
    model_path = os.path.join(cfg.data_dir, "probe_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Probe 模型已保存至 %s", model_path)


if __name__ == "__main__":
    main()
