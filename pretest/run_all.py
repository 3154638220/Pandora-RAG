"""
Pandora-RAG 预实验总运行入口
============================
按顺序执行三个步骤，并汇总最终结论。

用法：
    python -m pretest.run_all [--force]

    --force : 强制重新收集轨迹（即使 data/trajectories.json 已存在）
"""
import argparse
import json
import logging
import os
import sys
import time

from pretest.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 65


def run_step(name: str, module_path: str) -> float:
    """动态导入并运行某一步的 main()，返回耗时（秒）。"""
    logger.info("%s\n▶  %s\n%s", SEPARATOR, name, SEPARATOR)
    import importlib
    mod = importlib.import_module(module_path)
    t0 = time.time()
    mod.main()
    elapsed = time.time() - t0
    logger.info("✔  %s 完成（耗时 %.1f s）\n", name, elapsed)
    return elapsed


def print_summary():
    """读取各步结果文件，打印总结报告。"""
    print(f"\n{SEPARATOR}")
    print("  Pandora-RAG 预实验总结报告")
    print(SEPARATOR)

    # Step2 结果
    s2_path = os.path.join(cfg.results_dir, "step2_results.csv")
    if os.path.exists(s2_path):
        import pandas as pd
        df2 = pd.read_csv(s2_path)
        print("\n[Step 2] Oracle 停止效果（Pareto 表）：")
        print(df2.to_string(index=False, float_format="%.4f"))

        # 计算关键结论
        upper_row = df2[df2["strategy"] == f"Always-K={cfg.max_k}"]
        oracle_row = df2[df2["strategy"] == "Oracle-Pandora-RAG"]
        if not upper_row.empty and not oracle_row.empty:
            ub_f1 = upper_row["avg_f1"].values[0]
            ora_f1 = oracle_row["avg_f1"].values[0]
            ora_steps = oracle_row["avg_steps"].values[0]
            cost_save = (1 - ora_steps / cfg.max_k) * 100
            quality_ratio = ora_f1 / ub_f1 * 100 if ub_f1 > 0 else 0
            print(f"\n  ★ 结论：Oracle 用 {ora_steps:.2f} 步（节省 {cost_save:.1f}%），")
            print(f"         达到满血 F1 的 {quality_ratio:.1f}%。")
            print(f"         → Pandora's Box 理论在 RAG 中不是玄学！")

    # Step3 结果
    s3_path = os.path.join(cfg.results_dir, "step3_results.csv")
    if os.path.exists(s3_path):
        import pandas as pd
        df3 = pd.read_csv(s3_path)
        print("\n[Step 3] ML Probe vs 固定阈值 vs Oracle：")
        print(df3.to_string(index=False, float_format="%.4f"))

        probe_row = df3[df3["strategy"] == "Probe-Pandora-RAG"]
        oracle_row = df3[df3["strategy"] == "Oracle-Pandora-RAG"]
        ft_rows = df3[df3["strategy"].str.startswith("Fixed")]

        if not probe_row.empty and not ft_rows.empty:
            probe_f1 = probe_row["avg_f1"].values[0]
            best_ft_f1 = ft_rows["avg_f1"].max()
            delta = probe_f1 - best_ft_f1
            print(f"\n  ★ 结论：Probe 比最佳固定阈值 F1 高 {delta:+.4f}。")

        if not probe_row.empty and not oracle_row.empty:
            probe_f1 = probe_row["avg_f1"].values[0]
            ora_f1 = oracle_row["avg_f1"].values[0]
            ratio = probe_f1 / ora_f1 * 100 if ora_f1 > 0 else 0
            probe_steps = probe_row["avg_steps"].values[0]
            print(f"         Probe 达到 Oracle 的 {ratio:.1f}%，"
                  f"平均仅用 {probe_steps:.2f} 步。")
            print(f"         → 轻量级特征足够，无需复杂 PPO！")

    print(f"\n  输出文件目录：{cfg.results_dir}/")
    print(f"  轨迹缓存：    {cfg.trajectory_file}")
    print(SEPARATOR + "\n")


# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="运行 Pandora-RAG 预实验全流程")
    parser.add_argument("--force", action="store_true", help="强制重新收集轨迹")
    parser.add_argument("--skip-step1", action="store_true",
                        help="跳过轨迹收集（假设已存在）")
    args = parser.parse_args()

    if args.force and os.path.exists(cfg.trajectory_file):
        os.remove(cfg.trajectory_file)
        logger.info("已删除旧轨迹文件，将重新收集。")

    timings = {}

    # Step 1
    if not args.skip_step1:
        timings["step1"] = run_step(
            "Step 1: 离线 RAG 轨迹收集",
            "pretest.step1_collect_trajectories",
        )
    else:
        logger.info("跳过 Step 1（--skip-step1）。")

    # Step 2
    timings["step2"] = run_step(
        "Step 2: Oracle Pandora's Box 停止验证",
        "pretest.step2_oracle_experiment",
    )

    # Step 3
    timings["step3"] = run_step(
        "Step 3: ML Probe 轻量停止验证",
        "pretest.step3_probe_experiment",
    )

    print_summary()
    total = sum(timings.values())
    logger.info("全部步骤完成！总耗时 %.1f s（不含跳过步骤）。", total)


if __name__ == "__main__":
    main()
