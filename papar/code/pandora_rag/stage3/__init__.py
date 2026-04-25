"""
Stage 3：E-process / E-value 风险控制与探针停止策略的集成评估。

设计要点：
  - Stage2 的 checkpoint、特征拼接与推理细节集中在 ``stage3.adapters.stage2_probe``
  - ``evalue.py``：E-wealth tracker 与多种 betting multiplier（indicator / outcome-aware）
  - ``quality_model.py``：P(F1≥γ) 质量预测器（LogReg + 校准评估）
  - ``stopping.py``：停止策略仿真（Probe / E-value 门控 / CP 门控 / 分布漂移构造）
  - ``config.py``：Stage3Config（含 betting 策略、分布漂移、多 γ/α）
  - ``run_stage3.py``：主入口（CLI → JSON + 累积误差曲线 + wealth trace + Markdown 报告）
"""
