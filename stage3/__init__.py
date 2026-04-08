"""
Stage 3：E-process / E-value 风险控制与探针停止策略的集成评估。

设计要点：Stage2 的 checkpoint、特征拼接与推理细节集中在 ``stage3.adapters.stage2_probe``；
本包其余模块只依赖「探针给出的继续概率」「浅层特征向量」等抽象数据，便于 Stage2 改版时局部替换适配器。
"""
