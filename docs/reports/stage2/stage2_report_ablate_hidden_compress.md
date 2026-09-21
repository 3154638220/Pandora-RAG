# Stage2 消融：Hidden 分支压缩维度（4096→d）

对应 `docs/plan_claude.md` 中「Hidden states 压缩维度提升」：在 `ProbeMLP_v2` 中将 `compress_dim` 从默认 **64** 提升到 **256 / 512**，检验 test F1 是否随瓶颈放宽而提升。

## 复现实验

```bash
# 默认 d=64：与仓库当前主结果一致（见 docs/reports/stage2/stage2_final.md）
python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --compress-dim 256 --artifact-suffix ablate_compress_dim256

python -m stage2.run_stage2 --datasets hotpotqa,musique,2wiki \
  --compress-dim 512 --artifact-suffix ablate_compress_dim512
```

其余超参、Phase C（`gw_steps_cap_mult=1.05`）与默认 Stage2 一致。

## 结果汇总（test）


| compress_dim | HotpotQA F1 | MuSiQue F1 | 2Wiki F1 | 相对 d=64（Hotpot / MuSiQue / 2Wiki） |
| ------------ | ----------- | ---------- | -------- | --------------------------------- |
| **64**（默认）   | 0.5148      | 0.1794     | 0.3484   | —                                 |
| **256**      | **0.5273**  | **0.1826** | 0.3273   | +0.0125 / +0.0032 / −0.0211       |
| **512**      | 0.5251      | 0.1818     | 0.3047   | +0.0103 / +0.0024 / −0.0437       |


平均步数（test）：64 基线为 HotpotQA 2.532、MuSiQue 3.089、2Wiki 2.411；256 为 2.854 / 3.252 / 2.247；512 为 2.885 / 3.590 / 1.975（2Wiki 在 d=512 上阈值相同但停得更早，F1 明显变差）。

### 相对 Shallow-Only（w/o Deep Features）的增益

Shallow-Only：`docs/reports/stage2/stage2_report_shallow.md`（HotpotQA 0.4829，MuSiQue 0.1394，2Wiki 0.3552）。


| compress_dim | ΔF1 vs Shallow（Hotpot / MuSiQue / 2Wiki） |
| ------------ | ---------------------------------------- |
| 64           | +0.0319 / +0.0400 / −0.0068              |
| 256          | +0.0444 / +0.0432 / −0.0279              |
| 512          | +0.0422 / +0.0424 / −0.0505              |


## 结论

1. **HotpotQA / MuSiQue**：将 `compress_dim` 提到 **256** 带来小幅稳定的 test F1 提升；**512** 相对 256 **无一致收益**（HotpotQA 略降，MuSiQue 略降）。
2. **2Wiki**：增大压缩维度后 test F1 **持续下降**，且相对 Shallow-Only 的 deep 增益变负，更符合 **过拟合 / 与 Phase C 阈值–步数权衡耦合** 而非「越大越好」。
3. 与 plan 中「希望从 +0.02~0.05 拉到 +0.08+」相比：在 **HotpotQA 上相对 d=64 的额外增益约 +0.01**，**未达到 +0.08**；全三数据集 **不存在单一 d 同时 Pareto 最优**。

**论文/工程建议**：若只选一个折中，可优先考虑 **d=256** 作为「放宽瓶颈」的消融点；2Wiki 仍建议保持 **d=64** 或单独调参（如 `fuse_dim`、正则或按数据集选 d）。详细分项报告与 CSV/PNG：`docs/reports/stage2/stage2_report_ablate_compress_dim256.md`、`docs/reports/stage2/stage2_report_ablate_compress_dim512.md`。
