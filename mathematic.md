# Pandora-RAG 数学主线

> 目标：给论文提供一条与当前实现和实验结果一致的数学叙事。  
> 当前可安全支撑的主线是：**Bellman Oracle for utility optimization -> learned stopping signal for deployment -> E-process for anytime-valid risk monitoring**。

---

## 1. 问题设置

给定问题 \(q\)，多跳 RAG 系统按固定顺序最多执行 \(K\) 轮检索。第 \(k\) 轮后的可观测状态为

$$
s_k=(q,d_1,\ldots,d_k),\qquad k=0,1,\ldots,K,
$$

其中 \(s_0=q\)，\(d_k\) 是第 \(k\) 次检索得到的新证据。令

$$
\mathcal{F}_k=\sigma(s_0,\ldots,s_k)
$$

表示到第 \(k\) 步为止的历史信息。停止时刻 \(\tau\in\{0,\ldots,K\}\) 必须是关于过滤族 \((\mathcal{F}_k)_{k=0}^K\) 的 stopping time，即是否在第 \(k\) 步停止只能依赖当前及过去信息。

若系统在 \(s_k\) 停止并生成答案，答案质量记为

$$
Q(s_k)\in[0,1].
$$

实验中 \(Q\) 主要取 token-level F1，也同时汇报 EM。第 \(k+1\) 次检索的成本为 \(c_{k+1}>0\)。当前主实验使用常数成本 \(c_k\equiv0.05\)，代码也支持按 token 或 latency 归一化的步级成本。

---

## 2. 两层目标：效用优化与风险监控

Pandora-RAG 不把“最大化效用”和“错误率控制”揉成一个单一优化问题。更稳妥、也更符合当前实现的写法是两层：

**主目标：成本-质量效用优化。**

$$
U(\tau)=Q(s_\tau)-\sum_{j=1}^{\tau}c_j.
\tag{1}
$$

最优停止策略定义为

$$
\pi^*\in\arg\max_\pi
\mathbb{E}_\pi\left[
Q(s_{\tau_\pi})-\sum_{j=1}^{\tau_\pi}c_j
\right].
\tag{2}
$$

**部署安全层：在线风险监控。**

给定质量阈值 \(\gamma\) 和名义错误率 \(\alpha\)，部署流中用 E-process 检测“当前系统错误率是否持续高于 \(\alpha\)”。该层不承诺自动把错误率压到 \(\alpha\) 以下，而是在错误累积时给出 anytime-valid 的告警证据。

这个拆分是论文中最重要的口径：Stage 2 Probe 负责效用近似，Stage 3 E-value 负责风险监控。

---

## 3. Fixed-Order Optimal Stopping Oracle

### 3.1 Bellman 递推

定义从状态 \(s_k\) 出发的最优剩余价值

$$
V_k^*(s_k)=
\sup_{\tau\in\mathcal{T}_k}
\mathbb{E}\left[
Q(s_\tau)-\sum_{j=k+1}^{\tau}c_j
\mid \mathcal{F}_k
\right],
\tag{3}
$$

其中 \(\mathcal{T}_k\) 是所有取值于 \(\{k,\ldots,K\}\) 且关于 \((\mathcal{F}_t)\) 可选停止的 stopping times。边界条件为

$$
V_K^*(s_K)=Q(s_K).
\tag{4}
$$

对任意 \(k<K\)，最优性方程为

$$
V_k^*(s_k)=
\max\left\{
Q(s_k),\;
\mathbb{E}\!\left[V_{k+1}^*(s_{k+1})\mid\mathcal{F}_k\right]-c_{k+1}
\right\}.
\tag{5}
$$

记继续价值为

$$
C_k^*(s_k)=
\mathbb{E}\!\left[V_{k+1}^*(s_{k+1})\mid\mathcal{F}_k\right]-c_{k+1},
\tag{6}
$$

边际继续价值为

$$
\Delta_k^*(s_k)=C_k^*(s_k)-Q(s_k).
\tag{7}
$$

则最优动作具有阈值形式：

$$
a_k^*(s_k)=
\begin{cases}
\textsc{Continue}, & \Delta_k^*(s_k)>0,\\
\textsc{Stop}, & \Delta_k^*(s_k)\le 0.
\end{cases}
\tag{8}
$$

### 3.2 命题 1：Bellman 最优性

**命题 1.** 若 \(Q(s_k)\) 有界且每步成本有限，则式 (5) 给出了 fixed-order 多跳检索停止问题的最优递推；由式 (8) 导出的策略是最优停止策略。

**证明思路。** 对有限时域 \(K\) 做后向归纳。最后一步必须停止，故 \(V_K^*=Q(s_K)\)。若在第 \(k\) 步停止，收益为 \(Q(s_k)\)；若继续，则支付 \(c_{k+1}\) 并在下一状态按最优策略行动，条件期望收益为式 (6)。两者取大得到式 (5)，动作规则由比较两项直接得到。

### 3.3 轨迹级 DP Oracle

实际实验中，我们已经缓存了每条轨迹上每一步的真实 F1。因此 Stage 1 使用轨迹级后向 DP 生成 oracle 标签。对样本 \(i\)，令 \(Q_k^i=Q(s_k^i)\)。定义

$$
\widetilde V_K^i=Q_K^i,\qquad
\widetilde V_k^i=\max\left\{Q_k^i,\widetilde V_{k+1}^i-c_{k+1}^i\right\}.
\tag{9}
$$

轨迹级 oracle 在最小的 \(k<K\) 满足

$$
Q_k^i \ge \widetilde V_{k+1}^i-c_{k+1}^i
\tag{10}
$$

时停止，否则在 \(K\) 步停止。它同时为每个中间状态写出训练标签：

$$
m_k^i=(\widetilde V_{k+1}^i-c_{k+1}^i)-Q_k^i,
\qquad
y_k^i=\mathbf{1}\{m_k^i>0\}.
\tag{11}
$$

这里 \(m_k^i\) 是继续 margin，\(y_k^i\) 是 Continue/Stop 二分类标签。当前主实现使用 \(y_k^i\) 训练二分类 Probe，而不是直接回归 \(m_k^i\)。

---

## 4. 与 Pandora / Weitzman 的关系

原始 Weitzman Pandora's Box 问题在独立盒子假设下得到 reservation value：

$$
c_k=\int_{r_k^*}^{\infty}(x-r_k^*)\,dF_k(x).
\tag{12}
$$

多跳 RAG 与原始设定不同：第 \(k+1\) 步检索分布依赖前面检索到的证据和中间推理状态，因此不是独立盒子问题。论文中应当这样定位：

1. Pandora / Weitzman 提供“继续检索是否值得”的阈值型动机。
2. 本文真正可证明的主结构来自 fixed-order finite-horizon Bellman 递推。
3. `Global-Weitzman` 是结构化静态参照，测试时使用真实 F1，不可部署。
4. `Deployable-GW` 将真实 F1 替换为 self-consistency 等代理，但当前结果基本退化到一步停止，说明简单代理不足以表达多跳正确性。

因此，正文可以使用 `Pandora-RAG` 作为方法名，但不要把部署 Probe 宣称为严格 Weitzman reservation value 的精确学习器。

---

## 5. Learned Stopping Signal

部署时 \(Q(s_k)\)、\(C_k^*(s_k)\) 和 \(m_k^i\) 都不可观测。Pandora-RAG 使用轻量 Probe 从可观测表示中学习停止信号。令

$$
x_k=(h_k,\phi_k),
\tag{13}
$$

其中 \(h_k\) 是 Llama-3.1-8B-Instruct 在当前状态下的 hidden state，\(\phi_k\) 是检索分数、answer logprob、NLI 分数、历史变化等浅层特征。

### 5.1 当前实现：二分类停止探针

主实现训练

$$
p_\theta(k)=\Pr_\theta(y_k=1\mid x_k),
\tag{14}
$$

其中 \(y_k=1\) 表示 oracle 标签为 Continue。部署时使用 dev 上选择的阈值 \(\eta_k\)：

$$
\hat a_k=
\begin{cases}
\textsc{Continue}, & p_\theta(k)\ge \eta_k,\\
\textsc{Stop}, & p_\theta(k)<\eta_k.
\end{cases}
\tag{15}
$$

当前最终工作点采用 per-step thresholds；阈值选择以 dev 上的 Pareto 分数 \(F1-\lambda\cdot\text{normalized cost}\) 为目标，并约束 Probe 的平均步数不超过 `Global-Weitzman(dev)` 平均步数乘以 `1.05`。这使 Probe 的主结果更符合“低成本 Pareto 点”，而非纯 max-F1 点。

训练损失是带 margin 权重、focal weighting 与 label smoothing 的 BCE。抽象写作可表示为

$$
\mathcal{L}_{\mathrm{cls}}(\theta)=
\mathbb{E}\left[
w(m_k)\,
\ell_{\mathrm{focal}}\!\left(p_\theta(k),y_k\right)
\right],
\tag{16}
$$

其中 \(w(m_k)\) 强调更有决策意义的高 margin 样本。

### 5.2 可选理论写法：继续价值或 margin 估计

若正文希望与 Bellman 更直接对齐，可以把 Probe 表述为 learned stopping index，而不是严格的 reservation value：

$$
g_\theta(x_k)\approx \Delta_k^*(s_k)
\quad\text{or}\quad
g_\theta(x_k)\approx \Pr(\Delta_k^*(s_k)>0\mid x_k).
\tag{17}
$$

这与当前二分类实现一致：它学习的是 Continue/Stop 决策边界，而不是完整价值函数。

### 5.3 命题 2：估计误差只在小 margin 区域影响动作

设存在一个 margin 估计器 \(\hat m_k\)，并按 \(\hat m_k>0\) 继续、否则停止。对任意 \(\varepsilon>0\)，有

$$
\Pr(\hat a_k\ne a_k^*)
\le
\Pr(|m_k^*|\le\varepsilon)
+
\Pr(|\hat m_k-m_k^*|>\varepsilon).
\tag{18}
$$

**证明。** 若动作不同，则 \(\hat m_k\) 与 \(m_k^*\) 符号不同。若 \(|m_k^*|>\varepsilon\) 且 \(|\hat m_k-m_k^*|\le\varepsilon\)，估计值无法跨过 0 改变符号。因此错误动作只能发生在真实 margin 小于 \(\varepsilon\) 的区域，或估计误差超过 \(\varepsilon\) 的区域。取并集并用 union bound 得证。

**论文含义。** Probe 不需要完美估计所有状态的精确价值；它真正需要做对的是靠近停止边界、且会改变动作的状态。这也解释了为什么 margin filtering 和 per-step threshold refinement 是合理的工程选择。

---

## 6. E-value 在线风险监控

### 6.1 错误事件与原假设

部署流中第 \(n\) 个样本的最终停止质量为 \(Q_n=Q(s_{\tau_n})\)。给定质量阈值 \(\gamma\)，定义错误指标

$$
e_n=\mathbf{1}\{Q_n<\gamma\}.
\tag{19}
$$

令 \(\mathcal{G}_{n-1}\) 表示处理第 \(n\) 个样本前的所有历史信息，包括过去样本、过去停止时刻、模型输出、E-wealth 轨迹与 betting 选择。E-value 层检验的原假设为

$$
H_0:\quad
\mathbb{E}[e_n\mid\mathcal{G}_{n-1}]\le\alpha
\quad\text{for all }n.
\tag{20}
$$

### 6.2 E-process

选择任意可预测下注比例 \(\lambda_n\in[0,1]\)，要求 \(\lambda_n\) 对 \(\mathcal{G}_{n-1}\) 可测。定义

$$
M_n=1-\lambda_n+\lambda_n\frac{e_n}{\alpha},
\tag{21}
$$

以及 wealth process

$$
E_0=1,\qquad
E_n=\prod_{t=1}^{n}M_t.
\tag{22}
$$

当前主实验使用 predictive betting：

$$
\lambda_n=\mathrm{clip}(1-\hat p_n,\epsilon,1-\epsilon),
\tag{23}
$$

其中 \(\hat p_n\) 是质量模型对 \(Q_n\ge\gamma\) 的概率估计。质量模型是在 calib split 上训练的 Logistic Regression，输入为浅层特征并可拼接 Probe 的 continue probability。实现中还使用 quality bar 阻断最低质量分位的停止请求。

### 6.3 定理 1：Anytime-valid 告警保证

**定理 1.** 若 \(H_0\) 成立，且每个 \(\lambda_n\) 都是 \(\mathcal{G}_{n-1}\)-可测并取值于 \([0,1]\)，则 \((E_n)_{n\ge0}\) 是非负超鞅。因而对任意 \(\delta\in(0,1)\)，

$$
\Pr_{H_0}\left(\sup_{n\ge1}E_n\ge\frac{1}{\delta}\right)\le\delta.
\tag{24}
$$

实验中常取 \(\delta=\alpha\)，于是告警边界为 \(1/\alpha\)。

**证明。** 在 \(H_0\) 下，

$$
\mathbb{E}[M_n\mid\mathcal{G}_{n-1}]
=1-\lambda_n+\lambda_n
\frac{\mathbb{E}[e_n\mid\mathcal{G}_{n-1}]}{\alpha}
\le1.
\tag{25}
$$

由于 \(\lambda_n\) 可预测，

$$
\mathbb{E}[E_n\mid\mathcal{G}_{n-1}]
=E_{n-1}\mathbb{E}[M_n\mid\mathcal{G}_{n-1}]
\le E_{n-1}.
\tag{26}
$$

因此 \(E_n\) 是非负超鞅。Ville 不等式给出式 (24)。

### 6.4 实现口径与理论口径

论文必须明确以下三点：

1. E-value 保证的是“若条件错误率从未超过 \(\alpha\)，则错误触发强告警的概率受控”，不是“系统输出错误率自动低于 \(\alpha\)”。
2. 当前实验中的 Probe 错误率明显高于 \(\alpha=0.1\)，因此 E-value 的角色是检测和监控，而非修复底层 QA 能力。
3. 代码中为数值稳定会将 wealth trace 裁剪在 cap 附近；理论对象是未裁剪的乘积过程。裁剪后的 trace 用于可视化和保守告警展示，不改变 E-process 主结论。

---

## 7. 当前结果对应的安全表述

可以安全写：

1. 多跳检索停止可形式化为 fixed-order finite-horizon optimal stopping。
2. DP Oracle 提供实例级上界和监督标签。
3. 当前 Probe 是 learned stopping signal / continuation-decision estimator。
4. Probe 在三个数据集上恢复约 \(79.9\%\) 到 \(85.4\%\) 的 Oracle F1，并显著减少相对 Fixed-K=5 的步数。
5. Probe+E-value 是低开销在线风险监控层；在 HotpotQA 与 2Wiki 上小幅改善 F1/error，在 MuSiQue 上基本持平。
6. E-value 对 sudden / gradual / periodic shift 通常能积累风险证据，尤其在 MuSiQue 与 2Wiki 上响应更强。

不要写：

1. 部署系统是 provably optimal。
2. Probe 精确学习了 Weitzman reservation value。
3. E-value 把错误率控制到 \(\alpha\) 以下。
4. Split CP 在理论上“失效”所以一定比 E-value 差。更准确的说法是：在当前 adaptive stopping pipeline 中，CP 表现为高成本、数据集敏感的静态门控，而 E-value 提供跨样本、anytime-valid 的风险证据过程。

---

## 8. 正文和附录分工

**正文建议保留：**

1. 效用目标：式 (1)-(2)
2. Bellman 递推：式 (5)-(8)
3. 轨迹级 oracle 标签：式 (9)-(11)
4. Probe 停止规则：式 (14)-(16)
5. E-process：式 (19)-(24)

**附录建议展开：**

1. 命题 1 的完整后向归纳证明
2. 命题 2 的 margin-error 分解
3. 定理 1 的超鞅与 Ville 不等式证明
4. Weitzman reservation value 与 fixed-order Bellman 主线的关系
5. E-value cap clipping、selective prediction、CP-quantile gate 的实现细节

---

## 9. 一句话总结

Pandora-RAG 的数学闭环不是“把 Pandora、Probe、E-value 拼起来”，而是：

$$
\text{fixed-order Bellman Oracle}
\Longrightarrow
\text{learned stopping signal}
\Longrightarrow
\text{anytime-valid E-process monitor}.
$$

这条链条既符合当前代码实现，也能支撑论文中最重要的三项贡献：结构化问题定义、可部署停止器、以及自适应停止下的在线风险监控。
