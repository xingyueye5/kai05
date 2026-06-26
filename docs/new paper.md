# Ask the Policy Itself: Free Advantage Signals for Flow-Matching VLAs

> 论文核心叙事 + 痛点 + 解法 + 实验与指标的完整规划。
> 配套文档：`总分析.md`（含代码复用与改动文件清单）、`plan.md`（执行甘特图）。
>
> 生成时间：2026-06-26。

---

## 1. 论文一句话定位（Elevator Pitch）

> **Flow-matching VLAs already know what good behavior looks like — we just need to ask them.
> By combining the policy's own action-variance with a foundation-feature-based progress signal,
> we extract advantage labels from any static dataset, with zero reward, zero value model,
> zero preference, zero rollout, zero intervention.**

两个对仗钩子贯穿全文：

- **Ask the data** → RTAP：episode 终点是天然 goal anchor，foundation features 给出距离度量。
- **Ask the policy** → FlowVar：flow-matching 在 noise sampling 下的 action variance 是 policy 自身的不确定性打分。

两者分别捕获 **input-side（视觉）** 与 **output-side（动作）** 的 advantage 维度，理论上正交、实证上正交、机制上互补。

---

## 2. 主框架图（Figure 1，论文第一张图）

```
                Ask the data                               Ask the policy
                     │                                            │
            ┌────────┴────────┐                          ┌────────┴────────┐
            │      RTAP       │                          │     FlowVar     │
            │ visual progress │                          │  action-chunk   │
            │  to goal anchor │                          │ variance under  │
            │   (input-side)  │                          │ noise sampling  │
            │                 │                          │  (output-side)  │
            └────────┬────────┘                          └────────┬────────┘
                     │                                            │
                     │            Orthogonal Fusion               │
                     └──────────────────┬─────────────────────────┘
                                        │
                            percentile-normalize + α-blend
                                        │
                                        ▼
                          Advantage bins (binary / N-bins)
                                        │
                                        ▼
                        sincos embedding → prompt 末尾拼接
                                        │
                                        ▼
                 π0.5 PaliGemma + Gemma Action Expert (Flow Matching)
                                        │
                                        ▼
                            Inference: advantage = 1 + CFG
                                        │
                                        ▼
                                  Action Chunk

    External signals required: ❌ reward  ❌ value model  ❌ preference
                               ❌ rollout ❌ intervention
```

---

## 3. 论文核心叙事（四幕结构）

### 第一幕 — Setup：让 reviewer 同意这是真问题

1. **共识**：VLA 后训练已经不能只做 SFT，需要 advantage / value signal 区分 demo 质量。π\*0.6 (RECAP) 是这一范式的代表。
2. **隐藏假设**：π\*0.6 风格的方法默认你能拿到 reward、训得起 value model、有真机做 rollout、有人能做干预。
3. **现实**：绝大多数研究者只能拿到 HuggingFace 上的公开静态数据集（LIBERO / RoboCasa / OpenX）。在这种 setting 下，整个 advantage-conditioning 范式失效。
4. **gap**：FlowPRO 试图去掉 reward，但仍依赖 preference 与真机干预。**至今没有一个方法能在"只有一坨静态 demo"的设定下做有效的 advantage conditioning。**
5. **本文问题陈述**：*Where does advantage signal come from, when all you have is a static dataset?*

### 第二幕 — Insight：让 reviewer 拍大腿

统一 insight：advantage 信号其实早已隐藏在两个被忽视的地方——

- 数据集自己（**Ask the data**）：episode 终点是天然 goal anchor；
- 模型自己（**Ask the policy**）：flow-matching 在不同 noise 下的 action variance 是隐式的"我对这个状态多有把握"。

**两者都不需要任何外部信号**，且分别覆盖 advantage 的 input-side 与 output-side，**理论上正交**。

### 第三幕 — Evidence：让 reviewer 相信不是 cherry-picked

按"说服力顺序"（不是按时间）展开实验：sanity → orthogonality → resource-matched 主表 → 与 reward-based 比 → 与 π\*0.6 互补 → 失效分析 → 数据效率与 transfer → ablation。详见 §6。

### 第四幕 — Implication：让 reviewer 觉得社区会变好

- **重新定义** "advantage signal 来源"的边界：不必非来自 reward。
- **降低门槛**：让没有真机/标注预算的研究者也能做 advantage-conditioned 后训练。
- **加法而非减法**：FreeVAC 不替代 π\*0.6 / FlowPRO，而是作为它们的 cold-start 模块。
- **主动承认局限**：论文必须在 Discussion 写明 5 条已知局限（详见 §8），避免被 reviewer 挖出来。

---

## 4. 之前方法的痛点（每条对应我们一个解法）

| # | 之前方法痛点 | 代表工作 | 我们的解法 |
|---|---|---|---|
| **P1** | 必须有 reward 标注 / 训独立 value model | π\*0.6 (RECAP) | **RTAP**：从 episode 结构 + foundation features 中无训练提取 progress |
| **P2** | 即使 reward-free 也需要 preference + 真机干预 | FlowPRO (RPRO) | 全部由静态数据 + 已有 SFT checkpoint 完成，**零额外人工** |
| **P3** | 原 VC-Value 用"progress window 限制匹配范围"形成循环依赖，且强依赖 `progress_gt` 标注 | Kai05-VLA 原版 | **RTAP 用 episode 终点而非 progress 做 anchor**，无循环依赖、无 progress_gt 需求 |
| **P4** | advantage 信号来源单一（只来自视觉相似度），跨任务、长任务退化 | 原 VC-Value、单 anchor 方法 | **FlowVar**：从 policy 自身动作分布中提取与视觉信号正交的第二份 advantage |
| **P5** | flow-matching 的 action variance 一直只被当成"推理时调度变量"，没人用作训练时 advantage | AdaFlow (NeurIPS 2024) | **首次将 flow-matching variance 作为训练时 advantage 信号**（FlowVar 的真正 novelty） |
| **P6** | 单信号 advantage 方法在 failure mode 上不互补，没有兜底 | DT / IQL / AWR 等 | **正交融合** + 失效互补的实证证据（§6 Step 6 关键 figure） |

---

## 5. 三个核心贡献（每个都精准对位上面的 P）

### C1 — RTAP（Reverse-Time Anchored Progress）→ 解决 P1, P3
- **input-side** 信号
- 以 episode 最后一帧的 foundation feature 作为 goal anchor，计算每帧到 anchor 的余弦距离 → 反向归一化为 progress ∈ [0, 1]
- Multi-Anchor 版本：用 kmeans 聚类多个 episode 的终点以处理任务多种成功方式
- **理论叙述（修正）**：避免声称严格的 "$V^*$ 一致估计"——deterministic goal-conditioned MDP + Lipschitz 假设过强且无法验证。改写为 "RTAP is a foundation-feature-space soft analog of HER's goal relabeling; under mild assumptions it provides a Lipschitz upper bound on value error, validated empirically in §5"，把强 theorem 降级为 informal proposition + 实证验证
- **与 HER 的差别**：HER 做 transition relabeling，RTAP 提取连续 progress 信号
- **必须处理的失败 demo 鲁棒性问题（P1）**：若数据集包含失败 demo，其最后一帧并非 goal，会让 RTAP 给错误状态高 progress。论文必须：(a) 在 ablation 中人工注入 10% 失败 demo，对比 RTAP-only / FlowVar-only / Fusion 哪个更鲁棒；(b) 提供基于 Multi-Anchor outlier detection 的自动失败 demo 过滤策略

### C2 — FlowVar Advantage → 解决 P4, P5（最强 novelty）
- **output-side** 信号
- 用 vanilla SFT 的 π0.5（**不需要额外训练，本来就要做的 SFT**）作为 reference policy $\pi_{\text{ref}}$
- 在每个 state 上以 N=8 个不同 noise 执行 $\pi_{\text{ref}}$ 的 flow sampling，得到 8 条 action chunks
- 计算样本间的 chunk-wise std，聚合为标量 advantage：方差越小 → policy 越确定 → advantage 越高
- **直觉**：高 value 状态"该怎么做"很明确，policy 在不同 noise 下应收敛到相近 action；低 value 状态发散
- **理论叙述（重要修正）**：不写"variance ≈ policy entropy"——flow matching 不直接输出概率密度，这个等式不严谨。正确叙述是 "dispersion under noise sampling reflects the spread of the implicit density induced by the learned vector field"，并以 Continuous Normalizing Flow 文献中的 implicit density 框架给出一个 informal proposition（Proposition 2），刻画 "vector field 越确定 → 隐式 density 越尖锐 → action chunk variance 越小"。
- **与 AdaFlow 的精确区分**（论文中必须独立一张表）：

  | | AdaFlow (NeurIPS 2024) | FlowVar (ours) |
  |---|---|---|
  | variance 用途 | 推理时调度去噪步数 | 训练时作为 advantage 信号 |
  | 是否做 advantage conditioning | 否 | 是 |
  | 论文目标 | 加速推理 | 提升策略质量 |
  | 一句话 | "用 variance 做**快**" | "用 variance 做**好**" |

- **必须做的 chicken-and-egg 消融（P0）**：FlowVar 依赖 reference policy 质量，存在"先有好 policy 才能产生好 advantage"的循环担忧。论文必须给出以 10%/25%/50%/100% 训完的 SFT checkpoint 作为 reference 的下游 SR 曲线，证明：(a) under-trained reference 也能产生有用 FlowVar 信号；或 (b) 给出明确的 reference quality 下界。这是 reviewer 必问，不答会被毙。

### C3 — Orthogonal Signal Fusion → 解决 P6
- RTAP 是 input-side，FlowVar 是 output-side
- **Framing 修正**：避免使用强意义"理论正交"——两者都与 progress 正相关，严格独立不成立。正确表述为 **"empirically uncorrelated and complementary in failure modes"**，对应：(i) Pearson $|\rho|$ < 0.6（弱意义不相关，gate criterion）；(ii) 失效模式来源不同（RTAP 在"视觉混淆但任务进度不同"处失效，FlowVar 在"动作模糊但视觉清晰"处失效）
- 融合策略：各自 percentile-normalize 到 [0,1] 后加权平均
- **互补性论证**：在 RTAP 失效的 long-horizon multi-subgoal 任务上 FlowVar 顶上；在 FlowVar 早期不可靠的 cold-start 阶段 RTAP 顶上

---

## 6. 实验设计与指标

### 6.1 数据集

| 数据集 | 角色 | 备注 |
|---|---|---|
| **LIBERO-Spatial / Object / Goal / Long**（130 tasks × 50 demos） | 主战场 | 与 π0.5 / π\*0.6 同 benchmark，公开且标准 |
| **RoboCasa-Kitchen** 5–8 atomic tasks | 跨场景泛化 | 加分项，可裁 |
| **LIBERO-90 → LIBERO-10** | 跨任务 transfer | 附录 / 加分项 |

### 6.2 主表设计原则 — 按"需要什么资源"分组，避免被 reviewer 直接拿绝对 SR 横扫

#### Table 1 主表（Resource-Grouped）

| Resource Setting | Method | reward | value | pref. | rollout | Spatial | Object | Goal | Long | Avg |
|---|---|:-:|:-:|:-:|:-:|---|---|---|---|---|
| **SOTA reference** | OpenVLA / OFT-VLA（公开 best report） | varies | varies | – | – | – | – | – | – | – |
| **A. Zero external signal** | π0.5 vanilla SFT | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| | + time-index advantage | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| | + 原 VC-Value (需 progress_gt) | – | ❌ | ❌ | ❌ | × | × | × | × | × |
| | **+ FreeVAC (RTAP only)** | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| | **+ FreeVAC (FlowVar only)** | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| | **+ FreeVAC (Fusion + CFG)** | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| **B. Reward 可得** | + DT-style return cond. | ✅ | ❌ | ❌ | ❌ | – | – | – | – | – |
| | + π\*0.6-style trained value (**ours re-implementation, 必须明说**) | ✅ | ✅ | ❌ | ❌ | – | – | – | – | – |
| **C. Real-robot rollout 可用** | π\*0.6 full（参考原论文） | ✅ | ✅ | ❌ | ✅ | – | – | – | – | – |

**故事**：
- A 组内 **FreeVAC-Fusion 第一**（与 vanilla / time-index / 原 VC-Value 比较）
- A 组的 FreeVAC 接近 B 组（用 0 cost 拿到接近 reward-based 的效果）
- C 组只作为 reference，**明确不参与公平比较**

**主表数字必备项（P0）**：
- **5 seeds**（不是 3 seeds），每个 cell 报均值 ± std
- **配对统计显著性检验**：FreeVAC-Fusion vs 每条 baseline 做 paired bootstrap / paired t-test，主表的小幅 gain (+1~2 SR) 必须有 p-value 支撑
- **SOTA reference 行必须有**（OpenVLA / OFT-VLA / π0.5 官方报数），否则会被批 "baseline 偏弱"
- **π\*0.6 baseline 在论文中必须明确标注 "ours re-implementation"**，避免被指控复现不当导致 unfair comparison

#### Table 2 互补性表（决定论文命运的一张表）

| 阶段 1 (offline SFT) | 阶段 2 (online RL, 可选) | LIBERO-Long SR |
|---|---|---|
| vanilla SFT | — | $x$ |
| **FreeVAC SFT** | — | $x + \Delta_1$ |
| vanilla SFT | π\*0.6-style online stage | $y$ |
| **FreeVAC SFT** | π\*0.6-style online stage | $y + \Delta_2$ |

- 若 $\Delta_2$ 与 $\Delta_1$ 同量级 → FreeVAC 的 cold-start 优势在 online 阶段**不会被冲掉** → 与 π\*0.6 **互补而非竞争**
- 这一张表 + 一段 Discussion 直接拆解 "你 SR 没打过 π\*0.6" 的批评

### 6.3 关键图

| 编号 | 图 | 作用 | 在叙事中的位置 |
|---|---|---|---|
| Figure 1 | 主框架图（§2） | 一图讲完方法 | Introduction 末 |
| **Figure 2** | RTAP-adv vs FlowVar-adv 散点图 + Pearson $\rho$ | 证明正交（C3 命根子） | Section 5 实验第一张 |
| **Figure 3** | 失效互补可视化（LIBERO-Long 上 RTAP 失效区段 + FlowVar 失效区段 + Fusion 救场） | 证明机制意义上的互补 | Section 5 中段 |
| Figure 4 | Data efficiency 曲线（10/25/50/100% 数据 × 3 方法） | 证明非过拟合 trick | Section 5 后段 |
| Figure 5 | Pareto plot（外部 cost vs SR） | Discussion 收尾 | Section 6 |

### 6.4 评估指标

| 指标 | 用途 | 备注 |
|---|---|---|
| **Task Success Rate (SR)** | 主指标 | 每 task 100 episodes rollout, **5 seeds (P0, 不可少于此)**，报均值 ± std + paired bootstrap CI |
| **Per-suite Avg SR** | 主表汇总 | LIBERO 标准做法 |
| **Paired statistical significance (p-value)** | 显著性 | FreeVAC vs 每条 baseline 必报 |
| **Spearman correlation with oracle progress** | 信号有效性 sanity | 手工标 100 episodes oracle progress |
| **Pearson correlation between RTAP-adv & FlowVar-adv** | 不相关性 gate（$|\rho|$ < 0.6）| 决定 contribution 3 是否成立 |
| **Failure mode classification rate** | 失效互补量化 | 把 rollout 失败 case 按"RTAP 错"、"FlowVar 错"、"双错"分类 |
| **Sample efficiency (SR at X% data)** | 数据效率 | Figure 4 横轴 |
| **Cold-start gain $\Delta$ in online RL** | 互补性 | Table 2 关键数字 |
| **Wall-clock & VRAM (P1)** | 实用性 | 包含 FlowVar N=8 sampling 的 advantage labeling 一次性开销 + 训练 / 推理 latency / VRAM |

### 6.5 必做消融

| Ablation | 维度 | 目的 |
|---|---|---|
| RTAP 变体 | Time-index / Single-Anchor / Multi-Anchor / Hybrid | C1 内部消融 |
| **FlowVar reference policy 质量 (P0)** | **reference 训完 10% / 25% / 50% / 100% 时的下游 SR** | **拆解 chicken-and-egg 担忧，reviewer 必问** |
| FlowVar 样本数 N | {2, 4, 8, 16} | 计算-精度 trade-off |
| Fusion 权重 α | {0, 0.25, 0.5, 0.75, 1} | 端点对应 RTAP-only / FlowVar-only |
| Advantage 离散化 | binary / 2-bins / 5-bins / 10-bins | 粒度敏感性 |
| CFG scale | {1.0, 1.5, 2.0, 3.0} | 与条件强度相关 |
| Encoder（appendix） | SigLIP2 vs DINOv2 | RTAP 对 encoder 敏感性 |
| **失败 demo 鲁棒性 (P1)** | 人工注入 0/5/10/20% failure demos | RTAP / FlowVar / Fusion 谁更鲁棒，论文必须诚实回答 |
| **跨 backbone 验证 (P1)** | 至少在 OpenVLA 或 DP3 上验证 FlowVar 也成立 | 证明 FlowVar 不是 π0.5-specific hack，扩大方法影响范围 |
| **Negative result（P1）** | 报告 FreeVAC 在哪些任务上 gain ≈ 0 或为负 | 防 cherry-pick 指控 |

### 6.6 资源预算（详细见 `plan.md`）

| 项目 | 数值 |
|---|---|
| 总 GPU·hours (A100) | ~7,200（紧张时压到 ~4,000） |
| 存储 | ~600 GB |
| 时间 | 15 周 |
| 复用 Kai05-VLA 原有代码 | ≥ 70% |

---

## 7. 三个 Reviewer 必问问题的预防式回答

| Reviewer 问题 | 我们的预防式回答（写在论文里） |
|---|---|
| Q1：**FlowVar 不就是 AdaFlow？** | Section 3.3 第一句直接区分：AdaFlow 在**推理时**用 variance 调度去噪步数，FlowVar 在**训练时**用 variance 作 advantage 信号。两者完全正交，可组合。 |
| Q2：**RTAP 不就是 HER in feature space？** | Related Work 主动 cite HER + Goal Sets，明确两点 delta：(i) 我们不 relabel transitions，而是提取连续 progress；(ii) Multi-Anchor 通过 kmeans 解决 goal-set 的单点过约束。 |
| Q3：**为什么不直接和 π\*0.6 比绝对 SR？** | Intro 末段 + 主表前主动说明 resource-matched 设定。Table 2 提供互补性证据：FreeVAC 是 π\*0.6 的 cold-start 模块，二者收益叠加。 |

---

## 8. 主动承认的局限性（Discussion 必须包含）

Reviewer 一旦挖出未承认的局限，论文被动；主动写出来反而显得严谨。以下 5 条在 Discussion 章节按顺序展开。

| # | 局限 | 必备的 mitigation 叙述 |
|---|---|---|
| **L1** | **FlowVar 的 reference policy 依赖**（最大局限）：需要先做 vanilla SFT，存在 chicken-and-egg 担忧 | 引用 §6.5 中 "reference quality" 消融结果，证明 under-trained reference 也能产生有用信号；并指出未来方向（用通用预训练 VLA 作 universal reference） |
| **L2** | **RTAP 的"episode 终点 = goal"假设**：失败 demo 的终点不是 goal state | 引用 §6.5 中 "失败 demo 鲁棒性" 消融，证明 Multi-Anchor + outlier filtering 能缓解；坦诚指出在高失败率数据集上 RTAP 单独失效，需要靠 FlowVar 救场 |
| **L3** | **Long-horizon multi-stage 任务**：RTAP 会把视觉接近 goal 的中间帧给高 progress，无法区分阶段 | 在 LIBERO-Long 上诚实展示 RTAP 失效的可视化（即 Figure 3）；说明这正是 Fusion 互补的价值 |
| **L4** | **FlowVar 一次性计算开销**：每帧需要 N=8 次 flow sampling 给 advantage 打标签 | §6.4 报 wall-clock；指出在 LIBERO 量级（~10⁵ 帧）开销可接受，OpenX 量级（10⁷+）需 future work |
| **L5** | **Offline-only，不解决具身泛化**：与 "具身核心是 generalization" 的批评呼应 | 明确定位为 **data-centric advantage signal extraction**，而非通用 VLA 泛化方法；FreeVAC 与 world model / online RL 方向正交 |

---

## 9. Gate Criterion（Week 3 末决策，决定主路线 / Plan B）

满足 ≥ 3/5 走主路线，否则切 Plan B（见 §9）：

| 检查项 | 阈值 |
|---|---|
| π0.5 vanilla 复现 LIBERO-Spatial | ±2% 官方数字 |
| RTAP 与 oracle progress Spearman | > 0.5 |
| FlowVar 与 oracle progress Spearman | > 0.3 |
| **RTAP × FlowVar Pearson $|\rho|$** | **< 0.6**（C3 命根子，弱意义不相关 + 失效模式互补即可）|
| FreeVAC-Fusion vs vanilla on LIBERO-Spatial | +1.5 SR |

---

## 10. Plan B（兜底）

| 触发条件 | 退路 | 卖点 |
|---|---|---|
| RTAP × FlowVar 高度相关（$|\rho| > 0.6$） | **Plan B-1**：单押 RTAP 转 Data-Centric | "Foundation-feature-based dataset subset selection for VLA training" |
| FlowVar 信号弱、相关性低 | **Plan B-2**：单押 FlowVar 转诊断性论文 | "What does the variance of a flow-matching policy tell us?" → workshop |
| 主表收益不显著 | **Plan B-3**：转 landscape paper | "A landscape of free advantage signals in static robot datasets" → dataset & benchmark track |

---

## 11. 摘要模板

> Advantage-conditioned policies have emerged as a leading paradigm for VLA post-training,
> but existing methods (e.g., π\*0.6, FlowPRO) rely on reward labels, trained value models,
> real-robot rollouts, or human interventions — resources unavailable in the common case
> where one only has a static public dataset. We ask whether advantage signal can be
> extracted entirely from internal sources, and answer this affirmatively with two
> observations. First, **the dataset itself** provides a progress signal via
> foundation-feature distance to episode endpoints (**RTAP**). Second, **the policy itself**
> provides an advantage signal via the action-chunk variance under noise sampling — a free
> byproduct of flow matching that, surprisingly, has never been used for training-time
> advantage conditioning (**FlowVar**). We show these two signals are empirically and
> theoretically orthogonal, and their fusion enables advantage-conditioned fine-tuning of
> π0.5 on LIBERO and RoboCasa **without any reward, value model, preference, rollout, or
> intervention**. FreeVAC matches reward-based baselines under matched resource settings
> and provides a strictly additive cold-start for online RL methods like π\*0.6.

---

## 12. 投稿建议

| 会议 | 匹配度 | 卖点对应 |
|---|---|---|
| **CoRL 2026** | ★★★★★ | 机器人 + 落地导向 + 实验扎实 |
| **NeurIPS 2026** | ★★★★ | 理论（正交性 / RTAP 一致性 / FlowVar entropy） + 大规模实验 |
| **ICLR 2027** | ★★★★ | "Ask the policy itself" 这类 representation insight 受偏爱 |
| RSS 2027 | ★★★ | 若加实物实验 |
| ICML 2027 | ★★★ | 若加强理论 |

**推荐主投 CoRL 2026**，rebuttal 不过再投 NeurIPS / ICLR。

---

*本文档专注于"叙事 + 痛点 + 解法 + 实验指标"，工程细节与文件改动见 `总分析.md`。*
