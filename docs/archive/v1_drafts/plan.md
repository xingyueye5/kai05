# FreeVAC: Static-Data-Only Advantage Conditioning for Offline VLA Post-Training

> **目标**：在 4-6 个月内、仅用公开静态数据集、≤ 5000 GPU·hours，
> 完成一篇可投 **CoRL 2026 / NeurIPS 2026 / ICLR 2027** 的论文。
>
> **撰写日期**：2026-06-25 (v3 — 严格对标竞品后的差异化版本)

---

## 1. 我们在哪个 niche？（先把竞品摆清楚，再说自己做什么）

### 1.1 当前竞品全景图（2025.11 - 2026.06）

| 方法 | 时间 | reward 标注 | value model | preference 对 | 人在环干预 | 实机 rollout | 数据来源 |
|---|---|:-:|:-:|:-:|:-:|:-:|---|
| **π\*0.6 (RECAP)** | 2025.11 | ✅ 人工标 | ✅ 670M | ❌ | ✅ | ✅ | 自采 + 干预 |
| **FlowPRO (RPRO)** | 2026.06 | ❌ | ❌ | ✅ 干预生成 | ✅ | ✅ | 真实机器人自采 |
| **AdaFlow** | NeurIPS 2024 | — (BC only) | — | — | — | — | 任意 demo，但 variance 用于**推理时调度**而非 advantage |
| **我们 (FreeVAC)** | 目标 | ❌ | ❌ | ❌ | ❌ | ❌ | **公开静态数据集** |

### 1.2 关键洞察：我们守住了一个真空带

- **π\*0.6** 教会业界："advantage conditioning 是 VLA 后训练的关键范式"
- **FlowPRO** 教会业界："不要 reward 也能做，用 preference + 干预"
- **AdaFlow** 教会业界："flow matching 的 action variance 是有用信号"
- **没人做的事**：**完全静态、完全公开数据**情况下，如何在 π0.5 这种 flow-matching VLA 上做有效的 advantage conditioning？

我们的 niche 一句话：

> **"既不要 reward 也不要 preference，也不要 rollout、也不要干预——
> 仅用公开静态 demo，从 (i) 视觉特征到终点的距离 + (ii) flow-matching policy 自身的 action variance
> 两个互相独立的信号中提取 advantage，完成 π0.5 的离线后训练。"**

### 1.3 我们与 AdaFlow 的精确差别（必须写在论文里）

| | AdaFlow (NeurIPS 2024) | 我们 |
|---|---|---|
| variance 用法 | **推理时**自适应调整 flow 去噪步数 | **训练时**作为 advantage 条件信号注入 |
| 目标问题 | 加速推理 | 提升策略质量 |
| 是否引入 advantage conditioning | ❌ | ✅ |
| 与 advantage conditioning 范式的关系 | 无关 | 是该范式的低成本数据源 |

→ 一句话：**AdaFlow 用 variance 做"快"，我们用 variance 做"好"。**

---

## 2. 论文核心贡献（3 个，每个都对应一个具体竞品 gap）

### Contribution 1：**RTAP** — Reverse-Time Anchored Progress（视觉 advantage 信号）

**Why 要做**：替代原项目对人工 `progress_gt` 标注的依赖。这是 paper.md 中审稿意见 W4 的核心质疑，必须解决。

**与已有工作的差异**：
- **vs Hindsight Experience Replay (HER, NeurIPS 2017)**：HER 用真实 goal state 做 relabel；我们用 episode 终点 + foundation features 做特征空间的 *progress 估计*，目的不是 relabel 而是生成 advantage
- **vs Decision Transformer (NeurIPS 2021)**：DT 需要 reward-to-go；我们完全无 reward
- **vs π\*0.6**：π\*0.6 训练 670M value 模型预测剩余步数；我们用一行 kNN 实现近似

**做法**（3 个 level，论文用 Level-1 + Level-2）：

```python
# Level-1: Single-Anchor RTAP
def rtap_single(features_per_episode):
    f_goal = features_per_episode[-1]                    # episode 终点
    d = 1 - cos_sim(features_per_episode, f_goal)
    return 1 - d / d.max()                               # progress ∈ [0, 1]

# Level-2: Multi-Anchor RTAP（处理多种成功方式）
def rtap_multi(all_episodes):
    goals = [ep[-1] for ep in all_episodes]
    centers = kmeans(goals, K=num_tasks * 3)
    progress = [softmax(max(cos_sim(ep, c) for c in centers), τ=0.1)
                for ep in all_episodes]
    return progress
```

**理论支持**（1 页 appendix）：在 deterministic goal-conditioned MDP + 特征空间 Lipschitz 假设下，single-anchor RTAP 是 $V^*$ 的一致估计。本质是 HER 在特征空间的对偶。

**工程量**：~250 行 Python，0 GPU 成本。

---

### Contribution 2：**FlowVar Advantage** — 用 flow-matching policy 自身的 action variance 作为 advantage 信号（关键新意）

**Why 这是真正的算法新意**：
- **没人在 π0/π0.5 这类 flow-matching VLA 上系统地把 multi-noise 生成的 action chunk 方差作为 advantage 信号**
- AdaFlow 用 variance 做推理调度，**没有 advantage conditioning**
- π\*0.6 训练独立 value model，**没用到 flow matching 自身的特性**
- 这填补了一个明显的 gap：**flow matching 模型其实"自带"一个免费的 value 信号**

**直觉**（这条直觉本身就值一个 contribution）：
> Flow matching 用不同 noise 去采样同一状态的 action chunks。如果是高 value 状态（"应该这么做"很明确），policy 在不同 noise 下应该收敛到相近 action；如果是低 value 状态（"该做什么都可以"或"探索区域"），不同 noise 会生成发散的 action。
>
> **Action variance 就是 policy 自身对"我对这个状态多有把握"的隐式打分。**

**做法**：

```python
# 阶段 1：先 vanilla fine-tune π0.5（拿到一个 reference flow policy π_ref）
# 阶段 2：用 π_ref 给数据集每个状态打 advantage 分

def flowvar_advantage(state_s, pi_ref, n_samples=8):
    # 用 n 个不同 noise 在同一 state 上生成 action chunks
    chunks = [pi_ref.flow_sample(state_s, noise=ε_i) for i in range(n_samples)]
    # 在 chunk 维度上计算 dispersion
    var = mean_over_time(std_across_samples(chunks))     # action chunk std
    advantage = 1 / (1 + var)                            # 方差越小 advantage 越高
    return advantage
```

**为什么这是 strong contribution**：
1. **首次将 flow-matching policy 的 ensemble variance 作为 advantage 信号**注入到 advantage-conditioned 训练
2. 与 RTAP 完全**正交**：RTAP 是 "视觉接近成功"，FlowVar 是 "策略行为确信"——两个完全不同维度
3. 不依赖任何外部信号（reward / preference / progress / 干预）
4. 算法极简，但有清晰的 information-theoretic 解释：variance ≈ policy entropy ≈ -log p(a*|s)

**与已有工作的精确区分**：

| 已有工作 | 用的什么 | 用作什么 | 差别 |
|---|---|---|---|
| AdaFlow (NeurIPS 2024) | action variance | 推理调度 | 不做 advantage conditioning |
| Imitation Learning by Estimating Expertise (ICML 2022) | demonstration quality | 加权 BC loss | 估计的是 demonstrator 等级，不是 state-wise advantage |
| Decision Transformer | return-to-go | 条件输入 | 需要 reward 标注 |
| π\*0.6 | trained value | 条件输入 | 需要 reward + 训练 |
| **FlowVar (ours)** | **policy 的 ensemble variance** | **advantage 条件输入** | **新组合** |

**工程量**：~300 行 Python，需要 **1 次额外的前向推理 pass**（在数据集上跑 vanilla π0.5 推理生成 advantage 标签），约 50-80 GPU·h（一次性）。

---

### Contribution 3：**Orthogonal Signal Fusion** — RTAP × FlowVar 的正交融合（实验为核心）

**Why**：单个信号必然在某些场景失效（RTAP 在长任务失效；FlowVar 在 policy 还没训练好时失效）。但两个信号**理论上正交**：

- RTAP 看的是**视觉空间**（输入侧）
- FlowVar 看的是**动作空间**（输出侧）

**做法**：
```python
# 两个 advantage 信号独立 percentile 化后融合
adv_rtap = percentile_normalize(rtap_advantage)
adv_flowvar = percentile_normalize(flowvar_advantage)
adv_fused = (adv_rtap + adv_flowvar) / 2                 # 简单平均
# 也可学习权重：adv_fused = α * adv_rtap + (1-α) * adv_flowvar
```

**Why 这是 strong contribution**：
1. **正交性可被实验证实**：画 scatter plot (RTAP-adv, FlowVar-adv)，如果 Pearson correlation < 0.5 就是 strong evidence
2. **互补性可被实验证实**：在 RTAP 失效的任务（如 LIBERO-Long）上，单 FlowVar 应该顶上；反之亦然
3. **构成完整理论故事**：input-side signal + output-side signal = 充分覆盖 advantage 估计的两个维度

**工程量**：~150 行（融合策略 + percentile normalize）。

---

### ❌ 砍掉的内容（保证工程量可控）

| 砍掉 | 原因 |
|---|---|
| ~~Multi-encoder ensemble (SigLIP+DINOv2+Theia+VC-1+R3M)~~ | +1200 GPU·h，吸引力远低于 FlowVar |
| ~~Confidence-aware (kNN 方差)~~ | 已被 FlowVar 覆盖（FlowVar 本身就是一种 confidence）|
| ~~OpenX-Embodiment 大实验~~ | 时间不够，LIBERO + RoboCasa 够支撑 |
| ~~World Model / 想象未来帧~~ | π0.7 占赛道，技术风险高 |
| ~~严格收敛证明~~ | CoRL 不要求，命题级陈述够 |
| ~~实物机器人~~ | 无条件 |

---

## 3. 公开数据集选择

| 数据集 | 必要性 | 任务数 | 用途 |
|---|---|---|---|
| **LIBERO** (Spatial/Object/Goal/Long) | **必做** | 130 tasks × 50 demos | 主战场，与 π0.5、π\*0.6 同基准 |
| **RoboCasa-Kitchen 子集** | **推荐** | 5-8 atomic tasks | 跨场景泛化 |
| ~~OpenX-Embodiment~~ | 砍掉 | — | 时间允许再加 |

---

## 4. 实验设计

### 4.1 主表 Table 1 — 显式把"需要什么"做成列

| 方法 | reward | value model | preference | rollout | Spatial | Object | Goal | Long | Avg |
|---|:-:|:-:|:-:|:-:|---|---|---|---|---|
| π0.5 vanilla | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| π0.5 + Time-Index advantage (naive) | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| π0.5 + DT-style return cond. | ✅ | ❌ | ❌ | ❌ | – | – | – | – | – |
| π0.5 + π\*0.6-style trained value (offline re-impl) | ✅ | ✅ | ❌ | ❌ | – | – | – | – | – |
| **π0.5 + FreeVAC (RTAP only)** | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| **π0.5 + FreeVAC (FlowVar only)** | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |
| **π0.5 + FreeVAC (Fusion + CFG)** | ❌ | ❌ | ❌ | ❌ | – | – | – | – | – |

> 故事点：(a) FreeVAC-Fusion ≥ trained value baseline，但 0 标注 0 训练；(b) 单 RTAP 在 Long 上失效但 Fusion 救回来——证明正交融合的价值。

### 4.2 关键 Ablation 与分析

| Ablation / 分析 | Configs | GPU·h |
|---|---|---|
| **RTAP × FlowVar 正交性 scatter plot**（核心 figure）| 1 次计算 | 0 |
| RTAP 变体 | Time-index / Single / Multi / Hybrid × 2 suites | ~400 |
| FlowVar 样本数 N | N ∈ {2, 4, 8, 16} × 2 suites | ~300 |
| Fusion 权重 α | α ∈ {0, 0.25, 0.5, 0.75, 1} × 2 suites | ~250 |
| Bins / CFG / Confidence | 标准消融 | ~400 |
| **失效分析 "When does each signal fail?"** | 已有数据上的可视化 | 0 |
| Encoder ablation (appendix) | SigLIP2 vs DINOv2 × 1 suite | ~150 |

### 4.3 加分项

| 实验 | 用途 | GPU·h |
|---|---|---|
| Data efficiency 曲线 | 10/25/50/100% × 2 suites → Figure 2 | ~400 |
| LIBERO-90 → LIBERO-10 transfer | 跨任务泛化 → Table 6 | ~300 |
| RTAP / FlowVar 与 oracle 的 Spearman | 信号可靠性硬证据 | ~10 |

---

## 5. 工作计划（15 周，约 5000 GPU·hours）

### Phase 1 — 实现三个核心信号 + 复现 baseline（Week 1-3）

| Step | 工作 | GPU·h |
|---|---|---|
| 1.1 | LIBERO 4 suites + RoboCasa 转 LeRobot 格式 | 0 |
| 1.2 | 实现 RTAP Level-1 + Level-2 (~250 行) | 0 |
| 1.3 | SigLIP2 特征提取（已有代码）| ~80 |
| 1.4 | **π0.5 vanilla fine-tune LIBERO-Spatial（同时作为 FlowVar 的 reference policy）** | ~150 |
| 1.5 | **实现 FlowVar advantage 计算 (~300 行)** | 0 |
| 1.6 | 用 vanilla π0.5 跑 FlowVar 给所有数据打 advantage 分 | ~80 |
| 1.7 | 实现 Orthogonal Fusion (~150 行) | 0 |
| 1.8 | Sanity check：RTAP/FlowVar 与手工 oracle progress 的 Spearman | ~10 |
| 1.9 | **正交性 scatter plot**：(RTAP-adv, FlowVar-adv) 散点图 + Pearson correlation | 0 |

**Phase 1 GPU**：~320 h；时间：3 周
**🚧 Gate Criterion（Week 3 末）**：见 §7

### Phase 2 — 主实验（Week 4-9）

| Step | 工作 | GPU·h |
|---|---|---|
| 2.1 | π0.5 vanilla 4 suites 全跑 | ~600 |
| 2.2 | π\*0.6-style trained value baseline（用项目已有 TD Value Head）| ~800 |
| 2.3 | DT-style baseline | ~600 |
| 2.4 | **FreeVAC 主实验**：RTAP-only / FlowVar-only / Fusion × 4 suites × 3 seeds | ~1800 |
| 2.5 | RoboCasa 子集（3 主方法 × 6 tasks × 2 seeds）| ~600 |

**Phase 2 GPU**：~4400 h；时间：6 周

### Phase 3 — Ablation + Data Efficiency（Week 7-11，部分并行）

| Step | 工作 | GPU·h |
|---|---|---|
| 3.1 | RTAP / FlowVar / Fusion 各自变体 | ~950 |
| 3.2 | Bins / CFG / Encoder | ~550 |
| 3.3 | Data efficiency 曲线 | ~400 |

**Phase 3 GPU**：~1900 h

### Phase 4 — 跨任务 + 写作（Week 10-15）

| Step | 工作 | GPU·h |
|---|---|---|
| 4.1 | LIBERO-90 → LIBERO-10 transfer | ~300 |
| 4.2 | 论文撰写（main + appendix）| 0 |
| 4.3 | rebuttal buffer | ~300 |

**Phase 4 GPU**：~600 h

### 总计资源

| 项目 | 数值 |
|---|---|
| **GPU·hours (A100)** | **~7,200**（注：比之前略高，因为新增 FlowVar 主实验）|
| 若 GPU 紧张可砍 | -1500 h（砍 RoboCasa） / -2000 h（4 suites 只做 2 个） |
| **人力时间** | **15 周** |
| **存储** | **~600 GB** |
| **8×A100 集群纯计算时间** | **~38 天** |

> 资源压缩策略：如 < 4000 GPU·h 预算，只做 LIBERO-Spatial + LIBERO-Long，砍掉 RoboCasa 和 transfer 实验。

---

## 6. 论文叙事结构

```
1. Introduction
   Hook: "π*0.6 needs reward labels and trained value models.
          FlowPRO needs preference pairs from teleoperated interventions.
          Both still require real-robot rollouts.
          But what if you only have a static public dataset—say, LIBERO?
          We show that the flow-matching policy itself, combined with episode
          endpoint anchoring, already contains all the advantage signal you need."

   Contributions:
   (a) RTAP: visual progress proxy from episode endpoints, no labels needed
   (b) FlowVar: flow-matching policy's own action variance as advantage signal
   (c) Orthogonal Signal Fusion: empirically and theoretically independent
       advantage sources, robust where either signal alone fails

2. Related Work
   - VLA post-training: π0/π0.5, π*0.6 (RECAP), FlowPRO (RPRO), OpenVLA-OFT
   - Flow matching for action generation: π0, AdaFlow
   - Advantage-conditioned policies: DT, RvS, AWR, IQL
   - Goal-conditioned RL: HER
   - Imitation from suboptimal demos: ILMAR, ADR-BC, value-aligned BC

3. Method
   3.1 Preliminaries: advantage-conditioned flow matching (cite π*0.6)
   3.2 RTAP: visual progress via reverse-time anchoring  ← input-side signal
   3.3 FlowVar: policy ensemble variance as advantage   ← output-side signal
   3.4 Orthogonal Signal Fusion
   3.5 Training and Inference with CFG

4. Theoretical Analysis (1 页)
   Proposition 1: kNN-based RTAP is a consistent V* estimator under
                  goal-conditioned MDP assumption
   Proposition 2: For an optimal flow-matching policy, action variance is
                  proportional to advantage uncertainty (Boltzmann rationality)
   Proposition 3: RTAP and FlowVar are independent under mild assumptions
                  (sketch + empirical verification in §5)

5. Experiments
   5.1 Setup (LIBERO 4 suites + RoboCasa subset)
   5.2 Main Results — Table 1
   5.3 Orthogonality Analysis — Figure: scatter plot, Pearson correlation
   5.4 Ablations
   5.5 Data Efficiency
   5.6 Cross-Task Generalization
   5.7 When does each signal fail? — Figure with failure case analysis

6. Discussion & Limitations
   - 公开承认：FlowVar 需要一个 reference policy（虽然不需要新训练）
   - 单 RTAP 在 long-horizon multi-subgoal 任务上的局限
   - 我们的 niche：public offline data + zero auxiliary cost
   - 未来方向：与 π*0.6 / FlowPRO 互补（FreeVAC 提供 cold-start advantage，
              π*0.6/FlowPRO 提供 online improvement）

7. Conclusion
```

**核心叙事张力**：

> π\*0.6 / FlowPRO 的 advantage signal 都来自**外部世界**（reward / preference / 干预）。
> 我们证明 advantage signal 也可以完全来自**内部**：
>   - 数据自身的 episode 结构（RTAP）
>   - 模型自身的不确定性（FlowVar）
>
> 这两个"内部信号"虽然各自不完美，但**正交互补**，融合后达到与外部信号相当的性能。

---

## 7. Gate Criterion（Week 3 末决策）

满足 ≥ 3/5 继续；否则切 Plan B：

| 检查 | 阈值 | 含义 |
|---|---|---|
| π0.5 vanilla LIBERO-Spatial SR 复现 | ±2% 官方数字 | setup 正确 |
| RTAP 与 oracle progress Spearman | > 0.5 | RTAP 信号有效 |
| **FlowVar 与 oracle progress Spearman** | **> 0.3**（FlowVar 不一定要直接相关于 progress，门槛较低）| FlowVar 信号有意义 |
| **RTAP × FlowVar Pearson correlation** | **< 0.6** | **正交性成立 — 这是论文最关键的硬条件** |
| FreeVAC-Fusion vs vanilla on Spatial | +1.5 pts SR | 方法 work |

**最关键的是第 4 项**：如果 RTAP 和 FlowVar 高度相关，那 Fusion 就不是真贡献，整个第 3 contribution 塌方。

---

## 8. Plan B（任意 Gate 不过时的退路）

### Plan B-1：单押 RTAP，转向 Data-Centric
- 标题：**"Foundation-Feature-Based Subset Selection for Public Robot Datasets"**
- 卖点：用 RTAP advantage 筛选 top-30% 数据，达到 100% 数据 95% 性能

### Plan B-2：单押 FlowVar，转向 Flow Matching Analysis
- 标题：**"What does the variance of a flow-matching policy tell us?"**
- 把 FlowVar 作为诊断工具：分析在 LIBERO 上哪些 state 是高 variance 的，是否对应 task-critical decisions
- 投 ICLR / ICML workshop（成功率高）

### Plan B-3：Diagnostic Paper
- 转为分析性论文：研究"在没有外部 reward 信号时，哪些 free signals 能用作 advantage"
- 系统性枚举所有 free signal（time-index, RTAP, FlowVar, action consistency, neighbor diversity, etc.）
- 给出"signal landscape"分析

---

## 9. 主要风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| **审稿人："π\*0.6 / FlowPRO 已经做了 reward-free"** | 高 | Intro Table 1 直接对比"需要什么"列，凸显我们的 niche |
| **审稿人："FlowVar 就是 AdaFlow / 信号不强"** | 中高 | 明确区分（AdaFlow 推理调度 vs 我们训练 advantage）；用正交性 scatter plot 证明 FlowVar 与 RTAP 互补 |
| **RTAP × FlowVar 高度相关（>0.6）** | 中 | 切 Plan B-1 或 B-2（单押其一）|
| LIBERO 上 vanilla π0.5 已是 SOTA | 中 | 切 Plan B-1（data efficiency）|
| FlowVar 依赖 reference policy，被质疑"还是需要训练" | 中 | 强调：reference policy 就是 vanilla fine-tune，不需要额外训练 |
| GPU 不够 | 中 | 砍 RoboCasa 和 transfer；只做 LIBERO 2 个 suites |

---

## 10. 立即执行的 Action Items（Week 1）

```
□ Day 1-2:  下载 LIBERO，跑通 openpi 官方 LeRobot 转换
□ Day 3:    π0.5 vanilla LIBERO-Spatial 开始 fine-tune（后台跑，作为 FlowVar reference）
□ Day 4:    实现 Single-Anchor RTAP（~150 行）
□ Day 5:    实现 Multi-Anchor RTAP（~100 行），可视化 1 个 episode 的 progress 曲线
□ Day 6:    手工标 100 个 episode 的 oracle progress，计算 RTAP Spearman
□ Day 7:    Week 1 review
□ Week 2:   等 vanilla π0.5 训完 → 实现 FlowVar 推理 pass → 计算所有数据的 FlowVar advantage
□ Week 3:   计算 RTAP × FlowVar 正交性散点图 + Pearson correlation
            跑 FreeVAC-Fusion 在 LIBERO-Spatial 上的 1 次完整训练
            按 §7 Gate Criterion 决策
```

---

## 11. 参考来源（这次重写依据的关键工作）

| 工作 | 时间 | 我们的 framing 与它的关系 |
|---|---|---|
| **π\*0.6 (RECAP)** [arXiv:2511.14759] | 2025.11 | advantage conditioning 范式的奠基；我们移除其 reward/value/rollout 依赖 |
| **FlowPRO (RPRO)** [arXiv:2606.05468] | 2026.06 | 另一种 reward-free 路线（preference + 干预）；我们移除其 preference/干预/实机依赖 |
| **AdaFlow** [NeurIPS 2024] | 2024 | 首次用 flow matching action variance；我们用作 advantage 而非推理调度 |
| **HER (Hindsight Experience Replay)** [NeurIPS 2017] | 2017 | RTAP 的理论亲属；我们做特征空间的对偶 |
| **Decision Transformer** [NeurIPS 2021] | 2021 | return-conditioned policy 的奠基；我们用 advantage 而非 return |
| **ILMAR / ADR-BC / Value-Aligned BC** | 2024-2025 | weighted BC 的相关工作；我们做 advantage conditioning 而非 loss weighting |
| **π0.7** [arXiv:2604.15483] | 2026.04 | world model + 多模态上下文；与我们方向正交，仅 related work 引用 |

---

## 12. 总结：v3 vs v2 / v1 的关键升级

| 维度 | v1 (RTAP + multi-encoder) | v2 (RTAP + confidence) | **v3 (RTAP + FlowVar + Fusion)** |
|---|---|---|---|
| 核心新意 | 自监督 progress + ensemble | 自监督 progress + 方差 | **2 个正交 advantage 信号 + 融合** |
| 算法新意层级 | 数据预处理 | 数据预处理 | **算法层面（FlowVar 是新 signal）** |
| 与 AdaFlow 区分 | 不需要 | 不需要 | **明确（advantage vs scheduling）** |
| 与 FlowPRO 区分 | 部分 | 部分 | **明确（无 preference 无干预）** |
| 工程量 | 高 | 中 | 中（FlowVar 增加 ~80 GPU·h）|
| 抗审稿能力 | 弱（trick 组合）| 中 | **强（每个 contribution 都有清晰 gap）**|
| 风险点 | 多 encoder 收益不确定 | 创新弱 | **正交性可能不成立 → 但 Plan B 完备** |

**最终 pitch**：
> π\*0.6 让 advantage conditioning 成为 VLA 后训练范式。
> 之后 FlowPRO 用 preference + 干预把 reward 标注去掉了。
> 我们再进一步，**把所有外部信号都去掉**——advantage 信号完全来自数据自身（RTAP）和模型自身（FlowVar）。
> 这两个内部信号**正交互补**，让任何研究者都能在公开静态数据上做 π\*0.6-style 的后训练。

---

*Plan v3。下一步：执行 §10 Day 1-2 的 LIBERO 数据准备。*
