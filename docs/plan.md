# FreeVAC: Label-Free Advantage Conditioning for Offline VLA Post-Training

> **目标**：在 4-6 个月内、仅用公开数据集、不超过 5000 GPU·hours，
> 把 Kai05-VLA 项目改造成一篇可投 CoRL 2026 / NeurIPS 2026 的论文。
>
> **撰写日期**：2026-06-25 (final)

---

## 1. 形势：我们处在哪里？

### 1.1 当前项目（Kai05-VLA）的核心方法
- 基座：π0.5（PaliGemma + Gemma Action Expert + Flow Matching）
- 创新：用 **VC-Value**（SigLIP 特征空间 kNN + progress_gt 加权）替代显式 value model
- 把 advantage 二值化/分箱后通过 sinusoidal embedding 注入到 language tokens

### 1.2 致命弱点（来自 paper.md 审稿意见 + explain.md 分析）
1. **依赖人工 progress_gt 标注** → 99% 公开数据集没有
2. 只在叠衣服单任务验证 → 无泛化证据
3. 本质是 kNN，新颖性弱
4. 缺理论支撑

### 1.3 业界最新进展（必须正面回应）

**π\*0.6 (RECAP, 2025.11)**：advantage-conditioned policy + CFG 已被 PI 团队验证为 VLA 后训练的关键范式。但它需要：
- 人工标注 episode 成败 reward
- 训练 670M 参数的 distributional value model
- 多轮实机 on-policy rollout + 人在环干预

**π0.7 (2026.04)**：world model + 多模态上下文路线，与我们正交，不构成威胁。

### 1.4 我们的 Niche（一句话）

> **π\*0.6 证明 advantage conditioning 范式有效，但它的获取代价（reward 标注 + value 训练 + rollout）目前只有 PI 这种公司付得起。我们把这个代价降到 0：仅用公开静态数据 + 预训练特征 + episode 自身结构，实现同范式的后训练。**

这个定位**同时解决了项目原 4 个弱点**，并把 π\*0.6 从威胁转化为最强 reference baseline。

---

## 2. 论文核心贡献（只做 2 件事）

### Contribution 1：**RTAP** — Reverse-Time Anchored Progress（替代 progress_gt）

**Why**：原项目最致命的 progress_gt 依赖必须被消除，否则方法无法 work on 任何公开数据集。这是 paper.md 审稿意见 W4 的核心质疑。

**What**：把每个 episode 的最后一帧作为该任务的"成功锚点"，progress 由"当前帧到锚点的特征距离"自监督生成。

**How**（三个 level，按工作量递增；论文只用 Level-1 + Level-2）：

```python
# Level-1: Single-Anchor RTAP（必做）
def rtap_single(features_per_episode):
    f_goal = features_per_episode[-1]                       # 终点锚点
    d = 1 - cos_sim(features_per_episode, f_goal)           # 到锚点距离
    progress = 1 - d / d.max()                              # 归一化
    return progress

# Level-2: Multi-Anchor RTAP（论文主推）
def rtap_multi(all_episodes_features):
    goals = [ep[-1] for ep in all_episodes_features]
    centers = kmeans(goals, K=num_tasks * 3)                # 同任务多种成功方式
    for ep in all_episodes_features:
        sims = [cos_sim(ep, c) for c in centers]
        progress = softmax(max(sims, axis=0), tau=0.1)      # 软分配
    return progress
```

**理论支持**（1 页 appendix）：在 deterministic goal-conditioned MDP + 特征空间 task-Lipschitz 假设下，single-anchor RTAP 是 $V^*$ 的一致估计；本质是 Hindsight Experience Replay 的特征空间对偶。

**代码量**：~250 行 Python，几乎 0 GPU 成本（CPU 计算）。

---

### Contribution 2：**Confidence-Aware Advantage**（小创新点，让方法稳健）

**Why**：kNN 在数据稀疏区域噪声大；π\*0.6 没有这个机制。这是我们对 π\*0.6 范式的小改进，能让审稿人看到"不只是无 reward 版本，也有方法学新意"。

**What**：每个帧除了 advantage value，还附带一个 confidence score（top-N similarity 的方差），policy 在训练时学到"高 advantage + 高 confidence → 强模仿；低 confidence → 退化为普通 BC"。

**How**：
```python
adv, top_n_sims = kNN_value(s_t)
confidence = 1 / (1 + std(top_n_sims))                       # similarity 分布越聚集越自信
# 把 (advantage, confidence) 一起注入到 advantage embedding token
```

**代码量**：~150 行（修改 `pi0_pytorch.py` 的 advantage 注入逻辑 + `calculate_VC_value.py` 输出 confidence）。

---

### ❌ 不做什么（节省工程量的关键决策）

| 不做 | 原因 |
|---|---|
| ~~Multi-encoder ensemble (SigLIP+DINOv2+Theia+VC-1+R3M)~~ | 5 套特征提取 + 融合 = +1200 GPU·h + 1.4 TB 存储，对审稿人吸引力远低于 RTAP |
| ~~World Model / 想象未来帧~~ | π0.7 已占赛道，工程量极大，技术风险高 |
| ~~大规模 OpenX 实验~~ | 时间紧时砍掉，LIBERO + RoboCasa 已够支撑论文 |
| ~~严格的理论收敛证明~~ | CoRL 不要求；1 页 appendix 给出近似性陈述即可 |
| ~~实物机器人实验~~ | 无条件，且 LIBERO 仿真已是社区认可基准 |

---

## 3. 公开数据集（精简到 2 个）

| 数据集 | 必要性 | 任务数 | 用途 |
|---|---|---|---|
| **LIBERO**（Spatial / Object / Goal / Long 4 suites） | **必做** | 130 tasks × 50 demos | 主战场，与 π0.5 / π\*0.6 同基准 |
| **RoboCasa-Kitchen 子集** | **推荐** | 5-8 atomic tasks | 跨场景泛化验证 |
| ~~OpenX-Embodiment~~ | 砍掉 | — | 时间允许再加 |

---

## 4. 实验设计（够用就好）

### 4.1 主表 — Table 1

在 LIBERO 4 个 suites 上对比 success rate。**关键是把"需要什么标注"做成显式列**，让审稿人一眼看出我们的 niche：

| 方法 | 需 reward 标注 | 需训练 value | 需 rollout | Spatial | Object | Goal | Long | Avg |
|---|:-:|:-:|:-:|---|---|---|---|---|
| π0.5 vanilla | ❌ | ❌ | ❌ | – | – | – | – | – |
| π0.5 + Time-Index advantage（naive baseline）| ❌ | ❌ | ❌ | – | – | – | – | – |
| π0.5 + DT-style return conditioning | ✅(终态) | ❌ | ❌ | – | – | – | – | – |
| **π0.5 + π\*0.6-style trained value**（用项目 TD Value Head 复现）| ✅(终态) | **✅ +Value Net** | ❌ | – | – | – | – | – |
| **π0.5 + FreeVAC (ours, Level-1 RTAP)** | ❌ | ❌ | ❌ | – | – | – | – | – |
| **π0.5 + FreeVAC (ours, Level-2 RTAP + Conf + CFG)** | ❌ | ❌ | ❌ | – | – | – | – | – |

**故事点**：
- vs vanilla → 证明 advantage conditioning 在 offline 公开数据上有用
- vs trained value baseline → 证明无需训练 value 也能达到 comparable 性能
- vs DT → 证明 advantage 优于 return
- 跨 4 个 suites → 证明泛化性

### 4.2 关键 Ablation（每项只跑 1-2 个 suites，足以说明问题）

| Ablation | Configs | 估算 GPU·h |
|---|---|---|
| RTAP 变体 | Time-index / Single-anchor / Multi-anchor / Time-Visual hybrid（4 × 2 suites） | ~400 |
| Advantage bins | {2, 5, 10}（3 × 1 suite） | ~150 |
| CFG scale | {1.0, 1.5, 2.0, 3.0}（4 × 1 suite） | ~200 |
| Confidence-aware on/off | 2 × 2 suites | ~200 |
| Encoder（appendix）| SigLIP2 vs DINOv2 × 1 suite | ~150 |

### 4.3 加分项

| 实验 | 用途 | GPU·h |
|---|---|---|
| **Data efficiency 曲线** | 数据量 10/25/50/100% × 2 suites → Figure 2 | ~400 |
| **LIBERO-90 → LIBERO-10 transfer** | 跨任务泛化 → Table 6 | ~300 |
| **RTAP sanity check** | 手工标 100 episode 的 oracle progress，计算 Spearman 相关性 | ~10 |
| **"When does FreeVAC fail?" 章节** | 长任务 / 多模态目标失败案例分析（已有数据即可）| 0 |

### 4.4 RoboCasa 实验（泛化卖点）
3 个核心方法 × 6 tasks × 2 seeds = ~600 GPU·h

---

## 5. 工作计划（15 周，5070 GPU·h）

### Phase 1 — RTAP 落地 + 数据 + 复现 baseline（Week 1-3）

| Step | 工作 | GPU·h |
|---|---|---|
| 1.1 | LIBERO 4 suites 转 LeRobot 格式（用 openpi 官方脚本）| 0 |
| 1.2 | RoboCasa-Kitchen 子集转换（5-8 tasks）| 0 |
| 1.3 | 实现 RTAP Level-1 + Level-2（~250 行）| 0 |
| 1.4 | SigLIP2 特征提取（所有数据集）| ~80 |
| 1.5 | 用 RTAP 计算 VC-Value + 输出 confidence | ~30 |
| 1.6 | **RTAP sanity check**：手工标 100 个 episode 的 progress，验证 Spearman > 0.6 | ~10 |
| 1.7 | π0.5 vanilla LIBERO-Spatial 复现（确保 setup 没问题）| ~150 |

**Phase 1 GPU**：~270 h；时间：3 周
**🚧 Gate Criterion（Week 3 末）**：见 §7

---

### Phase 2 — 主实验（Week 3-9）

| Step | 工作 | GPU·h |
|---|---|---|
| 2.1 | π0.5 vanilla 4 suites（3 个剩余）| ~600 |
| 2.2 | π\*0.6-style trained value baseline（复用项目已有 TD Value Head）4 suites × 2 seeds | ~800 |
| 2.3 | DT-style baseline 4 suites × 2 seeds | ~600 |
| 2.4 | **FreeVAC 主实验**：Level-1 + Level-2(full) × 4 suites × 3 seeds = 24 runs | ~1100 |
| 2.5 | RoboCasa：vanilla + trained value + FreeVAC × 6 tasks × 2 seeds | ~600 |

**Phase 2 GPU**：~3700 h；时间：6 周

---

### Phase 3 — Ablation + Data Efficiency（Week 7-11，与 Phase 2 部分并行）

| Step | 工作 | GPU·h |
|---|---|---|
| 3.1 | RTAP 变体 ablation | ~400 |
| 3.2 | Bins / CFG / Confidence ablation | ~550 |
| 3.3 | Data efficiency 曲线 | ~400 |
| 3.4 | Encoder ablation (appendix) | ~150 |

**Phase 3 GPU**：~1500 h；时间：4 周

---

### Phase 4 — 跨任务泛化 + 论文写作（Week 10-15）

| Step | 工作 | GPU·h |
|---|---|---|
| 4.1 | LIBERO-90 → LIBERO-10 transfer | ~300 |
| 4.2 | 论文撰写（main + appendix）| 0 |
| 4.3 | rebuttal 准备 / 补充实验 buffer | ~300 |

**Phase 4 GPU**：~600 h

---

### 总计资源

| 项目 | 数值 |
|---|---|
| **GPU·hours (A100)** | **~6,000**（约 $12K） |
| **人力时间** | **15 周（≈ 3.5 月）** |
| **存储** | **~600 GB** |
| **8×A100 集群纯计算时间** | **~31 天** |

> Phase 2 是关键瓶颈，建议 Phase 2/3 并行（不同 GPU 跑不同实验），缩短墙钟时间。

---

## 6. 论文骨架（v2，正面对标 π\*0.6）

```
1. Introduction
   Hook: "PI's π*0.6 established advantage-conditioned policies as the key paradigm
          for VLA post-training. But its three costs—reward labels, a 670M value model,
          and on-robot rollouts—block adoption on public offline datasets.
          We remove all three."
   Contributions:
   (a) RTAP: self-supervised progress proxy via episode endpoint anchoring
   (b) FreeVAC: label-free, value-model-free, rollout-free advantage conditioning
   (c) Comparable to π*0.6-style trained-value baseline on LIBERO/RoboCasa with 0 auxiliary training

2. Related Work
   - VLA post-training: π0.5, π*0.6 (RECAP), OpenVLA-OFT
   - Advantage-conditioned policies: DT, RvS, AWR, IQL
   - Goal-conditioned RL / HER (RTAP 的理论亲属)
   - Foundation features for robotics: SigLIP, DINOv2, R3M, VC-1

3. Method
   3.1 Preliminaries: advantage-conditioned flow matching (cite π*0.6)
   3.2 Reverse-Time Anchored Progress (RTAP)  ← Core
   3.3 Feature-Space kNN with Confidence-Aware Advantage
   3.4 Training and Inference with CFG

4. Theoretical Analysis (1 页)
   Proposition 1: kNN value estimator consistency (cosine kernel)
   Proposition 2: Single-anchor RTAP ≈ V* under goal-conditioned MDP
   When does RTAP fail? (long sub-task chains, multi-modal goals)

5. Experiments
   5.1 Setup
   5.2 Main Results — Table 1
   5.3 Ablations — Table 2-3
   5.4 Data Efficiency — Figure 2
   5.5 Cross-Task Generalization — Table 4
   5.6 RTAP vs Oracle Progress Correlation — Figure 3
   5.7 Failure Analysis — when FreeVAC fails

6. Discussion & Limitations
   - 公开承认：需要 fine-grained reward signal 的任务上不如 π*0.6
   - 我们的 niche：public offline data 上的 zero-cost 后训练
   - 未来方向：与 π*0.6 互补（FreeVAC 做 cold-start, π*0.6 做 online improvement）

7. Conclusion
```

---

## 7. Gate Criterion（避免投入大量资源后才发现不 work）

**Week 3 末必须通过的 3 个检查（满足 ≥ 2 个继续，否则切 Plan B）**：

| 检查 | 阈值 | 含义 |
|---|---|---|
| π0.5 vanilla LIBERO-Spatial SR 复现 | ±2% 官方数字 | setup 正确 |
| RTAP 与手工 oracle progress 的 Spearman 相关性 | > 0.6 | RTAP 假设成立 |
| FreeVAC (Level-1) vs vanilla on Spatial | +1.5 pts SR | 方法 work |

---

## 8. Plan B（Gate 不过时的退路）

### Plan B-1：Data-Centric 角度
- 标题改为 **"Foundation Features Discover High-Value Subsets in Public Robot Datasets"**
- 主卖点：用 FreeVAC 做数据筛选（保留 top-30% advantage 数据），仅用 30% 数据达到 100% 数据 95% 的性能
- CoRL 2025 已有多篇类似工作验证此方向可发表

### Plan B-2：Diagnostic Paper
- 转为分析性论文 "What value signal can foundation features extract from offline data?"
- 所有 ablation + 失败案例作为正向贡献
- 目标 CoRL / RSS / ICLR workshop

---

## 9. 主要风险与应对

| 风险 | 应对 |
|---|---|
| 审稿人："π\*0.6 已经做了" | Intro 第一段就把 niche 写清楚：no reward + no value model + no rollout |
| LIBERO 上 vanilla π0.5 已是 SOTA | 切 Plan B（data efficiency 角度）|
| RTAP 在 Long-horizon 任务失效 | 主推 Multi-Anchor RTAP；坦诚写"long-horizon as future work" |
| Trained Value baseline 比 FreeVAC 强 | 卖点改成 "comparable performance, 0 auxiliary cost" |
| 审稿人："VC-Value 就是 kNN" | 强调 RTAP（self-supervised）+ Confidence-aware 是新的；π\*0.6 也是把 advantage 当文本注入，差别在 framework |
| GPU 不够 | 砍掉 RoboCasa；4 suites 只做 Spatial + Long |

---

## 10. 立即可执行的 Action Items（Week 1）

```
□ Day 1-2: 下载 LIBERO，跑通 openpi 官方 LeRobot 转换脚本
□ Day 3:   π0.5 vanilla LIBERO-Spatial 开始 fine-tune（后台跑）
□ Day 4-5: 实现 Single-Anchor RTAP（~150 行），在 1 个 episode 上可视化 progress 曲线
□ Day 6:   实现 Multi-Anchor RTAP（~100 行），用 K-means 跑通
□ Day 7:   手工标 100 个 episode 的 oracle progress，计算 Spearman 相关性
□ Week 2:  FreeVAC 端到端在 LIBERO-Spatial 跑 1 次完整 fine-tune，对比 vanilla
□ Week 3:  Gate Criterion 决策 → 全面铺开 / 切 Plan B
```

---

## 11. 为什么这个 plan 能 work — 五条核心理由

1. **聚焦**：只做 2 个贡献（RTAP + Confidence），不贪多
2. **正面应对 π\*0.6**：把它当 baseline 而不是竞争对手，把"无标注无 rollout"做成尖锐 niche
3. **最大化复用**：60-70% 项目代码直接复用，唯一必须新增的是 RTAP（250 行）+ Confidence（150 行）
4. **资源可控**：5,070 GPU·hours，3.5 个月，对得上 CoRL 2026 截稿（通常 6-7 月）
5. **风险可控**：Week 3 Gate Criterion 决策，3 周内能判定方向，Plan B 兜底

---

**最终 pitch**：
> π\*0.6 教会我们：advantage conditioning 是 VLA 后训练的关键钥匙。
> 但这把钥匙的获取代价（reward 标注 + value model 训练 + 实机 rollout）目前只有 PI 这样的公司付得起。
> **我们把这把钥匙的代价降到 0**——仅用公开静态数据 + 预训练特征 + episode 自身的终点信息，
> 让任何研究者都能在自己手头的数据上做 π\*0.6-style 的后训练。

---

*Plan final 版本。下一步：执行 §10 Day 1-2 的 LIBERO 数据准备。*
