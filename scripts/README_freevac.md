# FreeVAC quick start

> 三个核心脚本（C1 RTAP / C2 FlowVar / C3 Fusion）+ 单元测试。
> 所有脚本与现有 `calculate_VC_value.py` 同 IO 协议（读 `features/features_merged_*.pt`，
> 写回 LeRobot parquet 新列），下游接现有 `calculate_lerobot_advantage.py` 离散化即可。

---

## 1. 跑单元测试（CPU only, 秒级）

```bash
cd others/Kai05-VLA
python scripts/test_freevac.py
# 或
python -m pytest scripts/test_freevac.py -v
```

覆盖：
- RTAP single_anchor 单调性 + 端点 progress=1
- RTAP multi_anchor 范围 + 有效性
- FlowVar 零方差状态 advantage 最高
- FlowVar inverse / rank_pct 两种模式
- Fusion 在独立 / 共线输入下的 |ρ| 行为
- RTAP→Fusion 在临时 parquet 数据集上的端到端写回-读取

---

## 2. 在真实 LeRobot 数据集上运行（前置：已跑过 `01_extract_features.sh` + `merge_siglip_features.py`）

### Step 2.1 — RTAP（C1）

```bash
DATASET=/path/to/your/lerobot_dataset
python scripts/calculate_RTAP_advantage.py \
    --source_path "$DATASET" \
    --camera_keys top_head \
    --mode single_anchor \
    --device cpu

# Multi-Anchor 版本（用于多种 goal 视觉表征）：
python scripts/calculate_RTAP_advantage.py \
    --source_path "$DATASET" \
    --camera_keys top_head \
    --mode multi_anchor \
    --n_anchors 8 \
    --device cpu
```

产物：parquet 多出 `RTAP_progress_single_anchor_top_head` 列。

### Step 2.2 — FlowVar（C2）

FlowVar 需要 vanilla SFT 的 π0.5 reference checkpoint。脚本提供：
- **纯函数 `compute_flowvar_advantage_from_chunks(chunks)`**：单元测试中直接 mock 验证；
- **CLI 入口**：留有 `_build_pi_ref_from_checkpoint` stub，待真实实验时按现有项目
  `serve_policy.py` / `pi0_pytorch.Pi0Pytorch.sample_actions` 的方式注入即可。

最小驱动脚本片段（自写一个 driver）：

```python
from calculate_FlowVar_advantage import compute_flowvar_advantage_from_chunks
# ... 加载 π_ref 并按你的 dataloader 跑 N 次不同 noise 的 sample_actions ...
chunks = torch.stack(samples, dim=1)             # [B, N, H, A]
adv = compute_flowvar_advantage_from_chunks(chunks, mode="inverse")
# 把 (video_ids, frame_indices, adv) 累加，最后调用：
from calculate_FlowVar_advantage import write_back_to_parquet
write_back_to_parquet(
    {"video_ids": vids, "frame_indices": fids, "advantage": all_adv},
    source_path=Path(DATASET),
    column_name="FlowVar_advantage",
)
```

### Step 2.3 — Fusion（C3）

```bash
python scripts/fuse_advantages.py \
    --source_path "$DATASET" \
    --rtap_col RTAP_progress_single_anchor_top_head \
    --flowvar_col FlowVar_advantage \
    --out_col fused_advantage \
    --alpha 0.5 \
    --scatter_out /tmp/freevac_scatter.npz
```

终端会打印：
```
[Fusion] orthogonality stats:
  pearson_r:   ...
  spearman_r:  ...
  abs_pearson: ...
  ...
```

如果 `abs_pearson >= 0.6` → Gate Criterion 不过，切 Plan B（见 `new paper.md` §10）。

### Step 2.4 — 离散化（复用现有脚本，无改动）

```bash
python scripts/calculate_lerobot_advantage.py \
    --repo_id "$DATASET" \
    --advantage_source fused_advantage \
    --advantage_type 5bins
```

下游训练直接走 `08_train_torch_ali.sh`，模型代码 0 改动。

---

## 3. 与 `new paper.md` / `总分析.md` 的对应

| 文件 | 对应贡献 | 对应实验 |
|---|---|---|
| `calculate_RTAP_advantage.py` | C1 RTAP | §6.5 RTAP 变体消融 |
| `calculate_FlowVar_advantage.py` | C2 FlowVar | §6.5 FlowVar N / reference quality 消融 |
| `fuse_advantages.py` | C3 Orthogonal Fusion | Figure 2 正交性散点 + Table 1A 主行 |
| `test_freevac.py` | 单元测试 | CI / Gate Criterion 准入门槛 |
