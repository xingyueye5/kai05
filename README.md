# Installation of pytorch kai05-VLA@Lirui

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules https://github.com/Lirui-Zhao/Kai05-VLA.git
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies.

```bash
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

---

# FreeVAC: Free Advantage Signals for Flow-Matching VLAs

> **One-liner**: Flow-matching VLAs already know what good behavior looks like — we just need
> to ask them. By combining the policy's own action-variance with a foundation-feature-based
> progress signal, we extract advantage labels from any static dataset, with **zero reward,
> zero value model, zero preference, zero rollout, zero intervention**.

This repository extends `kai05-VLA` (π0.5 with advantage-conditioned flow matching) with the
**FreeVAC** pipeline: three reward-free advantage signals (**RTAP** / **FlowVar** /
**Orthogonal Fusion**) that drop into the existing π0.5 training stack with zero changes to
the model code.

Full method writeup: `docs/new paper.md`. Engineering & code-reuse details: `docs/总分析.md`.

---

## 1. Core idea (Figure 1)

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

Three contributions, each filling a precise gap left by prior work
(π\*0.6 / FlowPRO / AdaFlow):

| ID | Signal | Source | Solves |
|---|---|---|---|
| **C1 RTAP** (Reverse-Time Anchored Progress) | input-side | episode endpoint as goal anchor + SigLIP2 cosine distance | progress\_gt dependency, value-model training |
| **C2 FlowVar** | output-side | $N$-sample action-chunk std under different noise from vanilla SFT $\pi_{\text{ref}}$ | needs no reward / preference; novel training-time use of flow variance |
| **C3 Orthogonal Fusion** | fused | percentile-normalize + α-blend | failure-mode complementarity between input- and output-side signals |

---

## 2. Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 0 ─ Data prep                                                 │
│   00_lerobot_*.sh    →  LeRobot dataset (existing pipeline)         │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1 ─ Feature extraction (unchanged from upstream)              │
│   01_extract_features.sh                                            │
│   scripts/extract_siglip_features.py + merge_siglip_features.py     │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2 ─ Advantage labelling (NEW: FreeVAC)                        │
│   scripts/calculate_RTAP_advantage.py        (C1)                   │
│   scripts/calculate_FlowVar_advantage.py     (C2)                   │
│   scripts/fuse_advantages.py                 (C3 + orthogonality)   │
│                                                                     │
│   (baseline kept for paper: scripts/calculate_VC_value.py)          │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3 ─ Discretization (unchanged from upstream)                  │
│   03_calculate_lerobot_advantage.sh                                 │
│   scripts/calculate_lerobot_advantage.py                            │
│       --advantage_source fused_advantage                            │
│       --advantage_type 5bins                                        │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 4 ─ Training (unchanged; model code 0 modifications)          │
│   08_train_torch_ali.sh  →  scripts/train_pytorch.py                │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 5 ─ Serving (unchanged)                                       │
│   09_serve_policy.sh  →  scripts/serve_policy.py                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Reuse rate from upstream kai05-VLA: ≥ 70%**. FreeVAC adds 4 files
(3 advantage scripts + 1 test) and modifies **zero** model files.

---

## 3. Quick start

### 3.1 Unit tests (CPU only, seconds)

Sanity-check the three FreeVAC components against synthetic data with known ground truth:

```bash
python scripts/test_freevac.py
# or
python -m pytest scripts/test_freevac.py -v
```

Expected output:

```
✓ test_rtap_single_anchor_monotonic
✓ test_rtap_multi_anchor_runs_and_bounded
✓ test_flowvar_zero_variance_gives_max_advantage
✓ test_flowvar_rank_pct_mode
✓ test_flowvar_end_to_end_helper
✓ test_fusion_orthogonal_inputs
✓ test_fusion_collinear_inputs
✓ test_percentile_normalize_range
✓ test_end_to_end_rtap_fusion_on_tmp_parquet

9/9 tests passed
```

### 3.2 End-to-end on a real LeRobot dataset

Prerequisite: SigLIP2 features already extracted via the existing pipeline
(`01_extract_features.sh` + `merge_siglip_features.py`), so the file
`<DATASET>/features/features_merged_<camera>.pt` exists.

```bash
DATASET=/path/to/your/lerobot_dataset

# ── Step A: RTAP (C1) ────────────────────────────────────────────────
python scripts/calculate_RTAP_advantage.py \
    --source_path "$DATASET" \
    --camera_keys top_head \
    --mode single_anchor \
    --device cpu
# → adds parquet column: RTAP_progress_single_anchor_top_head

# ── Step B: FlowVar (C2) ─────────────────────────────────────────────
# Requires a vanilla SFT π0.5 reference checkpoint. The pure-function core
# (`compute_flowvar_advantage_from_chunks`) is fully tested; the CLI driver
# is environment-specific, see scripts/calculate_FlowVar_advantage.py
# docstring for the minimal driver pattern.
# → adds parquet column: FlowVar_advantage

# ── Step C: Orthogonal Fusion (C3) ───────────────────────────────────
python scripts/fuse_advantages.py \
    --source_path "$DATASET" \
    --rtap_col RTAP_progress_single_anchor_top_head \
    --flowvar_col FlowVar_advantage \
    --out_col fused_advantage \
    --alpha 0.5 \
    --scatter_out /tmp/freevac_scatter.npz
# Console prints orthogonality stats including abs Pearson |ρ|.
# Gate criterion: |ρ| < 0.6  → C3 holds, proceed.
#                 |ρ| ≥ 0.6  → fall back to Plan B (see docs/new paper.md §10).

# ── Step D: discretize into bins (UNCHANGED upstream script) ─────────
python scripts/calculate_lerobot_advantage.py \
    --repo_id "$DATASET" \
    --advantage_source fused_advantage \
    --advantage_type 5bins

# ── Step E: training (UNCHANGED upstream script) ─────────────────────
bash 08_train_torch_ali.sh

# ── Step F: serving (UNCHANGED upstream script) ──────────────────────
bash 09_serve_policy.sh
```

---

## 4. Repository map (after FreeVAC)

```
Kai05-VLA/
├── README.md                              ← this file
├── pyproject.toml                         ← upstream (Python ≥ 3.11, uv)
├── 00_lerobot_*.sh / 01_extract_features.sh / …   ← upstream pipeline drivers
├── 03_calculate_lerobot_advantage.sh      ← downstream of FreeVAC, unchanged
├── 08_train_torch_*.sh / 09_serve_policy*.sh
│
├── docs/
│   ├── new paper.md                       ★ paper narrative + experimental plan
│   ├── 总分析.md                          ★ engineering + code-reuse + file map
│   └── 临时文件/                          ─ early drafts (kept for trace)
│
├── scripts/
│   ├── ┌─ FreeVAC ────────────────────────────────────────────┐
│   │   │ calculate_RTAP_advantage.py        (C1, ~210 lines) │
│   │   │ calculate_FlowVar_advantage.py     (C2, ~200 lines) │
│   │   │ fuse_advantages.py                 (C3, ~165 lines) │
│   │   │ test_freevac.py                    (9 unit tests)   │
│   │   │ README_freevac.md                  (quick reference)│
│   │   └────────────────────────────────────────────────────┘
│   │
│   ├── ┌─ Upstream kai05-VLA (unchanged) ───────────────────────┐
│   │   │ extract_siglip_features.py / merge_siglip_features.py │
│   │   │ calculate_VC_value.py    ← baseline in paper Table 1A │
│   │   │ calculate_lerobot_advantage.py                        │
│   │   │ train_pytorch.py / serve_policy.py / …                │
│   │   └────────────────────────────────────────────────────────┘
│
├── src/openpi/                            ← upstream model code, UNCHANGED
│   └── models_pytorch/pi0_pytorch.py      ← advantage sincos embed + CFG (reused as-is)
│
└── configs/train/                         ← upstream + FreeVAC yaml configs
```

---

## 5. How FreeVAC plugs into the upstream stack

| Upstream module | What FreeVAC does | Modification |
|---|---|---|
| `extract_siglip_features.py` / `merge_siglip_features.py` | Provides `features_merged_<cam>.pt` as input to RTAP | None |
| `calculate_VC_value.py` | Kept as paper baseline (Table 1A row "原 VC-Value") | None |
| `calculate_lerobot_advantage.py` | Discretizes any per-frame scalar column (now `fused_advantage`) into `task_index` + writes `tasks.jsonl` | None — works out-of-the-box with the new column |
| `src/openpi/models_pytorch/pi0_pytorch.py` | Consumes `action_advantage` via sincos pos embedding → prompt suffix; supports CFG | **None** — FreeVAC only changes the *source* of the advantage value, not how it is injected |
| `train_pytorch.py` / `serve_policy.py` | Train and serve the advantage-conditioned policy | None |

This is the heart of FreeVAC's engineering claim: **a new family of advantage signals,
with zero model code changes** — the entire contribution lives upstream of
`calculate_lerobot_advantage.py`.

---

## 6. Paper artefacts produced by this code

| Paper element | Produced by |
|---|---|
| **Figure 1** (method overview) | this README + `docs/new paper.md` |
| **Figure 2** (RTAP × FlowVar orthogonality scatter) | `fuse_advantages.py --scatter_out …` |
| **Table 1A** (Zero external signal) — `+ FreeVAC (RTAP only)` row | Step A + D + E |
| **Table 1A** — `+ FreeVAC (FlowVar only)` row | Step B + D + E |
| **Table 1A** — `+ FreeVAC (Fusion + CFG)` row | Steps A → F with `--alpha 0.5` |
| **Table 2** (complementarity with π\*0.6 online RL) | re-run Step E with `with_value_head=true` after Steps A–D |
| **Gate criterion log** (Week-3 decision) | console output of Step C |

See `docs/new paper.md` §6 for the full experimental design and `docs/总分析.md` for
the engineering breakdown of every change.

---

## 7. License

Original kai05-VLA / openpi codebase: see `LICENSE` and `LICENSE_GEMMA.txt`.
FreeVAC additions inherit the same license.
