# `pipeline/` — Auxiliary shell entrypoints

This folder holds the **non-frontline** shell drivers that previously cluttered the
project root. The five highest-frequency entrypoints stay in the project root for
muscle-memory:

```
<root>/03_calculate_lerobot_advantage.sh   # advantage discretization (FreeVAC consumer)
<root>/08_train_torch_ali.sh               # main training driver (DDP)
<root>/08_train_torch_debug.sh             # single-GPU debug training
<root>/09_serve_policy.sh                  # websocket policy server (config-name style)
<root>/09_serve_policy_yaml.sh             # websocket policy server (YAML config style)
```

Everything else has moved here, grouped by purpose:

| Folder | Contents | Purpose |
|---|---|---|
| `data_prep/` | `00_lerobot_*.sh`, `01_compute_norm_stats.sh`, `01_compute_pi06_value_gt_TODO.sh`, `01_extract_features.sh` | LeRobot dataset merge/split + SigLIP feature extraction + normalization stats |
| `value/`     | `02_calculate_VC_value.sh`, `02-1_visualize_VC_value.sh`, `04_inference_lerobot_value.sh` | Legacy VC-Value pipeline (kept as paper baseline) and value-model inference |
| `utils/`     | `08-1_plot_actions.sh`, `clean_progress_predicted.sh` | Action plotting, dataset cleanup |

## How to run

All scripts internally `cd` into the project root, so they can be invoked from anywhere:

```bash
# from project root
bash pipeline/data_prep/01_extract_features.sh
bash pipeline/value/02_calculate_VC_value.sh
bash pipeline/utils/08-1_plot_actions.sh
```

## FreeVAC pipeline (new advantage signals)

The FreeVAC scripts (RTAP / FlowVar / Fusion) live in `scripts/` and have no
shell wrappers yet — invoke them directly with `python scripts/calculate_RTAP_advantage.py …`
(see `README.md` §3.2 and `scripts/README_freevac.md` for full commands).
