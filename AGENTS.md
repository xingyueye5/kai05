# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is **openpi (Kai05-VLA)** — a robotics Vision-Language-Action model framework. It provides π₀, π₀-FAST, and π₀.₅ models for robot control. The codebase uses JAX and PyTorch backends with a patched `transformers` library.

### Key services

| Service | Command | Notes |
|---------|---------|-------|
| openpi (core library) | `uv run python -c "from openpi.training import config"` | Core ML models and training |
| Policy server (WebSocket) | `uv run scripts/serve_policy.py --env=ALOHA_SIM` | Serves VLA policies on port 8000; requires GPU + checkpoint |
| openpi-client | Workspace package at `packages/openpi-client/` | Lightweight client for querying the policy server |

### Development commands

- **Install deps**: `GIT_LFS_SKIP_SMUDGE=1 uv sync`
- **Lint**: `uv run ruff check .` and `uv run ruff format --check .`
- **Tests**: `uv run pytest -v -m "not manual"` (see caveats below)
- **Training**: `uv run scripts/train_pytorch.py <config_name> --exp_name <name>`
- **Serve policy**: `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config> --policy.dir=<dir>`

### Non-obvious caveats

1. **Transformers patching**: After `uv sync`, the transformers library must be patched by copying files from `src/openpi/models_pytorch/transformers_replace/` into `.venv/lib/python3.11/site-packages/transformers/`. This is critical for PyTorch model support (AdaRMS, precision control, KV cache). The patch survives reinstalls due to uv's hardlink mode — run `uv cache clean transformers` to fully undo.

2. **Heavy model tests OOM**: `src/openpi/models/model_test.py` and `scripts/train_test.py` require significant memory and will be OOM-killed on machines with < 16GB RAM. Skip them with `--ignore` flags when running in constrained environments.

3. **Pre-existing test issue**: `scripts/test_yaml_config.py::test_class_registry` has a missing fixture (`config_loader`). This is a pre-existing issue in the codebase.

4. **Pre-existing lint issues**: The codebase has ~2358 ruff lint errors and ~35 formatting issues. These are pre-existing and not caused by agent changes.

5. **No GPU in cloud VM**: JAX falls back to CPU (`CpuDevice`) when no NVIDIA GPU is available. Tests and imports still work on CPU. Model inference/training requires an NVIDIA GPU with ≥8GB VRAM.

6. **Python version**: The project pins Python 3.11 (see `.python-version`). uv will manage this automatically.

7. **Git LFS**: Always use `GIT_LFS_SKIP_SMUDGE=1` when running `uv sync` to avoid issues with LeRobot dependency pulling.

8. **Submodules**: `third_party/aloha` and `third_party/libero` are git submodules needed for ALOHA/LIBERO examples. Initialize with `git submodule update --init --recursive`.
