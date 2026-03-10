#!/bin/bash
# Action 曲线可视化入口（仅 GT / GT vs 推理对比）
# 用法:
#   ./plot_actions.sh gt CONFIG_PATH                     # 只画数据集 GT
#   ./plot_actions.sh compare CONFIG_PATH CHECKPOINT_DIR # 画 GT 与推理对比
#
# 示例:
#   ./plot_actions.sh gt configs/train/xxx.yaml
#   ./plot_actions.sh compare configs/train/xxx.yaml /path/to/checkpoint/50000

set -e
set -o pipefail

MODE="${1:?用法: $0 gt CONFIG_PATH  或  $0 compare CONFIG_PATH CHECKPOINT_DIR}"

cd /cpfs01/user/zhaolirui/Kai05-VLA
source .venv/bin/activate

case "${MODE}" in
  gt)
    CONFIG_PATH="${2:?请提供 CONFIG_PATH}"
    python scripts/plot_actions.py --config-path "${CONFIG_PATH}"
    ;;
  compare)
    CONFIG_PATH="${2:?请提供 CONFIG_PATH}"
    CHECKPOINT_DIR="${3:?请提供 CHECKPOINT_DIR}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    python scripts/plot_actions.py \
      --config-path "${CONFIG_PATH}" \
      --checkpoint-dir "${CHECKPOINT_DIR}"
    ;;
  *)
    echo "未知模式: ${MODE}" >&2
    echo "用法: $0 gt CONFIG_PATH 或 $0 compare CONFIG_PATH CHECKPOINT_DIR" >&2
    exit 1
    ;;
esac
