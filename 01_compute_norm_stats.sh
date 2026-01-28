#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python scripts/compute_norm_stats_fast.py \
    --base_dir /cpfs01/user/zhaolirui/Kai05-VLA/test_lerobot_dataset/1031_20_200_v3-4_3000_lerobot_clip_200 \
    --robot_type agilex