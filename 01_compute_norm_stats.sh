#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python scripts/compute_norm_stats_fast.py \
    --base_dir /cpfs01/shared/kai05_data_train/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556/split_0_0.9_split_16_merge \
    --robot_type agilex