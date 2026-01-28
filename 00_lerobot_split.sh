#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python data_merge/lerobot_split.py \
    --source_path /cpfs01/shared/kai05_data/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556_reward \
    --dst_path /cpfs01/shared/kai05_data_train/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556 \
    --split_ratio "0.9,0.1" \
    --seed 42 