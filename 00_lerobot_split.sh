#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python data_merge_split/lerobot_split.py \
    --source_path /cpfs01/shared/kai05_data_train/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556/split_0_0.9 \
    --dst_path /cpfs01/shared/kai05_data_train/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556/split_0_0.9_split_16 \
    --split_num 16 \
    --seed 42 


# --split_ratio "0.9,0.1" \