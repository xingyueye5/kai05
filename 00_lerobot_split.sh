#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python data_merge_split/lerobot_split.py \
    --source_path /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_standard_all_lerobot_2012_split/split_1_0.9 \
    --dst_path /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_standard_all_lerobot_2012_split/split_1_0.9_split_16 \
    --split_num 16 \
    --seed 42 


# --split_num 16 \
# --split_ratio "0.1,0.9" \