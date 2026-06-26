#!/bin/bash
# just merge the lerobot data
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python just_merge_lerobot.py \
    --source_path \
    /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_standard_all_lerobot_2012 \
    /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_weitiao_1991 \
    /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_xihu_1996 \
    --dst_path \
    /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_3sops_baseline \
    --copy_video