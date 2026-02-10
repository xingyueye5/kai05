#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python scripts/compute_norm_stats_fast.py \
    --base_dir /cpfs01/shared/kai05_data_train/agilex/flatten_fold/short_sleeve/flatten_fold_3sops_5bins \
    --robot_type agilex