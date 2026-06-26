#!/bin/bash
set -xe
set -o pipefail

DATASET_PATH=${1}

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python scripts/compute_norm_stats_fast.py \
    --base_dir ${DATASET_PATH} \
    --robot_type agilex