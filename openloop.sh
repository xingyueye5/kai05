#!/bin/bash
set -xe
set -o pipefail

CONFIG_PATH=/cpfs01/user/zhaolirui/Kai05-VLA/configs/train/vla/flatten_fold_xihu_1996/vla_torch_flatten_fold_xihu_1996_baseline.yaml
CHECKPOINT_DIR=/nas/zhaolirui/Kai05-VLA/checkpoints/vla_torch_flatten_fold_xihu_1996_baseline/0227_vla_torch_flatten_fold_xihu_1996_baseline/50000
# CONFIG_PATH=${1}
# CHECKPOINT_DIR=${2}

# visible devices
export CUDA_VISIBLE_DEVICES=1

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python openloop.py \
    --config-path ${CONFIG_PATH} \
    --checkpoint-dir ${CHECKPOINT_DIR}