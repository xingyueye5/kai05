#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

# 超参数配置
DATASET_PATH=${1}
CAMERA_KEYS=${2:-top_head}
CKPT=${3:-/cpfs01/shared/zhaolirui/ckpts/siglip2-giant-opt-patch16-384}
BATCH_SIZE=${4:-1024}
FRAME_INTERVAL=${5:-1}
NUM_WORKERS=${6:-12}

echo "========== 配置信息 =========="
echo "DATASET_PATH: $DATASET_PATH"
echo "CKPT: $CKPT"
echo "BATCH_SIZE: $BATCH_SIZE (单卡)"
echo "FRAME_INTERVAL: $FRAME_INTERVAL"
echo "NUM_WORKERS: $NUM_WORKERS (单卡)"
echo "CAMERA_KEYS: $CAMERA_KEYS"
echo "=============================="

# python 01_extract_features_multi_thread.py \
#     ${DATASET_PATH} \
#     --ckpt ${CKPT} \
#     --batch_size ${BATCH_SIZE} \
#     --frame_interval ${FRAME_INTERVAL} \
#     --num_workers $ {NUM_WORKERS} \
#     --camera_keys ${CAMERA_KEYS}


python 01_merge_features.py \
    ${DATASET_PATH} \
    --num_workers ${NUM_WORKERS} \
    --camera_keys ${CAMERA_KEYS}