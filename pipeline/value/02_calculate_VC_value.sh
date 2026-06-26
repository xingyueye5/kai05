#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 超参数配置
SOURCE_PATH=${1}
# 支持逗号分隔的多个相机，例如: top_head,front_camera,side_camera
CAMERA_KEYS_INPUT=${2:-top_head}
CAMERA_KEYS=${CAMERA_KEYS_INPUT//,/ }  # 将逗号替换为空格
TOP_N=${3:--1}
TIME_RANGE=${4:-0.6}
QUERY_CHUNK_SIZE=${5:-128}
WORKERS_PER_GPU=${6:-1}

echo "========== 配置信息 =========="
echo "SOURCE_PATH: $SOURCE_PATH"
echo "CAMERA_KEYS: $CAMERA_KEYS"
echo "TOP_N: $TOP_N"
echo "TIME_RANGE: $TIME_RANGE"
echo "QUERY_CHUNK_SIZE: $QUERY_CHUNK_SIZE"
echo "WORKERS_PER_GPU: $WORKERS_PER_GPU"
echo "=============================="

python scripts/calculate_VC_value.py \
    --workers_per_gpu ${WORKERS_PER_GPU} \
    --source_path ${SOURCE_PATH} \
    --top_n ${TOP_N} \
    --exclude_self_episode \
    --exclude_self_frame_value \
    --time_range ${TIME_RANGE} \
    --query_chunk_size ${QUERY_CHUNK_SIZE} \
    --camera_keys ${CAMERA_KEYS}