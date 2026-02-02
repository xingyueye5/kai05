#!/bin/bash

# # 用法
# bash 04_inference_lerobot_value.sh <TASK_NAME> <MODEL_VERSION> <DATA_PATH>

# # 示例
# bash 04_inference_lerobot_value.sh Flatten-Fold 1T_TL /path/to/data

set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

# 超参数配置（必填）
TASK_NAME=${1}
MODEL_VERSION=${2}
DATA_PATH=${3}

# 参数检查
if [ -z "$TASK_NAME" ] || [ -z "$MODEL_VERSION" ] || [ -z "$DATA_PATH" ]; then
    echo "错误: 缺少必填参数"
    echo "用法: $0 <TASK_NAME> <MODEL_VERSION> <DATA_PATH>"
    echo "示例: $0 Flatten-Fold 1T_TL /path/to/data"
    exit 1
fi

echo "========== 配置信息 =========="
echo "TASK_NAME: $TASK_NAME"
echo "MODEL_VERSION: $MODEL_VERSION"
echo "DATA_PATH: $DATA_PATH"
echo "=============================="

# 执行推理
python scripts/lerobot_inference.py \
    ${TASK_NAME} \
    ${MODEL_VERSION} \
    ${DATA_PATH}
