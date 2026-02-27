#!/bin/bash
set -xe
set -o pipefail

# 使用方法: bash 02-1_visualize_VC_value.sh <start_episode> <end_episode> <dataset_path> <value_source>
# 示例: bash 02-1_visualize_VC_value.sh 0 10 /path/to/dataset top_head
# 示例: bash 02-1_visualize_VC_value.sh 0 10 /path/to/dataset top_head_front_camera

# 检查参数
if [ $# -lt 4 ]; then
    echo "使用方法: $0 <start_episode> <end_episode> <dataset_path> <value_source>"
    echo "示例: $0 0 10 /path/to/dataset top_head"
    echo "示例: $0 0 10 /path/to/dataset top_head_front_camera"
    exit 1
fi

START_EPISODE=$1
END_EPISODE=$2
DATASET_PATH=$3
VALUE_SOURCE=$4

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

echo "开始可视化 episode $START_EPISODE 到 $END_EPISODE"
echo "数据集路径: $DATASET_PATH"
echo "value_source (key): $VALUE_SOURCE"

# 循环处理每个 episode
for ((episode=START_EPISODE; episode<=END_EPISODE; episode++)); do
    echo "========================================"
    echo "正在处理 episode $episode / $END_EPISODE"
    echo "========================================"
    python scripts/visualize_VC_value.py \
        "$DATASET_PATH" \
        --episode "$episode" \
        --value_source "$VALUE_SOURCE"
done

echo "所有 episode 处理完成!"
