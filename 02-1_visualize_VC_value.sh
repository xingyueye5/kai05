#!/bin/bash
set -xe
set -o pipefail

# 使用方法: bash 03_visualize_progress.sh <start_episode> <end_episode> [dataset_path]
# 示例: bash 03_visualize_progress.sh 0 10
# 示例: bash 03_visualize_progress.sh 0 10 /path/to/dataset

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <start_episode> <end_episode> [dataset_path]"
    echo "示例: $0 0 10"
    exit 1
fi

START_EPISODE=$1
END_EPISODE=$2
DATASET_PATH=${3:-/cpfs01/shared/kai05_data/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556}

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

echo "开始可视化 episode $START_EPISODE 到 $END_EPISODE"
echo "数据集路径: $DATASET_PATH"

# 循环处理每个 episode
for ((episode=START_EPISODE; episode<=END_EPISODE; episode++)); do
    echo "========================================"
    echo "正在处理 episode $episode / $END_EPISODE"
    echo "========================================"
    
    python scripts/visualize_VC_value.py \
        "$DATASET_PATH" \
        --episode "$episode"
done

echo "所有 episode 处理完成!"
