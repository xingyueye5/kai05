#!/bin/bash
set -xe
set -o pipefail

# 原始数据集路径
ORIGINAL_REPO_ID=/cpfs01/shared/kai05_data/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556
# 复制后的数据集路径（加后缀）
SUFFIX="_reward"
REPO_ID="${ORIGINAL_REPO_ID}${SUFFIX}"
PARQUET_PATH=progress_predicted

# REPO_ID=$1
# PARQUET_PATH=$2
# CHUNK_SIZE=${3:-50}
# ADVANTAGE_SOURCE=${4:-"progress"}
# STAGE_NUMS=${5:-1}
# POSITIVE_RATE=${6:-30}

# ========== 复制数据集 ==========
echo "开始复制数据集..."
echo "原始路径: $ORIGINAL_REPO_ID"
echo "目标路径: $REPO_ID"


# # 如果目标目录已存在，先删除
if [ -d "$REPO_ID" ]; then
    echo "目标目录已存在，删除旧目录..."
    rm -rf "$REPO_ID"
fi

# 创建目标目录
mkdir -p "$REPO_ID"

# 复制除 videos 和 features 之外的所有文件和文件夹
for item in "$ORIGINAL_REPO_ID"/*; do
    basename_item=$(basename "$item")
    if [ "$basename_item" != "videos" ] && [ "$basename_item" != "features" ]; then
        echo "复制: $basename_item"
        cp -r "$item" "$REPO_ID/"
    fi
done

# 软链接 videos 和 features 文件夹
if [ -d "$ORIGINAL_REPO_ID/videos" ]; then
    echo "软链接: videos"
    ln -s "$ORIGINAL_REPO_ID/videos" "$REPO_ID/videos"
fi

if [ -d "$ORIGINAL_REPO_ID/features" ]; then
    echo "软链接: features"
    ln -s "$ORIGINAL_REPO_ID/features" "$REPO_ID/features"
fi

echo "数据集复制完成!"
echo "=========================================="

# ========== 执行主脚本 ==========
cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python 03_lerobot_value_reward.py \
    --repo_id "$REPO_ID" \
    --parquet_path "$PARQUET_PATH"
    # --chunk_size "$CHUNK_SIZE" \
    # --advantage_source "$ADVANTAGE_SOURCE" \
    # --stage_nums "$STAGE_NUMS" \
    # --positive_rate "$POSITIVE_RATE" \