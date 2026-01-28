#!/bin/bash
set -xe
set -o pipefail

# ============================================================
# 配置参数
# ============================================================

# 原始数据集路径
ORIGINAL_REPO_ID=/cpfs01/shared/kai05_data/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556

# ---------- 模式配置 ----------
# ADVANTAGE_TYPE 支持以下格式:
#   - "binary"       : 二分类 (negative/positive)
#   - "<N>bins"      : N分类，如 "10bins", "5bins"
ADVANTAGE_TYPE="100bins"

# ---------- 比例配置 (仅 binary 模式生效) ----------
# POSITIVE_RATE: positive 的比例 (百分比)，如 30 表示 top 30% 为 positive
POSITIVE_RATE=30

# ---------- 其他参数 ----------
PARQUET_PATH="progress_predicted"
CHUNK_SIZE=50
ADVANTAGE_SOURCE="progress_predicted"

# ============================================================
# 参数验证
# ============================================================

validate_params() {
    # 验证 ADVANTAGE_TYPE 格式
    if [[ "$ADVANTAGE_TYPE" == "binary" ]]; then
        # binary 模式：验证 POSITIVE_RATE
        if ! [[ "$POSITIVE_RATE" =~ ^[0-9]+$ ]] || [ "$POSITIVE_RATE" -lt 1 ] || [ "$POSITIVE_RATE" -gt 99 ]; then
            echo "错误: POSITIVE_RATE 必须是 1-99 之间的整数"
            echo "当前值: $POSITIVE_RATE"
            exit 1
        fi
    elif [[ "$ADVANTAGE_TYPE" =~ ^[0-9]+bins$ ]]; then
        # bins 模式：验证 bins 数量
        BINS_NUM="${ADVANTAGE_TYPE%bins}"
        if [ "$BINS_NUM" -lt 2 ]; then
            echo "错误: bins 数量必须 >= 2"
            echo "当前值: $BINS_NUM"
            exit 1
        fi
    else
        echo "错误: ADVANTAGE_TYPE 必须是 'binary' 或 '<N>bins' (如 '10bins')"
        echo "当前值: $ADVANTAGE_TYPE"
        exit 1
    fi
}

# ============================================================
# 生成目标路径后缀
# ============================================================

generate_suffix() {
    if [[ "$ADVANTAGE_TYPE" == "binary" ]]; then
        # binary 模式：后缀包含比例信息
        echo "binary_p${POSITIVE_RATE}"
    else
        # bins 模式：直接使用 bins 数量
        echo "$ADVANTAGE_TYPE"
    fi
}

# ============================================================
# 复制数据集函数
# ============================================================

copy_dataset() {
    local src="$1"
    local dst="$2"

    echo "=========================================="
    echo "开始复制数据集..."
    echo "原始路径: $src"
    echo "目标路径: $dst"
    echo "=========================================="

    # 如果目标目录已存在，先删除
    if [ -d "$dst" ]; then
        echo "目标目录已存在，删除旧目录..."
        rm -rf "$dst"
    fi

    # 创建目标目录
    mkdir -p "$dst"

    # 只复制 meta 和 PARQUET_PATH 文件夹
    if [ -d "$src/meta" ]; then
        echo "复制: meta"
        cp -r "$src/meta" "$dst/"
    fi

    # 复制 norm_stats.json
    if [ -f "$src/norm_stats.json" ]; then
        echo "复制: norm_stats.json"
        cp "$src/norm_stats.json" "$dst/"
    fi

    if [ -d "$src/$PARQUET_PATH" ]; then
        echo "复制: $PARQUET_PATH"
        cp -r "$src/$PARQUET_PATH" "$dst/"
    fi

    # 软链接 videos 文件夹
    if [ -d "$src/videos" ]; then
        echo "软链接: videos"
        ln -s "$src/videos" "$dst/videos"
    fi

    echo "数据集复制完成!"
}

# ============================================================
# 主流程
# ============================================================

main() {
    # 参数验证
    validate_params

    # 生成后缀和目标路径
    SUFFIX=$(generate_suffix)
    REPO_ID="${ORIGINAL_REPO_ID}_${SUFFIX}"

    # 打印配置信息
    echo "=========================================="
    echo "配置信息:"
    echo "  ADVANTAGE_TYPE:   $ADVANTAGE_TYPE"
    if [[ "$ADVANTAGE_TYPE" == "binary" ]]; then
        echo "  POSITIVE_RATE:    ${POSITIVE_RATE}%"
    fi
    echo "  CHUNK_SIZE:       $CHUNK_SIZE"
    echo "  ADVANTAGE_SOURCE: $ADVANTAGE_SOURCE"
    echo "  PARQUET_PATH:     $PARQUET_PATH"
    echo "  输出目录后缀:     $SUFFIX"
    echo "=========================================="

    # 复制数据集
    copy_dataset "$ORIGINAL_REPO_ID" "$REPO_ID"

    # 执行主脚本
    echo "=========================================="
    echo "执行 Python 脚本..."
    echo "=========================================="

    cd /cpfs01/user/zhaolirui/Kai05-VLA
    source .venv/bin/activate

    python scripts/calculate_lerobot_advantage.py \
        --repo_id "$REPO_ID" \
        --parquet_path "$PARQUET_PATH" \
        --chunk_size "$CHUNK_SIZE" \
        --advantage_source "$ADVANTAGE_SOURCE" \
        --advantage_type "$ADVANTAGE_TYPE" \
        --positive_rate "$POSITIVE_RATE"

    echo "=========================================="
    echo "处理完成!"
    echo "输出目录: $REPO_ID"
    echo "=========================================="
}

# 执行主函数
main