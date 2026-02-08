#!/bin/bash
set -xe
set -o pipefail

# ============================================================
# 配置参数
# ============================================================

# 原始数据集路径
ORIGINAL_REPO_ID=/cpfs01/shared/kai05_data_train/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556/split_0_0.9_split_16_merge
# 自动生成训练数据集路径（kai05_data -> kai05_data_train，如果已经是 kai05_data_train 则不变）
if [[ "$ORIGINAL_REPO_ID" == *"kai05_data_train"* ]]; then
    # 已经是 kai05_data_train，不做替换
    TRAIN_REPO_ID="$ORIGINAL_REPO_ID"
else
    # kai05_data -> kai05_data_train
    TRAIN_REPO_ID="${ORIGINAL_REPO_ID/kai05_data/kai05_data_train}"
fi

# ---------- 模式配置 ----------
# ADVANTAGE_TYPES 支持多个值，用逗号分隔:
#   - "binary"       : 二分类 (negative/positive)
#   - "<N>bins"      : N分类，如 "10bins", "5bins"
# 示例: "binary,5bins,10bins,100bins"
ADVANTAGE_TYPES="binary,5bins,10bins,100bins"

# ---------- 比例配置 (仅 binary 模式生效) ----------
# POSITIVE_RATE: positive 的比例 (百分比)，如 30 表示 top 30% 为 positive
POSITIVE_RATE=30

# ---------- 其他参数 ----------
PARQUET_PATH="data_1T_TL_100000"
CHUNK_SIZE=50
ADVANTAGE_SOURCE="absolute_value"

# ============================================================
# 参数验证
# ============================================================

validate_single_type() {
    local adv_type="$1"
    # 验证单个 ADVANTAGE_TYPE 格式
    if [[ "$adv_type" == "binary" ]]; then
        # binary 模式：验证 POSITIVE_RATE
        if ! [[ "$POSITIVE_RATE" =~ ^[0-9]+$ ]] || [ "$POSITIVE_RATE" -lt 1 ] || [ "$POSITIVE_RATE" -gt 99 ]; then
            echo "错误: POSITIVE_RATE 必须是 1-99 之间的整数"
            echo "当前值: $POSITIVE_RATE"
            return 1
        fi
    elif [[ "$adv_type" =~ ^[0-9]+bins$ ]]; then
        # bins 模式：验证 bins 数量
        local bins_num="${adv_type%bins}"
        if [ "$bins_num" -lt 2 ]; then
            echo "错误: bins 数量必须 >= 2"
            echo "当前值: $bins_num"
            return 1
        fi
    else
        echo "错误: ADVANTAGE_TYPE 必须是 'binary' 或 '<N>bins' (如 '10bins')"
        echo "当前值: $adv_type"
        return 1
    fi
    return 0
}

validate_all_params() {
    # 将逗号分隔的字符串转为数组
    IFS=',' read -ra TYPES_ARRAY <<< "$ADVANTAGE_TYPES"
    
    echo "验证所有 ADVANTAGE_TYPE..."
    for adv_type in "${TYPES_ARRAY[@]}"; do
        # 去除空格
        adv_type=$(echo "$adv_type" | xargs)
        if ! validate_single_type "$adv_type"; then
            exit 1
        fi
        echo "  ✓ $adv_type"
    done
    echo "验证通过!"
}

# ============================================================
# 生成目标路径后缀
# ============================================================

generate_suffix() {
    local adv_type="$1"
    if [[ "$adv_type" == "binary" ]]; then
        # binary 模式：后缀包含比例信息
        echo "binary_p${POSITIVE_RATE}"
    else
        # bins 模式：直接使用 bins 数量
        echo "$adv_type"
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
# 处理单个 ADVANTAGE_TYPE
# ============================================================

process_single_type() {
    local adv_type="$1"
    local index="$2"
    local total="$3"

    # 生成后缀和目标路径
    local suffix=$(generate_suffix "$adv_type")
    local repo_id="${TRAIN_REPO_ID}_${PARQUET_PATH}_${suffix}"

    echo ""
    echo "############################################################"
    echo "# 处理 [$index/$total]: $adv_type"
    echo "############################################################"

    # 打印配置信息
    echo "=========================================="
    echo "配置信息:"
    echo "  ADVANTAGE_TYPE:   $adv_type"
    if [[ "$adv_type" == "binary" ]]; then
        echo "  POSITIVE_RATE:    ${POSITIVE_RATE}%"
    fi
    echo "  CHUNK_SIZE:       $CHUNK_SIZE"
    echo "  ADVANTAGE_SOURCE: $ADVANTAGE_SOURCE"
    echo "  PARQUET_PATH:     $PARQUET_PATH"
    echo "  输出目录:         $repo_id"
    echo "=========================================="

    # 复制数据集
    copy_dataset "$ORIGINAL_REPO_ID" "$repo_id"

    # 执行主脚本
    echo "=========================================="
    echo "执行 Python 脚本..."
    echo "=========================================="

    python scripts/calculate_lerobot_advantage.py \
        --repo_id "$repo_id" \
        --parquet_path "$PARQUET_PATH" \
        --chunk_size "$CHUNK_SIZE" \
        --advantage_source "$ADVANTAGE_SOURCE" \
        --advantage_type "$adv_type" \
        --positive_rate "$POSITIVE_RATE"

    echo "=========================================="
    echo "[$index/$total] 处理完成: $adv_type"
    echo "输出目录: $repo_id"
    echo "=========================================="
}

# ============================================================
# 主流程
# ============================================================

main() {
    # 参数验证
    validate_all_params

    # 将逗号分隔的字符串转为数组
    IFS=',' read -ra TYPES_ARRAY <<< "$ADVANTAGE_TYPES"
    local total=${#TYPES_ARRAY[@]}

    # 打印总体信息
    echo ""
    echo "############################################################"
    echo "# 批量处理 ADVANTAGE_TYPE"
    echo "# 共 $total 个类型: $ADVANTAGE_TYPES"
    echo "############################################################"

    cd /cpfs01/user/zhaolirui/Kai05-VLA
    source .venv/bin/activate

    # 循环处理每个类型
    local index=1
    for adv_type in "${TYPES_ARRAY[@]}"; do
        # 去除空格
        adv_type=$(echo "$adv_type" | xargs)
        process_single_type "$adv_type" "$index" "$total"
        ((index++))
    done

    echo ""
    echo "############################################################"
    echo "# 全部处理完成! 共 $total 个类型"
    echo "############################################################"
    echo "处理的类型:"
    for adv_type in "${TYPES_ARRAY[@]}"; do
        adv_type=$(echo "$adv_type" | xargs)
        local suffix=$(generate_suffix "$adv_type")
        echo "  - $adv_type -> ${TRAIN_REPO_ID}_${suffix}"
    done
}

# 执行主函数
main