#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

# 超参数配置
CONFIG=${1}
LOG_FILE=${2:-}
LOGER_INFO_PATH=${3:-logs}

echo "========== 配置信息 =========="
echo "CONFIG: $CONFIG"
echo "LOG_FILE: $LOG_FILE"
echo "LOGER_INFO_PATH: $LOGER_INFO_PATH"
echo "=============================="

# 构建命令参数
CMD="python data_merge_split/merges_by_yaml.py --config ${CONFIG}"

if [ -n "$LOG_FILE" ]; then
    CMD="$CMD --log_file ${LOG_FILE}"
fi

if [ -n "$LOGER_INFO_PATH" ]; then
    CMD="$CMD --loger_info_path ${LOGER_INFO_PATH}"
fi

# 执行命令
eval $CMD