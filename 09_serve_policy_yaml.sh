#!/bin/bash

# 用法: bash serve_policy_yaml_copy.sh <MODEL_PATH> <CONFIG_PATH> [PORT]
# 示例: bash serve_policy_yaml_copy.sh /mnt/nas/Kai05-VLA/checkpoints/xxx/50000 /path/to/config.yaml 8001
# 若模型在 NAS 上会自动拷贝到本地再启动服务

set -e

cd /home/lirui/Kai05-VLA
source .venv/bin/activate

nas_root=/mnt/nas/Kai05-VLA
local_root=/home/lirui/Kai05-VLA

# 从外部传入：模型地址、config 地址，可选端口
# MODEL_PATH=/mnt/nas/Kai05-VLA/checkpoints/vla_torch_flatten_fold_standard_2012_5bins/0226_vla_torch_flatten_fold_standard_2012_5bins/25000
# CONFIG_PATH=/home/lirui/Kai05-VLA/configs/val/vla_torch_flatten_fold_standard_2012_baseline.yaml
# DEFAULT_PROMPT="Flatten and fold the short sleeve"
MODEL_PATH=${1}
CONFIG_PATH=${2}
DEFAULT_PROMPT=${3}
PORT=${4:-8001}

# 参数检查
if [ -z "$MODEL_PATH" ] || [ -z "$CONFIG_PATH" ]; then
    echo "错误: 缺少必填参数"
    echo "用法: $0 <MODEL_PATH> <CONFIG_PATH> [PORT]"
    echo "示例: $0 /mnt/nas/Kai05-VLA/checkpoints/xxx/50000 $local_root/configs/val/xxx.yaml 8001"
    exit 1
fi

echo "========== 配置信息 =========="
echo "MODEL_PATH: $MODEL_PATH"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "PORT: $PORT"
echo "DEFAULT_PROMPT: $DEFAULT_PROMPT"
echo "=============================="

# 若模型在 NAS 上，拷贝到本地（保持相对 checkpoints 的路径）
use_ckpt="$MODEL_PATH"
if [[ "$MODEL_PATH" == "$nas_root"* ]]; then
    rel="${MODEL_PATH#$nas_root/checkpoints/}"
    local_ckpt="$local_root/checkpoints/$rel"
    if [ ! -d "$local_ckpt" ]; then
        echo "本地 ckpt 不存在，从 NAS 复制: $MODEL_PATH -> $local_ckpt"
        mkdir -p "$(dirname "$local_ckpt")"
        rsync -ah --info=progress2 "$MODEL_PATH/" "$local_ckpt/"
    else
        echo "本地 ckpt 已存在: $local_ckpt"
    fi
    use_ckpt="$local_ckpt"
fi

CMD="uv run scripts/serve_policy_yaml.py --port=$PORT"
if [ -n "$DEFAULT_PROMPT" ]; then
    CMD="$CMD --default-prompt=\"$DEFAULT_PROMPT\""
fi
CMD="$CMD policy:checkpoint --policy.config_path=$CONFIG_PATH --policy.dir=$use_ckpt"

eval $CMD
