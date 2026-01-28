set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

export TZ='Asia/Shanghai'

RUNPATH=$1
RUNTIME=$2
RUNNAME=$(basename "${RUNPATH}" .yaml)
# RUNNAME=$1
# RUNTIME=$2
WORLD_SIZE=${WORLD_SIZE:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANK=${RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12345}

if [ -z "${RUNNAME+x}" ]; then  
    echo "[ERROR] RUNNAME is not set, please inject RUNNAME for experiment output directory" 
    exit 1
else  
    echo "RUNNAME is set to: $RUNNAME"  
fi

if [ -z "${RUNTIME+x}" ]; then  
    echo "[ERROR] RUNTIME is not set, please inject RUNTIME for experiment output directory" 
    exit 1
else  
    echo "RUNTIME is set to: $RUNTIME"  
fi

LOG_OUTPUT_DIR="experiment"/${RUNNAME}
if [ ! -d "$LOG_OUTPUT_DIR" ]; then
  mkdir -p "$LOG_OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LAUNCHER=pytorch
export NCCL_P2P_LEVEL=NVL

# smch: comment this if you want to run wandb
export WANDB_MODE=offline

# smch: other envs
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export JAX_PLATFORMS=cuda

# uv run python scripts/compute_norm_stats_fast.py --config-name ${RUNNAME}

torchrun \
    --nnodes=${WORLD_SIZE} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_pytorch.py ${RUNPATH} --exp_name=${RUNTIME} --overwrite > ./${LOG_OUTPUT_DIR}/${RUNTIME}.log 2>&1
