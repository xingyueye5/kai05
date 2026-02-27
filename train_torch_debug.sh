#!/bin/bash

# config_name=${1}
# ngpus_per_node=${2}

config_name=/cpfs01/user/zhaolirui/Kai05-VLA/configs/train/vla/flatten_fold_standard_2012/vla_torch_flatten_fold_standard_2012_debug.yaml
exp_name=debug
ngpus_per_node=1
# export TORCH_ELASTIC_LOG_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1  # ! DEBUG
# export PYTHONPATH=/gpfs/yangjiazhi/workspace/RoboRL/openpi:$PYTHONPATH

# * ON AlayaNew
cd /cpfs01/user/zhaolirui/Kai05-VLA
source .venv/bin/activate


export WANDB_MODE=offline
# which uv
# echo !!!!!!!!!!!!!!!!!!!!!!!!!

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $config_name --exp-name=$config_name ${PY_ARGS}
# uv run torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $config_name
torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $exp_name --overwrite



# ${PY_ARGS}


# --overwrite.

# uv run torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $config_name