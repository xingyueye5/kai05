#!/bin/bash
set -e

cd /cpfs01/user/zhaolirui/Kai05-VLA
source .venv/bin/activate

python plot_gt.py \
    --config-path /cpfs01/user/zhaolirui/Kai05-VLA/configs/train/debug_copy.yaml