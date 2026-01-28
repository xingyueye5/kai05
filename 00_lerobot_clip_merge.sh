#!/bin/bash
set -xe
set -o pipefail

cd /cpfs01/user/zhaolirui/Kai05-VLA

source .venv/bin/activate

python data_merge/merges_by_yaml.py \
    --config data_merge/config/merge_test.yaml \
    --log_file merge_clip_lerobot.txt