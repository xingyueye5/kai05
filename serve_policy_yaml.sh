cd /cpfs01/user/zhaolirui/Kai05-VLA
source .venv/bin/activate

uv run scripts/serve_policy_yaml.py \
    policy:checkpoint \
    --policy.config_path=configs/val/vla_torch_flatten_fold_standard_all_lerobot_2012_baseline.yaml \
    --policy.dir=/nas/zhaolirui/Kai05-VLA/checkpoints/vla_torch_flatten_fold_standard_all_lerobot_2012_baseline/0207_vla_torch_flatten_fold_standard_all_lerobot_2012_baselnie/30000