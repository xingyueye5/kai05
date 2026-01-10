python 01_extract_features_multi_thread.py \
    /cpfs01/shared/kai05_data/short_sleeve_shirt/arms_A/v2-3_1231_5017 \
    --ckpt /cpfs01/shared/zhaolirui/ckpts/siglip2-giant-opt-patch16-384 \
    --batch_size 4096 \
    --frame_interval 1 \
    --num_workers 32 \
    --camera_keys top_head