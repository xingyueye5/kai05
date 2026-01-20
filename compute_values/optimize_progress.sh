export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python optimize_progress_v4.py \
    --workers_per_gpu 1 \
    --source_path /cpfs01/shared/kai05_data/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556 \
    --top_n -1 \
    --exclude_self_episode \
    --exclude_self_frame_value \
    --time_range 0.6 \
    --query_chunk_size 128 \
    --camera_keys top_head