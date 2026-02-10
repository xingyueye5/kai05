'''
修改parquet文件

dataset_root/
  ├ data/
  │    chunk-000/
  │        episode_000000.parquet
  │        episode_000001.parquet
  │        ...
  ├ videos/
  │    chunk-000/
  │        observation.images.hand_left
  │            episode_000000.mp4
  │            ...
  │        observation.images.hand_right
  │            episode_000000.mp4
  │            ...
  │        observation.images.top_head
  │            episode_000000.mp4
  │            ...
  ├ meta/
  │    info.json
  │    episodes.jsonl
  │    tasks.jsonl
  │    episodes_stats.jsonl
  │    v2-短袖_record.csv
  │    error_record.csv
  └ README.md

用法:
    python lerobot_evaluation.py <model_type> <model_name> <repo_id>
    
参数:
    model_type: Flatten-Fold / demo_A / demo_B
    model_name: 1T_TL / 2T_TL / 2T_SL / 1T_SL
    repo_id: 数据集路径字符串
'''
import os
import argparse
from value_realtime_evaluator_video_fast_thread_yaml import SimpleValueEvaluator
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import List, Dict
import pyarrow as pa
from tqdm import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# 定义所有模型配置的映射
MODELS_CONFIG_MAP = {
    'Flatten-Fold': {
        '1T_TL': {
            'name': '1T_TL',
            'config_path': '/cpfs01/user/zhaolirui/Kai05-VLA/configs/train/value_model/pi05_value_flatten_fold_standard_all_lerobot_2012_split_1T_TL.yaml',
            'ckpt_dir': '/nas/zhaolirui/Kai05-VLA/checkpoints/pi05_value_flatten_fold_standard_all_lerobot_2012_split_1T_TL/pi05_value_flatten_fold_standard_all_lerobot_2012_split_1T_TL_0207',
            'ckpt_steps': 10000
        },
        '1T_TL_VC': {
            'name': '1T_TL_VC',
            'config_path': '/cpfs01/user/zhaolirui/Kai05-VLA/configs/train/value_model/pi05_value_flatten_fold_standard_all_lerobot_2012_split_1T_TL_VC.yaml',
            'ckpt_dir': '/nas/zhaolirui/Kai05-VLA/checkpoints/pi05_value_flatten_fold_standard_all_lerobot_2012_split_1T_TL_VC/0207_pi05_value_flatten_fold_standard_all_lerobot_2012_split_1T_TL_VC',
            'ckpt_steps': 10000
        },
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='LeRobot数据集评估工具')
    parser.add_argument('model_type', type=str, choices=['Flatten-Fold', 'demo_A', 'demo_B'],
                        help='模型类型: Flatten-Fold / demo_A / demo_B')
    parser.add_argument('model_name', type=str, choices=['1T_TL', '1T_TL_VC', '2T_SL', '1T_SL'],
                        help='模型配置名称: 1T_TL / 2T_TL / 2T_SL / 1T_SL')
    parser.add_argument('repo_id', type=str,
                        help='数据集路径字符串')
    return parser.parse_args()

def edit_parquet_file(src_parquet: Path, output_path: Path, advantages_dict: Dict[str, list]):
    # read parquet files
    table = pq.read_table(src_parquet)
    advantages_table = pa.Table.from_pylist(advantages_dict)
    # mask = pc.is_in(advantages_table["frame_idx"], table["frame_index"])
    # filtered_advantages_table = advantages_table.filter(mask)
    
    cols_to_add = ["relative_advantage", "absolute_value", "absolute_advantage"]
    new_columns = {}
    for col in cols_to_add:
        if col not in table.column_names and col in advantages_table.column_names:
            new_columns[col] = advantages_table[col]
    if new_columns:
        for name, column in new_columns.items():
            table = table.append_column(name, column)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 根据参数获取模型配置
    model_type = args.model_type
    model_name = args.model_name
    repo_id = Path(args.repo_id)
    
    # 获取对应的模型配置
    if model_type not in MODELS_CONFIG_MAP:
        raise ValueError(f"未知的模型类型: {model_type}，可选: {list(MODELS_CONFIG_MAP.keys())}")
    if model_name not in MODELS_CONFIG_MAP[model_type]:
        raise ValueError(f"未知的模型名称: {model_name}，可选: {list(MODELS_CONFIG_MAP[model_type].keys())}")
    
    models_config = [MODELS_CONFIG_MAP[model_type][model_name]]
    
    print(f"模型类型: {model_type}")
    print(f"模型名称: {model_name}")
    print(f"Using dataset at {repo_id}")
    # 模型配置
    relative_interval = 50

    for model_cfg in models_config:
        config_path = model_cfg['config_path']
        ckpt_dir = f"{model_cfg['ckpt_dir']}/{model_cfg['ckpt_steps']}"
        is_1timestep = True
        
        evaluator = evaluator = SimpleValueEvaluator(
            config_path=config_path,
            ckpt_dir=ckpt_dir,
            num_workers=64,  # 并行线程数，根据CPU核心数调整
        )
        dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id=repo_id,)
        for i in tqdm(range(dataset_metadata.total_episodes), desc="Evaluating videos"):
            parquet_file = repo_id/dataset_metadata.data_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i)
            if not parquet_file.exists():
                print(f"Parquet file {parquet_file} not found")
                continue
            top_video = repo_id/dataset_metadata.video_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i,video_key='observation.images.top_head')
            left_video = repo_id/dataset_metadata.video_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i,video_key='observation.images.hand_left')
            right_video = repo_id/dataset_metadata.video_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i,video_key='observation.images.hand_right')
            if not top_video.exists() or not left_video.exists() or not right_video.exists():
                print(f"Video file {top_video} or {left_video} or {right_video} not found")
                continue
            video_paths=(top_video, left_video, right_video)
            
            parquet_table = pq.read_table(parquet_file)
            min_frame_index = parquet_table['frame_index'].to_pylist()[0]
            max_frame_index = parquet_table['frame_index'].to_pylist()[-1]

            output_path=repo_id / f"data_{model_cfg['name']}_{model_cfg['ckpt_steps']}" / parquet_file.relative_to(repo_id / "data")
            if output_path.exists():
                print(f"文件 {output_path} 已存在，跳过...")
                continue

            if evaluator.config.data.default_prompt is not None:
                prompt = evaluator.config.data.default_prompt
            elif evaluator.config.data.base_config.prompt_from_task is True:
                task_index = int(parquet_table['task_index'].to_pylist()[0])
                prompt = dataset_metadata.tasks.get(task_index)
                if prompt is None:
                    raise ValueError(f"task_index={task_index} 不在 dataset_metadata.tasks 中: {dataset_metadata.tasks}")
            else:
                raise ValueError(f"未知的prompt类型: {evaluator.config.data.base_config.prompt_from_task}")
            if is_1timestep:
                results = evaluator.evaluate_video_1timestep_advantage(
                    video_paths=video_paths,
                    prompt=prompt,
                    batch_size=400,
                    frame_interval=1,  # 1为全评估，2为隔一帧评估，3为每3帧评估一次
                    min_frame_index=min_frame_index,
                    max_frame_index=max_frame_index,
                    prefetch=True,  # 启用数据预取
                )
            else:
                results = evaluator.evaluate_video_2timesteps_advantages(
                    video_paths=video_paths,
                    prompt=prompt,
                    batch_size=160,
                    frame_interval=1,  # 1为全评估，2为隔一帧评估，3为每3帧评估一次
                    relative_interval=relative_interval,
                    min_frame_index=min_frame_index,
                    max_frame_index=max_frame_index,
                    prefetch=True,  # 启用数据预取
                )
            edit_parquet_file(
                src_parquet=parquet_file,
                output_path=output_path,
                advantages_dict=results
            )
        

if __name__ == "__main__":
    main()