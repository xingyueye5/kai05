"""
由 01_modify_task_index_based_on_progress.py 分析判断得到如下规则：
reward计算规则：
1. 如果advantage_source为"absolute_advantage"或"relative_advantage"，直接取*.parquet文件中的对应列作为奖励；
2. 如果advantage_source为"progress"，则按照以下规则计算奖励：
    - 如果i+chunk_size未超出范围，则直接计算progress[i+chunk_size] - progress[i]
    - 如果i+chunk_size超出范围，则用最后一个值计算，并进行数值调整，即rewards[i] = (progress[-1] - progress[i]) / (len(progress) - i) * chunk_size
    - progress为*.parquet文件中的progress列，chunk_size为预看帧数，默认为50，用于计算当前帧与未来chunk_size帧的progress差值，作为reward值

操作执行 Advantage 划分规则
1.包含Advantage: negative和Advantage: positive，分别对应0和1
2.negative与positive的划分依据为：
    - 划分阈值（threshold）计算：
        - 计算数据集中所有操作帧的reward
        - 计算上步中所有reward的N%分位数作为划分阈值，N为划分阈值百分比，默认为70
        - 如果任务具有多阶段，则需要计算每个阶段的划分阈值；每个阶段按照该阶段的阈值进行划分
    - advantage: negative: reward < threshold
    - advantage: positive: reward >= threshold
------------------------------------------------------------------------------------------------
!!!注意：目前reward的计算，均假设只有1个任务!!!
!!!注意：目前仅划分advantage: negative和advantage: positive，未考虑其他划分方式!!!
"""
from ctypes import Array
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import pyarrow.parquet as pq
import pyarrow as pa

def calculate_rewards(advantage_source_values: np.ndarray, chunk_size: int = 50) -> np.ndarray:
    """
    Calculate rewards based on progress differences.
    
    Args:
        data: DataFrame containing 'progress' column
        chunk_size: Number of frames to look ahead for progress calculation
        
    Returns:
        Array of rewards for each frame
    """
    n_frames = len(advantage_source_values)
    rewards = np.zeros(n_frames, dtype=np.float32)
    # 计算规则：progress[i+chunk_size] - progress[i]；如果i+chunk_size超出范围，则用最后一个值计算，并进行数值调整
    rewards[:-chunk_size] = advantage_source_values[chunk_size:] - advantage_source_values[:-chunk_size]
    # TODO: 有没有更好的处理尾部数据的方式？
    rewards[-chunk_size:] = (advantage_source_values[-1] - advantage_source_values[-chunk_size:]) / np.linspace(chunk_size, 1, chunk_size)* chunk_size
    return rewards

# 更新episode_stats.jsonl，补充stats字段额外
def compute_reward_statistics(rewards: List[float], percentiles: List[int] = list(range(0, 101, 10))) -> dict:
    """
    Compute reward distribution statistics.
    
    Args:
        rewards: List of all rewards
        
    Returns:
        Dictionary containing percentile information
    """
    if len(rewards) == 0:  # 如果奖励列表为空，则返回0
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        },{p: 0.0 for p in percentiles}
    
    rewards_array = np.array(rewards)
    
    # Compute percentiles points
    percentile_values = np.percentile(rewards_array, percentiles)
    
    stats = {
        'mean': [np.mean(rewards_array)],
        'std': [np.std(rewards_array)],
        'min': [np.min(rewards_array)],
        'max': [np.max(rewards_array)],
        'count': [len(rewards_array)]
    }
    return stats, dict(zip(percentiles, percentile_values)),

def compute_threshold_points(rewards: Dict[int, List[float] | np.ndarray], positive_rate: float = 30) -> float | List[float]:
    """
    Compute threshold points based on positive rate.
    Args:
        rewards: List of all rewards
        positive_rate: Positive rate of the task, default is 30%
    Returns:
        Threshold points
    """
    if len(rewards) == 1:
        return np.percentile(rewards[0], 100 - positive_rate)
    else:
        return [np.percentile(rewards[i], 100 - positive_rate) for i in range(len(rewards))]

def collect_all_rewards(
    parquet_path: Path,
    chunk_size: int = 50,
    advantage_source: str = "progress",
) -> Tuple[Dict[int, List[float]], List[str]]:
    parquet_files = list(parquet_path.glob("chunk-*/*.parquet"))
    assert len(parquet_files) > 0, f"No parquet files found in {parquet_path}/chunk-*/"

    rewards_by_stage = {0: []}

    for parquet_file in tqdm(parquet_files, desc="Collecting rewards"):
        advantage_source_values = pd.read_parquet(parquet_file)[advantage_source].values
        rewards = calculate_rewards(advantage_source_values, chunk_size)

        rewards_by_stage[0].extend(rewards.tolist())
    return rewards_by_stage, parquet_files

def update_tasks_jsonl(tasks_path: Path) -> None:
    """
    Update tasks.jsonl file with reward statistics.
    
    Args:
        tasks_path: Base directory path containing tasks.jsonl file
    """
    tasks_jsonl_path = tasks_path / "tasks.jsonl"
    assert tasks_jsonl_path.exists(), f"Tasks.jsonl file not found at {tasks_jsonl_path}"
    
    ##################################### 
    with open(tasks_jsonl_path, 'r') as f:
        tasks = [json.loads(i) for i in f]
    task_desc = tasks[0]['task'].strip()# TODO: 目前假设只有1个任务，后续需要修改
    #####################################

    if not task_desc[-1].isalpha(): # 如果最后一个字符不是字母，则去掉最后一个字符
        task_desc = task_desc[:-1] # TODO: 行，假定后面有句号，去掉句号是吧
    
    with open(tasks_jsonl_path, 'w') as f:
        f.write(json.dumps({
            'task_index': 0,
            'task': task_desc+', advantage: negative',
        }) + '\n')
        f.write(json.dumps({
            'task_index': 1,
            'task': task_desc+', advantage: positive',
        }) + '\n')

def update_info_json(info_path: Path) -> None:
    """
    Update info.jsonl file, update total_tasks from 1 to 2.
    
    Args:
        info_path: Base directory path containing info.jsonl file
    """
    info_jsonl_path = info_path / "info.json"
    assert info_jsonl_path.exists(), f"Info.json file not found at {info_jsonl_path}"
    with open(info_jsonl_path, 'r') as f:
        info = json.load(f)
        info['total_tasks'] = 2
    with open(info_jsonl_path, 'w') as f:
        json.dump(info, f, indent=4)

def update_parquet_file(parquet_file: Path, threshold_points: float|List[float], chunk_size: int = 50, advantage_source: str = "progress") -> np.ndarray:
    """
    Update parquet file, add new column 'reward', update task_index based on threshold_points.
    
    Args:
        parquet_file: Path to the parquet file
        threshold_points: List of threshold points
    return:
        Array of reward values
    """
    df = pd.read_parquet(parquet_file)
    rewards = calculate_rewards(df[advantage_source].values, chunk_size)
    df['reward'] = rewards
    # 更新task_index based on threshold_points
    # 如果threshold_points是个单元素的列表，则使用该元素，视为单阶段任务
    if isinstance(threshold_points, list) and len(threshold_points) == 1:
        threshold_points = threshold_points[0]

    if isinstance(threshold_points, float):
        task_index = (rewards >= threshold_points).astype(np.int32)
    elif isinstance(threshold_points, List[float]):
        task_index = np.zeros(len(rewards), dtype=np.int32)
        stage_nums = len(threshold_points)
        stage_cut_points = [i/stage_nums for i in range(stage_nums)] # 初始化阶段划分点
        stage_progress_gt_values = df['stage_progress_gt'].values
        for frame_idx, spg in enumerate(stage_progress_gt_values):
            stage_idx = np.where(spg >= stage_cut_points)[0][-1] # 找到当前帧属于哪个阶段
            task_index[frame_idx] = rewards[frame_idx] >= threshold_points[stage_idx]
    else:
        raise ValueError(f"Unknown threshold_points type: {type(threshold_points)}")
    df['task_index'] = task_index

    df.to_parquet(parquet_file, index=False)
    return rewards

def update_episode_stats_jsonl(episode_stats_path: Path, rewards: Dict[int, List[float] | np.ndarray]) -> None:
    """
    Update episode_stats.jsonl file, add new column 'reward'.
    don't update the task_index as new task is assigned to each frame not episode.
    Args:
        episode_stats_path: Base directory path containing episode_stats.jsonl file
        rewards: Dictionary of rewards by episode_index
    """
    episode_stats_jsonl_path = episode_stats_path / "episodes_stats.jsonl"
    assert episode_stats_jsonl_path.exists(), f"Episode_stats.jsonl file not found at {episode_stats_jsonl_path}"
    with open(episode_stats_jsonl_path, 'r') as f:
        episode_stats = [json.loads(line) for line in f]
    for episode_stat in episode_stats:
        episode_index = episode_stat['episode_index']
        if rewards.get(episode_index) is not None:
            reward, percentiles = compute_reward_statistics(rewards[episode_index])
            episode_stat['stats']['reward'] = reward
    with open(episode_stats_jsonl_path, 'w') as f:
        for episode_stat in episode_stats:
            f.write(json.dumps(episode_stat) + '\n')

def update_all_advantage(repo_id:Path, parquet_path, chunk_size: int = 50, advantage_source: str = "progress", positive_rate: float = 30):
    """
    读取repo_id下的所有parquet文件，计算奖励，更新所有parquet文件，更新tasks.jsonl，info.jsonl，episode_stats.jsonl文件
    Args:
        repo_id: Path to the repository
        parquet_path: Path to the parquet files, default is 'data'
        chunk_size: Number of frames to look ahead for progress calculation
        advantage_source: Source of advantage values
        positive_rate: Positive rate of the task, default is 30%
    """
    rewards_all, parquet_files = collect_all_rewards(repo_id/parquet_path, chunk_size=chunk_size, advantage_source=advantage_source)
    update_tasks_jsonl(repo_id/'meta')
    update_info_json(repo_id/'meta')
    # 计算阈值点,用于划分advantage: negative和advantage: positive
    threshold_points = compute_threshold_points(rewards_all, positive_rate)
    all_rewards = {0: []}
    for parquet_file in tqdm(parquet_files, desc="Updating parquet files"):
        rewards = update_parquet_file(parquet_file, threshold_points, chunk_size=chunk_size, advantage_source=advantage_source)
        episode_index = int(parquet_file.name.split('_')[-1].split('.')[0])
        all_rewards[episode_index] = rewards
    update_episode_stats_jsonl(repo_id/'meta', all_rewards)

def build_args():
    """
    Build arguments for the program.
    """
    parser = argparse.ArgumentParser(description='Update tasks.jsonl and info.jsonl files.')
    parser.add_argument('--repo_id', type=str, default='/cpfs01/user/baidexiang/test_data_40/', help='Path to the repository')
    parser.add_argument('--parquet_path', type=str, default='data', help='Path to the parquet files')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of frames to look ahead for progress calculation')
    parser.add_argument('--advantage_source', type=str, default='progress_predicted', help='Source of advantage values')
    parser.add_argument('--stage_nums', type=int, default=1, help='Number of stages to divide data into based on stage_progress_gt')
    parser.add_argument('--positive_rate', type=float, default=30, help='Positive rate of the task, default is 30%')
    args = parser.parse_args()
    return args

def main():
    args = build_args()
    repo_id = Path(args.repo_id)
    update_all_advantage(repo_id=repo_id, parquet_path=args.parquet_path, chunk_size=args.chunk_size, advantage_source=args.advantage_source, positive_rate=args.positive_rate)

if __name__ == "__main__":
    main()
    # uv run python lerobot_value_reward.py --chunk_size 50 --advantage_source absolute_advantage --stage_nums 1 --positive_rate 30