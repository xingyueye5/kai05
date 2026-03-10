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
import shutil

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
    
    # 转换为 Python 原生类型，避免 JSON 序列化问题
    stats = {
        'mean': [float(np.mean(rewards_array))],
        'std': [float(np.std(rewards_array))],
        'min': [float(np.min(rewards_array))],
        'max': [float(np.max(rewards_array))],
        'count': [int(len(rewards_array))]
    }
    return stats, {p: float(v) for p, v in zip(percentiles, percentile_values)}

def compute_threshold_points(
    rewards: Dict[int, List[float] | np.ndarray], 
    advantage_type: str = "binary",
    positive_rate: float = 30
) -> float | List[float]:
    """
    Compute threshold points based on advantage type and positive rate.
    
    Args:
        rewards: Dictionary of rewards by stage
        advantage_type: Type of advantage ('binary' or '<N>bins')
        positive_rate: Positive rate for binary mode (default 30%)
    
    Returns:
        Threshold points:
            - binary: single float (percentile threshold)
            - bins: list of floats (bin edges, N-1 values for N bins)
    """
    # 合并所有 stage 的 rewards
    all_rewards = np.concatenate([np.array(r) for r in rewards.values()])
    
    if advantage_type == "binary":
        # 二分类: 使用 positive_rate 计算阈值
        # top positive_rate% 为 positive
        return float(np.percentile(all_rewards, 100 - positive_rate))
    elif advantage_type.endswith("bins") and advantage_type.split('bins')[0].isdigit():
        # N分类: 计算 N-1 个分位点作为 bin 边界
        bins_num = int(advantage_type.split('bins')[0])
        percentiles = [100.0 * i / bins_num for i in range(1, bins_num)]
        return [float(np.percentile(all_rewards, p)) for p in percentiles]
    else:
        raise ValueError(f"Unknown advantage_type: {advantage_type}")

def collect_all_rewards(
    parquet_path: Path,
    chunk_size: int,
    advantage_source: str,
) -> Tuple[Dict[int, List[float]], List[str]]:
    parquet_files = list(parquet_path.glob("chunk-*/*.parquet"))
    assert len(parquet_files) > 0, f"No parquet files found in {parquet_path}/chunk-*/"

    rewards_by_stage = {0: []}

    for parquet_file in tqdm(parquet_files, desc="Collecting rewards"):
        advantage_source_values = pd.read_parquet(parquet_file)[advantage_source].values
        rewards = calculate_rewards(advantage_source_values, chunk_size)

        rewards_by_stage[0].extend(rewards.tolist())
    return rewards_by_stage, parquet_files

def update_tasks_jsonl(tasks_path: Path, advantage_type: str) -> int:
    """
    Update tasks.jsonl file with reward statistics.
    
    Args:
        tasks_path: Base directory path containing tasks.jsonl file
        advantage_type: Type of advantage ('binary' or '<N>bins')
    
    Returns:
        Number of tasks created
    
    Note:
        advantage 值为 (0, 1] 区间的数字：
        - binary: 0.5, 1
        - 5bins: 0.2, 0.4, 0.6, 0.8, 1
        - 100bins: 0.01, 0.02, ..., 1
        公式: 第 i 个 bin 的值为 (i+1)/N，其中 i 从 0 到 N-1
    """
    import re
    tasks_jsonl_path = tasks_path / "tasks.jsonl"
    assert tasks_jsonl_path.exists(), f"Tasks.jsonl file not found at {tasks_jsonl_path}"
    
    with open(tasks_jsonl_path, 'r') as f:
        tasks = [json.loads(i) for i in f]
    task_desc = tasks[0]['task'].strip()  # TODO: 目前假设只有1个任务，后续需要修改

    task_desc = re.sub(r'[^\w\s]+$', '', task_desc)  # 去掉末尾所有标点符号
    
    if advantage_type == "all_positive":
        # 全正: 只有 1 个 task，advantage 为 1
        with open(tasks_jsonl_path, 'w') as f:
            f.write(json.dumps({
                'task_index': 0,
                'task': f'{task_desc}, advantage: 1',
            }) + '\n')
        return 1
    elif advantage_type == "binary":
        # 二分类: 0.5 和 1
        with open(tasks_jsonl_path, 'w') as f:
            f.write(json.dumps({
                'task_index': 0,
                'task': f'{task_desc}, advantage: 0.5',
            }) + '\n')
            f.write(json.dumps({
                'task_index': 1,
                'task': f'{task_desc}, advantage: 1',
            }) + '\n')
        return 2
    elif advantage_type.endswith("bins") and advantage_type.split('bins')[0].isdigit():
        # N分类: advantage: (i+1)/N，即 1/N, 2/N, ..., N/N
        # 例如 5bins: 0.2, 0.4, 0.6, 0.8, 1
        bins_num = int(advantage_type.split('bins')[0])
        with open(tasks_jsonl_path, 'w') as f:
            for i in range(bins_num):
                advantage_value = (i + 1) / bins_num
                # 格式化数字，去掉不必要的尾部零
                if advantage_value == 1:
                    advantage_str = "1"
                else:
                    advantage_str = f"{advantage_value:.10g}"  # 使用 g 格式去掉尾部零
                f.write(json.dumps({
                    'task_index': i,
                    'task': f'{task_desc}, advantage: {advantage_str}',
                }) + '\n')
        return bins_num
    else:
        raise ValueError(f"Unknown advantage_type: {advantage_type}")

def update_info_json(info_path: Path, total_tasks: int) -> None:
    """
    Update info.json file, update total_tasks.
    
    Args:
        info_path: Base directory path containing info.json file
        total_tasks: Number of total tasks
    """
    info_jsonl_path = info_path / "info.json"
    assert info_jsonl_path.exists(), f"Info.json file not found at {info_jsonl_path}"
    with open(info_jsonl_path, 'r') as f:
        info = json.load(f)
        info['total_tasks'] = total_tasks
    with open(info_jsonl_path, 'w') as f:
        json.dump(info, f, indent=4)

def update_parquet_file(
    parquet_file: Path, 
    threshold_points: float | List[float], 
    chunk_size: int, 
    advantage_source: str,
    advantage_type: str,
    output_file: Path | None = None
) -> np.ndarray:
    """
    Update parquet file, add new column 'reward', update task_index based on threshold_points.
    
    Args:
        parquet_file: Path to the parquet file (input)
        threshold_points: Threshold point(s) for classification
            - binary: single float value
            - bins: list of float values (bin edges)
        chunk_size: Number of frames to look ahead
        advantage_source: Source column for advantage calculation
        advantage_type: Type of advantage ('binary' or '<N>bins')
        output_file: Path to the output parquet file. If None, overwrite the input file.
    
    Returns:
        Array of reward values
    """
    df = pd.read_parquet(parquet_file)
    rewards = calculate_rewards(df[advantage_source].values, chunk_size)
    df['reward'] = rewards
    
    if advantage_type == "binary":
        # 二分类: reward >= threshold -> 1 (positive), else -> 0 (negative)
        if isinstance(threshold_points, list):
            threshold_points = threshold_points[0]
        task_index = (rewards >= threshold_points).astype(np.int32)
    elif advantage_type.endswith("bins") and advantage_type.split('bins')[0].isdigit():
        # N分类: 根据 percentile 分配到不同的 bin
        # threshold_points 是 bin 边界列表 (N-1 个值)
        bins_num = int(advantage_type.split('bins')[0])
        # np.digitize 返回 1 到 bins_num，需要减 1 得到 0 到 bins_num-1
        task_index = np.digitize(rewards, threshold_points, right=True).astype(np.int32)
        # 确保 task_index 在有效范围内
        task_index = np.clip(task_index, 0, bins_num - 1)
    else:
        raise ValueError(f"Unknown advantage_type: {advantage_type}")
    
    df['task_index'] = task_index
    
    # 写入到指定的输出文件或覆盖原文件
    save_path = output_file if output_file is not None else parquet_file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
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

def update_all_advantage(
    repo_id: Path, 
    parquet_path: str, 
    chunk_size: int, 
    advantage_source: str, 
    advantage_type: str, 
    positive_rate: float,
    output_parquet_path: str = "data"
):
    """
    读取 repo_id 下的所有 parquet 文件，计算奖励，更新所有 parquet 文件，
    更新 tasks.jsonl，info.json，episode_stats.jsonl 文件
    
    Args:
        repo_id: Path to the repository
        parquet_path: Path to the input parquet files (e.g., 'data_raw', 'data_v1')
        chunk_size: Number of frames to look ahead for progress calculation
        advantage_source: Source of advantage values
        advantage_type: Type of advantage ('binary' or '<N>bins')
        positive_rate: Positive rate of the task (only for binary mode), default is 30%
        output_parquet_path: Path to the output parquet files, default is 'data'
    """
    # 判断是否需要重命名文件夹
    need_rename = parquet_path != output_parquet_path
    
    print(f"[INFO] Advantage type: {advantage_type}")
    
    if advantage_type == "all_positive":
        # all_positive 模式：只修改 meta，不动 parquet
        total_tasks = update_tasks_jsonl(repo_id / 'meta', advantage_type=advantage_type)
        update_info_json(repo_id / 'meta', total_tasks=total_tasks)
        print(f"[INFO] all_positive mode: updated tasks.jsonl and info.json only (total_tasks={total_tasks})")
        return
    
    print(f"[INFO] Reading from: {repo_id / parquet_path}")
    if need_rename:
        print(f"[INFO] Will rename folder to: {repo_id / output_parquet_path}")
    else:
        print(f"[INFO] Writing to: {repo_id / output_parquet_path}")
    if advantage_type == "binary":
        print(f"[INFO] Positive rate: {positive_rate}%")
    elif advantage_type.endswith("bins"):
        bins_num = int(advantage_type.split('bins')[0])
        print(f"[INFO] Number of bins: {bins_num}")
    
    # 收集所有 rewards
    rewards_all, parquet_files = collect_all_rewards(
        repo_id / parquet_path, 
        chunk_size=chunk_size, 
        advantage_source=advantage_source
    )
    
    # 更新 tasks.jsonl，获取任务数量
    total_tasks = update_tasks_jsonl(repo_id / 'meta', advantage_type=advantage_type)
    print(f"[INFO] Total tasks created: {total_tasks}")
    
    # 更新 info.json
    update_info_json(repo_id / 'meta', total_tasks=total_tasks)
    
    # 计算阈值点
    threshold_points = compute_threshold_points(
        rewards_all, 
        advantage_type=advantage_type,
        positive_rate=positive_rate
    )
    print(f"[INFO] Threshold points: {threshold_points}")
    
    # 计算输入路径的基础目录
    input_base = repo_id / parquet_path
    
    # 更新所有 parquet 文件（原地修改）
    all_rewards = {}
    for parquet_file in tqdm(parquet_files, desc="Updating parquet files"):
        rewards = update_parquet_file(
            parquet_file, 
            threshold_points, 
            chunk_size=chunk_size, 
            advantage_source=advantage_source,
            advantage_type=advantage_type,
            output_file=None  # 原地修改
        )
        episode_index = int(parquet_file.name.split('_')[-1].split('.')[0])
        all_rewards[episode_index] = rewards
    
    # 如果 parquet_path 和 output_parquet_path 不一致，重命名文件夹
    if need_rename:
        output_base = repo_id / output_parquet_path
        if output_base.exists():
            print(f"[WARNING] Output folder already exists, removing: {output_base}")
            shutil.rmtree(output_base)
        shutil.move(str(input_base), str(output_base))
        print(f"[INFO] Renamed folder: {input_base} -> {output_base}")
    
    # 更新 episode_stats.jsonl
    update_episode_stats_jsonl(repo_id / 'meta', all_rewards)

def build_args():
    """
    Build arguments for the program.
    """
    parser = argparse.ArgumentParser(description='Update tasks.jsonl and info.jsonl files.')
    parser.add_argument('--repo_id', type=str, default='/cpfs01/user/baidexiang/test_data_40/', help='Path to the repository')
    parser.add_argument('--parquet_path', type=str, default='data', help='Path to the input parquet files (e.g., data_raw, data_v1)')
    parser.add_argument('--output_parquet_path', type=str, default='data', help='Path to the output parquet files, default is "data"')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of frames to look ahead for progress calculation')
    parser.add_argument('--advantage_source', type=str, default='progress_predicted', help='Source of advantage values')
    parser.add_argument('--advantage_type', type=str, default='binary', help='Type of advantage, default is binary')
    parser.add_argument('--positive_rate', type=float, default=30, help='Positive rate of the task, default is 30%')
    args = parser.parse_args()
    return args

def main():
    args = build_args()
    repo_id = Path(args.repo_id)
    update_all_advantage(
        repo_id=repo_id, 
        parquet_path=args.parquet_path, 
        chunk_size=args.chunk_size, 
        advantage_source=args.advantage_source, 
        advantage_type=args.advantage_type, 
        positive_rate=args.positive_rate,
        output_parquet_path=args.output_parquet_path
    )

if __name__ == "__main__":
    main()
    # uv run python lerobot_value_reward.py --chunk_size 50 --advantage_source absolute_advantage --stage_nums 1 --positive_rate 30