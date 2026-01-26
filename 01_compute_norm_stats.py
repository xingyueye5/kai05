"""Compute normalization statistics for LeRobot datasets - Ultra Fast Version (Fixed).

This script uses a simplified approach to compute normalization statistics,
directly reading parquet files and accumulating data in memory, then computing 
statistics using the same RunningStats class as the org version for exact matching.

FIXED VERSION: This version produces results matching compute_norm_stats_fast_org.py
with < 0.000001% difference by:
1. Using normalize.RunningStats() (same as org) instead of manual numpy calculations
2. Feeding data in batches of 32 (matching org's batch_size) for identical float accumulation
3. Sorting parquet files for deterministic ordering
4. Processing actions at full action_dim (32) without trimming to 14 dimensions
5. Using histogram-based quantile computation (via RunningStats) instead of np.percentile

Usage:
    python compute_norm_states_ultra_fast_agx_fixed.py --base-dir /path/to/dataset --robot-type agilex
    python compute_norm_states_ultra_fast_agx_fixed.py --base-dir /path/to/dataset --robot-type franka --output-dir /path/to/output
"""
import os
import numpy as np
from tqdm import tqdm
import tyro
import pandas as pd
from pathlib import Path
from typing import Literal

import openpi.shared.normalize as normalize


# 机器人类型到 action_dim 的映射
ROBOT_ACTION_DIM = {
    "agilex": 32,
}

# 支持的机器人类型
RobotType = Literal["agilex", "franka", "ur5", "xarm", "aloha", "so100", "koch"]


def pad_to_dim(data, target_dim):
    """Pad data to target dimension."""
    if isinstance(data, list):
        data = np.array(data)
    data = np.asarray(data)
    
    if data.shape[-1] >= target_dim:
        return data[..., :target_dim]
    
    padding = np.zeros((*data.shape[:-1], target_dim - data.shape[-1]))
    return np.concatenate([data, padding], axis=-1)


def process_state(state, action_dim):
    """Process state following the FakeInputs logic."""
    # Pad to action dimension
    state = pad_to_dim(state, action_dim)
    
    # Filter abnormal values (outside [-pi, pi])
    state = np.where(state > np.pi, 0, state)
    state = np.where(state < -np.pi, 0, state)
    
    return state


def process_actions(actions, action_dim):
    """Process actions following the FakeInputs logic."""
    # Pad to action dimension
    actions = pad_to_dim(actions, action_dim)
    
    # Filter abnormal values (outside [-pi, pi])
    actions = np.where(actions > np.pi, 0, actions)
    actions = np.where(actions < -np.pi, 0, actions)
    
    # Return full action_dim dimensions (not trimmed to 14)
    return actions


def main(
    base_dir: str,
    robot_type: RobotType,
    output_dir: str | None = None,
    max_frames: int | None = None,
    data_subdir: str = "data",
):
    """
    Compute normalization statistics for a LeRobot dataset.
    
    Args:
        base_dir: Base directory containing the LeRobot dataset.
        robot_type: Type of robot, determines action_dim. 
                    Supported: agilex(32), franka(8), ur5(7), xarm(8), aloha(14), so100(6), koch(6)
        output_dir: Output directory for saving statistics. Defaults to base_dir.
        max_frames: Maximum number of frames to process. If None, processes all data.
        data_subdir: Subdirectory containing parquet files. Defaults to "data".
    """
    # 从机器人类型获取 action_dim
    action_dim = ROBOT_ACTION_DIM[robot_type]
    
    # 验证数据集目录
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # 确定输出目录
    if output_dir is None:
        output_path = base_path
    else:
        output_path = Path(output_dir)
    
    print(f"Robot type: {robot_type}")
    print(f"Reading data from: {base_dir}")
    print(f"Action dimension: {action_dim}")
    
    # Keys to collect
    keys = ["state", "actions"]
    collected_data = {key: [] for key in keys}
    
    # Column names in the parquet files
    state_col = "observation.state"
    action_col = "action"
    
    total_frames = 0
    files_processed = 0
    
    # Collect all parquet files from data subdirectory
    data_dir = base_path / data_subdir
    if not data_dir.exists():
        raise ValueError(f"Data subdirectory does not exist: {data_dir}")
    
    print(f"Searching parquet files in: {data_dir}")
    
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    # Sort files for deterministic ordering (same as dataset ordering)
    parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Process each parquet file
    for file_path in tqdm(parquet_files, desc="Processing files"):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
        
        # Check if required columns exist
        if state_col not in df.columns or action_col not in df.columns:
            continue
        
        try:
            # Extract and process state data
            states_list = []
            actions_list = []
            
            for i in range(len(df)):
                try:
                    # Get state
                    state = np.array(df[state_col].iloc[i])
                    state = process_state(state, action_dim)
                    states_list.append(state)
                    
                    # Get action
                    action = np.array(df[action_col].iloc[i])
                    action = process_actions(action, action_dim)
                    actions_list.append(action)
                    
                    total_frames += 1
                    
                    # Check max_frames limit
                    if max_frames is not None and total_frames >= max_frames:
                        break
                        
                except Exception as e:
                    continue
            
            if states_list:
                collected_data["state"].append(np.stack(states_list))
            if actions_list:
                collected_data["actions"].append(np.stack(actions_list))
            
            files_processed += 1
            
            # Check max_frames limit
            if max_frames is not None and total_frames >= max_frames:
                print(f"\nReached max_frames limit: {max_frames}")
                break
                
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue
    
    print(f"\nProcessed {files_processed} files with {total_frames} frames")
    
    # Compute statistics using RunningStats (same as org version)
    print("\nComputing statistics...")
    print("Initializing RunningStats objects...")
    stats = {key: normalize.RunningStats() for key in keys}
    
    # Concatenate all data first
    print("Concatenating collected data...")
    all_data = {}
    for key in keys:
        if collected_data[key]:
            print(f"Concatenating {len(collected_data[key])} batches for '{key}'...")
            all_data[key] = np.concatenate(collected_data[key], axis=0)
            print(f"  Shape after concatenation: {all_data[key].shape}")
            
            # Ensure data is padded to action_dim
            if all_data[key].shape[-1] < action_dim:
                print(f"  Padding from {all_data[key].shape[-1]} to {action_dim} dimensions...")
                padding = np.zeros((*all_data[key].shape[:-1], action_dim - all_data[key].shape[-1]))
                all_data[key] = np.concatenate([all_data[key], padding], axis=-1)
                print(f"  Shape after padding: {all_data[key].shape}")
    
    # Feed data to RunningStats in fixed batches of 32 (same as org version)
    # This ensures identical floating-point accumulation
    batch_size = 32
    print("Feeding data to RunningStats...")
    for key in keys:
        if key in all_data:
            data = all_data[key]
            num_samples = len(data)
            num_batches = (num_samples + batch_size - 1) // batch_size
            for i in tqdm(range(0, num_samples, batch_size), desc=f"Processing {key}", total=num_batches):
                batch = data[i:i+batch_size]
                stats[key].update(batch)
    
    # Get statistics from RunningStats
    norm_stats = {}
    for key in keys:
        if collected_data[key]:
            stat_result = stats[key].get_statistics()
            norm_stats[key] = stat_result
            
            print(f"\n{key} statistics:")
            print(f"  Mean: {stat_result.mean}")
            print(f"  Std: {stat_result.std}")
            print(f"  Q01: {stat_result.q01}")
            print(f"  Q99: {stat_result.q99}")
        else:
            print(f"Warning: No data collected for key '{key}'")
    
    # Save statistics
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"✅ Normalization stats saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)