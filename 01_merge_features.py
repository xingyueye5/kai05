"""
将分散的 .pt 特征文件按 camera 合并成一个大矩阵，同时加载 progress_gt

原始结构:
    features/observation.images.top_head/episode_000000.pt
    features/observation.images.top_head/episode_000001.pt
    data/chunk-000/episode_000000.parquet  (包含 progress_gt)
    ...

合并后:
    features/features_merged_top_head.pt = {
        "features": tensor[M, D],      # M = 所有视频帧数总和, D = 特征维度
        "video_ids": tensor[M],        # 每帧对应的视频索引 (int64)
        "frame_indices": tensor[M],    # 每帧在原视频中的帧索引 (int64)
        "progress_gt": tensor[M],      # 每帧的 progress_gt (float32, 0-1)
        "video_names": list[str],      # 视频名称列表，按索引查找
    }

使用方法:
    python merge_features.py /path/to/dataset
    python merge_features.py /path/to/dataset --camera_keys top_head

加载示例:
    data = torch.load("features_merged_top_head.pt")
    F_all = data["features"]             # [M, D]
    video_ids = data["video_ids"]        # [M]
    frame_indices = data["frame_indices"] # [M]
    progress_gt = data["progress_gt"]    # [M]
    video_names = data["video_names"]    # list[str]
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from tqdm import tqdm


def get_parquet_path(dataset_path: Path, episode_id: str, chunk_size: int = 1000) -> Path:
    """根据 episode_id 计算 parquet 文件路径"""
    episode_idx = int(episode_id.split('_')[1])
    chunk_idx = episode_idx // chunk_size
    return dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"{episode_id}.parquet"


def load_parquet_progress(parquet_path: Path) -> Dict[int, float]:
    """加载 parquet 文件中的 progress_gt"""
    df = pd.read_parquet(parquet_path)
    return {int(row['frame_index']): float(row['progress_gt']) for _, row in df.iterrows()}


def load_single_feature(pt_file: Path) -> tuple[str, str, torch.Tensor]:
    """
    加载单个特征文件
    
    Returns:
        (episode_id, camera_key, features_tensor)
    """
    data = torch.load(pt_file, map_location="cpu", weights_only=True)
    
    # 从文件名提取 episode_id (例如: episode_000000.pt -> episode_000000)
    episode_id = pt_file.stem
    
    # 从路径提取 camera_key (例如: observation.images.top_head -> top_head)
    parent_name = pt_file.parent.name
    if "." in parent_name:
        camera_key = parent_name.split(".")[-1]
    else:
        camera_key = parent_name
    
    features = data["features"]
    
    return episode_id, camera_key, features


def merge_features(
    dataset_path: str,
    camera_keys: list[str] | None = None,
    num_workers: int = 8,
    chunk_size: int = 1000,
):
    """
    合并特征文件（每个 camera 生成一个合并文件）
    
    Args:
        dataset_path: 数据集路径
        camera_keys: 要合并的相机视角列表，None表示所有
        num_workers: 加载线程数
        chunk_size: parquet chunk 大小
    """
    dataset_path = Path(dataset_path)
    features_dir = dataset_path / "features"
    
    if not features_dir.exists():
        raise FileNotFoundError(f"特征目录不存在: {features_dir}")
    
    # 获取所有 .pt 文件（只搜索子目录，排除根目录下的合并文件）
    # 使用 */**/*.pt 确保至少有一层子目录
    pt_files = list(features_dir.glob("*/**/*.pt"))
    
    # 按 camera 分组
    camera_files: dict[str, list[Path]] = defaultdict(list)
    for f in pt_files:
        parent_name = f.parent.name
        if "." in parent_name:
            camera_key = parent_name.split(".")[-1]
        else:
            camera_key = parent_name
        camera_files[camera_key].append(f)
    
    # 过滤指定的 camera
    if camera_keys is not None:
        camera_files = {k: v for k, v in camera_files.items() if k in camera_keys}
    
    if len(camera_files) == 0:
        raise ValueError(f"未找到任何特征文件: {features_dir}")
    
    print(f"检测到 {len(camera_files)} 个相机视角:")
    for cam, files in camera_files.items():
        print(f"  - {cam}: {len(files)} 个文件")
    
    # 逐个 camera 合并
    output_paths = []
    for camera_key, files in camera_files.items():
        print(f"\n{'='*50}")
        print(f"正在合并 {camera_key} ({len(files)} 个文件)...")
        
        episode_features = {}
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_single_feature, f): f for f in files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"加载 {camera_key}"):
                try:
                    episode_id, _, features = future.result()
                    episode_features[episode_id] = features
                except Exception as e:
                    breakpoint()
                    pt_file = futures[future]
                    print(f"警告: 加载 {pt_file} 失败: {e}")
        
        # 按 episode_id 排序
        sorted_episodes = sorted(episode_features.keys())
        
        # 加载所有 parquet 的 progress_gt
        print(f"  加载 progress_gt...")
        all_progress_dicts = {}
        for episode_id in tqdm(sorted_episodes, desc="加载 parquet", leave=False):
            parquet_path = get_parquet_path(dataset_path, episode_id, chunk_size)
            if parquet_path.exists():
                try:
                    all_progress_dicts[episode_id] = load_parquet_progress(parquet_path)
                except Exception as e:
                    breakpoint()
                    print(f"    警告: 加载 {parquet_path} 失败: {e}")
        
        # 构建大矩阵和 video_id 数组
        all_features = []
        all_video_ids = []
        all_frame_indices = []
        all_progress_gt = []
        video_names = []
        
        for video_idx, episode_id in enumerate(sorted_episodes):
            features = episode_features[episode_id]  # [num_frames, D]
            num_frames = features.shape[0]
            progress_dict = all_progress_dicts.get(episode_id, {})
            
            # 构建该 episode 的 progress_gt 数组
            if len(progress_dict) != num_frames:
                raise ValueError(f"{episode_id}: progress_dict 有 {len(progress_dict)} 帧, 特征有 {num_frames} 帧")
            episode_progress = torch.tensor([progress_dict[i] for i in range(num_frames)], dtype=torch.float32)
            
            all_features.append(features)
            all_video_ids.append(torch.full((num_frames,), video_idx, dtype=torch.int64))
            all_frame_indices.append(torch.arange(num_frames, dtype=torch.int64))
            all_progress_gt.append(episode_progress)
            video_names.append(episode_id)
        
        # 拼接成大矩阵
        F_all = torch.cat(all_features, dim=0)  # [M, D]
        video_ids = torch.cat(all_video_ids, dim=0)  # [M]
        frame_indices = torch.cat(all_frame_indices, dim=0)  # [M]
        progress_gt = torch.cat(all_progress_gt, dim=0)  # [M]
        
        # 统计信息
        M, D = F_all.shape
        print(f"  - Episode 数量: {len(video_names)}")
        print(f"  - 总帧数 M: {M}")
        print(f"  - 特征维度 D: {D}")
        print(f"  - progress_gt 范围: [{progress_gt.min():.3f}, {progress_gt.max():.3f}]")
        
        # 保存
        merged_data = {
            "features": F_all,             # [M, D]
            "video_ids": video_ids,        # [M]
            "frame_indices": frame_indices, # [M]
            "progress_gt": progress_gt,    # [M]
            "video_names": video_names,    # list[str], len = num_videos
        }
        
        output_name = f"features_merged_{camera_key}.pt"
        output_path = features_dir / output_name
        
        print(f"  - 保存到: {output_path}")
        torch.save(merged_data, output_path)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  - 文件大小: {file_size_mb:.2f} MB")
        
        output_paths.append(output_path)
    
    print(f"\n{'='*50}")
    print("合并完成!")
    print("生成的文件:")
    for p in output_paths:
        print(f"  - {p}")
    
    return output_paths


def verify_merged_file(merged_path: str):
    """验证合并后的文件"""
    print(f"\n验证合并文件: {merged_path}")
    
    data = torch.load(merged_path, map_location="cpu", weights_only=False)
    
    F_all = data["features"]
    video_ids = data["video_ids"]
    frame_indices = data["frame_indices"]
    progress_gt = data["progress_gt"]
    video_names = data["video_names"]
    
    M, D = F_all.shape
    num_videos = len(video_names)
    
    print(f"F_all shape: [{M}, {D}] (M=总帧数, D=特征维度)")
    print(f"video_ids shape: [{video_ids.shape[0]}]")
    print(f"frame_indices shape: [{frame_indices.shape[0]}]")
    print(f"progress_gt shape: [{progress_gt.shape[0]}]")
    print(f"视频数量: {num_videos}")
    print(f"progress_gt 范围: [{progress_gt.min():.3f}, {progress_gt.max():.3f}]")
    print(f"前5个视频名称: {video_names[:5]}")
    
    # 统计每个视频的帧数
    print("\n每个视频的帧数:")
    for vid in range(min(5, num_videos)):
        mask = video_ids == vid
        frame_count = mask.sum().item()
        print(f"  - {video_names[vid]}: {frame_count} 帧")
    if num_videos > 5:
        print(f"  - ... (共 {num_videos} 个视频)")
    
    # 验证 video_ids 范围
    assert video_ids.min() >= 0, "video_ids 最小值应 >= 0"
    assert video_ids.max() < num_videos, f"video_ids 最大值应 < {num_videos}"
    print(f"\nvideo_ids 范围: [{video_ids.min().item()}, {video_ids.max().item()}] ✓")


def main():
    parser = argparse.ArgumentParser(description="合并特征文件为大矩阵 F_all[M,D] + progress_gt[M]")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集路径",
    )
    parser.add_argument(
        "--camera_keys",
        type=str,
        nargs="+",
        default=None,
        help="要合并的相机视角列表 (默认: 所有)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="加载线程数 (默认: 8)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="parquet chunk 大小 (默认: 1000)",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="验证指定的合并文件路径",
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_merged_file(args.verify)
    else:
        merge_features(
            dataset_path=args.dataset_path,
            camera_keys=args.camera_keys,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
