"""
基于SigLIP特征的视频帧Progress预测工具（矩阵版 - 极简）

功能：
1. 直接加载合并后的大矩阵 F_all[M, D] 和 progress_gt[M]
2. 一次性计算所有帧之间的相似度（完全向量化）
3. 多相机融合预测 progress
4. 结果保存到 parquet

输入数据格式（来自 01_merge_features.py 合并后的特征文件）:
    {
        "features": tensor[M, D],       # 所有视频帧拼成的大矩阵
        "video_ids": tensor[M],         # 每帧对应的视频索引 (int64)
        "frame_indices": tensor[M],     # 每帧在原视频中的帧索引 (int64)
        "progress_gt": tensor[M],       # 每帧的 progress_gt (0-1)
        "video_names": list[str],       # 视频名称列表
    }

使用示例：
python script.py /path/to/dataset --top_n 5
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse
from dataclasses import dataclass
import pandas as pd


@dataclass
class ProgressPrediction:
    """Progress预测结果"""
    query_frame_idx: int
    query_progress_gt: float
    predicted_progress: float
    top_n_matches: List[Tuple[str, int, float, float]]  # [(episode_id, frame_idx, similarity, progress_gt), ...]


def get_available_merged_cameras(features_dir: Path) -> List[str]:
    """获取可用的合并特征文件对应的相机列表"""
    cameras = []
    for f in features_dir.glob("features_merged_*.pt"):
        camera_key = f.stem.replace("features_merged_", "")
        cameras.append(camera_key)
    return cameras


def get_parquet_path(dataset_path: Path, episode_id: str, chunk_size: int = 1000) -> Path:
    """根据 episode_id 计算 parquet 文件路径"""
    episode_idx = int(episode_id.split('_')[1])
    chunk_idx = episode_idx // chunk_size
    return dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"{episode_id}.parquet"


def load_merged_features(features_dir: Path, camera_key: str) -> Dict:
    """
    加载合并后的特征文件
    
    Returns:
        {
            "features": tensor[M, D],
            "video_ids": tensor[M],
            "frame_indices": tensor[M],
            "progress_gt": tensor[M],
            "video_names": list[str],
        }
    """
    merged_file = features_dir / f"features_merged_{camera_key}.pt"
    return torch.load(merged_file, map_location='cpu', weights_only=False)


def predict_progress_per_episode(
    dataset_path: Path,
    camera_keys: List[str],
    top_n: int = 5,
    batch_size: int = 256,
    device: str = 'cuda',
    time_range: float = 1.0,
    exclude_self: bool = True,
    chunk_size: int = 1000,
    verbose: bool = True
):
    """
    逐 episode 计算并保存预测结果，避免内存爆炸
    
    计算完一个 episode 的所有帧后，立即保存到 parquet 并释放内存
    """
    features_dir = dataset_path / "features"
    output_base_dir = dataset_path / "progress_predicted"
    
    # ============ 加载所有相机特征到 GPU 并获取元数据 ============
    if verbose:
        print(f"\n加载 {len(camera_keys)} 个相机的特征...")
    
    all_camera_features = []
    for camera_key in camera_keys:
        if verbose:
            print(f"  加载 {camera_key} 特征...")
        camera_data = load_merged_features(features_dir, camera_key)
        if len(all_camera_features) == 0:
            video_names = camera_data["video_names"]
            video_ids = camera_data["video_ids"]
            frame_indices = camera_data["frame_indices"]
            progress_gt = camera_data["progress_gt"]
            M, D = camera_data["features"].shape
            
            if verbose:
                print(f"  总帧数 M: {M}, 特征维度 D: {D}")
                print(f"  视频数量: {len(video_names)}")
            
            top_n_actual = min(top_n, M)
        all_camera_features.append(camera_data["features"])
        del camera_data
    
    # 将特征移到 GPU
    if device == 'cuda' and torch.cuda.is_available():
        all_camera_features = [f.to(device) for f in all_camera_features]
        gpu_video_ids = video_ids.to(device)
        gpu_progress_gt = progress_gt.to(device)
    else:
        gpu_video_ids = video_ids
        gpu_progress_gt = progress_gt
    
    half_range = time_range / 2 if time_range < 1.0 else None
    
    # ============ 构建每个 episode 的帧索引范围 ============
    video_ids_np = video_ids.numpy()
    frame_indices_np = frame_indices.numpy()
    progress_gt_np = progress_gt.numpy()
    
    # 找到每个 episode 在大矩阵中的起止索引
    episode_ranges = {}  # {episode_id: [global_idx1, global_idx2, ...]}
    for global_idx in range(M):
        vid_idx = video_ids_np[global_idx]
        ep_id = video_names[vid_idx]
        if ep_id not in episode_ranges:
            episode_ranges[ep_id] = []
        episode_ranges[ep_id].append(global_idx)
    
    if verbose:
        print(f"\n逐 episode 计算并保存 (共 {len(episode_ranges)} 个 episode)...")
    
    # ============ 统计变量 ============
    total_mae, total_rmse = [], []
    total_processed = 0
    
    # ============ 逐 episode 处理 ============
    for episode_id in tqdm(episode_ranges.keys(), desc="处理 episode", disable=not verbose):
        ep_global_indices = episode_ranges[episode_id]
        ep_global_indices = sorted(ep_global_indices)  # 确保顺序
        
        # 为当前 episode 的帧分批计算 top-n
        ep_results = []
        
        for i in range(0, len(ep_global_indices), batch_size):
            batch_global_indices = ep_global_indices[i:i + batch_size]
            batch_len = len(batch_global_indices)
            batch_indices_tensor = torch.tensor(batch_global_indices, dtype=torch.long, device=device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
            
            # 多相机融合相似度
            batch_sim = torch.zeros(batch_len, M, device=all_camera_features[0].device)
            for features in all_camera_features:
                batch_feat = features[batch_indices_tensor]
                sim = torch.mm(batch_feat, features.t())
                batch_sim += sim
            batch_sim = batch_sim / len(camera_keys)
            
            # 应用 exclude_self 掩码
            if exclude_self:
                batch_vids = gpu_video_ids[batch_indices_tensor]
                same_mask = batch_vids.unsqueeze(1) == gpu_video_ids.unsqueeze(0)
                batch_sim.masked_fill_(same_mask, float('-inf'))
            
            # 应用 time_range 掩码
            if half_range is not None:
                batch_prog = gpu_progress_gt[batch_indices_tensor]
                time_mask = (gpu_progress_gt.unsqueeze(0) < batch_prog.unsqueeze(1) - half_range) | \
                            (gpu_progress_gt.unsqueeze(0) > batch_prog.unsqueeze(1) + half_range)
                batch_sim.masked_fill_(time_mask, float('-inf'))
            
            # 取 Top-n
            top_vals, top_idxs = torch.topk(batch_sim, top_n_actual, dim=1)
            top_vals = top_vals.cpu().numpy()
            top_idxs = top_idxs.cpu().numpy()
            
            # 立即构建该 batch 的预测结果
            for j, global_idx in enumerate(batch_global_indices):
                query_frame_idx = frame_indices_np[global_idx]
                query_gt = progress_gt_np[global_idx]
                
                progress_values = []
                top_n_matches = []
                
                for k in range(top_n_actual):
                    db_idx = top_idxs[j, k]
                    similarity = top_vals[j, k]
                    
                    if similarity == float('-inf'):
                        continue
                    
                    db_vid_idx = video_ids_np[db_idx]
                    db_ep_id = video_names[db_vid_idx]
                    db_frame_idx = frame_indices_np[db_idx]
                    db_prog_gt = progress_gt_np[db_idx]
                    
                    top_n_matches.append((db_ep_id, int(db_frame_idx), float(similarity), float(db_prog_gt)))
                    progress_values.append(db_prog_gt)
                
                progress_values.append(query_gt)
                predicted_progress = float(np.mean(progress_values))
                
                pred = ProgressPrediction(
                    query_frame_idx=int(query_frame_idx),
                    query_progress_gt=float(query_gt),
                    predicted_progress=predicted_progress,
                    top_n_matches=top_n_matches
                )
                ep_results.append(pred)
            
            # 释放临时变量
            del batch_sim, top_vals, top_idxs
        
        # ============ 立即保存该 episode 的结果 ============
        parquet_path = get_parquet_path(dataset_path, episode_id, chunk_size)
        
        # 计算误差
        errors = [abs(r.predicted_progress - r.query_progress_gt) for r in ep_results]
        total_mae.append(np.mean(errors))
        total_rmse.append(np.sqrt(np.mean([e**2 for e in errors])))
        
        # 保存到 parquet
        chunk_name = parquet_path.parent.name
        output_chunk_dir = output_base_dir / chunk_name
        output_chunk_dir.mkdir(parents=True, exist_ok=True)
        save_predictions_to_parquet(ep_results, parquet_path, output_chunk_dir / f"{episode_id}.parquet")
        
        total_processed += 1
        
        # 释放该 episode 的结果内存
        del ep_results
    
    # 释放 GPU 内存
    del all_camera_features
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # 返回统计结果
    return total_processed, total_mae, total_rmse


def save_predictions_to_parquet(
    results: List[ProgressPrediction],
    original_parquet_path: Path,
    output_parquet_path: Path,
):
    """将预测结果保存到新的parquet文件"""
    df = pd.read_parquet(original_parquet_path)
    
    predicted_progress = {r.query_frame_idx: r.predicted_progress for r in results}
    
    if 'frame_index' in df.columns:
        df['progress_predicted'] = df['frame_index'].map(predicted_progress)
    else:
        df['progress_predicted'] = df.index.map(lambda i: predicted_progress.get(i))
    
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="基于SigLIP特征的视频帧Progress预测（矩阵版）",
        epilog="示例: python script.py /path/to/dataset --top_n 5"
    )
    
    parser.add_argument("dataset_path", type=str, help="数据集路径")
    parser.add_argument("--top_n", type=int, default=5, help="top-n匹配数 (默认: 5)")
    parser.add_argument("--top_n_all", action="store_true", help="使用所有视频作为top_n")
    parser.add_argument("--exclude_self", action="store_true", help="排除自己这条视频")
    parser.add_argument("--batch_size", type=int, default=256, help="批处理大小 (默认: 256)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--time_range", type=float, default=1.0, help="时间范围限制 (0-1)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="parquet chunk大小 (默认: 1000)")
    parser.add_argument("--camera_keys", type=str, nargs="+", default=None, help="要使用的相机列表")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    features_dir = dataset_path / "features"
    
    # 获取相机列表
    available_cameras = get_available_merged_cameras(features_dir)
    if not available_cameras:
        raise ValueError(f"未找到合并特征文件: {features_dir}")
    
    if args.camera_keys:
        camera_keys = [c for c in args.camera_keys if c in available_cameras]
    else:
        camera_keys = sorted(available_cameras)
    
    print(f"数据集: {dataset_path.name}")
    print(f"使用相机: {camera_keys}")
    
    # 计算 top_n
    top_n = args.top_n
    if args.top_n_all:
        first_data = load_merged_features(features_dir, camera_keys[0])
        top_n = len(first_data["video_names"])
        del first_data
        print(f"使用所有视频: top_n = {top_n}")
    
    # 逐 episode 计算并保存（内存友好）
    print("\n开始逐 episode 预测并保存...")
    total_processed, total_mae, total_rmse = predict_progress_per_episode(
        dataset_path=dataset_path,
        camera_keys=camera_keys,
        top_n=top_n,
        batch_size=args.batch_size,
        device=args.device,
        time_range=args.time_range,
        exclude_self=args.exclude_self,
        chunk_size=args.chunk_size,
        verbose=True
    )
    
    # 打印统计
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"  成功: {total_processed}")
    if total_mae:
        print(f"  平均 MAE: {np.mean(total_mae):.4f}")
        print(f"  平均 RMSE: {np.mean(total_rmse):.4f}")


if __name__ == "__main__":
    main()
