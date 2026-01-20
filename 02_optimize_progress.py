"""
每次加载一对视频，计算其中任意两帧之间的相似性，之后将对应序列压入限制大小的小顶堆，最后将堆中平均similarity作为预测结果

多进程共享版本：
- 只从硬盘读取一次数据到CPU共享内存
- 支持多个GPU各自从共享内存获取数据
- 每个GPU进程可以独立工作
"""
import torch
import torch.multiprocessing as mp
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import argparse
import pandas as pd
import os
import time

class SharedFeatureStore:
    """
    共享特征存储器
    - 只从硬盘读取一次数据到CPU共享内存
    - 多个GPU进程可以各自获取数据副本
    """
    
    def __init__(self, source_path: Path, camera_keys: List[str]):
        """
        初始化并加载数据到CPU共享内存（只读一次硬盘）
        
        Args:
            source_path: 特征文件路径
        """
        print(f"从硬盘加载特征数据...")
        t0 = time.time()
        
        # 只读一次硬盘
        try:
            features_all = [
                torch.load(
                    source_path / 'features' / f'features_merged_{camera_key}.pt',
                    map_location='cpu',
                    weights_only=False
                    ) for camera_key in camera_keys
            ]
        except Exception as e:
            print(f"加载特征数据失败: {e}")
            raise e
        # 加载第一个相机特征文件的元数据
        self._video_ids = features_all[0]["video_ids"]
        self._frame_indices = features_all[0]["frame_indices"]
        self._progress_gt = features_all[0]["progress_gt"]
        self._video_names = features_all[0]["video_names"]

        for i in range(len(camera_keys)):
            assert torch.equal(self._video_ids, features_all[i]["video_ids"]), f"video_ids不一致: {camera_keys[i]}，请确保所有相机特征文件的video_ids一致"
            assert torch.equal(self._frame_indices, features_all[i]["frame_indices"]), f"frame_indices不一致: {camera_keys[i]}，请确保所有相机特征文件的frame_indices一致"
            assert torch.equal(self._progress_gt, features_all[i]["progress_gt"]), f"progress_gt不一致: {camera_keys[i]}，请确保所有相机特征文件的progress_gt一致"
            assert self._video_names == features_all[i]["video_names"], f"video_names不一致: {camera_keys[i]}，请确保所有相机特征文件的video_names一致"
        # 存储所有数据
        self._features: torch.Tensor = torch.cat([features_all[i]["features"] for i in range(len(camera_keys))], dim=1)
        del features_all
        
        # 将tensor放入CPU共享内存，多进程可直接访问
        self._features.share_memory_()
        
        print(f"数据加载完成，耗时: {time.time() - t0:.2f}s")
        print(f'   - features shape: {self._features.shape}, dtype: {self._features.dtype}')
        print(f"   - 共享内存状态: {self._features.is_shared()}")
    
    @property
    def features(self) -> torch.Tensor:
        """CPU共享内存中的features（所有进程共享）"""
        return self._features
    
    @property
    def video_ids(self):
        return self._video_ids
    
    @property
    def frame_indices(self):
        return self._frame_indices
    
    @property
    def progress_gt(self):
        return self._progress_gt
    
    @property
    def video_names(self):
        return self._video_names
    
    def get_features_on_gpu(self, gpu_id: int) -> torch.Tensor:
        """
        获取指定GPU上的features副本
        
        Args:
            gpu_id: GPU编号
            
        Returns:
            该GPU上的features张量
        """
        device = torch.device(f'cuda:{gpu_id}')
        return self._features.to(device)
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有共享数据"""
        return {
            "features": self._features,
            "video_ids": self._video_ids,
            "frame_indices": self._frame_indices,
            "progress_gt": self._progress_gt,
            "video_names": self._video_names,
        }


# 全局共享存储（用于多进程访问）
_shared_store: Optional[SharedFeatureStore] = None

# Worker进程的全局GPU数据缓存
_worker_gpu_data: Dict[str, Any] = {}


def init_shared_store(source_path: Path, camera_keys: List[str]) -> SharedFeatureStore:
    """
    初始化全局共享存储（在主进程中调用一次）
    """
    global _shared_store
    if _shared_store is None:
        _shared_store = SharedFeatureStore(source_path, camera_keys)
    return _shared_store


def get_shared_store() -> SharedFeatureStore:
    """获取全局共享存储"""
    global _shared_store
    if _shared_store is None:
        raise RuntimeError("共享存储未初始化，请先调用 init_shared_store()")
    return _shared_store


def worker_init_fn(gpu_id: int, features: torch.Tensor, progress_gts: torch.Tensor, video_ids: torch.Tensor):
    """
    Worker进程初始化函数
    在worker启动时将数据加载到对应GPU，后续该worker的所有任务共享这份GPU数据
    
    Args:
        gpu_id: 该worker使用的GPU编号
        features: CPU共享内存中的features
        progress_gts: progress ground truth
        video_ids: video ids
    """
    global _worker_gpu_data
    
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        # 将数据加载到GPU并缓存
        _worker_gpu_data['features'] = features.to(device)
        _worker_gpu_data['progress_gts'] = progress_gts.to(device)
        _worker_gpu_data['video_ids'] = video_ids.to(device)
        _worker_gpu_data['device'] = device
        _worker_gpu_data['gpu_id'] = gpu_id
    else:
        # CPU模式
        _worker_gpu_data['features'] = features
        _worker_gpu_data['progress_gts'] = progress_gts
        _worker_gpu_data['video_ids'] = video_ids
        _worker_gpu_data['device'] = torch.device('cpu')
        _worker_gpu_data['gpu_id'] = None


def process_single_episode(args: Tuple, exclude_self_episode: bool = True, exclude_self_frame_value: bool = True, query_chunk_size: int = 128):
    """
    处理单个episode
    使用worker进程初始化时缓存的GPU数据
    分块处理以避免显存溢出
    
    Args:
        args: (episode_index, top_n, window)
    """
    global _worker_gpu_data
    
    episode_index, top_n, window = args
    
    # 使用worker初始化时缓存的GPU数据
    features = _worker_gpu_data['features']
    progress_gts = _worker_gpu_data['progress_gts']
    video_ids = _worker_gpu_data['video_ids']
    
    query_mask = video_ids == episode_index
    query_features = features[query_mask]
    query_progress_gts = progress_gts[query_mask]
    num_query = len(query_features)
    
    if num_query == 0:
        return [], episode_index
    
    # 分块处理参数：控制每次处理的query帧数
    # 30%显存使用率 -> 可增大chunk_size提升并行度
    # 设置为128，约占用90%显存
    chunk_size = query_chunk_size
    
    all_predictions = []
    all_valid_mask = []
    
    # 预先计算用于排除自身的mask（只计算一次）
    if exclude_self_episode:
        self_mask = video_ids == episode_index  # [num_all_frames]
    
    for chunk_start in range(0, num_query, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_query)
        chunk_query_features = query_features[chunk_start:chunk_end]
        chunk_query_progress = query_progress_gts[chunk_start:chunk_end]
        
        # 计算当前chunk的相似度 [chunk_num, num_all_frames]
        similarity_scores = chunk_query_features @ features.T
        del chunk_query_features
        
        # 排除自身（使用预计算的mask）
        if exclude_self_episode:
            similarity_scores.masked_fill_(self_mask.unsqueeze(0), float('-inf'))
        
        # 向量化mask计算（比逐行处理快得多）
        # lower_bounds, upper_bounds: [chunk_num, 1]
        # progress_gts: [1, num_all_frames] -> broadcast to [chunk_num, num_all_frames]
        lower_bounds = (chunk_query_progress - window).unsqueeze(1)
        upper_bounds = (chunk_query_progress + window).unsqueeze(1)
        progress_gts_exp = progress_gts.unsqueeze(0)
        
        # 创建并应用mask（向量化操作）
        mask_matrix = (progress_gts_exp < lower_bounds) | (progress_gts_exp > upper_bounds)
        similarity_scores.masked_fill_(mask_matrix, float('-inf'))
        del lower_bounds, upper_bounds, progress_gts_exp, mask_matrix
        
        # 处理无效值（全是-inf的行）
        valid_mask = ~torch.all(similarity_scores == float('-inf'), dim=1)
        
        # 批量topk
        actual_top_n = min(top_n, similarity_scores.shape[1])
        _, top_indices = torch.topk(similarity_scores, actual_top_n, dim=1)
        del similarity_scores
        
        # 获取top progress_gts并计算均值
        top_progress_values = progress_gts[top_indices]
        del top_indices
        
        predictions = top_progress_values.mean(dim=1)
        del top_progress_values
        
        all_predictions.append(predictions)
        all_valid_mask.append(valid_mask)
    
    # 合并所有chunk的结果
    all_predictions = torch.cat(all_predictions, dim=0)
    if not exclude_self_frame_value:
        all_predictions = (all_predictions*actual_top_n + query_progress_gts) / (actual_top_n + 1)
    all_valid_mask = torch.cat(all_valid_mask, dim=0)
    
    # 只保留有效预测
    progress_predictions = all_predictions[all_valid_mask].tolist()
    del all_predictions, all_valid_mask, query_features, query_progress_gts
    
    torch.cuda.empty_cache()
    return progress_predictions, episode_index
        


def run_gpu_worker(gpu_id: int, worker_id: int, features: Dict[str, torch.Tensor], progress_gts: torch.Tensor, 
                   video_ids: torch.Tensor, tasks: List[Tuple], result_queue: mp.Queue, 
                   exclude_self_episode: bool = True, exclude_self_frame_value: bool = True, query_chunk_size: int = 128):
    """
    GPU worker进程：初始化GPU数据后处理分配的任务
    
    Args:
        gpu_id: GPU编号
        worker_id: 该GPU上的worker编号
        features: CPU共享内存中的features, Dict[str, torch.Tensor]
        progress_gts: progress ground truth, torch.Tensor
        video_ids: video ids
        tasks: 该worker需要处理的任务列表 [(episode_index, top_n, window), ...]
        result_queue: 结果队列
        exclude_self_episode: 是否排除自身episode
        exclude_self_frame_value: 是否排除自身frame 的progress_gt value
        query_chunk_size: query chunk大小
    """
    # 初始化GPU数据（只加载一次到该GPU）
    try:
        worker_init_fn(gpu_id, features, progress_gts, video_ids)
        print(f"GPU {gpu_id} Worker {worker_id}: 数据已加载到显存，分配 {len(tasks)} 个任务")
        
        # 处理所有分配给该worker的任务
        for task in tasks:
            result = process_single_episode(task, exclude_self_episode, exclude_self_frame_value, query_chunk_size)
            result_queue.put(result)
    except Exception as e:
        print(f"GPU {gpu_id} Worker {worker_id}: 发生错误: {e}")
        raise e
    finally:
        # 清理GPU显存
        global _worker_gpu_data
        _worker_gpu_data.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU {gpu_id} Worker {worker_id}: 已完成并清理显存")

def write_parquet(result: Tuple, source_path: Path, chunk_size: int = 1000):
    dst_path = source_path / "progress_predicted"
    data_parquet = source_path / "data"
    progress_predictions, episode_index = result
    old_parquet_path = data_parquet / f"chunk-{episode_index//chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"
    new_parquet_path = dst_path / f"chunk-{episode_index//chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(old_parquet_path)
    df['progress_predicted'] = progress_predictions
    if not new_parquet_path.parent.exists():
        new_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(new_parquet_path, index=False)

def main(workers_per_gpu: int = 1, user_args: argparse.Namespace = None):
    """
    Args:
        workers_per_gpu: 每个GPU上的worker进程数量，默认为1
    """
    source_path = Path(user_args.source_path)
    shared_store = init_shared_store(source_path, user_args.camera_keys)
    features = shared_store.features
    video_ids = shared_store.video_ids
    progress_gts = shared_store.progress_gt
    
    # 将tensor放入共享内存
    video_ids.share_memory_() 
    progress_gts.share_memory_()

    
    window = user_args.window
    episode_indexes = list(set(video_ids.tolist()))
    episode_indexes.sort()
    top_n = len(episode_indexes) if user_args.top_n <= 0 else user_args.top_n
    
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"可用GPU数量: {num_gpus}")
    print(f"每GPU进程数: {workers_per_gpu}")
    print(f"任务数量: {len(episode_indexes)}")
    
    mp.set_start_method('spawn', force=True)
    
    # 创建任务列表 (不再包含features等大数据)
    all_tasks = [(episode_index, top_n, window) for episode_index in episode_indexes]
    
    if num_gpus > 0:
        # 多GPU模式：每个GPU启动多个worker进程
        total_workers = num_gpus * workers_per_gpu
        
        # 将任务分配给各个worker (轮询分配)
        worker_tasks = [[] for _ in range(total_workers)]
        for idx, task in enumerate(all_tasks):
            worker_tasks[idx % total_workers].append(task)
        
        # 结果队列
        result_queue = mp.Queue()
        
        # 启动每个GPU的多个worker进程
        workers = []
        for gpu_id in range(num_gpus):
            for worker_idx in range(workers_per_gpu):
                global_worker_id = gpu_id * workers_per_gpu + worker_idx
                p = mp.Process(
                    target=run_gpu_worker,
                    args=(gpu_id, worker_idx, features, progress_gts, video_ids, 
                          worker_tasks[global_worker_id], result_queue, user_args.exclude_self_episode, 
                          user_args.exclude_self_frame_value, user_args.query_chunk_size)
                )
                p.start()
                workers.append(p)
        
        print(f"共启动 {total_workers} 个worker进程")
        
        # 收集结果并显示进度
        results = []
        with tqdm(total=len(all_tasks), desc="Processing episodes") as pbar:
            while len(results) < len(all_tasks):
                result = result_queue.get()
                results.append(result)
                pbar.update(1)
                write_parquet(result, source_path, chunk_size=user_args.chunk_size)
        
        # 等待所有worker完成
        for p in workers:
            p.join()
    else:
        # CPU模式：使用进程池
        process_count = os.cpu_count()
        print(f"CPU模式，进程数量: {process_count}")
        
        # CPU模式下需要在每个进程初始化数据
        def cpu_worker_init():
            worker_init_fn(None, features, progress_gts, video_ids)
        
        results = []
        with mp.Pool(processes=process_count, initializer=cpu_worker_init) as pool:
            for result in tqdm(pool.imap_unordered(process_single_episode, all_tasks), 
                              total=len(all_tasks), 
                              desc="Processing episodes"):
                results.append(result)
                write_parquet(result, source_path, chunk_size=user_args.chunk_size)

    print(f"处理完成，共 {len(results)} 个episode")

def build_parsers():
    parser = argparse.ArgumentParser(description="优化Progress预测")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="每个GPU上的worker进程数量")
    parser.add_argument("--source_path", type=str, default="/cpfs01/shared/kai05_data/kai0_data/short_sleeve/flatten_fold/v9-3/v9-3_0108_4556", help="数据集路径")
    parser.add_argument("--time_range", type=float, default=None, help="时间窗口")
    parser.add_argument("--window", type=float, default=0.3, help="时间窗口")
    parser.add_argument("--top_n", type=int, default=-1, help="top-n匹配数")
    parser.add_argument("--exclude_self_episode", action="store_true", help="是否排除自身episode")
    parser.add_argument("--exclude_self_frame_value", action="store_true", help="是否排除自身frame value")
    parser.add_argument("--chunk_size", type=int, default=1000, help="parquet chunk大小")
    parser.add_argument("--query_chunk_size", type=int, default=64, help="query chunk大小")
    parser.add_argument("--camera_keys", type=str, nargs="+", default=None, help="要使用的相机列表")
    return parser.parse_args()
if __name__ == "__main__":
    args = build_parsers()
    # print(args)
    if args.time_range is not None:
        args.window = args.time_range / 2 
    main(workers_per_gpu=args.workers_per_gpu, user_args=args)