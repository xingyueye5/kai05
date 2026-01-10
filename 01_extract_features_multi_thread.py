import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
from dataclasses import dataclass
import time


class SigLIPWrapper(torch.nn.Module):
    """包装SigLIP模型，使get_image_features可以被DataParallel并行"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, **inputs):
        return self.model.get_image_features(**inputs)


@dataclass
class FrameTask:
    """单帧任务"""
    video_path: Path
    frame_idx: int
    frame_data: Optional[np.ndarray] = None


@dataclass
class VideoTask:
    """视频任务"""
    video_path: Path
    feature_path: Path
    camera_key: str
    total_frames: int = 0


def get_video_frame_count(video_path: str, frame_interval: int = 1) -> int:
    """
    快速获取视频帧数（考虑采样间隔）
    
    Args:
        video_path: 视频路径
        frame_interval: 帧采样间隔
    
    Returns:
        采样后的帧数
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 计算采样后的帧数
    return (frame_count + frame_interval - 1) // frame_interval


def count_total_frames(video_tasks: List, frame_interval: int = 1, num_workers: int = 8) -> Tuple[int, Dict[str, int]]:
    """
    多线程统计所有视频的总帧数
    
    Args:
        video_tasks: 视频任务列表
        frame_interval: 帧采样间隔
        num_workers: 线程数
    
    Returns:
        (总帧数, {video_path: frame_count})
    """
    frame_counts = {}
    
    def count_single(task):
        count = get_video_frame_count(str(task.video_path), frame_interval)
        return str(task.video_path), count
    
    print("正在统计视频帧数...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(count_single, task) for task in video_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="统计帧数"):
            video_path, count = future.result()
            frame_counts[video_path] = count
    
    total_frames = sum(frame_counts.values())
    print(f"总帧数: {total_frames}")
    
    return total_frames, frame_counts


def load_video_frames_batch(video_path: str, start_frame: int, num_frames: int) -> List[np.ndarray]:
    """
    加载视频的指定帧范围
    
    Args:
        video_path: 视频路径
        start_frame: 起始帧索引
        num_frames: 要加载的帧数
    
    Returns:
        帧列表 (RGB格式)
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames


def load_single_video_all_frames(video_path: str, frame_interval: int = 1) -> List[np.ndarray]:
    """加载单个视频的所有帧"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
    finally:
        cap.release()
    
    return frames


class VideoFrameLoader:
    """
    按顺序加载视频帧，预加载固定数量的视频
    
    内存占用 = prefetch_videos 个视频的帧
    """
    
    def __init__(
        self,
        video_tasks: List[VideoTask],
        batch_size: int,
        frame_interval: int = 1,
        num_workers: int = 4,
        prefetch_videos: int = 3
    ):
        """
        Args:
            video_tasks: 视频任务列表（按顺序处理）
            batch_size: 批处理大小
            frame_interval: 帧采样间隔
            num_workers: 加载线程数
            prefetch_videos: 预加载视频数量（内存中最多同时存在的视频数）
        """
        self.video_tasks = video_tasks
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.num_workers = num_workers
        self.prefetch_videos = prefetch_videos
        
        # 预加载的视频帧缓存: {video_idx: [(frame_idx, frame), ...]}
        self.video_cache: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.video_total_frames: Dict[int, int] = {}
        
        # 当前处理到的视频索引
        self.current_video_idx = 0
        # 当前视频已取出的帧索引
        self.current_frame_idx = 0
        
        # 加载完成标记
        self.loading_done = False
        self.all_done = False
        
        # 线程锁
        from threading import Lock, Condition
        self.lock = Lock()
        self.condition = Condition(self.lock)
        
        # 下一个要加载的视频索引
        self.next_load_idx = 0
    
    def _load_single_video(self, video_idx: int) -> Tuple[int, List[Tuple[int, np.ndarray]], int]:
        """加载单个视频的所有帧"""
        if video_idx >= len(self.video_tasks):
            return video_idx, [], 0
        
        task = self.video_tasks[video_idx]
        frames = []
        
        cap = cv2.VideoCapture(str(task.video_path))
        frame_count = 0
        sampled_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % self.frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((sampled_idx, frame_rgb))
                    sampled_idx += 1
                
                frame_count += 1
        finally:
            cap.release()
        
        task.total_frames = sampled_idx
        return video_idx, frames, sampled_idx
    
    def _prefetch_worker(self):
        """预加载工作线程"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while True:
                # 确定要加载哪些视频
                videos_to_load = []
                
                with self.lock:
                    # 计算需要预加载的视频
                    while (self.next_load_idx < len(self.video_tasks) and 
                           len(self.video_cache) + len(videos_to_load) < self.prefetch_videos):
                        videos_to_load.append(self.next_load_idx)
                        self.next_load_idx += 1
                
                if not videos_to_load:
                    # 检查是否所有视频都已加载完成
                    with self.lock:
                        if self.next_load_idx >= len(self.video_tasks):
                            self.loading_done = True
                            self.condition.notify_all()
                            break
                    time.sleep(0.01)
                    continue
                
                # 并行加载这些视频
                futures = {executor.submit(self._load_single_video, idx): idx for idx in videos_to_load}
                
                for future in as_completed(futures):
                    video_idx, frames, total_frames = future.result()
                    
                    with self.lock:
                        if frames:  # 只缓存有帧的视频
                            self.video_cache[video_idx] = frames
                            self.video_total_frames[video_idx] = total_frames
                        self.condition.notify_all()
                
                # 等待当前视频被消费后再继续加载
                with self.lock:
                    while (len(self.video_cache) >= self.prefetch_videos and 
                           self.current_video_idx < len(self.video_tasks)):
                        self.condition.wait(timeout=0.1)
    
    def start_loading(self):
        """启动后台预加载线程"""
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def get_batch(self) -> Tuple[List[VideoTask], List[int], List[np.ndarray]]:
        """
        按顺序获取一个batch的帧
        
        Returns:
            (video_tasks, frame_indices, frames) 或 (None, None, None) 表示结束
        """
        video_tasks = []
        frame_indices = []
        frames = []
        
        while len(frames) < self.batch_size:
            # 检查是否全部完成
            if self.current_video_idx >= len(self.video_tasks):
                break
            
            # 等待当前视频加载完成
            with self.lock:
                while (self.current_video_idx not in self.video_cache and 
                       not self.loading_done and
                       self.current_video_idx < len(self.video_tasks)):
                    self.condition.wait(timeout=0.1)
                
                # 再次检查
                if self.current_video_idx >= len(self.video_tasks):
                    break
                
                if self.current_video_idx not in self.video_cache:
                    # 可能是空视频，跳过
                    self.current_video_idx += 1
                    self.current_frame_idx = 0
                    continue
                
                cached_frames = self.video_cache[self.current_video_idx]
                task = self.video_tasks[self.current_video_idx]
                
                # 从当前视频取帧
                while self.current_frame_idx < len(cached_frames) and len(frames) < self.batch_size:
                    frame_idx, frame = cached_frames[self.current_frame_idx]
                    video_tasks.append(task)
                    frame_indices.append(frame_idx)
                    frames.append(frame)
                    self.current_frame_idx += 1
                
                # 当前视频取完了
                if self.current_frame_idx >= len(cached_frames):
                    # 释放内存
                    del self.video_cache[self.current_video_idx]
                    self.current_video_idx += 1
                    self.current_frame_idx = 0
                    self.condition.notify_all()  # 通知预加载线程可以继续加载
        
        if len(frames) == 0:
            return None, None, None
        
        return video_tasks, frame_indices, frames


class FeatureAccumulator:
    """
    特征累积器
    按视频收集特征，当一个视频的所有帧都处理完后保存
    """
    
    def __init__(self):
        self.video_features: Dict[str, Dict] = {}  # video_path -> {features, indices, total_frames, ...}
    
    def add_features(
        self,
        video_tasks: List[VideoTask],
        frame_indices: List[int],
        features: torch.Tensor
    ):
        """添加一批特征"""
        for i, (task, frame_idx) in enumerate(zip(video_tasks, frame_indices)):
            video_key = str(task.video_path)
            
            if video_key not in self.video_features:
                self.video_features[video_key] = {
                    'task': task,
                    'features': [],
                    'indices': [],
                    'total_frames': task.total_frames
                }
            
            self.video_features[video_key]['features'].append(features[i:i+1])
            self.video_features[video_key]['indices'].append(frame_idx)
            # 更新total_frames（可能在后续更新）
            if task.total_frames > 0:
                self.video_features[video_key]['total_frames'] = task.total_frames
    
    def get_completed_videos(self) -> List[Tuple[VideoTask, torch.Tensor]]:
        """获取已完成的视频（所有帧都已处理）"""
        completed = []
        to_remove = []
        
        for video_key, data in self.video_features.items():
            if len(data['features']) == data['total_frames'] and data['total_frames'] > 0:
                # 按帧索引排序
                sorted_pairs = sorted(zip(data['indices'], data['features']))
                sorted_features = [f for _, f in sorted_pairs]
                
                # 合并特征
                combined_features = torch.cat(sorted_features, dim=0)
                completed.append((data['task'], combined_features))
                to_remove.append(video_key)
        
        # 移除已完成的
        for key in to_remove:
            del self.video_features[key]
        
        return completed
    
    def flush_remaining(self) -> List[Tuple[VideoTask, torch.Tensor]]:
        """强制输出剩余的所有视频特征"""
        remaining = []
        
        for video_key, data in self.video_features.items():
            if len(data['features']) > 0:
                sorted_pairs = sorted(zip(data['indices'], data['features']))
                sorted_features = [f for _, f in sorted_pairs]
                combined_features = torch.cat(sorted_features, dim=0)
                remaining.append((data['task'], combined_features))
        
        self.video_features.clear()
        return remaining


def save_features(task: VideoTask, features: torch.Tensor):
    """保存特征到文件"""
    task.feature_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'features': features,
        'video_path': str(task.video_path),
        'camera_key': task.camera_key,
        'num_frames': features.shape[0],
        'feature_dim': features.shape[-1]
    }, task.feature_path)


def extract_features_from_dataset(
    dataset_path: str,
    ckpt: str = "/cpfs01/user/zhaolirui/siglip2/siglip2-giant-opt-patch16-384",
    batch_size: int = 1024,
    frame_interval: int = 1,
    camera_keys: List[str] = None,
    num_workers: int = 8
):
    """
    从数据集中提取SigLIP特征（多线程优化版）
    
    Args:
        dataset_path: 数据集路径
        ckpt: SigLIP模型检查点路径
        batch_size: 批处理大小
        frame_interval: 帧间隔
        camera_keys: 要处理的相机视角列表
        num_workers: 数据加载线程数
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    
    # 加载模型
    print(f"加载模型: {ckpt}")
    processor = AutoProcessor.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)
    model = SigLIPWrapper(model)  # 包装模型，使forward调用get_image_features
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 多GPU支持
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"检测到 {num_gpus} 个GPU，使用 DataParallel 并行")
        model = torch.nn.DataParallel(model)
    
    print(f"使用设备: {device} (GPU数量: {num_gpus})")
    model = model.to(device)
    model.eval()
    
    # 获取所有视频文件
    video_dir = dataset_path / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")
    
    video_files = list(video_dir.glob("**/*.mp4"))
    video_files.sort()
    
    if len(video_files) == 0:
        raise ValueError(f"未找到任何视频文件: {video_dir}")
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 创建特征保存目录
    features_dir = dataset_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建视频任务列表
    video_tasks = []
    skipped = 0
    
    for video_file in video_files:
        camera_key = video_file.parent.name.split('.')[-1]
        
        # 过滤相机视角
        if camera_keys is not None and camera_key not in camera_keys:
            continue
        
        # 构建特征保存路径
        relative_video_path = video_file.relative_to(dataset_path / "videos")
        feature_filename = video_file.stem + ".pt"
        feature_path = features_dir / relative_video_path.parent / feature_filename
        
        # 检查是否已存在
        if feature_path.exists():
            skipped += 1
            continue
        
        video_tasks.append(VideoTask(
            video_path=video_file,
            feature_path=feature_path,
            camera_key=camera_key
        ))
    
    print(f"待处理: {len(video_tasks)} 个视频, 已跳过: {skipped} 个")
    
    if len(video_tasks) == 0:
        print("没有需要处理的视频!")
        return
    
    # 预先统计总帧数
    total_frames, frame_counts = count_total_frames(
        video_tasks, 
        frame_interval=frame_interval, 
        num_workers=num_workers
    )
    num_batches = (total_frames + batch_size - 1) // batch_size
    print(f"预计batch数: {num_batches}")
    
    # 创建多线程加载器
    # prefetch_videos: 预加载几个视频（内存占用 = prefetch_videos 个视频的帧）
    loader = VideoFrameLoader(
        video_tasks=video_tasks,
        batch_size=batch_size,
        frame_interval=frame_interval,
        num_workers=num_workers,
        prefetch_videos=3  # 预加载3个视频，内存可控
    )
    
    # 创建特征累积器
    accumulator = FeatureAccumulator()
    
    # 启动后台加载
    loader.start_loading()
    
    # 处理循环
    total_frames_processed = 0
    total_videos_saved = 0
    
    pbar = tqdm(total=total_frames, desc="处理帧", unit="帧")
    
    while True:
        # 获取一个batch
        video_tasks_batch, frame_indices, frames = loader.get_batch()
        
        if frames is None:
            break
        
        # GPU推理
        inputs = processor(images=frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 通过wrapper的forward调用，DataParallel会自动分发到多卡
            batch_features = model(**inputs)
            batch_features = F.normalize(batch_features, p=2, dim=-1)
            batch_features = batch_features.cpu()
        
        # 累积特征
        accumulator.add_features(video_tasks_batch, frame_indices, batch_features)
        
        # 检查并保存已完成的视频
        completed = accumulator.get_completed_videos()
        for task, features in completed:
            save_features(task, features)
            total_videos_saved += 1
            tqdm.write(f"已保存: {task.feature_path.name} (shape: {features.shape})")
        
        total_frames_processed += len(frames)
        pbar.update(len(frames))
    
    pbar.close()
    
    # 保存剩余的视频
    remaining = accumulator.flush_remaining()
    for task, features in remaining:
        save_features(task, features)
        total_videos_saved += 1
        print(f"已保存: {task.feature_path.name} (shape: {features.shape})")
    
    print(f"\n特征提取完成!")
    print(f"总处理帧数: {total_frames_processed}")
    print(f"保存视频数: {total_videos_saved}")
    print(f"特征保存在: {features_dir}")


def main():
    parser = argparse.ArgumentParser(description="使用SigLIP从数据集提取视觉特征（多线程优化版）")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集路径，例如: /cpfs01/user/zhaolirui/siglip2/1020_27_442_v9-3_3000_lerobot_full"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/cpfs01/user/zhaolirui/siglip2/siglip2-giant-opt-patch16-384",
        help="SigLIP模型检查点路径"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="批处理大小 (默认: 1024)"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="帧采样间隔，1表示每帧都处理 (默认: 1)"
    )
    parser.add_argument(
        "--camera_keys",
        type=str,
        nargs="+",
        default=None,
        help="要处理的相机视角列表，例如: top_head wrist (默认: 处理所有视角)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="数据加载线程数 (默认: 8)"
    )
    
    args = parser.parse_args()
    
    extract_features_from_dataset(
        dataset_path=args.dataset_path,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        camera_keys=args.camera_keys,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()