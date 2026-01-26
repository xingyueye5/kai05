"""
视频处理类，用于处理视频数据,包括视频读取、视频保存、根据给定起始和结束帧，进行视频切片
"""

import numpy as np
from pathlib import Path
from multiprocessing import Queue, Process
import multiprocessing as mp
from video_save import save_video_by_cv, encode_video_frames
from logger import setup_logging, start_logging_listener, LOG_FILE
import cv2
import logging
import gc

def cal_video_stats_by_cv(frames: np.array):
    """
    计算视频统计信息
    此处假设视频是由cv读取，此时frames是BGR格式；进行数据统计与记录时，需要倒序处理
    """
    x0 = frames[...,0]
    x1 = frames[...,1]
    x2 = frames[...,2]
    x0 = (x0 - x0.min()) / (x0.max() - x0.min())
    x1 = (x1 - x1.min()) / (x1.max() - x1.min())
    x2 = (x2 - x2.min()) / (x2.max() - x2.min())
    results = {
        'min': x2.min(keepdims=True).tolist() + x1.min(keepdims=True).tolist()+ x0.min(keepdims=True).tolist(),
        'max': x2.max(keepdims=True).tolist() + x1.max(keepdims=True).tolist() + x0.max(keepdims=True).tolist(),
        'mean': x2.mean(keepdims=True).tolist() + x1.mean(keepdims=True).tolist() + x0.mean(keepdims=True).tolist(),
        'std': x2.std(keepdims=True).tolist() + x1.std(keepdims=True).tolist() + x0.std(keepdims=True).tolist(),
        'count': [len(frames)],
    }
    return results

def read_video_by_cv(video_path: Path, logger: logging.Logger):
    """
    读取视频
    """
    logger.info('read video %s', video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error('read video %s failed', video_path)
        return {}
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    logger.info('read video %s done', video_path)
    return frames

class VideoProcessor(Process):
    def __init__(self, queue:Queue, dst_path:Path, episode_stats_queue:Queue, log_queue=None, chunks_size:int=1000, save_method:str='cv'):
        super().__init__()
        assert save_method in ['cv', 'av'], f"save_method must be 'cv' or 'av', but got {save_method}"
        self.queue = queue
        # self.episode_stats = []
        self.dst_path = dst_path
        self.episode_stats_queue = episode_stats_queue
        self.chunks_size = chunks_size
        self.log_queue = log_queue
        self.logger = logging.getLogger(self.__class__.__name__)
        self.save_method = save_method
    def run(self):
        setup_logging(log_queue=self.log_queue)
        while True:
            item = self.queue.get()

            if item is None:
                self.queue.put(None) # 通知数据处理器结束
                break
            self.logger.info('get info from data_parquet_processor')
            source_path, video, old_start_frame_index, old_end_frame_index, global_episode_index, global_index, parquet_results = item
            self.logger.info('process video %s, global_episode_index %s', video, global_episode_index)
            video_results = self.process_video(source_path, video, old_start_frame_index, old_end_frame_index, global_episode_index)
            self.logger.info('process video %s done', video)
            # self.episode_stats.append(cal_episodex_stats_line(global_episode_index, results, video_results))
            self.episode_stats_queue.put((global_episode_index, parquet_results, video_results))
        self.logger.info(f"{self.name}, {self.pid}, all video processed")
        self.episode_stats_queue.put(None) # 通知episode_stats处理器结束

    def process_video(self, source_path: Path, video: str, old_start_frame_index: int, old_end_frame_index: int, global_episode_index:int):

        episode_index = int(video.split('.')[0].split('_')[-1])
        chunk_index = episode_index // self.chunks_size
        video_keys = ['observation.images.hand_left', 'observation.images.hand_right', 'observation.images.top_head']
        video_results = {}
        for video_key in video_keys:
            video_path = source_path/'videos'/f'chunk-{chunk_index:03d}'/f'{video_key}'/f'{video}'
            if not video_path.exists():
                self.logger.warning('%s not found', video_path)
                continue
            frames = read_video_by_cv(video_path, self.logger)
            frames = frames[old_start_frame_index:old_end_frame_index+1] # 去除非记录帧
            frames = np.stack(frames)
            video_results[video_key] = cal_video_stats_by_cv(frames)
            save_video_path = self.dst_path/'videos'/f'chunk-{global_episode_index//self.chunks_size:03d}'/f'{video_key}'/f'episode_{global_episode_index:06d}.mp4'
            if self.save_method == 'cv':
                save_video_by_cv(frames, dst=save_video_path, fps=30, logger=self.logger, overwrite=True)
            elif self.save_method == 'av':
                encode_video_frames(frames, dst=save_video_path, fps=30, color_format='bgr', logger=self.logger, overwrite=True)
            # 立即释放内存：删除 frames 引用
            del frames
            # 处理完所有视频后，强制进行垃圾回收以释放内存
            gc.collect()
        return video_results