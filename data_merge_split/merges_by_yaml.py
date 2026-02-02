from math import log
from process_data_parquet import DataParquetProcessor
from video_process import VideoProcessor
from episodex_stats import EpisodexStatsProcessor
from pathlib import Path
from multiprocessing import Queue, Process
import multiprocessing as mp
import logging
from logger import setup_logging, start_logging_listener, TqdmToLogger
from tqdm import tqdm
from typing import List, Dict
import os
import yaml
import argparse
import json

from drop_duplicated import drop_duplicated_data2, get_path2episodes

def build_parser():
    parser = argparse.ArgumentParser(description='Merge data based on .yaml config file')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--log_file', type=str, required=False, default=None, help='Log file path. If not provided, will use the same relative path as config yaml under logs/')
    parser.add_argument('--compute_progress', required=False, help='Store progress', default=False, action='store_true')
    parser.add_argument('--loger_info_path', type=str, default='logs', required=False, help='Part of the data path to save loger info')
    args = parser.parse_args()
    return args

def drop_duplicated_data(source_path: List[Path], logger: logging.Logger, args: argparse.Namespace):
    unique_hash_values, count, duplicated_hash_values = drop_duplicated_data2(source_path)
    path2episodes = get_path2episodes(unique_hash_values)
    with open(Path(__file__).parent/args.loger_info_path/'duplicated_hash_values.json', 'w') as f:
        json.dump(duplicated_hash_values, f, indent=4)
    logger.info(f'has {count} duplicated data')
    with open(Path(__file__).parent/args.loger_info_path/'path2episodes.json', 'w') as f:
        json.dump(path2episodes, f, indent=4)
    logger.info(f'path2episodes: {path2episodes}')
    return path2episodes

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 如果没有指定 log_file，则使用和 config yaml 相同的相对路径结构
    if args.log_file is None:
        config_path = Path(args.config)
        # 找到 config 目录的位置，提取相对路径
        config_parts = config_path.parts
        if 'config' in config_parts:
            config_idx = config_parts.index('config')
            relative_path = Path(*config_parts[config_idx + 1:])
        else:
            # 如果没有 config 目录，就用文件名
            relative_path = Path(config_path.name)
        # 将扩展名改为 .log
        log_file_name = relative_path.with_suffix('.log')
        log_file = Path(__file__).parent / args.loger_info_path / log_file_name
    else:
        log_file = Path(__file__).parent / args.loger_info_path / args.log_file
    
    # 确保日志文件的父目录存在
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_queue, log_listener, log_handlers = start_logging_listener(log_file=log_file,console=True)
    setup_logging(log_queue=log_queue)  # 先配置logging，才能正确写入日志
    logger = logging.getLogger(__name__)
    logger.info(f'Log file: {log_file}')
    import json
    tmp = json.dumps(config, indent=4)
    logger.info(f'Config: {tmp}')
    source_path = [Path(item) for item in config['source_path']]
    dst_path = config['dst_path']
    tasks = config['tasks']
    combine_task = config['combine_task']
    task_description = config['task_description']
    save_method = config['save_method']
    process_count = config['process_count']

    path2episodes = drop_duplicated_data(source_path, logger, args)
    # change the key of path2episodes to Path object
    path2episodes = {Path(key): value for key, value in path2episodes.items()}

    if not process_count or process_count <= 0:
        process_count = mp.cpu_count()
    else:
        process_count = int(process_count)

    logger.info(f'source_path: {source_path}')
    logger.info(f'dst_path: {dst_path}')
    logger.info(f'tasks: {tasks}')
    logger.info(f'save_method: {save_method}')
    logger.info(f'video process_count: {process_count}')
    logger.info('start merge data')
    # source_path = [Path(item) for item in source_path]
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)
    merge_transfer_queue = Queue() # 基于*parquet*数据处理进程和视频处理进程之间的数据传输队列，用于传递处理数据队列和视频截断信息
    episode_stats_queue = Queue()
    logger.info('prepare data_parquet_processor')
    data_parquet_processor = DataParquetProcessor(path2episodes, dst_path, merge_transfer_queue, 
                                                    tasks=tasks, 
                                                    combine_task=combine_task, 
                                                    task_description=task_description, 
                                                    log_queue=log_queue)
    video_workers = []
    logger.info('prepare video_workers')
    for _ in range(process_count):
        video_workers.append(VideoProcessor(merge_transfer_queue, dst_path, episode_stats_queue, log_queue=log_queue, chunks_size=1000, save_method=save_method))
    
    logger.info('prepare episode_stats_processor')
    episode_stats_processor = EpisodexStatsProcessor(episode_stats_queue, dst_path, video_process_count=len(video_workers), log_queue=log_queue)
    
    try:
        logger.info('start data_parquet_processor and video_workers')
        data_parquet_processor.start()
        for i, video_worker in enumerate(video_workers):
            video_worker.name = f'VideoProcessor-{i}'
            video_worker.start()

        episode_stats_processor.start()
        data_parquet_processor.join()
        for video_worker in video_workers:
            video_worker.join()
        logger.info(f'the episode_stats_queue size is {episode_stats_queue.qsize()}')
        episode_stats_processor.join()
    finally:
        log_listener.stop()
        for handler in log_handlers:
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
    import json
    with open(dst_path/'meta'/'info.json', 'r') as f:
        info = json.load(f)
    total_episodes = info['total_episodes']

    new_name = dst_path.parent/f'{dst_path.name}_{total_episodes}'
    dst_path.rename(new_name)
    print(f'rename {dst_path} to {new_name}')

    

if __name__ == '__main__':
    args = build_parser()
    main(args)