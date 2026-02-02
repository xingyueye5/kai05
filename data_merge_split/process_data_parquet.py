"""
相较于process_data_parquet.py，本文件增加多任务合并功能，即多个任务，提供相同的task_index，
处理data/*.parquet文件，计算统计信息，包括：
1. 根据标注文件，进行数据截取，并记录截取后保留的起始帧和结束帧
2. 更新frame_index、timestamp、episode_index、task_index等字段
3. 计算统计信息，包括：
    - index
    - observation.state
    - action
    - timestamp
    - frame_index
    - episode_index
    - task_index
4. 返回处理后的数据，起始帧和结束帧，全局索引，统计信息
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import tempfile
import re
from multiprocessing import Queue, Process
import multiprocessing as mp
import logging
from logger import setup_logging, start_logging_listener, LOG_FILE, TqdmToLogger
from tqdm import tqdm
from typing import List, Dict, Union
import atomic_write

def cal_stats(arr:np.array, keepdims=True):
        return {
            'min': arr.min(axis=0,keepdims=keepdims).tolist(),
            'max': arr.max(axis=0,keepdims=keepdims).tolist(),
            'mean': arr.mean(axis=0,keepdims=keepdims).tolist(),
            'std': arr.std(axis=0,keepdims=keepdims).tolist(),
            'count': [len(arr)],
        }

def cal_data_parquet_stats(data_path: Path, episode_index:int, global_index:int, task_index:int, target_time:np.ndarray, logger:logging.Logger=None, fps:int=30):
    """
    计算data/*.parquet文件的统计信息
    episode_index: episode_index序号，同一个块一个编号
    global_index: 数据索引，全局索引，从给定index开始，逐帧加1
    task_index: 任务索引
    start_time: 开始时间，打标文件记录开始时间
    end_time: 结束时间，打标文件记录结束时间
    return: 
        data : data为处理后的数据
        old_start_frame_index : 用于进行视频剪辑，除去非记录帧,最小值
        old_end_frame_index : 用于进行视频剪辑，除去非记录帧,最大值
        global_index : 全局索引
        results : results为统计信息
    """
    data = pd.read_parquet(data_path)
    target_time = (target_time * fps).astype(np.int32)
    start_index, end_index = target_time[0], target_time[-1]
    data = data.loc[(data['frame_index']>=start_index)&(data['frame_index']<end_index)]
    old_start_frame_index, old_end_frame_index = data['frame_index'].min(), data['frame_index'].max() # 用于进行视频剪辑，除去非记录帧,最小值和最大值
    # 更新参数
    data['frame_index'] = data['frame_index'] - old_start_frame_index
    target_time= target_time - old_start_frame_index
    data['timestamp'] = data['timestamp'] - data['timestamp'].min()
    data['episode_index'] = episode_index
    # print(task_index)
    data['task_index'] = [task_index] * len(data)
    data['index'] = data['index'] - data['index'].min() + global_index
    # progress_gt 整体长度的变化
    data['progress_gt'] = [i/len(data) for i in range(len(data))]
    # stage_progress_gt，几个标注开始结束对，就有多少个阶段
    
    target_time = target_time.reshape(-1, 2)
    # 此处计算，是假设连续标签的时间也是连续的
    stage_progress_list = []
    num_stages = len(target_time)
    for idx, (stage_start, stage_end) in enumerate(target_time):
        base = idx / num_stages
        scale = 1 / num_stages
        stage_length = stage_end - stage_start
        stage_progress_list = stage_progress_list + [base + (i / stage_length) * scale for i in range(stage_length)]
        
    stage_progress_array = np.array(stage_progress_list, dtype=np.float32)
    try:
        data['stage_progress_gt'] = stage_progress_array
    except Exception as e:
        logger.error(f'Error calculating stage_progress_gt: data_path: {data_path}, episode_index: {episode_index}, global_index: {global_index}, task_index: {task_index}, target_time: {target_time}, error: {e}')
        # raise ValueError(f'Error calculating stage_progress_gt: data_path: {data_path}, episode_index: {episode_index}, global_index: {global_index}, task_index: {task_index}, target_time: {target_time}, error: {e}')
    results = {
        'index': cal_stats(data['index'].values),
        'observation.state':cal_stats(np.stack(data['observation.state'].values), keepdims=False),
        'action':cal_stats(np.stack(data['action'].values), keepdims=False), 
        'timestamp': cal_stats(data['timestamp'].values), 
        'frame_index': cal_stats(data['frame_index'].values), 
        'episode_index': cal_stats(data['episode_index'].values), 
        'task_index': cal_stats(data['task_index'].values),
    }
    global_index = data['index'].max() + 1 # 新全局索引，用于下一个数据集的索引
    return data, old_start_frame_index, old_end_frame_index, global_index, results


class DataParquetProcessor(Process):
    def __init__(self, data_path: Dict[str, List[int]], dst_path:Path, 
                        queue:Queue, chunks_size:int=1000, 
                        tasks:[Dict[str, [int|str|list]]|List[str]|None]=None, 
                        combine_task:List[List[str]]|None=None,
                        task_description:Dict[str, str]|None=None,
                        log_queue=None):
        """
        data_path: 待合并的多个数据集路径
        dst_path: 合并后数据保存路径
        queue: 共享队列，用于将处理后的Parquet数据次数传给视频处理进程
        chunks_size: 每个chunk的大小，默认为1000；用于计算chunk索引，可以从meta/info.json中获取
        tasks: 任务列表，如果为None或者为空，表示合并所有数据集；如果非空，则合并给定任务的数据集
        log_queue: 日志队列
        tasks 可以是一个字典，也可以是一个列表，列表中每一个元素是任务名称;如果为字典，则需要包含以下关键字：task_index, task
        tasks 示例：
        {
            'v2-3': {
                'task_index': 0,
                'task': 'v2-3',
            },
            'v2-4': {
                'task_index': 1,
                'task': 'v2-4',
            },
        }
        or
        [
            'v2-3',
            'v2-4',
        ]
        combine_task: 任务合并信息，如果为None，则不进行任务合并；如果非空，则进行任务合并，示例：
        [
            ['v2-3', 'v2-4'],
            ['v3-5', 'v3-6'],
        ]
        表示将v2-3和v2-4任务合并，v3-5和v3-6任务合并，合并后的任务索引为0和1
        """
        super().__init__()
        self.data_path = data_path
        self.global_index = 0
        self.queue = queue  # 使用共享队列
        self.global_episode_index = 0
        self.task_description = task_description if isinstance(task_description, dict) else {}
        self.all_task_name = set(list(self.task_description.values())) # 所有已给定任务名称
        self.task_index_map = {}# 任务描述到任务索引的映射
        self.combine_task = combine_task if combine_task is not None else {}
        self.log_queue = log_queue
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f'combine_task: {combine_task}')
        self.logger.info(f'task_description: {task_description}')
        self.logger.info(f'tasks: {tasks}')
        self.current_max_task = -1
        self.__init_user_tasks(tasks)
        self.NAME_PATTERN = re.compile(
            r"(?P<version>v\d+(?:-\d+)?|mixed\d+\.\d+)"  # version 可能缺失
            r"(?:-(?P<cloth>(衬衫|短袖)*))?"
            r"_record.csv$"
        )
        self.chunks_size = chunks_size
        self.dst_path = dst_path
        self.global_episodes = []
        self.merge_info = {} # 记录合并前，数据集的索引，以及合并后的数据集索引，用于合并后的数据集拆分

    def run(self):
        setup_logging(log_queue=self.log_queue)
        self.traverse_versions()
        for data_path in self.data_path.keys():
            try:
                self.run_one_task(data_path) # 处理一个单独lerobot数据集
            except Exception as e:
                self.logger.error(f'While running run_one_task, Error processing dataset {data_path}: {e}')
                continue
        self.queue.put(None) # 通知视频处理器结束
        self.logger.info(f'put None to queue, queue size: {self.queue.qsize()}')
        self.logger.info('*'*1000)
        self.logger.info('All datasets processed')
        # self.logger.info(f'Processing {self.global_episodes} done')
        self.logger.info('read meta/info.json')
        with open(str(data_path/'meta'/'info.json'), 'r') as f:
            info = json.load(f)

        self.logger.info('update info.json')
        # self.logger.info(info)
        info['total_episodes'] = self.global_episode_index
        info['total_frames'] = self.global_index
        info['total_tasks'] = len(self.tasks)   
        info['total_videos'] = self.global_episode_index * 3
        info['total_chunks'] = self.global_episode_index // self.chunks_size
        info['splits'] = {  
            'train': f'0:{self.global_episode_index}'
        }
        info['chunks_size'] = self.chunks_size
        self.logger.info('write meta/info.json')
        
        path = self.dst_path/'meta'
        path.mkdir(parents=True, exist_ok=True)
        self.logger.info(str(self.dst_path/'meta'/'info.json'))
        
        # 使用原子写入避免网络文件系统锁定问题
        atomic_write.atomic_write_json(self.dst_path/'meta'/'info.json', info, indent=4)

        self.logger.info('write meta/episodes.jsonl')
        atomic_write.atomic_write_jsonl(self.dst_path/'meta'/'episodes.jsonl', self.global_episodes)
        
        self.logger.info('write meta/tasks.jsonl')
    
        task = [
            {'task_index': index, 'task': task_name} for task_name, index in self.task_index_map.items()
        ]
        atomic_write.atomic_write_jsonl(self.dst_path/'meta'/'tasks.jsonl', list(task))

        for key in self.tasks.keys():
            df = pd.DataFrame(self.tasks[key]['task_episodes'])
            atomic_write.atomic_write_csv(self.dst_path/'meta'/f'{key}_record.csv', df)
        self.logger.info('write meta/tasks.jsonl done')
        
        self.logger.info('write merge_info.json')
        print('merge_info', self.merge_info)
        atomic_write.atomic_write_json(self.dst_path/'merge_info.json', self.merge_info, indent=4)
        self.logger.info('write merge_info.json done')

    def run_one_task(self,source_path: Path):
        self.logger.info('Start processing dataset %s', source_path)
        old_episodes_index = self.data_path[source_path]
        old_episodes_index.sort()
        pairs_data_index = [] # 记录合并前后数据索引，包括合并前的数据名称，以及合并后的数据名称

        # 新的 CSV 查找方式：{父文件夹名称}_record.csv
        parent_folder_name = source_path.parent.name
        record_csv_name = f"{parent_folder_name}_record.csv"
        record_csv_path = source_path / 'meta' / record_csv_name
        
        if not record_csv_path.exists():
            self.logger.warning(f'{record_csv_path} not found, skip {source_path}')
            return
        
        # 从 source_path.name 提取版本号（如 v1_260120_001_day_shift_395_10000_lerobot 中的 v1）
        dataset_name = source_path.name.lower()
        version_pattern = re.compile(r"^(?P<version>v\d+(?:-\d+)?|mixed\d+\.\d+)")
        version_match = version_pattern.match(dataset_name)
        
        if not version_match:
            self.logger.warning(f'{source_path.name} 未匹配到版本号，使用默认版本 None_version')
            version = 'None_version'
        else:
            version = version_match.group('version')
            self.logger.info(f'{source_path.name} 匹配到版本号：{version}')
        
        try:
            current_task = self.tasks[version]['task_index']
        except Exception as e:
            self.logger.error(f'While running run_one_task, miss version {version} in tasks, {source_path}: {e}, current_task: 赋值-1')
            current_task = -1

        with open(source_path/'meta'/'info.json', 'r') as f:
            info = json.load(f)
        chunks_size = info.get('chunks_size', 1000)

        with open(source_path/'meta'/'episodes.jsonl', 'r') as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        episodes = {
            item['episode_index']:item for item in episodes if item['episode_index'] in old_episodes_index
        }
        
        record_data = pd.read_csv(record_csv_path)
        record_data = record_data.sort_values(by='video')  # 原始视频可能乱序
        record_data = record_data.dropna(axis=1)
        num_cols = record_data.columns.tolist()
        if 'failure_points' in num_cols:
            num_cols.remove('failure_points')
        if 'video' in num_cols:
            num_cols.remove('video')
        if 'note' in num_cols:
            num_cols.remove('note')
        for col in num_cols:
            record_data[col] = pd.to_numeric(record_data[col], errors='coerce')
        # 仅保留去重后的数据，对应的episode_index
        record_data.loc[:,"episode_index"] = [int(item.split('.')[0].split('_')[-1]) for item in record_data['video'].values]
        record_data = record_data.loc[record_data['episode_index'].isin(old_episodes_index)]

        start_time_name = num_cols[0]  # 假设数据排列按照规定顺序
        tqdm_logger = TqdmToLogger(self.logger)

        # 处理每个parquet数据，并记录合并前后数据索引
        for _, row in tqdm(record_data.iterrows(), total=len(record_data), desc=f'Processing {record_csv_name}', file=tqdm_logger):
            video = row.iloc[0]
            target_time = row.loc[num_cols].values
            if not all(np.diff(target_time) >= 0):
                self.logger.error(f'record time error, {source_path}/meta/{record_csv_name}, {video} label time is not increasing, skip, label time: {target_time}')
                continue
            episode_index = int(video.split('.')[0].split('_')[-1])
            chunk_index = episode_index // chunks_size

            data_path = source_path/'data'/f'chunk-{chunk_index:03d}'/f'episode_{episode_index:06d}.parquet'
            try:
                data, old_start_frame_index, old_end_frame_index, self.global_index, results = cal_data_parquet_stats(data_path, self.global_episode_index, self.global_index, current_task, target_time=target_time, logger=self.logger)
            except Exception as e:
                self.logger.error(f'While running cal_data_parquet_stats, Error processing dataset {data_path}: {e}')
                continue
            # 保存.parquet数据
            atomic_write.atomic_write_parquet(self.dst_path/f'data/chunk-{self.global_episode_index//self.chunks_size:03d}'/f'episode_{self.global_episode_index:06d}.parquet', 
                                            data)                                    
            self.global_episodes.append({
                'episode_index': self.global_episode_index,
                'tasks': episodes.get(episode_index, {}).get('tasks', ['']),
                'length': old_end_frame_index - old_start_frame_index + 1,
            })
            self.queue.put((source_path, video, old_start_frame_index, old_end_frame_index, self.global_episode_index, self.global_index, results))
            pairs_data_index.append({
                'old_index': episode_index,
                'new_index': self.global_episode_index,
                'task_index': current_task,
            })
            # 重新调整video编号
            row.loc['video'] = f'episode_{self.global_episode_index:06d}.mp4'
            row.loc[num_cols] = row.loc[num_cols] - row.loc[start_time_name]
            self.tasks[version]['task_episodes'].append(row)
            self.global_episode_index += 1 # 更新全局数据集索引
        self.logger.info('Finish processing dataset %s', source_path)
        self.merge_info[str(source_path)] = pairs_data_index
    
    def __init_user_tasks(self, tasks:[Dict[str, [int|str|list]]|List[str]|None]):
        """
        初始化用户任务，根据用户选定给任务列表，初始化任务
        """
        if tasks is None or not tasks or len(tasks) == 0:
            self.tasks = {}
            self.appoint_tasks = False
            self.logger.info('tasks is None or not tasks or len(tasks) == 0,合并所有任务')
        elif isinstance(tasks, dict):
            try:
                self.tasks = {
                    task: {'task_index': task['task_index'], 
                            'task': task['task'], 
                            'task_episodes': []} for task in tasks.keys()
                    }
                self.appoint_tasks = len(self.tasks) > 0
            except Exception as e:
                self.logger.error(f'Error initializing tasks: {e}')
                raise ValueError(f'Invalid tasks {tasks}, error: {e}')
        elif isinstance(tasks, list):
            self.tasks = {task: {'task_index': i, 'task': task, 'task_episodes': []} for i, task in enumerate(tasks)}
            self.appoint_tasks = len(self.tasks) > 0
        else:
            self.logger.error(f'Invalid tasks type: {type(tasks)}')
            raise ValueError(f'Invalid tasks type: {type(tasks)}')
        if self.appoint_tasks:
            self.current_max_task = max([item['task_index'] for item in self.tasks.values()])
            self.logger.info(f'合并给定的所有任务，忽略其它未包含任务，合并任务为: {self.tasks}')
        else:
            self.current_max_task = -1
        
    def traverse_versions(self):
        """
        遍历，任务检索，获取任务版本号，进行task_index的分配，并更新self.tasks
        """
        task_index = self.current_max_task + 1 
        temp_tasks = {}
        dump_tasks_check = set()
        # 根据合并任务列表，预先分配任务编号，并更新temp_tasks
        for combine_task in self.combine_task:
            if len(dump_tasks_check & set(combine_task)) > 0:
                raise ValueError(f'存在相同任务被归并到不同任务列表，请检查combine_task参数，任务列表：{combine_task}')
            dump_tasks_check |= set(combine_task)
            add_task = False
            # 同一个合并序列中，给同一个标号
            for version in combine_task:
                if not self.appoint_tasks:
                    add_task = True
                    temp_tasks[version] = {
                        'task_index': task_index,
                        'task': self.task_description.get(version, version),
                        'task_episodes': [],
                    }
                elif version in self.tasks.keys():
                    add_task = True
                    temp_tasks[version] = {
                        'task_index': self.tasks[version]['task_index'],
                        'task': self.tasks[version]['task'],
                        'task_episodes': [],
                    }
            if add_task:
                task_index += 1
        
        tqdm_logger = TqdmToLogger(self.logger)
        version_pattern = re.compile(r"^(?P<version>v\d+(?:-\d+)?|mixed\d+\.\d+)")
        
        # 遍历所有数据路径，从路径名中提取版本号
        print('self.data_path', self.data_path)
        for source_path in tqdm(self.data_path.keys(), desc='Traversing data paths, allocating task indices', file=tqdm_logger):
            # 检查 CSV 文件是否存在
            parent_folder_name = source_path.parent.name
            record_csv_name = f"{parent_folder_name}_record.csv"
            record_csv_path = source_path / 'meta' / record_csv_name
            
            if not record_csv_path.exists():
                self.logger.warning(f'{record_csv_path} not found, skip {source_path}')
                continue
            
            # 从 source_path.name 提取版本号
            dataset_name = source_path.name.lower()
            version_match = version_pattern.match(dataset_name)
            
            if not version_match or not version_match.group('version'):
                self.logger.warning(f'{source_path.name} 未匹配到版本号，跳过')
                continue
            version = version_match.group('version')
            if version not in temp_tasks.keys() : # 非需要合并任务，则直接添加到self.tasks
                if version not in self.tasks.keys(): # 如果tasks里不存在
                    self.tasks[version] = {
                        'task_index': task_index,
                        'task': self.task_description.get(version, version),
                        'task_episodes': [],
                    }
                    task_index += 1
            else: # 需要合并任务，使用预计算task_index
                self.tasks[version] = {
                    'task_index': temp_tasks[version]['task_index'],
                    'task': temp_tasks[version]['task'],
                    'task_episodes': [],
                }
            # 更新任务描述到任务索引的映射;如果任务描述中没有该任务，则添加到任务描述中
            task_name = self.tasks[version]['task']
            if task_name == version: # 未提供任务描述，则从meta/tasks.jsonl中获取任务描述
                with open(source_path/'meta'/'tasks.jsonl', 'r') as f:
                    tasks = [json.loads(line) for line in f if line.strip()]
                # 假设一个数据集仅有一个任务
                task_name = tasks[0]['task'] if isinstance(tasks[0]['task'], str) else ''.join(tasks[0]['task'])
            if task_name not in self.task_index_map.keys(): # 如果存在版本号不同但任务相同的情况，则pass
                self.task_index_map[task_name] = self.tasks[version]['task_index']
            else:
                self.tasks[version]['task_index'] = self.task_index_map[task_name] # 如果存在版本号不同但任务相同的情况，则更新task_index
            self.all_task_name.add(task_name)
        self.current_max_task = task_index - 1 # 更新当前最大任务索引，因为预分配task_index时，已经加1
        self.logger.info(f'Traversing data paths, allocated task indices: {self.tasks}')