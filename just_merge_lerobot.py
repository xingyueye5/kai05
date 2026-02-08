"""
将给定数据集，链接到新目录，并进行一定命名调整，用于最后开源
视频文件采用软链接方式，其余文件进行调整后重新保存

"""
import argparse
from pathlib import Path
import shutil
from typing import List, Dict
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def link4open(source_path: Path, dst_path: Path):
    """
    将source_path链接到dst_path，source_path为原始数据位置
    dst_path为新数据位置
    """
    if not dst_path.exists():
        dst_path.mkdir(parents=True, exist_ok=True)
        dst_path.symlink_to(source_path)
    else:
        raise FileExistsError(f'{dst_path} already exists')
def copy2new(source_file: Path, dst_file: Path):
    """
    将source_path复制到dst_path，source_path为原始数据位置
    dst_path为新数据位置
    """
    if not dst_file.parent.exists():
        dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source_file, dst_file)

class MergeLerobots:
    def __init__(self, source_paths: List[Path], dst_path: Path,  copy_videos: bool = False):
        self.source_paths = source_paths
        self.dst_path = dst_path
        self.path2episode_index = self.get_path2episode_index()  # 获得每个数据集对应的episode_index
        self.new_chunks_size = 1000
        self.total_frames = 0
        
        self.end_episode_index = 0
        self.new_episodes_stats = []
        self.new_episodes = []
        self.should_copy_videos = copy_videos
        
        # 获取所有以 data 开头的文件夹名（从第一个 source_path 获取）
        self.data_folder_names = self.get_data_folder_names()
    
    def get_data_folder_names(self):
        """获取所有以 data 开头的文件夹名"""
        data_folders = []
        for source_path in self.source_paths:
            for item in source_path.iterdir():
                if item.is_dir() and item.name.startswith('data'):
                    if item.name not in data_folders:
                        data_folders.append(item.name)
        data_folders.sort()
        print(f"检测到以下 data 文件夹: {data_folders}")
        return data_folders
    def get_path2episode_index(self):
        path2episode_index = defaultdict(list)
        remove_info = []  # 需要排除的 episode index 列表
        for source_path in self.source_paths:
            with open(source_path/'meta'/'info.json', 'r') as f:
                info = json.load(f)
            path2episode_index[source_path].extend(list(range(info['total_episodes'])))
            path2episode_index[source_path] = [item for item in path2episode_index[source_path] if item not in remove_info]
            # path2episode_index[source_path].sort()
        return path2episode_index
    
    def process_all_datasets(self):
        source_path_old2new_episode_index = defaultdict(dict)
        (self.dst_path/'meta').mkdir(parents=True, exist_ok=True)
        # 为所有 data 开头的文件夹创建目标目录
        for data_folder in self.data_folder_names:
            (self.dst_path/data_folder).mkdir(parents=True, exist_ok=True)
        (self.dst_path/'videos').mkdir(parents=True, exist_ok=True)

        for source_path, old_episode_indexes in self.path2episode_index.items():
            new_episodes_stats, new_episodes, self.end_episode_index, old2new_episode_index = self.process_one_dataset(source_path, self.end_episode_index, old_episode_indexes)
            source_path_old2new_episode_index[str(source_path)] = old2new_episode_index
            self.new_episodes_stats.extend(new_episodes_stats)
            self.new_episodes.extend(new_episodes)
        
        with open(self.dst_path/'meta'/'episodes_stats.jsonl', 'w') as f:
            for episode_stat in self.new_episodes_stats:
                f.write(json.dumps(episode_stat) + '\n')
        with open(self.dst_path/'meta'/'episodes.jsonl', 'w') as f:
            for episode in self.new_episodes:
                f.write(json.dumps(episode) + '\n')
        with open(self.source_paths[0]/'meta'/'info.json', 'r') as f:
            info = json.load(f)
        info['total_episodes'] = self.end_episode_index
        info['total_frames'] = self.total_frames
        info['total_videos'] = self.end_episode_index * 3
        info['total_chunks'] = self.end_episode_index // self.new_chunks_size + 1
        info['splits'] = {
            'train': f"0:{self.end_episode_index}",
        }
        with open(self.dst_path/'meta'/'info.json', 'w') as f:
            json.dump(info, f, indent=4)
        shutil.copy(source_path/'meta'/'tasks.jsonl', self.dst_path/'meta'/'tasks.jsonl')

        with open(self.dst_path/'source_path_old2new_episode_index.json', 'w') as f:
            json.dump(source_path_old2new_episode_index, f, indent=4)
        return True

    def process_one_dataset(self, source_path: Path, start_episode_index: int, old_episode_indexes: List[int]):
        
        try:
            with open(source_path/'meta'/'info.json', 'r') as f:
                info = json.load(f)
            old_chunks_size = info['chunks_size']
            self.__check_files_exist__(source_path, old_episode_indexes, old_chunks_size)

            new_episode_indexes = list(range(start_episode_index, start_episode_index + len(old_episode_indexes)))
            old2new_episode_index = dict(zip(old_episode_indexes, new_episode_indexes))
        
            new_episodes_stats, old_episode_new_start_index = self.process_new_episode_stats(source_path, old_episode_indexes, old2new_episode_index)
            self.process_new_parquet(source_path, old_episode_indexes, old_chunks_size, old2new_episode_index, old_episode_new_start_index)
            new_episodes = self.process_new_episodes(source_path, old_episode_indexes, old2new_episode_index, old_episode_new_start_index)
            if not self.should_copy_videos:
                self.link_videos(source_path, old_episode_indexes, old_chunks_size, old2new_episode_index)
            else:
                self.copy_videos(source_path, old_episode_indexes, old_chunks_size, old2new_episode_index)

        except Exception as e:
            print(f'Error processing {source_path}: {e}')
            raise
        return new_episodes_stats, new_episodes, new_episode_indexes[-1] + 1, old2new_episode_index
    
    def process_new_episode_stats(self, source_path: Path, old_episode_indexes: List[int], old2new_episode_index: Dict[int, int]):
        with open(source_path/'meta'/'episodes_stats.jsonl', 'r') as f:
            episodes_stats = [json.loads(line) for line in f if line.strip()]
        # 当前在用的episode_index
        episodes_stats = [episode for episode in episodes_stats if episode['episode_index'] in old_episode_indexes]
        assert len(episodes_stats) == len(old_episode_indexes), f'episode_index: {old_episode_indexes}, episodes_stats: {episodes_stats}'
        episodes_stats.sort(key=lambda x: x['episode_index'])
        new_episodes_stats = []
        old_episode_new_start_index = defaultdict(int)
        new_frame_index = self.total_frames
        for episode_stat in tqdm(episodes_stats, desc='Processing new episode stats'):
            new_episode_index = old2new_episode_index[episode_stat['episode_index']]
            index_count = episode_stat['stats']['index']['count'][0]
            old_episode_new_start_index[episode_stat['episode_index']] = new_frame_index
            episode_stat['episode_index'] = new_episode_index
            episode_stat['stats']['index']['min'] = [new_frame_index]
            episode_stat['stats']['index']['max'] = [new_frame_index + index_count - 1]
            episode_stat['stats']['index']['mean'] = [(new_frame_index + new_frame_index + index_count - 1) / 2]
            episode_stat['stats']['index']['std'] = [np.std(range(new_frame_index, new_frame_index + index_count))]
            episode_stat['stats']['index']['count'] = [index_count]
            new_frame_index += index_count
            new_episodes_stats.append(episode_stat)
        self.total_frames = new_frame_index
        return new_episodes_stats, old_episode_new_start_index

    def process_new_parquet(self, source_path: Path, 
                                old_episode_indexes: List[int], 
                                old_chunks_size: int, 
                                old2new_episode_index: Dict[int, int], 
                                old_episode_new_start_index: Dict[int, int]):
        # 处理所有以 data 开头的文件夹
        for data_folder in self.data_folder_names:
            # 检查该 source_path 下是否存在这个 data 文件夹
            if not (source_path / data_folder).exists():
                print(f"跳过 {source_path / data_folder}，文件夹不存在")
                continue
            
            for old_episode_index in tqdm(old_episode_indexes, desc=f'Processing {data_folder} parquet'):
                new_episode_index = old2new_episode_index[old_episode_index]
                new_chunk_index = new_episode_index // self.new_chunks_size
                old_parquet_path = source_path/data_folder/f'chunk-{old_episode_index//old_chunks_size:03d}'/f'episode_{old_episode_index:06d}.parquet'
                new_parquet_path = self.dst_path/data_folder/f'chunk-{new_chunk_index:03d}'/f'episode_{new_episode_index:06d}.parquet'
                
                if not old_parquet_path.exists():
                    # 如果源文件不存在，跳过
                    continue
                    
                if not new_parquet_path.parent.exists():
                    new_parquet_path.parent.mkdir(parents=True, exist_ok=True)
                
                parquet = pd.read_parquet(old_parquet_path)
                parquet['index'] = parquet['index'] - parquet['index'].min() + old_episode_new_start_index[old_episode_index]
                parquet['episode_index'] = new_episode_index
                parquet.to_parquet(new_parquet_path, index=False)
        return True

    def process_new_episodes(self, source_path: Path, old_episode_indexes: List[int], old2new_episode_index: Dict[int, int], old_episode_new_start_index: Dict[int, int]):
        with open(source_path/'meta'/'episodes.jsonl', 'r') as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        episodes = [episode for episode in episodes if episode['episode_index'] in old_episode_indexes]
        assert len(episodes) == len(old_episode_indexes), f'episode_index: {old_episode_indexes}, episodes: {episodes}'
        episodes.sort(key=lambda x: x['episode_index'])
        new_episodes = []
        for episode in episodes:
            new_episode_index = old2new_episode_index[episode['episode_index']]
            episode['episode_index'] = new_episode_index
            new_episodes.append(episode)
        return new_episodes

    def link_videos(self, source_path: Path, old_episode_indexes: List[int], old_chunks_size: int, old2new_episode_index: Dict[int, int]):
        for old_episode_index in tqdm(old_episode_indexes, desc='Linking videos'):
            new_episode_index = old2new_episode_index[old_episode_index]
            new_chunk_index = new_episode_index // self.new_chunks_size
            for video_key in ['observation.images.hand_left', 'observation.images.hand_right', 'observation.images.top_head']:
                old_video_path = source_path/'videos'/f'chunk-{old_episode_index//old_chunks_size:03d}'/f'{video_key}'/f'episode_{old_episode_index:06d}.mp4'
                new_video_path = self.dst_path/'videos'/f'chunk-{new_chunk_index:03d}'/f'{video_key}'/f'episode_{new_episode_index:06d}.mp4'
                if not new_video_path.parent.exists():
                    new_video_path.parent.mkdir(parents=True, exist_ok=True)
                if not new_video_path.exists():
                    new_video_path.symlink_to(old_video_path)
                else:
                    print(old2new_episode_index)
                    raise FileExistsError(f'{new_video_path} already exists')
    def copy_videos(self, source_path: Path, old_episode_indexes: List[int], old_chunks_size: int, old2new_episode_index: Dict[int, int]):
        for old_episode_index in tqdm(old_episode_indexes, desc='Copying videos'):
            new_episode_index = old2new_episode_index[old_episode_index]
            new_chunk_index = new_episode_index // self.new_chunks_size
            for video_key in ['observation.images.hand_left', 'observation.images.hand_right', 'observation.images.top_head']:
                old_video_path = source_path/'videos'/f'chunk-{old_episode_index//old_chunks_size:03d}'/f'{video_key}'/f'episode_{old_episode_index:06d}.mp4'
                new_video_path = self.dst_path/'videos'/f'chunk-{new_chunk_index:03d}'/f'{video_key}'/f'episode_{new_episode_index:06d}.mp4'
                copy2new(old_video_path, new_video_path)


    def __check_files_exist__(self, source_path: Path, old_episode_indexes: List[int], old_chunks_size: int):
        miss_data_episode_indexes = []
        for old_episode_index in tqdm(old_episode_indexes, desc='Checking files exist'):
            for video_key in ['observation.images.hand_left', 'observation.images.hand_right', 'observation.images.top_head']:
                old_video_path = source_path/'videos'/f'chunk-{old_episode_index//old_chunks_size:03d}'/f'{video_key}'/f'episode_{old_episode_index:06d}.mp4'
                if not old_video_path.exists():
                    miss_data_episode_indexes.append(old_episode_index)
                    break
            old_parquet_path = source_path/'data'/f'chunk-{old_episode_index//old_chunks_size:03d}'/f'episode_{old_episode_index:06d}.parquet'
            if not old_parquet_path.exists():
                miss_data_episode_indexes.append(old_episode_index)
        for miss_data_episode_index in set(miss_data_episode_indexes):
            old_episode_indexes.remove(miss_data_episode_index)
        print(f"{source_path}，以下数据缺失：{miss_data_episode_indexes}")
        return True

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, nargs='+', required=True, help='原始数据集路径，可以为一个或多个')
    parser.add_argument('--dst_path', type=str, required=True, help='新数据集路径')
    parser.add_argument('--copy_video', required=False, default=False, action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = build_parser()
    source_paths = [Path(item) for item in args.source_path]
    dst_path = Path(args.dst_path)
    merge_lerobots = MergeLerobots(source_paths, dst_path, copy_videos=args.copy_video)
    merge_lerobots.process_all_datasets()