from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm
from typing import List
from collections import defaultdict
from typing import Dict, List
import hashlib

CHUNK = 1024 * 1024
def calculate_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file using the given hash algorithm."""
    hasher = hashlib.new(algorithm)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_hash(path):
    """计算单个文件的哈希值，返回 (path, hash) 元组"""
    return (path, calculate_hash(path))

def drop_duplicated_data2(source_paths: List[Path]):
    video_paths = []
    for source_path in source_paths:
        video_paths.extend(list(source_path.glob('videos/chunk-*/observation.images.hand_right/episode_*.mp4')))
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_hash, video_paths), total=len(video_paths), desc="计算Drop Data哈希值"))
    hash_values = dict(results)
    hash_to_paths = defaultdict(list)
    for path, hash_value in hash_values.items():
        hash_to_paths[hash_value].append(path)
    unique_hash_values = {}
    duplicated_hash_values = {}
    count = 0
    for hash_value, paths in hash_to_paths.items():
        unique_hash_values[hash_value] = paths[0]
        count += len(paths) - 1
        if len(paths) > 1:
            duplicated_hash_values[hash_value] = [str(path) for path in paths]
    return unique_hash_values, count, duplicated_hash_values

def get_path2episodes(hash_values: Dict[str, List[Path]]):
    path2episodes = defaultdict(list)
    for hash_value, path in hash_values.items():
        path2episodes[str(path).split('videos/')[0]].append(int(path.stem.split('_')[-1]))
    return path2episodes