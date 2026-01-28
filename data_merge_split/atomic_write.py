import numpy as np
import pandas as pd
import json
from pathlib import Path
import tempfile

class NumpyEncoder(json.JSONEncoder):
        """处理 numpy 和 pandas 类型的 JSON 编码器"""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            return super().default(obj)


def atomic_write_json(file_path: Path, data: dict, indent=None):
    """原子写入JSON文件，避免网络文件系统锁定问题"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=file_path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=indent, cls=NumpyEncoder)
        tmp_path = Path(tmp.name)
    tmp_path.rename(file_path)

def atomic_write_jsonl(file_path: Path, items: list):
    """原子写入JSONL文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', dir=file_path.parent, delete=False) as tmp:
        for item in items:
            json.dump(item, tmp, cls=NumpyEncoder)
            tmp.write('\n')
        tmp_path = Path(tmp.name)
    tmp_path.rename(file_path)

def atomic_write_csv(file_path: Path, df: pd.DataFrame):
    """原子写入CSV文件"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', dir=file_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_csv(str(tmp_path), index=False)
    tmp_path.rename(file_path)

def atomic_write_parquet(file_path: Path, data: pd.DataFrame):
    # save_path = self.dst_path/f'data/chunk-{episode_index//self.chunks_size:03d}'/f'episode_{episode_index:06d}.parquet'
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.parquet', dir=file_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    data.to_parquet(str(tmp_path), index=False)
    tmp_path.rename(file_path)