"""
计算episodex的统计信息
"""
import json
from pathlib import Path
from multiprocessing import Queue, Process
import multiprocessing as mp
import logging
from logger import setup_logging, start_logging_listener, LOG_FILE

def cal_episodex_stats_line(episode_index: int, parquet_stats: dict, video_stats: dict):
    results = {
        'episode_index': episode_index,
        'stats': {
            # from data/*.parquet
            # from videos/*.mp4
        }
    }
    results['stats'].update(parquet_stats)
    results['stats'].update(video_stats)
    return results

class EpisodexStatsProcessor(Process):
    def __init__(self, queue:Queue, dst_path:Path, video_process_count:int, log_queue=None):
        super().__init__()
        self.queue = queue
        self.dst_path = dst_path
        self.video_process_count = video_process_count
        self.log_queue = log_queue
        self.logger = logging.getLogger(self.__class__.__name__)
        self.episode_stats = []

    def run(self):
        setup_logging(log_queue=self.log_queue)
        count_none = 0
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    count_none += 1
                    if count_none >= self.video_process_count:
                        self.logger.info('all video_process done, no more data to EpisodexStatsProcessor process')
                        break
                    continue
                self.logger.info('get info from video_process')
                global_episode_index, data_parquet_results, video_results = item
                self.logger.info(f'process episode_index {global_episode_index}', )
                episode_stats = cal_episodex_stats_line(global_episode_index, data_parquet_results, video_results)
                self.episode_stats.append(episode_stats)
                self.logger.info(f'process episode_index  {global_episode_index} done, now {len(self.episode_stats)} episodes processed', )
        except Exception as e:
            self.logger.error(f'Error in EpisodexStatsProcessor: {e}')
        finally:
            self.episode_stats.sort(key=lambda x: x['episode_index']) # 按episode_index排序
            with open(str(self.dst_path/'meta'/'episodes_stats.jsonl'), 'w') as f:
                for item in self.episode_stats:
                    json.dump(item, f)
                    f.write('\n')
            self.logger.info(f'EpisodexStatsProcessor done, {len(self.episode_stats)} episodes processed')