from collections.abc import Callable
import logging
from pathlib import Path
import random

from pyarrow.util import _break_traceback_cycle_from_frame

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
import torch


class CustomLeRobotDataset(LeRobotDataset):
    """
    A custom extension of LeRobotDataset for progress estimation and value function training.

    This class extends LeRobotDataset with additional features:
    - History frame sampling (n_history)
    - Episode start frame inclusion (with_episode_start)
    - Timestep difference mode for contrastive learning
    - Stage progress supervision mode
    - Skip sampling within episodes
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        # * Custom parameters
        n_history: int = 0,
        n_future: int = 0,
        with_episode_start: bool = False,
        skip_sample_ratio_within_episode: float = 0.0,  # * 0 for no skipping, 0.5 for skipping first 50% samples in an episode
        timestep_difference_mode: bool = False,  # * Selecting samples from two different timesteps and do comparison as learning target.
        stage_process_mode: bool = False,  # * Using stage progress supervision.
        use_progress_predicted: bool = False,  # * Using progress predicted as learning target.
    ):
        """
        Initialize CustomLeRobotDataset.

        Args:
            repo_id: Repository ID for the dataset
            root: Local directory for dataset files
            episodes: List of episode indices to load
            image_transforms: Image transformation functions
            delta_timestamps: Delta timestamps configuration
            tolerance_s: Timestamp tolerance in seconds
            revision: Git revision/version
            force_cache_sync: Force sync with remote cache
            download_videos: Whether to download video files
            video_backend: Video decoding backend

            # Custom args:
            n_history: Number of history frames to include
            with_episode_start: Include episode start frame
            skip_sample_ratio_within_episode: Ratio of samples to skip at episode start
            timestep_difference_mode: Enable timestep difference learning
            stage_process_mode: Enable stage progress supervision
        """
        # 初始化继承自LeRobotDataset的初始化参数
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend,
        )
        # Store custom parameters before calling parent __init__
        self.n_history = n_history
        self.n_future = n_future
        self.with_episode_start = with_episode_start
        self.skip_sample_ratio_within_episode = skip_sample_ratio_within_episode
        self.timestep_difference_mode = timestep_difference_mode
        self.stage_process_mode = stage_process_mode
        self.use_progress_predicted = use_progress_predicted
        
        # Validation
        assert self.skip_sample_ratio_within_episode <= 0.5
        if self.timestep_difference_mode:
            assert not self.with_episode_start, "Cannot use episode start when using timestep difference mode."

        # * Custom: Episode index to array index mapping
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)} if episodes else {}

    def get_sample_with_imgs_from_idx(self, idx: int) -> dict:
        """
        Get a sample with video frames decoded from the given index.
        This is a helper method used by __getitem__ for fetching individual samples.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
            query_indices, padding = self._get_query_indices(idx, arr_idx)

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)

            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                print(f"Error decoding video frames for ep_idx {ep_idx} at idx {idx}: {e}")

            item = {**video_frames, **item}
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def __getitem__(self, idx: int) -> dict:
        """
        Get item with custom features for progress estimation.

        Returns a dict containing:
        - Base sample data
        - History frames (if n_history > 0)
        - Episode start frame (if with_episode_start)
        - Random timestep comparison frame (if timestep_difference_mode)
        - Progress labels
        """
        episode_level_dict = {}
        output_item = {}

        # Get main sample
        item = self.get_sample_with_imgs_from_idx(idx)
        output_item.update(item)

        ep_idx = item["episode_index"].item()
        cur_timestamp = item["timestamp"].item()

        if self.timestep_difference_mode:
            random_item = self.handle_timestep_difference_mode(idx, ep_idx, item)
            output_item.update(random_item)
        
        if self.with_episode_start:
            episode_start_item = self.handle_episode_start_frame(idx, ep_idx, item)
            output_item.update(episode_start_item)
        
        if self.n_history > 0:
            for his_idx in range(-self.n_history, 0):
                his_item = self.handle_history_frame(idx, ep_idx, his_idx, item, cur_timestamp)
                output_item.update(his_item)

        if self.n_future > 0:
            for fut_idx in range(1, self.n_future + 1):
                fut_item = self.handle_future_frame(idx, ep_idx, fut_idx, item, cur_timestamp)
                output_item.update(fut_item)
        
        # * Add is_failure_data to output_item
        if "is_failure_data" in item:
            output_item["is_failure_data"] = item["is_failure_data"]
        else:
            output_item["is_failure_data"] = False

        episode_level_dict["episode_length"] = self.meta.episodes[ep_idx]["length"]

        # Handle delta indices for action sequences
        if self.delta_indices is not None:
            episode_level_dict = self.handle_delta_indices(idx, ep_idx, episode_level_dict)
        if 'action_advantage' in item:
            episode_level_dict["action_advantage"] = item["action_advantage"]

        # Add task as a string
        task_idx = item["task_index"].item()
        episode_level_dict["task"] = self.meta.tasks[task_idx]

        # Append episode-level data
        output_item.update(episode_level_dict)

        # Compute progress labels
        if self.timestep_difference_mode:
            if self.stage_process_mode:
                stage_progress_gt_random = output_item["his_-100_stage_progress_gt"].item()
                stage_progress_gt = output_item["stage_progress_gt"].item()
                output_item["progress"] = stage_progress_gt - stage_progress_gt_random
            else:
                progress_gt_random = output_item["his_-100_progress_gt"].item()
                progress_gt = output_item["progress_gt"].item()
                output_item["progress"] = progress_gt - progress_gt_random
        elif self.stage_process_mode:
            stage_progress_gt = output_item["stage_progress_gt"].item()
            output_item["progress"] = stage_progress_gt
        elif self.use_progress_predicted:
            progress_predicted = output_item["VC_value_top_head"].item()
            output_item["progress"] = progress_predicted
        else:
            progress_gt = output_item["progress_gt"].item()
            output_item["progress"] = progress_gt

        return output_item
    
    def handle_delta_indices(self, idx, ep_idx, episode_level_dict) -> dict:
        query_indices = None
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        query_indices, padding = self._get_query_indices(idx, arr_idx)
        query_result = self._query_hf_dataset(query_indices)

        episode_level_dict = {**episode_level_dict, **padding}
        for key, val in query_result.items():
            episode_level_dict[key] = val
        return episode_level_dict

    def handle_timestep_difference_mode(self, idx, item) -> dict:
        """
        同一个视频片段里的不同两帧？
        """
        ep_idx = item["episode_index"].item()
        cur_timestamp = item["timestamp"].item()
        random_timestep_name = -100
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        ep_start_idx = self.episode_data_index["from"][arr_idx].item()
        ep_end_idx = self.episode_data_index["to"][arr_idx].item()
        while True:
            random_idx = random.randint(ep_start_idx, ep_end_idx - 1)
            if random_idx == idx:
                continue

            random_item = self.get_sample_with_imgs_from_idx(random_idx)

            ep_idx_check = random_item["episode_index"].item()
            cur_timestamp_check = random_item["timestamp"].item()
            if ep_idx_check != ep_idx or cur_timestamp_check == cur_timestamp:
                print(
                    f"Randomly selected invalid timestep, re-sampling. For global idx: {random_idx}, ep_idx: {ep_idx_check}, cur_timestamp: {cur_timestamp_check}"
                )
                continue
            break

        _keys = list(random_item.keys())
        for key in _keys:
            new_key = f"his_{random_timestep_name}_{key}"
            random_item[new_key] = random_item.pop(key)

        return random_item

    def handle_episode_start_frame(self, idx, ep_idx, item, final_item) -> dict:
        start_frame_name = -100

        cur_frame_index = item["frame_index"].item()
        start_index = idx - cur_frame_index
        cur_progress_gt = item["progress_gt"].item()
        if cur_progress_gt == 0:
            start_index = idx
        start_item = self.get_sample_with_imgs_from_idx(start_index)
        start_episode_ind = start_item["episode_index"].item()
        start_progress_gt = start_item["progress_gt"].item()

        while start_episode_ind != ep_idx or start_progress_gt != 0:
            start_index += 1
            start_item = self.get_sample_with_imgs_from_idx(start_index)
            start_episode_ind = start_item["episode_index"].item()
            start_progress_gt = start_item["progress_gt"].item()

        _keys = list(start_item.keys())
        for key in _keys:
            new_key = f"his_{start_frame_name}_{key}"
            start_item[new_key] = start_item.pop(key)

        return start_item

    def handle_history_frame(self, idx, ep_idx, his_idx, item, cur_timestamp) -> dict:
        check_his_idx = his_idx + idx
        check_item = self.get_sample_with_imgs_from_idx(check_his_idx)
        check_episode_ind = check_item['episode_index'].item()
        check_timestamp   = check_item['timestamp'].item()

        if check_his_idx >= 0 and check_episode_ind == ep_idx and check_timestamp < cur_timestamp:
            his_item = check_item
        else:
            # * idx is the start of a new episode, repeat it as its history
            his_item = item.copy()

        # * add prefix and merge to final_item
        for key in list(his_item.keys()):
            new_key = f"his_{his_idx}_{key}"
            his_item[new_key] = his_item.pop(key)
        return his_item
    
    def handle_future_frame(self, idx, ep_idx, fut_idx, item, cur_timestamp) -> dict:
        check_fut_idx = idx + fut_idx
        if check_fut_idx < self.num_frames:
            check_item = self.get_sample_with_imgs_from_idx(check_fut_idx)
            check_episode_ind = check_item['episode_index'].item()
            check_timestamp   = check_item['timestamp'].item()
            if check_episode_ind == ep_idx and check_timestamp > cur_timestamp:
                fut_item = check_item
            else:
                # * idx is the start of a new episode, repeat it as its future
                fut_item = item.copy()
        else:
            # * End of episode reached or dataset bound exceeded: repeat current frame (padding)
            fut_item = item.copy()
        
        for key in list(fut_item.keys()):
            new_key = f"fut_{fut_idx}_{key}"
            fut_item[new_key] = fut_item.pop(key)
        return fut_item


class CustomMultiLeRobotDataset(MultiLeRobotDataset, torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `CustomLeRobotDataset`s.

    The underlying `CustomLeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        **custom_kwargs,
    ):
        # super().__init__()
        torch.utils.data.Dataset.__init__(self)
        self.repo_ids = repo_ids

        # Use more relaxed tolerance
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.1)

        self._datasets = [
            CustomLeRobotDataset(
                repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=del_ts,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                **custom_kwargs,
            )
            for repo_id, del_ts in zip(repo_ids, delta_timestamps)
        ]

        # Disable any data keys that are not common across all of the datasets
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if len(extra_keys) > 0:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
