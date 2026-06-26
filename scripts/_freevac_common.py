"""
FreeVAC common IO utilities.

Shared by:
    - calculate_RTAP_advantage.py
    - calculate_FlowVar_advantage.py
    - fuse_advantages.py

Two responsibilities:

1. `load_merged_features(source_path, camera_keys)` — load and concatenate
   per-camera SigLIP2 features (produced by `merge_siglip_features.py`).
   Mirrors the IO contract of `calculate_VC_value.py::SharedFeatureStore` but
   without the multiprocess / GPU-shared-memory machinery (those are only
   needed for the heavy VC-Value kNN computation).

2. `write_per_episode_column(...)` — write a per-frame scalar column back to
   the corresponding LeRobot parquet under `<source>/data/chunk-XXX/episode_XXXXXX.parquet`.

Keeping both here ensures that any future advantage-signal script in the FreeVAC
family follows the exact same on-disk contract as the legacy VC-Value pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_merged_features(source_path: Path, camera_keys: list[str]) -> dict:
    """Load `features_merged_<camera>.pt` for each camera and concat along dim=1.

    Returns
    -------
    {
        "features":      tensor[M, D_total]   concat over cameras
        "video_ids":     tensor[M]
        "frame_indices": tensor[M]
        "progress_gt":   tensor[M]            (legacy column; unused by RTAP/FlowVar)
        "video_names":   list[str]
    }

    Sanity-checks across cameras: video_ids / frame_indices / progress_gt /
    video_names must agree across all camera_keys, matching the protocol used by
    `calculate_VC_value.py`.
    """
    feats_per_cam = []
    meta: Optional[dict] = None
    for camera_key in camera_keys:
        fp = source_path / "features" / f"features_merged_{camera_key}.pt"
        if not fp.exists():
            raise FileNotFoundError(f"missing feature file: {fp}")
        d = torch.load(fp, map_location="cpu", weights_only=False)
        feats_per_cam.append(d["features"])
        if meta is None:
            meta = {
                "video_ids":     d["video_ids"],
                "frame_indices": d["frame_indices"],
                "progress_gt":   d.get("progress_gt"),
                "video_names":   d.get("video_names"),
            }
        else:
            assert torch.equal(meta["video_ids"], d["video_ids"]), \
                f"video_ids mismatch in {fp}"
            assert torch.equal(meta["frame_indices"], d["frame_indices"]), \
                f"frame_indices mismatch in {fp}"
    features = torch.cat(feats_per_cam, dim=1)
    return {"features": features, **meta}


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------

def _parquet_path_for(source_path: Path, episode_index: int, chunk_size: int) -> Path:
    return (
        source_path
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def write_per_episode_column(
    values: torch.Tensor | np.ndarray,
    video_ids: torch.Tensor | np.ndarray,
    source_path: Path,
    column_name: str,
    chunk_size: int = 1000,
    *,
    desc: Optional[str] = None,
) -> None:
    """Group `values` by `video_ids` and write each episode's slice into the
    corresponding parquet file under `<source>/data/`. Skips episodes whose
    length mismatches the parquet on disk (with a warning).

    The relative order within each episode is preserved as given by the input
    arrays (which must already match the parquet row order — typically true
    because feature extraction iterates each episode's frames in order).
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    if isinstance(video_ids, torch.Tensor):
        video_ids = video_ids.cpu().numpy()
    unique_eps = sorted(set(int(v) for v in video_ids.tolist()))

    iter_eps: Iterable[int] = tqdm(unique_eps, desc=desc or f"Writing {column_name}")
    for ep in iter_eps:
        mask = video_ids == ep
        ep_values = values[mask]
        parquet_path = _parquet_path_for(source_path, ep, chunk_size)
        if not parquet_path.exists():
            print(f"  [WARN] parquet not found: {parquet_path}, skip")
            continue
        df = pd.read_parquet(parquet_path)
        if len(df) != len(ep_values):
            print(
                f"  [WARN] length mismatch for ep {ep}: "
                f"parquet={len(df)} vs values={len(ep_values)}, skip"
            )
            continue
        df[column_name] = ep_values
        df.to_parquet(parquet_path, index=False)
