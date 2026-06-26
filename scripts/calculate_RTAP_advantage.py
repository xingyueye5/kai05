"""
RTAP: Reverse-Time Anchored Progress
====================================

输入：features_merged_<camera>.pt（来自 merge_siglip_features.py），形如
    {
        "features":      tensor[M, D]   L2-normalized SigLIP2 features
        "video_ids":     tensor[M]      episode index of each frame
        "frame_indices": tensor[M]      frame index within each episode
        "progress_gt":   tensor[M]      [unused by RTAP, kept for sanity check only]
        "video_names":   list[str]
    }

输出：把每帧的 RTAP progress (float, ~[0, 1]) 写回对应的 LeRobot parquet 文件的
新列 `RTAP_progress_<camera_suffix>`，供下游 `calculate_lerobot_advantage.py`
通过 `--advantage_source RTAP_progress_<suffix>` 消费。

两种模式：
- single_anchor: 每个 episode 自身最后一帧 feature 作 goal anchor
- multi_anchor : 全数据集所有 episode 终点 feature 做 kmeans，公共 K 个 anchors

设计原则：
- 完全对齐 `calculate_VC_value.py` 的 IO 协议（同样的 features_merged_*.pt
  输入 + 写回 parquet 同一目录），最大化代码与流程复用。
- 仅依赖 numpy / torch / sklearn，无第三方重型 dep。
- CPU 即可跑（数据集量级 < 10^6 帧时）；提供 GPU 选项做加速。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Core RTAP computation
# ----------------------------------------------------------------------------

def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对最后一维做 L2 normalize；输入若已归一化则结果不变。"""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def compute_rtap_single_anchor(
    features: torch.Tensor,
    video_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Single-Anchor RTAP：
        对每个 episode i：
            anchor_i  = features[ last_frame_of_episode_i ]
            d(t)      = 1 - cos_sim( features[t], anchor_i )      for t in episode i
            progress  = 1 - d / d.max()                            ∈ [0, 1]
        其中"最后一帧"由 frame_indices 在该 episode 内最大值决定。

    Returns
    -------
    progress : tensor[M], float32, in [0, 1]
        与输入 features 行号一一对应；每个 episode 的最后一帧 progress = 1。
    """
    features = _l2_normalize(features.float())
    M = features.shape[0]
    progress = torch.zeros(M, dtype=torch.float32, device=features.device)

    unique_eps = torch.unique(video_ids).tolist()
    for ep in unique_eps:
        mask = video_ids == ep
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue

        # 找到该 episode 中 frame_index 最大者作为 goal anchor
        local_frames = frame_indices[idxs]
        anchor_local = torch.argmax(local_frames)
        anchor_global = idxs[anchor_local]
        anchor_feat = features[anchor_global]  # [D]

        ep_feats = features[idxs]                                # [N_ep, D]
        sims = ep_feats @ anchor_feat                            # [N_ep]
        dists = 1.0 - sims                                       # [N_ep]
        dist_max = dists.max().clamp(min=eps)
        prog = 1.0 - dists / dist_max                            # [N_ep]
        progress[idxs] = prog.float()

    return progress


def compute_rtap_multi_anchor(
    features: torch.Tensor,
    video_ids: torch.Tensor,
    frame_indices: torch.Tensor,
    n_anchors: int = 8,
    softmax_tau: float = 0.1,
    eps: float = 1e-8,
    seed: int = 0,
) -> torch.Tensor:
    """
    Multi-Anchor RTAP：
        Step 1: 取所有 episode 的"最后一帧"特征，做 kmeans 聚成 K 个 anchors
        Step 2: 对每帧 t，对 K 个 anchor 计算 cos_sim，取 softmax-weighted 最大值
                作为 progress（softmax_tau 越小越像 max）

    适用于任务有多种成功姿态/多种 goal 视觉表征的情形（Goal Sets, 2026）。
    """
    from sklearn.cluster import KMeans

    features = _l2_normalize(features.float())
    M, D = features.shape

    # 收集每个 episode 的终点特征
    unique_eps = torch.unique(video_ids).tolist()
    endpoint_feats = []
    for ep in unique_eps:
        mask = video_ids == ep
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        local_frames = frame_indices[idxs]
        anchor_local = torch.argmax(local_frames)
        anchor_global = idxs[anchor_local]
        endpoint_feats.append(features[anchor_global])
    endpoint_feats = torch.stack(endpoint_feats, dim=0)  # [n_episodes, D]

    # kmeans
    K = min(n_anchors, endpoint_feats.shape[0])
    kmeans = KMeans(n_clusters=K, n_init=4, random_state=seed)
    kmeans.fit(endpoint_feats.cpu().numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).to(features.device).float()
    centers = _l2_normalize(centers)  # [K, D]

    # 每帧到 K 个 anchor 的 cos_sim
    sims = features @ centers.T  # [M, K]
    # softmax-weighted "best" similarity
    weights = torch.softmax(sims / softmax_tau, dim=-1)            # [M, K]
    soft_sim = (weights * sims).sum(dim=-1)                        # [M]
    # 用全局最大/最小做 progress 归一化
    s_min = soft_sim.min()
    s_max = soft_sim.max().clamp(min=s_min + eps)
    progress = (soft_sim - s_min) / (s_max - s_min)
    return progress.float()


# ----------------------------------------------------------------------------
# IO  (delegates to scripts/_freevac_common.py to share with FlowVar / Fusion)
# ----------------------------------------------------------------------------

from _freevac_common import (
    load_merged_features as _load_merged_features,
    write_per_episode_column as _write_per_episode_column,
)


def load_features(source_path: Path, camera_keys: list[str]) -> dict:
    """Backward-compat thin wrapper over `_freevac_common.load_merged_features`."""
    return _load_merged_features(source_path, camera_keys)


def write_back_to_parquet(
    progress: torch.Tensor,
    video_ids: torch.Tensor,
    source_path: Path,
    column_name: str,
    chunk_size: int = 1000,
) -> None:
    """Backward-compat thin wrapper over `_freevac_common.write_per_episode_column`."""
    _write_per_episode_column(
        progress, video_ids, source_path,
        column_name=column_name, chunk_size=chunk_size,
    )


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Optional[dict]:
    """主入口；返回 dict 供单元测试断言（CLI 模式下不使用）。"""
    source_path = Path(args.source_path)
    print(f"[RTAP] loading features from {source_path / 'features'}")
    data = load_features(source_path, args.camera_keys)

    features = data["features"]
    video_ids = data["video_ids"]
    frame_indices = data["frame_indices"]
    print(f"[RTAP] features={tuple(features.shape)} eps={len(set(video_ids.tolist()))}")

    if args.device == "cuda" and torch.cuda.is_available():
        features = features.cuda()
        video_ids = video_ids.cuda()
        frame_indices = frame_indices.cuda()

    t0 = time.time()
    if args.mode == "single_anchor":
        progress = compute_rtap_single_anchor(features, video_ids, frame_indices)
    elif args.mode == "multi_anchor":
        progress = compute_rtap_multi_anchor(
            features, video_ids, frame_indices,
            n_anchors=args.n_anchors, softmax_tau=args.softmax_tau,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    print(f"[RTAP] mode={args.mode} compute_time={time.time() - t0:.2f}s")

    # 与 progress_gt 的 Spearman 作为 sanity（如果 progress_gt 有意义）
    if data["progress_gt"] is not None and data["progress_gt"].numel() == progress.numel():
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(progress.cpu().numpy(), data["progress_gt"].cpu().numpy())
            print(f"[RTAP] Spearman(RTAP, progress_gt) = {rho:.3f}")
        except ImportError:
            pass

    # 写回 parquet
    camera_suffix = "_".join(sorted(args.camera_keys))
    column_name = f"RTAP_progress_{args.mode}_{camera_suffix}"
    if not args.dry_run:
        write_back_to_parquet(
            progress.cpu(), video_ids.cpu(), source_path,
            column_name=column_name, chunk_size=args.chunk_size,
        )
    print(f"[RTAP] done -> parquet column `{column_name}`")
    return {"progress": progress.cpu(), "column_name": column_name}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RTAP: Reverse-Time Anchored Progress")
    parser.add_argument("--source_path", type=str, required=True,
                        help="LeRobot dataset root (contains data/ + features/)")
    parser.add_argument("--camera_keys", type=str, nargs="+", required=True,
                        help="同 calculate_VC_value.py 的 --camera_keys")
    parser.add_argument("--mode", type=str, default="single_anchor",
                        choices=["single_anchor", "multi_anchor"])
    parser.add_argument("--n_anchors", type=int, default=8,
                        help="multi_anchor 模式下的 kmeans K")
    parser.add_argument("--softmax_tau", type=float, default=0.1,
                        help="multi_anchor 软最大化温度，越小越像 max")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="parquet 分块大小，与原 pipeline 一致")
    parser.add_argument("--dry_run", action="store_true",
                        help="不写回 parquet，仅打印 stats")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
