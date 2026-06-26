"""
FlowVar Advantage
=================

核心思路（论文 C2）：
    用 vanilla SFT 的 π0.5 作为 reference policy π_ref。
    对数据集每个 state s，以 N 个不同 noise 调用 π_ref.sample_actions(s),
    得到 N 条 action chunks ∈ R^[N, H, A]。
    advantage(s) = 1 / (1 + mean( std_across_N(action_chunks) ))
        含义：N 个采样方差越小 → policy 越确定 → s 处 advantage 越高。

工程实现策略：
    - 与 `calculate_VC_value.py` / `calculate_RTAP_advantage.py` 同协议：
      把每帧的 advantage 写回 LeRobot parquet 的新列 `FlowVar_advantage`。
    - 推理通过项目 `pi0_pytorch.py` 中标准的 `sample_actions` 完成；
      本脚本不重新实现 flow matching，只组织批量推理 + 聚合。
    - 由于真实推理需要加载 ~3B 参数的 π0.5，本文件提供两种入口：
        (i)  CLI 模式：传入 checkpoint 路径，加载 model 跑真实数据；
        (ii) `compute_flowvar_advantage_from_chunks(...)` 纯函数：
              输入已 sample 好的 action chunks，仅做聚合 + 归一化；
              方便单元测试与解耦验证（mock 数据可用）。

设计上 (i) 调用 (ii)，二者保持一致语义。
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
# Pure-function core (decoupled from π0.5 model, testable on CPU with mock)
# ----------------------------------------------------------------------------

def aggregate_chunk_std(action_chunks: torch.Tensor) -> torch.Tensor:
    """
    给定 N 次采样的 action chunks，按"先时间-动作维平均，再 sample 维标准差"
    的方式聚合成每个 state 的 scalar dispersion。

    Parameters
    ----------
    action_chunks : tensor[B, N, H, A]
        B 个 state，每个 state 采样 N 次，每次输出 H 步 × A 维动作。

    Returns
    -------
    dispersion : tensor[B]  scalar dispersion per state, >= 0
    """
    assert action_chunks.dim() == 4, "expected [B, N, H, A]"
    # 在 sample (N) 维上算 std，再在时间+动作维上取均值得到 scalar
    std_over_samples = action_chunks.std(dim=1, unbiased=False)   # [B, H, A]
    dispersion = std_over_samples.mean(dim=(-1, -2))              # [B]
    return dispersion


def dispersion_to_advantage(
    dispersion: torch.Tensor,
    mode: str = "inverse",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    把 dispersion ∈ [0, +∞) 映射成 advantage ∈ [0, 1]，方差越小 advantage 越高。

    mode :
        inverse:   1 / (1 + dispersion)
        rank_pct:  按 dispersion 升序的 percentile → 1 - pct  ∈ [0, 1]
    """
    if mode == "inverse":
        return 1.0 / (1.0 + dispersion + eps)
    elif mode == "rank_pct":
        order = torch.argsort(dispersion)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(len(dispersion), dtype=torch.float32, device=dispersion.device)
        pct = ranks / max(len(dispersion) - 1, 1)
        return 1.0 - pct
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_flowvar_advantage_from_chunks(
    action_chunks: torch.Tensor,
    mode: str = "inverse",
) -> torch.Tensor:
    """端到端：从 action_chunks [B, N, H, A] 直接得 advantage [B] ∈ [0, 1]。"""
    dispersion = aggregate_chunk_std(action_chunks)
    return dispersion_to_advantage(dispersion, mode=mode)


# ----------------------------------------------------------------------------
# Real-data entry: requires a loaded π0.5 reference policy
# ----------------------------------------------------------------------------

class _ReferencePolicyInterface:
    """
    最小化接口：任何能从 observation 在不同 noise 下采样 action chunk 的对象都可作为 π_ref。
    项目里 `pi0_pytorch.Pi0Pytorch.sample_actions(device, observation, noise=...)` 即满足。
    """

    @torch.no_grad()
    def sample_actions(self, device, observation, noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@torch.no_grad()
def compute_flowvar_for_dataset(
    pi_ref: _ReferencePolicyInterface,
    dataloader,
    n_samples: int = 8,
    device: str = "cuda",
    mode: str = "inverse",
) -> dict:
    """
    在整个数据集上跑 FlowVar 标注。

    Returns
    -------
    {
        "video_ids":      tensor[M],
        "frame_indices":  tensor[M],
        "advantage":      tensor[M] in [0, 1],
    }

    实现细节：
        每个 batch 里 B 个 state，扩展为 [B*N, ...] 后调用 sample_actions，
        重塑回 [B, N, H, A]，再走 aggregate_chunk_std + dispersion_to_advantage。
    """
    all_vids, all_fids, all_advs = [], [], []
    for batch in tqdm(dataloader, desc="FlowVar sampling"):
        obs = batch["observation"]                    # 模型期望的 obs 字典
        vids = batch["video_ids"]                     # [B]
        fids = batch["frame_indices"]                 # [B]
        B = vids.shape[0]

        # 用 N 个不同 noise 跑 sample_actions
        chunks = []
        for n in range(n_samples):
            noise = torch.randn_like(batch["noise_template"], device=device)
            act = pi_ref.sample_actions(device, obs, noise=noise)
            chunks.append(act)
        chunks = torch.stack(chunks, dim=1)           # [B, N, H, A]

        adv = compute_flowvar_advantage_from_chunks(chunks, mode=mode)
        all_vids.append(vids.cpu())
        all_fids.append(fids.cpu())
        all_advs.append(adv.cpu())

    return {
        "video_ids":     torch.cat(all_vids),
        "frame_indices": torch.cat(all_fids),
        "advantage":     torch.cat(all_advs),
    }


def write_back_to_parquet(
    result: dict,
    source_path: Path,
    column_name: str = "FlowVar_advantage",
    chunk_size: int = 1000,
) -> None:
    """把 FlowVar advantage 写回 LeRobot parquet（与 RTAP 完全相同协议）。"""
    data_root = source_path / "data"
    vids = result["video_ids"].numpy()
    fids = result["frame_indices"].numpy()
    advs = result["advantage"].numpy()
    unique_eps = sorted(set(vids.tolist()))

    for ep in tqdm(unique_eps, desc=f"Writing {column_name}"):
        mask = vids == ep
        ep_fids = fids[mask]
        ep_advs = advs[mask]
        # 按 frame_index 排序，保持与 parquet 行顺序一致
        order = np.argsort(ep_fids)
        ep_advs = ep_advs[order]

        parquet_path = data_root / f"chunk-{ep // chunk_size:03d}" / f"episode_{ep:06d}.parquet"
        if not parquet_path.exists():
            print(f"  [WARN] parquet not found: {parquet_path}, skip")
            continue
        df = pd.read_parquet(parquet_path)
        if len(df) != len(ep_advs):
            print(f"  [WARN] length mismatch for ep {ep}: parquet={len(df)} vs flowvar={len(ep_advs)}, skip")
            continue
        df[column_name] = ep_advs
        df.to_parquet(parquet_path, index=False)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _build_pi_ref_from_checkpoint(checkpoint_path: str, device: str):
    """
    加载 Kai05-VLA 训练好的 π0.5 checkpoint 作为 reference policy。
    本函数有意保持简单：详细的 dataloader 构造请按需扩展。
    """
    raise NotImplementedError(
        "Loading π0.5 reference policy from a checkpoint is environment-specific. "
        "Implement this when wiring up real-data experiments. "
        "Unit tests should use `compute_flowvar_advantage_from_chunks` directly."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FlowVar advantage labeling for static datasets")
    parser.add_argument("--source_path", type=str, required=True,
                        help="LeRobot dataset root")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Vanilla SFT π0.5 reference policy checkpoint dir/file")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--mode", type=str, default="inverse", choices=["inverse", "rank_pct"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=1000)
    return parser


def run(args: argparse.Namespace) -> None:
    pi_ref = _build_pi_ref_from_checkpoint(args.checkpoint, args.device)
    # 项目内 dataloader 已存在；这里仅占位。
    raise NotImplementedError(
        "Real-data CLI requires the project's dataloader to be wired in. "
        "Use `from calculate_FlowVar_advantage import "
        "compute_flowvar_advantage_from_chunks` in your own driver script."
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
