"""
FreeVAC unit tests
==================

完全使用合成数据 (np.random + 已知 ground truth) 的 CPU-only 单元测试，
覆盖：
    1. RTAP single_anchor       — 端点 progress=1、单调向终点上升
    2. RTAP multi_anchor        — 处理多种 goal 表征
    3. FlowVar aggregate        — 零方差状态 advantage 最高
    4. FlowVar mode mapping     — inverse / rank_pct 行为正确
    5. Fusion + orthogonality   — 已知正交输入 → |ρ| ≈ 0；已知共线 → |ρ| ≈ 1
    6. End-to-end on tmp parquet — RTAP → Fusion 全流程能把列写回 parquet 并被读回

跑法：
    cd others/Kai05-VLA
    python -m pytest scripts/test_freevac.py -v
    或：python scripts/test_freevac.py        # 自带 main，无需 pytest
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 让脚本可以独立运行（无需把目录加进 PYTHONPATH）
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from calculate_RTAP_advantage import (
    compute_rtap_single_anchor,
    compute_rtap_multi_anchor,
    write_back_to_parquet as write_rtap,
)
from calculate_FlowVar_advantage import (
    aggregate_chunk_std,
    dispersion_to_advantage,
    compute_flowvar_advantage_from_chunks,
)
from fuse_advantages import (
    percentile_normalize,
    fuse_advantages,
    orthogonality_stats,
    collect_arrays_from_parquets,
    write_fused_back,
)


# ----------------------------------------------------------------------------
# Helpers: build mock features and parquet dataset
# ----------------------------------------------------------------------------

def _make_synthetic_episode_features(n_frames: int, dim: int, seed: int):
    """
    构造单 episode 特征：在 dim 维度的"单位向量"上做线性插值。
    起点 = e_0, 终点 = e_1，中间帧线性过渡 (然后 L2 归一化)。
    这样: 与终点的余弦相似度沿帧号严格单调上升 → RTAP progress 应严格单调上升。
    """
    rng = np.random.default_rng(seed)
    start = rng.standard_normal(dim).astype(np.float32)
    end = rng.standard_normal(dim).astype(np.float32)
    start /= np.linalg.norm(start)
    end /= np.linalg.norm(end)
    ts = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[:, None]
    feats = start[None, :] * (1 - ts) + end[None, :] * ts
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(feats)


def _build_mock_features(n_eps: int = 5, n_frames: int = 20, dim: int = 32):
    feats_list, vids_list, fids_list, progress_list = [], [], [], []
    for ep in range(n_eps):
        f = _make_synthetic_episode_features(n_frames, dim, seed=ep)
        feats_list.append(f)
        vids_list.append(torch.full((n_frames,), ep, dtype=torch.long))
        fids_list.append(torch.arange(n_frames, dtype=torch.long))
        progress_list.append(torch.linspace(0, 1, n_frames))
    return {
        "features":      torch.cat(feats_list, dim=0),
        "video_ids":     torch.cat(vids_list, dim=0),
        "frame_indices": torch.cat(fids_list, dim=0),
        "progress_gt":   torch.cat(progress_list, dim=0),
    }


def _build_mock_parquet_dataset(tmpdir: Path, n_eps: int = 3, n_frames: int = 10) -> Path:
    """在 tmpdir 下复现 LeRobot 数据集目录骨架 + parquet 文件。"""
    data_dir = tmpdir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(n_eps):
        df = pd.DataFrame({
            "frame_index": np.arange(n_frames, dtype=np.int64),
            "episode_index": np.full(n_frames, ep, dtype=np.int64),
            "progress_gt": np.linspace(0, 1, n_frames, dtype=np.float32),
        })
        df.to_parquet(data_dir / f"episode_{ep:06d}.parquet", index=False)
    return tmpdir


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_rtap_single_anchor_monotonic():
    """RTAP single_anchor: 在合成线性插值数据上应严格单调上升，且终点=1。"""
    mock = _build_mock_features(n_eps=4, n_frames=15)
    progress = compute_rtap_single_anchor(
        mock["features"], mock["video_ids"], mock["frame_indices"],
    )
    assert progress.shape == (4 * 15,), f"got {progress.shape}"
    assert torch.all((progress >= -1e-5) & (progress <= 1 + 1e-5)), \
        f"progress out of range: min={progress.min()} max={progress.max()}"

    # 检查每个 episode 单调 + 终点为 1
    for ep in range(4):
        mask = mock["video_ids"] == ep
        ep_prog = progress[mask]
        diffs = ep_prog[1:] - ep_prog[:-1]
        assert torch.all(diffs >= -1e-5), f"ep {ep} not monotonic: {ep_prog.tolist()}"
        assert abs(ep_prog[-1].item() - 1.0) < 1e-5, f"ep {ep} endpoint != 1: {ep_prog[-1].item()}"
    print("✓ test_rtap_single_anchor_monotonic")


def test_rtap_multi_anchor_runs_and_bounded():
    mock = _build_mock_features(n_eps=6, n_frames=12, dim=24)
    progress = compute_rtap_multi_anchor(
        mock["features"], mock["video_ids"], mock["frame_indices"],
        n_anchors=3,
    )
    assert progress.shape == (6 * 12,)
    assert torch.all((progress >= -1e-5) & (progress <= 1 + 1e-5))
    # 由于 multi-anchor 是软化最大相似度，再做全局 min-max 归一化，
    # 全局 min/max 应严格触及 0 和 1。
    assert progress.min().item() <= 1e-4
    assert progress.max().item() >= 1 - 1e-4
    print("✓ test_rtap_multi_anchor_runs_and_bounded")


def test_flowvar_zero_variance_gives_max_advantage():
    """
    所有 sample 完全相同 → dispersion = 0 → advantage = 1 (inverse mode)。
    """
    B, N, H, A = 4, 8, 5, 7
    # state 0: 完全 deterministic（零方差）
    fixed = torch.zeros(N, H, A) + 0.3
    # state 1: 高方差
    noisy = torch.randn(N, H, A) * 5.0
    # state 2: 中等方差
    mid = torch.randn(N, H, A) * 0.5
    # state 3: 更小方差
    tiny = torch.randn(N, H, A) * 0.05

    chunks = torch.stack([fixed, noisy, mid, tiny], dim=0)  # [4, N, H, A]
    dispersion = aggregate_chunk_std(chunks)
    assert dispersion.shape == (B,)
    assert abs(dispersion[0].item()) < 1e-6, f"deterministic state should be 0, got {dispersion[0]}"
    # 排序：固定 < tiny < mid < noisy
    order = torch.argsort(dispersion).tolist()
    assert order[0] == 0 and order[-1] == 1, f"unexpected order: {order}"

    adv = dispersion_to_advantage(dispersion, mode="inverse")
    assert abs(adv[0].item() - 1.0) < 1e-4, f"adv[0] should be ≈ 1, got {adv[0]}"
    assert adv[1].item() < adv[2].item() < adv[3].item() < adv[0].item()
    print("✓ test_flowvar_zero_variance_gives_max_advantage")


def test_flowvar_rank_pct_mode():
    dispersion = torch.tensor([0.1, 5.0, 1.0, 0.01])
    adv = dispersion_to_advantage(dispersion, mode="rank_pct")
    # rank_pct: dispersion 最小者 advantage = 1, 最大者 advantage = 0
    assert torch.argmax(adv).item() == 3
    assert torch.argmin(adv).item() == 1
    assert abs(adv.max().item() - 1.0) < 1e-6
    assert abs(adv.min().item() - 0.0) < 1e-6
    print("✓ test_flowvar_rank_pct_mode")


def test_flowvar_end_to_end_helper():
    chunks = torch.randn(10, 4, 3, 2)
    adv = compute_flowvar_advantage_from_chunks(chunks, mode="inverse")
    assert adv.shape == (10,)
    assert torch.all((adv >= 0) & (adv <= 1 + 1e-4))
    print("✓ test_flowvar_end_to_end_helper")


def test_fusion_orthogonal_inputs():
    """两组独立采样的 noise → Pearson 应接近 0。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000)
    b = rng.standard_normal(5000)
    stats = orthogonality_stats(a, b)
    assert stats["abs_pearson"] < 0.1, f"expected near-zero correlation, got {stats}"
    fused = fuse_advantages(a, b, alpha=0.5)
    assert fused.shape == (5000,)
    assert fused.min() >= -1e-6 and fused.max() <= 1 + 1e-6  # 经过 percentile_normalize
    print("✓ test_fusion_orthogonal_inputs")


def test_fusion_collinear_inputs():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000)
    b = a * 1.7 + 0.5  # 完全共线
    stats = orthogonality_stats(a, b)
    assert stats["abs_pearson"] > 0.99, f"expected near-1 correlation, got {stats}"
    print("✓ test_fusion_collinear_inputs")


def test_percentile_normalize_range():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    p = percentile_normalize(x)
    assert abs(p.min() - 0.0) < 1e-6 and abs(p.max() - 1.0) < 1e-6
    # 严格单调（无 ties）
    assert (np.argsort(p) == np.argsort(x)).all()
    print("✓ test_percentile_normalize_range")


def test_end_to_end_rtap_fusion_on_tmp_parquet():
    """
    RTAP 写回 parquet → Fusion 读出 RTAP + 自造 FlowVar → 写回 fused 列 → 再读验证。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _build_mock_parquet_dataset(tmpdir, n_eps=3, n_frames=10)

        # 构造与 parquet 行数对齐的 mock features
        mock = _build_mock_features(n_eps=3, n_frames=10, dim=16)
        progress = compute_rtap_single_anchor(
            mock["features"], mock["video_ids"], mock["frame_indices"],
        )
        # 用 RTAP 提供的 IO 接口写回 parquet
        rtap_col = "RTAP_progress_test"
        write_rtap(progress, mock["video_ids"], tmpdir,
                   column_name=rtap_col, chunk_size=1000)

        # 自造 FlowVar 列：与 RTAP 加上独立噪声，期望 |ρ| < 1
        flowvar_col = "FlowVar_advantage_test"
        rng = np.random.default_rng(7)
        for p in (tmpdir / "data" / "chunk-000").glob("episode_*.parquet"):
            df = pd.read_parquet(p)
            df[flowvar_col] = rng.uniform(0, 1, size=len(df))
            df.to_parquet(p, index=False)

        # Fusion
        rtap_all, flowvar_all, files, lens = collect_arrays_from_parquets(
            tmpdir, rtap_col, flowvar_col,
        )
        assert len(rtap_all) == 3 * 10
        stats = orthogonality_stats(rtap_all, flowvar_all)
        assert "abs_pearson" in stats
        fused = fuse_advantages(rtap_all, flowvar_all, alpha=0.5)
        write_fused_back(files, lens, fused, out_col="fused_advantage")

        # 再读一遍验证写回成功
        df = pd.read_parquet(files[0])
        assert "fused_advantage" in df.columns
        assert df["fused_advantage"].notna().all()
    print("✓ test_end_to_end_rtap_fusion_on_tmp_parquet")


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

ALL_TESTS = [
    test_rtap_single_anchor_monotonic,
    test_rtap_multi_anchor_runs_and_bounded,
    test_flowvar_zero_variance_gives_max_advantage,
    test_flowvar_rank_pct_mode,
    test_flowvar_end_to_end_helper,
    test_fusion_orthogonal_inputs,
    test_fusion_collinear_inputs,
    test_percentile_normalize_range,
    test_end_to_end_rtap_fusion_on_tmp_parquet,
]


def main():
    import traceback
    passed, failed = 0, 0
    for t in ALL_TESTS:
        try:
            t()
            passed += 1
        except Exception:
            print(f"✗ {t.__name__} FAILED")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
