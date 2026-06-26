"""
Orthogonal Signal Fusion (FreeVAC contribution C3)
==================================================

读取每个 LeRobot 数据集 parquet 中已经计算好的 RTAP 与 FlowVar 列，
做 percentile-normalize + 加权平均，写回新列 `fused_advantage`。

下游：
    `calculate_lerobot_advantage.py --advantage_source fused_advantage \
        --advantage_type 5bins` 即可消费此列。

附带 orthogonality 分析：
    - 两列在数据集全局上的 Pearson |ρ|
    - 散点采样（写到 .npz 供 Figure 2 复用）
    - 各自的均值/std/分位数
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Pure-function core
# ----------------------------------------------------------------------------

def percentile_normalize(x: np.ndarray) -> np.ndarray:
    """把 x 转成全局 percentile ∈ [0, 1]；处理 ties 用 average 排名。"""
    from scipy.stats import rankdata
    ranks = rankdata(x, method="average")
    return (ranks - 1) / max(len(ranks) - 1, 1)


def fuse_advantages(
    rtap: np.ndarray,
    flowvar: np.ndarray,
    alpha: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    fused = alpha * percentile(rtap) + (1 - alpha) * percentile(flowvar)

    alpha = 1.0  → RTAP only
    alpha = 0.0  → FlowVar only
    alpha = 0.5  → 等权（默认）
    """
    assert rtap.shape == flowvar.shape, f"shape mismatch: {rtap.shape} vs {flowvar.shape}"
    if normalize:
        r = percentile_normalize(rtap)
        f = percentile_normalize(flowvar)
    else:
        r, f = rtap, flowvar
    return alpha * r + (1.0 - alpha) * f


def orthogonality_stats(rtap: np.ndarray, flowvar: np.ndarray) -> dict:
    """Pearson / Spearman 相关性 + 描述性统计。"""
    from scipy.stats import pearsonr, spearmanr
    pr, _ = pearsonr(rtap, flowvar)
    sr, _ = spearmanr(rtap, flowvar)
    return {
        "pearson_r":  float(pr),
        "spearman_r": float(sr),
        "abs_pearson": float(abs(pr)),
        "rtap_mean":      float(rtap.mean()),
        "rtap_std":       float(rtap.std()),
        "flowvar_mean":   float(flowvar.mean()),
        "flowvar_std":    float(flowvar.std()),
    }


# ----------------------------------------------------------------------------
# IO over LeRobot parquet
# ----------------------------------------------------------------------------

def collect_arrays_from_parquets(
    source_path: Path,
    rtap_col: str,
    flowvar_col: str,
) -> tuple[np.ndarray, np.ndarray, list[Path], list[int]]:
    """遍历所有 parquet，concat 出全局 rtap[] / flowvar[]；并记录每文件长度。"""
    data_root = source_path / "data"
    files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    assert len(files) > 0, f"no parquet under {data_root}"
    rtap_list, flowvar_list, lens = [], [], []
    for p in tqdm(files, desc="Reading parquets"):
        df = pd.read_parquet(p)
        assert rtap_col in df.columns, f"{p}: missing column {rtap_col}"
        assert flowvar_col in df.columns, f"{p}: missing column {flowvar_col}"
        rtap_list.append(df[rtap_col].to_numpy())
        flowvar_list.append(df[flowvar_col].to_numpy())
        lens.append(len(df))
    return np.concatenate(rtap_list), np.concatenate(flowvar_list), files, lens


def write_fused_back(
    files: list[Path],
    lens: list[int],
    fused: np.ndarray,
    out_col: str,
) -> None:
    """把全局 fused 数组按 lens 切回每个 parquet，写入 out_col 列。"""
    offset = 0
    for p, L in zip(tqdm(files, desc=f"Writing {out_col}"), lens):
        df = pd.read_parquet(p)
        df[out_col] = fused[offset:offset + L]
        df.to_parquet(p, index=False)
        offset += L
    assert offset == len(fused), f"length sanity failed: {offset} vs {len(fused)}"


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Optional[dict]:
    source_path = Path(args.source_path)
    rtap, flowvar, files, lens = collect_arrays_from_parquets(
        source_path, args.rtap_col, args.flowvar_col,
    )

    stats = orthogonality_stats(rtap, flowvar)
    print("[Fusion] orthogonality stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    if stats["abs_pearson"] >= 0.6:
        print(f"[Fusion] WARNING: |ρ|={stats['abs_pearson']:.3f} ≥ 0.6, "
              f"orthogonality gate FAILED — consider Plan B in new paper.md")

    fused = fuse_advantages(rtap, flowvar, alpha=args.alpha, normalize=True)

    if args.scatter_out:
        # 随机降采样写到 .npz，供后续 Figure 2 / scatter plot 使用
        rng = np.random.default_rng(0)
        n_keep = min(len(rtap), args.scatter_max)
        idx = rng.choice(len(rtap), size=n_keep, replace=False)
        np.savez(
            args.scatter_out,
            rtap=rtap[idx], flowvar=flowvar[idx],
            **{k: v for k, v in stats.items()},
        )
        print(f"[Fusion] scatter samples saved -> {args.scatter_out}")

    if not args.dry_run:
        write_fused_back(files, lens, fused, out_col=args.out_col)
    print(f"[Fusion] done -> parquet column `{args.out_col}` (alpha={args.alpha})")
    return {"stats": stats, "fused": fused}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orthogonal fusion of RTAP and FlowVar advantages")
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--rtap_col", type=str, required=True,
                        help="parquet 中 RTAP 列名，例如 RTAP_progress_single_anchor_top_head")
    parser.add_argument("--flowvar_col", type=str, required=True,
                        help="parquet 中 FlowVar 列名，例如 FlowVar_advantage")
    parser.add_argument("--out_col", type=str, default="fused_advantage")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="alpha=1: RTAP only; alpha=0: FlowVar only")
    parser.add_argument("--scatter_out", type=str, default="",
                        help="非空时把降采样的 (rtap, flowvar) 对存为 .npz")
    parser.add_argument("--scatter_max", type=int, default=50000)
    parser.add_argument("--dry_run", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
