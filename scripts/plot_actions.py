"""
Action 曲线可视化：支持仅 GT 或 GT vs 推理对比两种模式。
- 仅 GT：不传 checkpoint_dir，只绘制数据集的 Ground Truth actions。
- 对比模式：传入 checkpoint_dir，绘制 GT 与模型推理结果对比。
"""
import dataclasses
import logging
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config_loader
from openpi.training import data_loader as _data_loader

# 默认将图片保存到当前目录下的该文件夹；gt 与 compare 各占子目录
PLOT_ACTIONS_OUTPUT_DIR = "plot_actions"


@dataclasses.dataclass
class Args:
    """训练/数据配置 YAML 路径。"""
    config_path: str
    """Checkpoint 目录；不传则仅绘制 GT，传入则绘制 GT vs 推理对比。"""
    checkpoint_dir: Optional[str] = None
    """数据集 window size。"""
    dataset_window_size: int = 50
    """采样的段数（每段长度由 chunk_size 决定）。"""
    num_segments: int = 10
    """每段采样的步数。"""
    chunk_size: int = 50
    """输出图像路径；不传则自动生成。"""
    output_path: str = ""
    """日志级别: INFO / DEBUG。"""
    log_level: str = "INFO"
    """是否弹出显示图像窗口。"""
    show_plot: bool = False


def _array_stats(name: str, x: np.ndarray) -> str:
    return (
        f"{name}: shape={tuple(x.shape)}, "
        f"min={np.min(x):.4f}, max={np.max(x):.4f}, mean={np.mean(x):.4f}, std={np.std(x):.4f}"
    )


def main(args: Args) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    t0 = time.time()
    mode = "compare" if args.checkpoint_dir else "gt_only"
    logging.info("启动 action 可视化 (mode=%s)", mode)
    logging.info("config_path=%s", args.config_path)
    if args.checkpoint_dir:
        logging.info("checkpoint_dir=%s", args.checkpoint_dir)

    # -------------------------------------------------------------------------
    # Step 1: 加载 config 与数据集
    # -------------------------------------------------------------------------
    logging.info("Step 1: 加载 config 与数据集 ...")
    config = config_loader.load_config(args.config_path)
    data_config = config.data.create(config.assets_dirs, config.model)
    base_dataset = _data_loader.create_torch_dataset(
        data_config, args.dataset_window_size, config.model
    )
    dataset = _data_loader.TransformedDataset(
        base_dataset,
        [*data_config.repack_transforms.inputs],
    )
    dataset_len = len(dataset)
    max_index_needed = (args.num_segments - 1) * args.chunk_size
    logging.info("dataset_len=%d, max_index_needed=%d", dataset_len, max_index_needed)
    if max_index_needed >= dataset_len:
        raise ValueError(
            f"索引会越界: 需要访问到 {max_index_needed}, 但数据集长度仅 {dataset_len}。"
            "请减小 num_segments 或 chunk_size。"
        )

    # 对比模式：加载 policy
    policy = None
    if args.checkpoint_dir:
        logging.info("加载 policy (checkpoint) ...")
        policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)

    # -------------------------------------------------------------------------
    # Step 2: 采集 GT，对比模式下同时推理
    # -------------------------------------------------------------------------
    logging.info("Step 2: 采集数据 ...")
    all_gt_actions = []
    all_inferred_actions = [] if policy else None

    for i in range(args.num_segments):
        sample_idx = i * args.chunk_size
        data = dataset[sample_idx]
        gt_actions = data["actions"].squeeze()
        all_gt_actions.append(gt_actions)

        if policy is not None:
            inferred = policy.infer(data)["actions"]
            all_inferred_actions.append(inferred)
            if i == 0:
                logging.info(_array_stats("gt_actions[0]", np.asarray(gt_actions)))
                logging.info(_array_stats("inferred_actions[0]", np.asarray(inferred)))

        if (i + 1) % max(1, args.num_segments // 10) == 0 or (i + 1) == args.num_segments:
            logging.info("进度: %d/%d (sample_idx=%d)", i + 1, args.num_segments, sample_idx)

    if not all_gt_actions:
        logging.warning("未采集到任何数据，退出。")
        return

    # -------------------------------------------------------------------------
    # Step 3: 绘图并保存
    # -------------------------------------------------------------------------
    logging.info("Step 3: 绘图并保存 ...")
    gt_actions_continuous = np.concatenate(all_gt_actions, axis=0)
    total_steps, num_dims = gt_actions_continuous.shape
    time_steps_per_segment = total_steps // args.num_segments
    x_axis = np.arange(total_steps)

    fig, axes = plt.subplots(7, 2, figsize=(20, 28), sharex=True)
    axes = axes.flatten()

    for dim_idx in range(num_dims):
        ax = axes[dim_idx]
        ax.plot(
            x_axis,
            gt_actions_continuous[:, dim_idx],
            label="Ground Truth",
            color="cornflowerblue",
            alpha=0.9,
        )
        start_indices = np.arange(0, total_steps, time_steps_per_segment)
        ax.scatter(
            start_indices,
            gt_actions_continuous[start_indices, dim_idx],
            c="blue",
            marker="o",
            s=40,
            zorder=5,
            label="Segment Start",
        )

        if all_inferred_actions is not None:
            inferred_continuous = np.concatenate(all_inferred_actions, axis=0)
            ax.plot(
                x_axis,
                inferred_continuous[:, dim_idx],
                label="Inferred",
                color="tomato",
                linestyle="--",
                alpha=0.9,
            )
            ax.scatter(
                start_indices,
                inferred_continuous[start_indices, dim_idx],
                c="darkred",
                marker="x",
                s=40,
                zorder=5,
                label="Inferred Start",
            )

        ax.set_title(f"Action Dimension {dim_idx}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    fig.supxlabel(f"Continuous Timestep (across {args.num_segments} segments)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    if all_inferred_actions is not None:
        steps = int(os.path.basename(args.checkpoint_dir))
        fig.suptitle(f"GT vs Inferred Actions @ Step {steps} — {config_name}", fontsize=18)
        default_output = os.path.join(PLOT_ACTIONS_OUTPUT_DIR, "compare", f"{config_name}_{steps}.png")
    else:
        fig.suptitle(f"Ground Truth Actions — {config_name}", fontsize=18)
        default_output = os.path.join(PLOT_ACTIONS_OUTPUT_DIR, "gt", f"{config_name}.png")

    output_path = args.output_path or default_output
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info("图像已保存 -> %s", output_path)
    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)
    logging.info("完成，总耗时 %.2fs", time.time() - t0)


if __name__ == "__main__":
    main(tyro.cli(Args))
