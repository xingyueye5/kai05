import dataclasses
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.training import config_loader
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_path: str
    dataset_window_size: int = 50
    max_samples: int = 10
    chunk_size: int = 50
    output_path: str = ""
    log_level: str = "INFO"
    show_plot: bool = False


def main(args: Args) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    t0 = time.time()
    logging.info("启动 GT 可视化脚本")
    logging.info("config_path=%s", args.config_path)

    logging.info("Step 1/3: 加载 config 与数据集 ...")
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
    max_index_needed = (args.max_samples - 1) * args.chunk_size
    logging.info("dataset_len=%d, max_index_needed=%d", dataset_len, max_index_needed)
    if max_index_needed >= dataset_len:
        raise ValueError(
            f"索引会越界: 需要访问到 {max_index_needed}, 但数据集长度仅 {dataset_len}。"
            "请减小 max_samples 或 chunk_size。"
        )

    logging.info("Step 2/3: 采集 GT actions ...")
    all_gt_actions = []
    for i in range(args.max_samples):
        sample_idx = i * args.chunk_size
        data = dataset[sample_idx]
        gt_actions = data["actions"].squeeze()
        all_gt_actions.append(gt_actions)
        if (i + 1) % max(1, args.max_samples // 10) == 0 or (i + 1) == args.max_samples:
            logging.info("采集进度: %d/%d (sample_idx=%d)", i + 1, args.max_samples, sample_idx)

    logging.info("Step 3/3: 绘图 ...")
    gt_actions_continuous = np.concatenate(all_gt_actions, axis=0)
    total_steps, num_dims = gt_actions_continuous.shape
    time_steps_per_sample = total_steps // args.max_samples

    fig, axes = plt.subplots(7, 2, figsize=(20, 28), sharex=True)
    axes = axes.flatten()
    x_axis = np.arange(total_steps)

    for dim_idx in range(num_dims):
        ax = axes[dim_idx]
        ax.plot(
            x_axis,
            gt_actions_continuous[:, dim_idx],
            label="Ground Truth",
            color="cornflowerblue",
            alpha=0.9,
        )
        start_indices = np.arange(0, total_steps, time_steps_per_sample)
        ax.scatter(
            start_indices,
            gt_actions_continuous[start_indices, dim_idx],
            c="blue",
            marker="o",
            s=40,
            zorder=5,
            label="Sample Start",
        )
        ax.set_title(f"Action Dimension {dim_idx}")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    fig.supxlabel(f"Continuous Timestep (across {args.max_samples} samples)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    fig.suptitle(f"Ground Truth Actions — {config_name}", fontsize=18)

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f"gt_actions_{config_name}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logging.info("图像已保存 -> %s", output_path)
    if args.show_plot:
        plt.show()
    else:
        plt.close(fig)
    logging.info("完成，总耗时 %.2fs", time.time() - t0)


if __name__ == "__main__":
    main(tyro.cli(Args))
