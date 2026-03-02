import dataclasses
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config_loader
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args:
    # 训练配置 YAML 路径，例如 "configs/train/pi05_aloha.yaml"
    config_path: str
    # checkpoint 目录，例如 ".../1022/10000"
    checkpoint_dir: str
    # dataset window size（原脚本固定为 50）
    dataset_window_size: int = 50
    # 采样推理次数
    max_inferences: int = 10
    # 每次推理采样间隔
    chunk_size: int = 50
    # 日志级别: INFO / DEBUG
    log_level: str = "INFO"
    # 是否显示图像窗口（仅保存时可设为 False）
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
    logging.info("启动 openloop 对比脚本")
    logging.info("config_path=%s", args.config_path)
    logging.info("checkpoint_dir=%s", args.checkpoint_dir)

    logging.info("Step 1/5: 加载 config 与 policy ...")
    config = config_loader.load_config(args.config_path)
    policy = _policy_config.create_trained_policy(config, args.checkpoint_dir)

    ckpt_root = os.path.dirname(args.checkpoint_dir)
    steps = int(os.path.basename(args.checkpoint_dir))

    logging.info("Step 2/5: 构建数据集 ...")
    data_config = config.data.create(config.assets_dirs, config.model)
    base_dataset = _data_loader.create_torch_dataset(
        data_config, args.dataset_window_size, config.model
    )
    dataset = _data_loader.TransformedDataset(
        base_dataset,
        [
            *data_config.repack_transforms.inputs,
            # *data_config.data_transforms.inputs,  # test data feed to model
            # *data_config.model_transforms.inputs, # test data feed to model
        ],
    )
    dataset_len = len(dataset)
    max_index_needed = (args.max_inferences - 1) * args.chunk_size
    logging.info("dataset_len=%d, max_index_needed=%d", dataset_len, max_index_needed)
    if max_index_needed >= dataset_len:
        raise ValueError(
            f"索引会越界: 需要访问到 {max_index_needed}, 但数据集长度仅 {dataset_len}。"
            "请减小 max_inferences 或 chunk_size。"
        )

    logging.info("Step 3/5: 开始推理与采样 ...")
    all_gt_actions = []
    all_inferred_actions = []
    for i in range(args.max_inferences):
        sample_idx = i * args.chunk_size
        data = dataset[sample_idx]

        inferred_actions = policy.infer(data)["actions"]  # Shape: (50, 14)
        gt_actions = data["actions"].squeeze()  # Shape: (50, 14)

        if i == 0:
            logging.info(_array_stats("gt_actions[0]", np.asarray(gt_actions)))
            logging.info(_array_stats("inferred_actions[0]", np.asarray(inferred_actions)))
        if (i + 1) % max(1, args.max_inferences // 10) == 0 or (i + 1) == args.max_inferences:
            logging.info("推理进度: %d/%d (sample_idx=%d)", i + 1, args.max_inferences, sample_idx)

        all_inferred_actions.append(inferred_actions)
        all_gt_actions.append(gt_actions)

    if all_gt_actions:
        logging.info("Step 4/5: 整理数据并绘图 ...")
        gt_actions_continuous = np.concatenate(all_gt_actions, axis=0)
        inferred_actions_continuous = np.concatenate(all_inferred_actions, axis=0)

        total_steps, num_dims = gt_actions_continuous.shape
        time_steps_per_inference = gt_actions_continuous.shape[0] // args.max_inferences

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
            ax.plot(
                x_axis,
                inferred_actions_continuous[:, dim_idx],
                label="Inferred",
                color="tomato",
                linestyle="--",
                alpha=0.9,
            )

            start_indices = np.arange(0, total_steps, time_steps_per_inference)
            ax.scatter(
                start_indices,
                gt_actions_continuous[start_indices, dim_idx],
                c="blue",
                marker="o",
                s=40,
                zorder=5,
                label="GT Start",
            )
            ax.scatter(
                start_indices,
                inferred_actions_continuous[start_indices, dim_idx],
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

        fig.supxlabel(f"Continuous Timestep (across {args.max_inferences} inferences)")
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig.suptitle(f"Comparison of Ground Truth and Inferred Actions @Step {steps}", fontsize=18)
        output_path = f"{ckpt_root}/inferred_vs_gt_actions-{steps}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logging.info("Step 5/5: 图像已保存 -> %s", output_path)
        if args.show_plot:
            plt.show()
        else:
            plt.close(fig)
        logging.info("完成，总耗时 %.2fs", time.time() - t0)
    else:
        logging.warning("No data was collected for plotting.")


if __name__ == "__main__":
    main(tyro.cli(Args))