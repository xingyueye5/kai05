"""
预测结果可视化工具：按 key（parquet 列名）读取数值，生成曲线图与叠加视频。

使用示例：
python visualize_VC_value.py /path/to/dataset --episode 0 --value_source top_head
python visualize_VC_value.py /path/to/dataset --episode 0 --value_source top_head --camera observation.images.top_head
python visualize_VC_value.py /path/to/dataset --list_columns   # 查看 parquet 列名
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm


def get_episode_paths(dataset_path: Path, episode_idx: int, camera_key: str, chunk_size: int = 1000):
    """根据 episode 索引获取相关文件路径"""
    chunk_idx = episode_idx // chunk_size
    episode_id = f"episode_{episode_idx:06d}"
    chunk_name = f"chunk-{chunk_idx:03d}"
    
    # 预测结果 parquet（改为 data 目录）
    parquet_path = dataset_path / "data" / chunk_name / f"{episode_id}.parquet"
    
    # 视频文件
    video_path = dataset_path / "videos" / chunk_name / camera_key / f"{episode_id}.mp4"
    
    return {
        "episode_id": episode_id,
        "chunk_name": chunk_name,
        "parquet_path": parquet_path,
        "video_path": video_path,
    }


def load_progress_data(parquet_path: Path, value_source: str):
    """加载 progress 数据。value_source 即 parquet 中的列名（key）。"""
    df = pd.read_parquet(parquet_path)
    if value_source not in df.columns:
        raise ValueError(f"列 '{value_source}' 不存在。可用列: {list(df.columns)}")
    return {
        "frame_index": df["frame_index"].values,
        "progress_gt": df["progress_gt"].values,
        "progress_predicted": df[value_source].values,
        "column_name": value_source,
        "key": value_source,
    }


def plot_progress_curve(progress_data: dict, output_path: Path, episode_id: str):
    """绘制进度曲线对比图"""
    frame_idx = progress_data["frame_index"]
    progress_gt = progress_data["progress_gt"]
    progress_pred = progress_data["progress_predicted"]
    
    # 计算误差
    mae = np.mean(np.abs(progress_pred - progress_gt))
    rmse = np.sqrt(np.mean((progress_pred - progress_gt) ** 2))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 上图：进度曲线对比
    ax1 = axes[0]
    ax1.plot(frame_idx, progress_gt, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(frame_idx, progress_pred, 'r-', linewidth=2, label='Predicted', alpha=0.8)
    ax1.set_ylabel('Progress', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_title(f'{episode_id} - Progress Comparison\nMAE: {mae:.4f}, RMSE: {rmse:.4f}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 下图：误差曲线
    ax2 = axes[1]
    error = progress_pred - progress_gt
    ax2.fill_between(frame_idx, error, 0, alpha=0.5, color='orange', label='Error (Pred - GT)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_ylim(-0.5, 0.5)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"进度曲线图已保存: {output_path}")


def create_progress_overlay(frame: np.ndarray, progress_gt: float, progress_pred: float, 
                           current_frame: int, total_frames: int) -> np.ndarray:
    """在视频帧上叠加进度信息"""
    h, w = frame.shape[:2]
    
    # 创建叠加层
    overlay = frame.copy()
    
    # 绘制进度条背景
    bar_height = 20
    bar_y = h - 50
    bar_margin = 20
    bar_width = w - 2 * bar_margin
    
    # 背景条
    cv2.rectangle(overlay, (bar_margin, bar_y), (bar_margin + bar_width, bar_y + bar_height), 
                  (50, 50, 50), -1)
    
    # Ground Truth 进度条 (蓝色)
    gt_width = int(bar_width * progress_gt)
    cv2.rectangle(overlay, (bar_margin, bar_y), (bar_margin + gt_width, bar_y + bar_height // 2), 
                  (255, 100, 100), -1)
    
    # Predicted 进度条 (红色)
    pred_width = int(bar_width * progress_pred)
    cv2.rectangle(overlay, (bar_margin, bar_y + bar_height // 2), 
                  (bar_margin + pred_width, bar_y + bar_height), 
                  (100, 100, 255), -1)
    
    # 混合叠加
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 帧信息
    text_frame = f"Frame: {current_frame}/{total_frames}"
    cv2.putText(result, text_frame, (bar_margin, bar_y - 35), font, font_scale, (255, 255, 255), thickness)
    
    # 进度信息
    text_gt = f"GT: {progress_gt:.3f}"
    text_pred = f"Pred: {progress_pred:.3f}"
    text_error = f"Error: {progress_pred - progress_gt:+.3f}"
    
    cv2.putText(result, text_gt, (bar_margin, bar_y - 10), font, font_scale, (255, 150, 150), thickness)
    cv2.putText(result, text_pred, (bar_margin + 150, bar_y - 10), font, font_scale, (150, 150, 255), thickness)
    cv2.putText(result, text_error, (bar_margin + 300, bar_y - 10), font, font_scale, (255, 255, 255), thickness)
    
    # 标签
    cv2.putText(result, "GT", (bar_margin + bar_width + 5, bar_y + bar_height // 2 - 2), 
                font, 0.4, (255, 150, 150), 1)
    cv2.putText(result, "Pred", (bar_margin + bar_width + 5, bar_y + bar_height - 2), 
                font, 0.4, (150, 150, 255), 1)
    
    return result


def create_side_by_side_video(
    video_path: Path,
    progress_data: dict,
    output_path: Path,
    episode_id: str,
    fps: int = 30
):
    """创建带进度曲线的并排视频（使用 OpenCV 高效渲染）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 输出尺寸：视频 + 曲线图
    chart_width = 400
    output_width = frame_width + chart_width
    output_height = frame_height
    
    # 创建视频写入器
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (output_width, output_height))
    
    # 准备数据
    progress_gt = progress_data["progress_gt"]
    progress_pred = progress_data["progress_predicted"]
    
    # 计算误差统计
    mae = np.mean(np.abs(progress_pred - progress_gt))
    rmse = np.sqrt(np.mean((progress_pred - progress_gt) ** 2))
    
    # 创建高效曲线图渲染器（预计算所有静态元素）
    chart_renderer = ChartRenderer(
        progress_gt, progress_pred,
        chart_size=(chart_width, output_height),
        episode_id=episode_id,
        mae=mae, rmse=rmse
    )
    
    print(f"生成视频: {total_frames} 帧, {video_fps:.1f} fps")
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="生成视频")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 查找当前帧的 progress 数据
        if frame_idx < len(progress_gt):
            gt = progress_gt[frame_idx]
            pred = progress_pred[frame_idx]
        else:
            gt = progress_gt[-1]
            pred = progress_pred[-1]
        
        # 在视频帧上叠加进度条
        frame_with_overlay = create_progress_overlay(frame, gt, pred, frame_idx, total_frames)
        
        # 使用高效渲染器创建曲线图
        chart = chart_renderer.render(frame_idx)
        
        # 合并视频和曲线图
        combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        combined[:, :frame_width] = frame_with_overlay
        combined[:, frame_width:] = chart
        
        out.write(combined)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    print(f"视频已保存: {output_path}")


class ChartRenderer:
    """
    使用 OpenCV 高效绘制曲线图（比 matplotlib 快 50 倍以上）
    预计算静态元素，每帧只更新动态部分
    """
    
    def __init__(self, progress_gt, progress_pred, chart_size, episode_id, mae, rmse):
        self.width, self.height = chart_size
        self.progress_gt = progress_gt
        self.progress_pred = progress_pred
        self.error = progress_pred - progress_gt
        self.total_frames = len(progress_gt)
        self.episode_id = episode_id
        self.mae = mae
        self.rmse = rmse
        
        # 颜色定义 (BGR)
        self.bg_color = (26, 26, 26)
        self.gt_color = (255, 255, 0)      # 青色
        self.pred_color = (0, 165, 255)    # 橙色
        self.error_color = (0, 165, 255)   # 橙色
        self.grid_color = (60, 60, 60)
        self.text_color = (255, 255, 255)
        self.line_color = (200, 200, 200)
        
        # 图表区域定义
        self.margin_left = 50
        self.margin_right = 20
        self.margin_top = 50
        self.margin_bottom = 30
        self.chart_gap = 40
        
        # 上下两个图的区域
        half_height = (self.height - self.margin_top - self.margin_bottom - self.chart_gap) // 2
        self.chart1_top = self.margin_top
        self.chart1_bottom = self.margin_top + half_height
        self.chart2_top = self.chart1_bottom + self.chart_gap
        self.chart2_bottom = self.chart2_top + half_height
        
        self.chart_left = self.margin_left
        self.chart_right = self.width - self.margin_right
        self.chart_width = self.chart_right - self.chart_left
        
        # 预计算曲线像素坐标
        self._precompute_curves()
        
        # 预渲染静态背景
        self._render_static_background()
    
    def _precompute_curves(self):
        """预计算曲线的像素坐标"""
        x_coords = np.linspace(self.chart_left, self.chart_right, self.total_frames).astype(np.int32)
        
        # 上图：progress (0-1 映射到 chart1_bottom - chart1_top)
        chart1_height = self.chart1_bottom - self.chart1_top
        gt_y = (self.chart1_bottom - (self.progress_gt * chart1_height * 0.9 + chart1_height * 0.05)).astype(np.int32)
        pred_y = (self.chart1_bottom - (self.progress_pred * chart1_height * 0.9 + chart1_height * 0.05)).astype(np.int32)
        
        self.gt_points = np.column_stack([x_coords, gt_y])
        self.pred_points = np.column_stack([x_coords, pred_y])
        
        # 下图：error (-0.3 到 0.3 映射)
        chart2_height = self.chart2_bottom - self.chart2_top
        chart2_mid = (self.chart2_top + self.chart2_bottom) // 2
        error_y = (chart2_mid - (self.error / 0.3 * chart2_height * 0.4)).astype(np.int32)
        error_y = np.clip(error_y, self.chart2_top, self.chart2_bottom)
        
        self.error_points = np.column_stack([x_coords, error_y])
        self.error_zero_y = chart2_mid
        self.x_coords = x_coords
    
    def _render_static_background(self):
        """预渲染静态背景（网格、标签等）"""
        self.static_bg = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 标题
        title = f"{self.episode_id}"
        cv2.putText(self.static_bg, title, (self.margin_left, 20), font, 0.5, self.text_color, 1)
        stats = f"MAE: {self.mae:.4f}  RMSE: {self.rmse:.4f}"
        cv2.putText(self.static_bg, stats, (self.margin_left, 38), font, 0.45, (180, 180, 180), 1)
        
        # 上图网格和边框
        cv2.rectangle(self.static_bg, (self.chart_left, self.chart1_top), 
                      (self.chart_right, self.chart1_bottom), self.grid_color, 1)
        for i in range(1, 5):
            y = self.chart1_top + (self.chart1_bottom - self.chart1_top) * i // 5
            cv2.line(self.static_bg, (self.chart_left, y), (self.chart_right, y), self.grid_color, 1)
        
        # 上图 Y 轴标签
        cv2.putText(self.static_bg, "1.0", (5, self.chart1_top + 10), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, "0.0", (5, self.chart1_bottom), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, "Progress", (5, (self.chart1_top + self.chart1_bottom) // 2), font, 0.35, self.text_color, 1)
        
        # 图例
        legend_y = self.chart1_top + 15
        cv2.line(self.static_bg, (self.chart_right - 80, legend_y), (self.chart_right - 60, legend_y), self.gt_color, 2)
        cv2.putText(self.static_bg, "GT", (self.chart_right - 55, legend_y + 4), font, 0.35, self.gt_color, 1)
        cv2.line(self.static_bg, (self.chart_right - 80, legend_y + 15), (self.chart_right - 60, legend_y + 15), self.pred_color, 2)
        cv2.putText(self.static_bg, "Pred", (self.chart_right - 55, legend_y + 19), font, 0.35, self.pred_color, 1)
        
        # 下图网格和边框
        cv2.rectangle(self.static_bg, (self.chart_left, self.chart2_top), 
                      (self.chart_right, self.chart2_bottom), self.grid_color, 1)
        cv2.line(self.static_bg, (self.chart_left, self.error_zero_y), 
                 (self.chart_right, self.error_zero_y), (100, 100, 100), 1)
        
        # 下图 Y 轴标签
        cv2.putText(self.static_bg, "+0.3", (5, self.chart2_top + 10), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, "0", (20, self.error_zero_y + 4), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, "-0.3", (5, self.chart2_bottom), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, "Error", (5, self.error_zero_y - 20), font, 0.35, self.text_color, 1)
        
        # X 轴标签
        cv2.putText(self.static_bg, "0", (self.chart_left, self.chart2_bottom + 15), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, str(self.total_frames), (self.chart_right - 30, self.chart2_bottom + 15), font, 0.35, self.text_color, 1)
        cv2.putText(self.static_bg, "Frame", ((self.chart_left + self.chart_right) // 2 - 20, self.chart2_bottom + 15), font, 0.35, self.text_color, 1)
    
    def render(self, current_frame: int) -> np.ndarray:
        """渲染当前帧的曲线图"""
        # 复制静态背景
        img = self.static_bg.copy()
        
        # 绘制曲线（使用 polylines 一次性绘制，很快）
        cv2.polylines(img, [self.gt_points], False, self.gt_color, 2, cv2.LINE_AA)
        cv2.polylines(img, [self.pred_points], False, self.pred_color, 2, cv2.LINE_AA)
        
        # 绘制误差填充区域
        fill_points = []
        for i in range(self.total_frames):
            fill_points.append([self.x_coords[i], self.error_zero_y])
        for i in range(self.total_frames - 1, -1, -1):
            fill_points.append([self.x_coords[i], self.error_points[i, 1]])
        fill_points = np.array(fill_points, dtype=np.int32)
        cv2.fillPoly(img, [fill_points], (0, 100, 180))
        cv2.polylines(img, [self.error_points], False, self.error_color, 1, cv2.LINE_AA)
        
        # 绘制当前帧指示线
        if current_frame < self.total_frames:
            x = self.x_coords[current_frame]
            cv2.line(img, (x, self.chart1_top), (x, self.chart1_bottom), self.line_color, 1)
            cv2.line(img, (x, self.chart2_top), (x, self.chart2_bottom), self.line_color, 1)
            
            # 当前点
            cv2.circle(img, (x, self.gt_points[current_frame, 1]), 5, self.gt_color, -1)
            cv2.circle(img, (x, self.pred_points[current_frame, 1]), 5, self.pred_color, -1)
            cv2.circle(img, (x, self.error_points[current_frame, 1]), 5, (0, 0, 255), -1)
        
        return img


def list_available_episodes(dataset_path: Path, chunk_size: int = 1000):
    """列出可用的 episode"""
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        print(f"目录不存在: {data_dir}")
        return []
    
    episodes = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
            for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
                episode_id = parquet_file.stem
                episode_idx = int(episode_id.split("_")[1])
                episodes.append(episode_idx)
    
    return sorted(episodes)


def list_available_cameras(dataset_path: Path):
    """列出可用的相机"""
    videos_dir = dataset_path / "videos"
    cameras = []
    
    # 查找第一个 chunk 目录下的相机
    for chunk_dir in sorted(videos_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
            for camera_dir in chunk_dir.iterdir():
                if camera_dir.is_dir():
                    cameras.append(camera_dir.name)
            break
    
    return sorted(cameras)


def main():
    parser = argparse.ArgumentParser(
        description="Progress 预测结果可视化工具",
        epilog="示例: python 03_visualize_progress.py /path/to/dataset --episode 0"
    )
    
    parser.add_argument("dataset_path", type=str, help="数据集路径")
    parser.add_argument("--episode", type=int, default=None, help="要可视化的 episode 索引")
    parser.add_argument("--camera", type=str, default="observation.images.top_head", 
                        help="用于显示视频的相机名称 (默认: observation.images.top_head)")
    parser.add_argument("--value_source", type=str, default=None,
                        help="要可视化的列名（key），即 parquet 中的列名，可视化时必填")
    parser.add_argument("--chunk_size", type=int, default=1000, help="chunk 大小 (默认: 1000)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录 (默认: dataset_path/visualizations)")
    parser.add_argument("--list_episodes", action="store_true", help="列出可用的 episode")
    parser.add_argument("--list_cameras", action="store_true", help="列出可用的相机")
    parser.add_argument("--list_columns", action="store_true", help="列出 parquet 中的列名")
    parser.add_argument("--no_video", action="store_true", help="只生成曲线图，不生成视频")
    parser.add_argument("--no_plot", action="store_true", help="只生成视频，不生成曲线图")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # 列出可用的 episode
    if args.list_episodes:
        episodes = list_available_episodes(dataset_path, args.chunk_size)
        print(f"可用的 episode ({len(episodes)} 个):")
        if len(episodes) <= 20:
            print(episodes)
        else:
            print(f"  前 10 个: {episodes[:10]}")
            print(f"  后 10 个: {episodes[-10:]}")
        return
    
    # 列出可用的相机
    if args.list_cameras:
        cameras = list_available_cameras(dataset_path)
        print(f"可用的相机 ({len(cameras)} 个):")
        for c in cameras:
            print(f"  - {c}")
        return
    
    # 列出 parquet 列名
    if args.list_columns:
        data_dir = dataset_path / "data"
        if not data_dir.exists():
            print(f"目录不存在: {data_dir}")
            return
        for chunk_dir in sorted(data_dir.iterdir()):
            if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
                    df = pd.read_parquet(parquet_file)
                    print(f"列名 ({len(df.columns)} 个):")
                    for col in df.columns:
                        print(f"  - {col}")
                    return
        print("没有找到 parquet 文件")
        return
    
    # 检查 episode 和 value_source 参数
    if args.episode is None:
        print("请指定 --episode 参数，或使用 --list_episodes 查看可用的 episode")
        return
    if args.value_source is None:
        print("请指定 --value_source 参数（key，如 top_head）")
        return
    
    # 获取路径
    paths = get_episode_paths(dataset_path, args.episode, args.camera, args.chunk_size)
    episode_id = paths["episode_id"]
    
    print(f"数据集: {dataset_path.name}")
    print(f"Episode: {episode_id}")
    print(f"视频相机: {args.camera}")
    print(f"value_source (key): {args.value_source}")
    
    # 检查文件是否存在
    if not paths["parquet_path"].exists():
        print(f"Parquet 文件不存在: {paths['parquet_path']}")
        return
    
    if not paths["video_path"].exists():
        print(f"视频文件不存在: {paths['video_path']}")
        if not args.no_video:
            print("将只生成曲线图")
            args.no_video = True
    
    # 加载 progress 数据
    print(f"\n加载数据: {paths['parquet_path']}")
    progress_data = load_progress_data(paths["parquet_path"], args.value_source)
    print(f"  使用列: {progress_data['column_name']}")
    print(f"  帧数: {len(progress_data['frame_index'])}")
    print(f"  progress_gt 范围: [{progress_data['progress_gt'].min():.4f}, {progress_data['progress_gt'].max():.4f}]")
    print(f"  数值范围: [{progress_data['progress_predicted'].min():.4f}, {progress_data['progress_predicted'].max():.4f}]")
    
    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    key = progress_data['key']
    
    # 生成曲线图
    if not args.no_plot:
        plot_path = output_dir / f"{episode_id}_{key}.jpg"
        print(f"\n生成曲线图...")
        plot_progress_curve(progress_data, plot_path, episode_id)
    
    # 生成视频
    if not args.no_video:
        video_output_path = output_dir / f"{episode_id}_{key}.mp4"
        print(f"\n生成视频...")
        create_side_by_side_video(
            paths["video_path"],
            progress_data,
            video_output_path,
            episode_id
        )
    
    print("\n完成!")


if __name__ == "__main__":
    main()
