import cv2
from pathlib import Path
import os

os.environ["SVT_LOG"] = "1"

from typing import List
from pathlib import Path
import shutil
import logging
from functools import partial

import numpy as np
import av

# code from https://codeup.aliyun.com/68a82b32efefeec54ed5eb5f/kai-vla/mini_lerobot/blob/master/convert_mcap.py
def encode_video_frames(
        images: np.ndarray | List[np.ndarray], 
        dst: Path,
        fps: int,
        vcodec: str = "libsvtav1",
        pix_fmt: str = "yuv420p",
        logger: logging.Logger = None,
        g: int | None = 2,
        crf: int | None = 30,
        fast_decode: int = 0,
        log_level: int | None = av.logging.ERROR,
        overwrite: bool = False,
        color_format: str = "rgb",
) -> bytes:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["libx264", "h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(dst)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        if logger is not None:
            logger.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        else:
            print(
                f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
            )
        pix_fmt = "yuv420p"

    # Define video output frame size (assuming all input frames are the same size)

    dummy_image = images[0]
    height, width, _ = dummy_image.shape
    # print('width, height', width, height)

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    # if log_level is not None:
    #     # "While less efficient, it is generally preferable to modify logging with Python’s logging"
    #     logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_image in images:
            # input_image = Image.open(input_data).convert("RGB")
            # input_frame = av.VideoFrame.from_image(input_image)
            if color_format.lower() == "bgr":
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_frame = av.VideoFrame.from_ndarray(input_image, format="rgb24", channel_last=True)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def save_video_by_cv(images: np.ndarray | List[np.ndarray], dst: Path, fps: int, logger: logging.Logger, overwrite: bool = False):
    logger.info(f'save video to {dst}')
    video_path = Path(dst)
    
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 H.264 编码（更通用）
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (images.shape[2], images.shape[1]))
    # 添加检查
    if not out.isOpened():
        logger.error(f'Failed to open VideoWriter for {video_path}')
        raise ValueError(f'Failed to open VideoWriter for {video_path}')
    try:
        for image in images:
            out.write(image)
        out.release()
        logger.info(f'save video {video_path} done')
    except Exception as e:
        logger.error(f'Failed to save video {video_path}: {e}')
        raise e
    return video_path