"""
图像处理工具：统一预处理、格式校验、后处理鲁棒性模拟。
所有模型适配器共享此模块，保证输入一致性。
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

# 支持的图片格式
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


@dataclass
class ImageMeta:
    """图片元信息，用于日志与可观测性。"""
    width: int
    height: int
    mode: str
    format: str
    size_bytes: int


def load_image(data: bytes) -> Tuple[Image.Image, ImageMeta]:
    """
    从字节流加载图片，自动处理 EXIF 方向、转 RGB。
    抛出 ValueError 表示非法图片。
    """
    if not data:
        raise ValueError("empty image data")

    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format or "UNKNOWN"
        if fmt.upper() not in SUPPORTED_FORMATS:
            raise ValueError(f"unsupported format: {fmt}")
        # 修正 EXIF 方向
        img = ImageOps.exif_transpose(img)
        # 统一为 RGB（丢弃 alpha 通道，避免模型输入维度不一致）
        img = img.convert("RGB")
        meta = ImageMeta(
            width=img.width,
            height=img.height,
            mode=img.mode,
            format=fmt,
            size_bytes=len(data),
        )
        return img, meta
    except (ValueError, OSError) as e:
        raise ValueError(f"invalid image: {e}") from e


def resize_to_tensor(
    img: Image.Image,
    size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    标准 ImageNet 归一化预处理，输出 NCHW float32。
    大多数开源检测模型基于此预处理。
    """
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    # 归一化
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean_arr) / std_arr
    # 增加 batch 维
    return np.expand_dims(arr, axis=0).astype(np.float32)


def compute_frequency_spectrum(img: Image.Image, max_size: int = 256) -> np.ndarray:
    """
    计算灰度图的对数幅度谱（频域特征），供频域检测器使用。
    返回中心化后的 2D 频谱（已 log 压缩）。
    """
    gray = img.convert("L")
    # 缩放到统一尺寸，保证频谱维度一致
    w, h = gray.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    gray = gray.resize(new_size, Image.BILINEAR)

    arr = np.asarray(gray, dtype=np.float32) / 255.0
    # 汉宁窗减少频谱泄漏
    window = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    arr = arr * window
    fft = np.fft.fft2(arr)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    # 对数压缩，避免动态范围过大
    log_mag = np.log1p(magnitude)
    # 归一化到 [0, 1]
    log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    return log_mag.astype(np.float32)


def simulate_postprocessing(
    img: Image.Image,
    jpeg_quality: Optional[int] = None,
    resize_scale: Optional[float] = None,
) -> Image.Image:
    """
    模拟社交平台常见后处理（JPEG 压缩 + 缩放），用于鲁棒性测试。
    生产链路不调用，仅在评估脚本中使用。
    """
    result = img
    if resize_scale and resize_scale != 1.0:
        new_w = max(1, int(img.width * resize_scale))
        new_h = max(1, int(img.height * resize_scale))
        result = result.resize((new_w, new_h), Image.BILINEAR)
    if jpeg_quality is not None:
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        result = Image.open(buf).convert("RGB")
    return result
