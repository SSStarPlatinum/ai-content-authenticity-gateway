"""
ELA (Error Level Analysis) 检测器：无权重，基于 JPEG 压缩误差分析。

原理：
  将原图以固定质量重新保存为 JPEG，计算原图与重压缩图的像素差异（ELA 图）。
  - 真实照片：不同区域压缩误差差异大（平滑区域误差小，纹理/边缘区域误差大）
  - AI 生成图像：压缩误差分布更均匀（生成模型倾向于产生"过度平滑"的统计特性）

判决依据：ELA 图的方差、高频能量占比、空间不均匀性。
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult
from .registry import register_model


@register_model("ela_detector")
class ELADetector(BaseDetector):
    """Error Level Analysis fake detector (weight-free)."""

    model_name = "ELA Detector (Compression Error)"
    version = "1.0.0"
    default_weight = 0.7
    strong_postprocessing = ["jpeg", "none"]

    # ELA 重压缩质量（越低，差异越明显，但噪声也越大）
    ELA_QUALITY = 90
    # 分析时的统一尺寸
    ANALYSIS_SIZE = (256, 256)

    def _load(self) -> None:
        self._loaded = True

    def _predict(self, img: Image.Image) -> DetectionResult:
        ela_map = self._compute_ela(img)

        # 特征1：ELA 全局均值（AI 生成通常整体误差偏低或偏高）
        ela_mean = float(ela_map.mean())

        # 特征2：ELA 方差（真实照片方差大，AI 生成方差小）
        ela_std = float(ela_map.std())

        # 特征3：高频能量占比（ELA 图的高频分量）
        h, w = ela_map.shape
        # 分块计算局部方差，衡量空间不均匀性
        block_size = 16
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        block_vars = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                block = ela_map[
                    i * block_size:(i + 1) * block_size,
                    j * block_size:(j + 1) * block_size,
                ]
                block_vars.append(float(block.var()))
        block_vars = np.array(block_vars)
        # 不均匀性 = 块间方差的变异系数
        spatial_unevenness = float(block_vars.std() / (block_vars.mean() + 1e-8))

        # 特征4：高误差像素占比（超过均值+1std 的像素比例）
        threshold = ela_mean + ela_std
        high_error_ratio = float((ela_map > threshold).mean())

        # 多特征打分
        # AI 生成图像特征：低方差、低空间不均匀性、高误差像素占比异常
        score = 0.0
        # 方差越低越可能是 AI 生成（归一化到 [0,1]）
        score += 0.35 * (1.0 - min(ela_std / 15.0, 1.0))
        # 空间不均匀性越低越可能是 AI 生成
        score += 0.30 * (1.0 - min(spatial_unevenness / 1.5, 1.0))
        # 高误差像素占比异常（过高或过低都可疑）
        score += 0.20 * min(abs(high_error_ratio - 0.15) * 3, 1.0)
        # 全局均值异常
        score += 0.15 * min(abs(ela_mean - 5.0) / 10.0, 1.0)

        fake_prob = float(np.clip(score, 0.0, 1.0))
        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "ela_mean": round(ela_mean, 4),
                "ela_std": round(ela_std, 4),
                "spatial_unevenness": round(spatial_unevenness, 4),
                "high_error_ratio": round(high_error_ratio, 4),
                "ela_quality": self.ELA_QUALITY,
            },
        )

    def _compute_ela(self, img: Image.Image) -> np.ndarray:
        """计算 ELA 误差图。"""
        # 统一尺寸
        src = img.convert("RGB").resize(self.ANALYSIS_SIZE, Image.BILINEAR)
        original = np.asarray(src, dtype=np.float32)

        # 以固定质量重新压缩
        buf = io.BytesIO()
        src.save(buf, format="JPEG", quality=self.ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        recompressed_arr = np.asarray(recompressed, dtype=np.float32)

        # 计算绝对误差，取 RGB 通道均值
        diff = np.abs(original - recompressed_arr)
        ela_map = diff.mean(axis=2)
        return ela_map
