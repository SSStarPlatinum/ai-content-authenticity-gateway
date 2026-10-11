"""
色彩分布检测器：无权重，基于 HSV 色彩空间统计分析。

原理：
  AI 生成模型在色彩分布上与真实照片存在系统性差异：
  - 生成图像通常饱和度偏高或偏低（取决于训练数据和模型偏好）
  - 色彩丰富度（独特颜色数量）可能异常
  - 色相分布可能过于集中或过于均匀
  - 亮度-饱和度相关性模式不同

判决依据：饱和度统计、色相熵、色彩丰富度、亮度-饱和度联合分布。
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult
from .registry import register_model


@register_model("color_detector")
class ColorDistributionDetector(BaseDetector):
    """Color distribution statistical fake detector (weight-free)."""

    model_name = "Color Distribution Detector"
    version = "1.0.0"
    default_weight = 0.8
    strong_postprocessing = ["none", "jpeg", "resize"]

    ANALYSIS_SIZE = (256, 256)

    def _load(self) -> None:
        self._loaded = True

    def _predict(self, img: Image.Image) -> DetectionResult:
        arr = np.asarray(
            img.convert("RGB").resize(self.ANALYSIS_SIZE, Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0

        # 转换到 HSV 空间（向量化）
        hsv = self._rgb_to_hsv_vectorized(arr)
        h = hsv[:, :, 0]  # 色相 [0, 1]
        s = hsv[:, :, 1]  # 饱和度 [0, 1]
        v = hsv[:, :, 2]  # 亮度 [0, 1]

        # 特征1：饱和度统计
        sat_mean = float(s.mean())
        sat_std = float(s.std())
        # 高饱和像素占比（>0.6）
        high_sat_ratio = float((s > 0.6).mean())
        # 低饱和像素占比（<0.1，近灰度）
        low_sat_ratio = float((s < 0.1).mean())

        # 特征2：色相熵（色彩多样性）
        hue_entropy = self._compute_hue_entropy(h, s, threshold=0.15)

        # 特征3：色彩丰富度（量化后的唯一颜色数占比）
        color_richness = self._compute_color_richness(arr)

        # 特征4：亮度-饱和度相关性
        # 真实照片中亮部通常饱和度低（高光过曝），AI 生成可能不遵循
        vs_corr = self._correlation(v.flatten(), s.flatten())

        # 特征5：色相分布均匀性（AI 生成可能色相分布异常均匀或集中）
        hue_uniformity = self._compute_hue_uniformity(h, s, threshold=0.15)

        # 多特征打分
        score = 0.0
        # 饱和度过高或过低都可疑
        score += 0.20 * min(abs(sat_mean - 0.35) / 0.3, 1.0)
        # 高饱和像素占比异常
        score += 0.15 * min(abs(high_sat_ratio - 0.15) * 3, 1.0)
        # 色相处在异常区间（过高或过低）
        score += 0.20 * min(abs(hue_entropy - 5.5) / 3.0, 1.0)
        # 色彩丰富度异常（AI 生成可能颜色数偏少或偏多）
        score += 0.15 * min(abs(color_richness - 0.4) / 0.3, 1.0)
        # 亮度-饱和度负相关性弱（真实照片通常负相关）
        score += 0.15 * max(0.0, min((vs_corr + 0.2) / 0.3, 1.0))
        # 色相分布过于均匀
        score += 0.15 * hue_uniformity

        fake_prob = float(np.clip(score, 0.0, 1.0))
        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "saturation_mean": round(sat_mean, 4),
                "saturation_std": round(sat_std, 4),
                "high_sat_ratio": round(high_sat_ratio, 4),
                "low_sat_ratio": round(low_sat_ratio, 4),
                "hue_entropy": round(hue_entropy, 4),
                "color_richness": round(color_richness, 4),
                "value_sat_correlation": round(vs_corr, 4),
                "hue_uniformity": round(hue_uniformity, 4),
            },
        )

    @staticmethod
    def _rgb_to_hsv_vectorized(rgb: np.ndarray) -> np.ndarray:
        """向量化 RGB -> HSV 转换。"""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        delta = maxc - minc

        # 饱和度
        s = np.where(maxc > 0, delta / (maxc + 1e-8), 0.0)

        # 色相
        h = np.zeros_like(r)
        mask = delta > 1e-8
        # r 是最大值
        idx = mask & (maxc == r)
        h[idx] = ((g[idx] - b[idx]) / (delta[idx] + 1e-8)) % 6
        # g 是最大值
        idx = mask & (maxc == g)
        h[idx] = (b[idx] - r[idx]) / (delta[idx] + 1e-8) + 2
        # b 是最大值
        idx = mask & (maxc == b)
        h[idx] = (r[idx] - g[idx]) / (delta[idx] + 1e-8) + 4
        h = h / 6.0
        h = h % 1.0

        return np.stack([h, s, v], axis=2)

    @staticmethod
    def _compute_hue_entropy(hue: np.ndarray, sat: np.ndarray, threshold: float = 0.15) -> float:
        """计算有色彩像素的色相熵（忽略低饱和像素）。"""
        mask = sat > threshold
        if mask.sum() < 100:
            return 0.0
        valid_hues = hue[mask]
        # 量化到 36 个 bin（每 10 度）
        hist, _ = np.histogram(valid_hues, bins=36, range=(0, 1))
        probs = hist / (hist.sum() + 1e-8)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)

    @staticmethod
    def _compute_hue_uniformity(hue: np.ndarray, sat: np.ndarray, threshold: float = 0.15) -> float:
        """色相分布均匀性（1 = 完全均匀，0 = 高度集中）。"""
        mask = sat > threshold
        if mask.sum() < 100:
            return 0.5
        valid_hues = hue[mask]
        hist, _ = np.histogram(valid_hues, bins=12, range=(0, 1))
        probs = hist / (hist.sum() + 1e-8)
        # 均匀分布时每个 bin 概率 = 1/12，计算与均匀分布的偏差
        uniform = np.ones(12) / 12
        deviation = np.abs(probs - uniform).mean()
        # deviation=0 时 uniformity=1, deviation=1/6(最大)时 uniformity=0
        uniformity = max(0.0, 1.0 - deviation * 6.0)
        return float(uniformity)

    @staticmethod
    def _compute_color_richness(rgb: np.ndarray) -> float:
        """色彩丰富度：量化后唯一颜色数占总像素比例。"""
        # 量化到 5x5x5 = 125 色
        quantized = (rgb * 4).astype(np.int32)
        colors = quantized.reshape(-1, 3)
        # 用元组集合统计唯一颜色
        unique = set()
        for c in colors[::10]:  # 采样 10% 加速
            unique.add((c[0], c[1], c[2]))
        richness = len(unique) / 125.0
        return float(min(richness, 1.0))

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray) -> float:
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
        if denom < 1e-8:
            return 0.0
        return float((a_c * b_c).sum() / denom)
