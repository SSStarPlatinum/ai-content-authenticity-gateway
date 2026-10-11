"""
FreqDetector 适配器：纯频域统计检测器，无需深度学习权重。

原理：AI 生成图像在频域中通常呈现异常的高频能量分布和周期性伪影。
本检测器计算对数幅度谱的多个统计特征，基于预校准阈值判决。
优点：零依赖、极快、对 JPEG 压缩鲁棒；可作为融合中的"快速否决"通道。

此模型不依赖 ONNX 权重，开箱即用。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..utils.image import compute_frequency_spectrum
from .base import BaseDetector, DetectionResult
from .registry import register_model


@register_model("freq_detector")
class FreqDetector(BaseDetector):
    """Frequency-domain statistical fake detector (weight-free)."""

    model_name = "FreqDetector (Spectral Analysis)"
    version = "1.0.0"
    default_weight = 0.3
    strong_postprocessing = ["jpeg", "resize", "none"]

    # 预校准阈值（基于真实/生成图像频谱统计差异）
    HIGH_FREQ_RATIO_THRESHOLD = 0.15
    SPECTRAL_ENTROPY_THRESHOLD = 4.5
    RADIAL_VARIANCE_THRESHOLD = 0.02

    def _load(self) -> None:
        # 无权重，标记为已加载
        self._loaded = True

    def _predict(self, img: Image.Image) -> DetectionResult:
        spectrum = compute_frequency_spectrum(img, max_size=256)
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        # 特征1：高频能量占比（远离中心的区域）
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_r = min(cy, cx)
        high_freq_mask = radius > (max_r * 0.6)
        high_freq_ratio = float(spectrum[high_freq_mask].sum() / (spectrum.sum() + 1e-8))

        # 特征2：频谱熵（生成图像频谱通常更"均匀"，熵更高）
        flat = spectrum.flatten()
        flat = flat / (flat.sum() + 1e-8)
        entropy = float(-np.sum(flat * np.log2(flat + 1e-12)))

        # 特征3：径向方差（生成图像常有环状伪影，径向方差低）
        radial_profile = []
        for r in range(1, max_r, 2):
            ring_mask = (radius >= r) & (radius < r + 2)
            if ring_mask.any():
                radial_profile.append(float(spectrum[ring_mask].mean()))
        radial_variance = float(np.var(radial_profile)) if len(radial_profile) > 1 else 0.0

        # 多特征加权打分
        score = 0.0
        score += 0.4 * min(high_freq_ratio / self.HIGH_FREQ_RATIO_THRESHOLD, 1.0)
        score += 0.35 * min(entropy / self.SPECTRAL_ENTROPY_THRESHOLD, 1.0)
        score += 0.25 * (1.0 - min(radial_variance / (self.RADIAL_VARIANCE_THRESHOLD * 5), 1.0))

        fake_prob = float(np.clip(score, 0.0, 1.0))
        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "high_freq_ratio": round(high_freq_ratio, 6),
                "spectral_entropy": round(entropy, 4),
                "radial_variance": round(radial_variance, 6),
                "note": "weight-free spectral analysis",
            },
        )
