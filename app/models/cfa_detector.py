"""
CFA (Color Filter Array) 插值伪影检测器：无权重。

原理：
  真实数码相机通过 Bayer 色彩滤波阵列采集，每个像素只有一种颜色，
  其余两种通过插值获得。这会在像素间留下周期性的相关性伪影。
  AI 生成图像没有这种采集过程，因此缺乏 CFA 插值伪影。

检测方法：
  1. 估计每个通道的插值残差（同色像素预测误差）
  2. 分析残差的周期性（Bayer 模式周期为 2）
  3. 周期性强 = 真实照片，周期性弱 = AI 生成

参考：基于 "Determining Image Origin and Integrity Using Sensor Noise" 的简化实现。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult
from .registry import register_model


@register_model("cfa_detector")
class CFADetector(BaseDetector):
    """CFA interpolation artifact detector (weight-free)."""

    model_name = "CFA Artifact Detector"
    version = "1.0.0"
    default_weight = 0.2
    strong_postprocessing = ["none"]

    ANALYSIS_SIZE = (256, 256)

    def _load(self) -> None:
        self._loaded = True

    def _predict(self, img: Image.Image) -> DetectionResult:
        arr = np.asarray(
            img.convert("RGB").resize(self.ANALYSIS_SIZE, Image.BILINEAR),
            dtype=np.float32,
        )

        # 对每个通道计算 CFA 插值残差的周期性
        periodicity_scores = []
        for ch in range(3):
            channel = arr[:, :, ch]
            period = self._compute_cfa_periodicity(channel)
            periodicity_scores.append(period)

        avg_periodicity = float(np.mean(periodicity_scores))

        # 额外特征：绿色通道与红/蓝通道的相关性差异
        # Bayer 模式中 G 通道采样率是 R/B 的 2 倍，相关性模式不同
        cross_corr = self._compute_cross_channel_correlation(arr)

        # 判决：周期性强 = 真实照片（fake_prob 低），周期性弱 = AI 生成（fake_prob 高）
        # avg_periodicity 范围约 [0, 0.3]，真实照片通常 > 0.08
        # 上限设为0.85，避免无CFA伪影时直接给1.0导致过度自信
        fake_prob = float(np.clip(0.85 - (avg_periodicity - 0.02) / 0.18, 0.1, 0.85))

        # 跨通道相关性异常也增加可疑度
        if cross_corr < 0.3:
            fake_prob = min(1.0, fake_prob + 0.1)

        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "cfa_periodicity_r": round(periodicity_scores[0], 6),
                "cfa_periodicity_g": round(periodicity_scores[1], 6),
                "cfa_periodicity_b": round(periodicity_scores[2], 6),
                "avg_periodicity": round(avg_periodicity, 6),
                "cross_channel_corr": round(cross_corr, 6),
            },
        )

    def _compute_cfa_periodicity(self, channel: np.ndarray) -> float:
        """
        计算单通道的 CFA 插值周期性。

        方法：对通道做 2x2 下采样（模拟 Bayer 同色像素间距），
        计算原始通道与上采样预测之间的残差，然后分析残差在 2 像素周期上的能量。
        """
        h, w = channel.shape

        # 取偶数行偶数列的像素（模拟 Bayer 同色位置）
        sampled = channel[::2, ::2]

        # 双线性上采样回原尺寸（模拟插值）
        from PIL import Image as PILImage
        sampled_img = PILImage.fromarray(sampled.astype(np.uint8))
        upsampled = np.asarray(
            sampled_img.resize((w, h), PILImage.BILINEAR),
            dtype=np.float32,
        )

        # 插值残差
        residual = channel - upsampled

        # 分析残差的 2 像素周期性（频谱中 (pi, pi) 附近的能量）
        fft = np.fft.fft2(residual)
        magnitude = np.abs(np.fft.fftshift(fft))
        cy, cx = h // 2, w // 2

        # Bayer 伪影在频谱中表现为 (0, pi), (pi, 0), (pi, pi) 处的峰值
        # 对应 fftshift 后的位置
        peak_positions = [
            (cy, 0),          # (0, pi)
            (cy, w - 1),      # (0, -pi)
            (0, cx),          # (pi, 0)
            (h - 1, cx),      # (-pi, 0)
            (0, 0),           # (pi, pi)
            (h - 1, w - 1),   # (-pi, -pi)
            (0, w - 1),       # (pi, -pi)
            (h - 1, 0),       # (-pi, pi)
        ]

        peak_energy = 0.0
        for py, px in peak_positions:
            peak_energy += magnitude[py, px]

        total_energy = magnitude.sum() + 1e-8
        periodicity = peak_energy / total_energy

        return float(periodicity)

    @staticmethod
    def _compute_cross_channel_correlation(arr: np.ndarray) -> float:
        """计算 G 通道与 R/B 通道的平均相关性（Bayer 模式特征）。"""
        r = arr[:, :, 0].flatten()
        g = arr[:, :, 1].flatten()
        b = arr[:, :, 2].flatten()

        def corr(a, b):
            a_c = a - a.mean()
            b_c = b - b.mean()
            denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
            if denom < 1e-8:
                return 0.0
            return float((a_c * b_c).sum() / denom)

        rg = corr(r, g)
        gb = corr(g, b)
        return float((abs(rg) + abs(gb)) / 2.0)
