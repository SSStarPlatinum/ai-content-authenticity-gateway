"""
噪声残差检测器：无权重，基于图像噪声模式分析。

原理：
  用中值滤波提取噪声残差（原图 - 中值滤波图）。
  - 真实照片：包含传感器噪声，噪声残差近似高斯分布，随机性强
  - AI 生成图像：上采样/反卷积产生周期性伪影，噪声残差有结构性、自相关性强

判决依据：噪声残差的偏度、峰度、自相关系数、频域周期性。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult
from .registry import register_model


@register_model("noise_detector")
class NoiseResidualDetector(BaseDetector):
    """Noise residual pattern fake detector (weight-free)."""

    model_name = "Noise Residual Detector"
    version = "1.0.0"
    default_weight = 0.6
    strong_postprocessing = ["none", "resize"]

    ANALYSIS_SIZE = (256, 256)
    MEDIAN_KERNEL = 3  # 中值滤波核大小

    def _load(self) -> None:
        self._loaded = True

    def _predict(self, img: Image.Image) -> DetectionResult:
        residual = self._extract_noise_residual(img)

        # 特征1：噪声残差统计量
        mean_val = float(residual.mean())
        std_val = float(residual.std())
        # 偏度（高斯噪声偏度≈0，结构化噪声偏度偏离0）
        skewness = float(self._skewness(residual))
        # 峰度（高斯噪声峰度≈3，结构化噪声峰度偏离）
        kurtosis = float(self._kurtosis(residual))

        # 特征2：水平/垂直自相关（AI 生成上采样伪影导致自相关高）
        autocorr_h = float(self._autocorrelation(residual, axis=1))
        autocorr_v = float(self._autocorrelation(residual, axis=0))
        avg_autocorr = (autocorr_h + autocorr_v) / 2.0

        # 特征3：噪声残差频谱的周期性能量占比
        spectral_periodicity = float(self._spectral_periodicity(residual))

        # 特征4：噪声能量分布均匀性（分块方差变异系数）
        energy_uniformity = float(self._energy_uniformity(residual))

        # 多特征打分
        score = 0.0
        # 自相关性越高越可疑（AI 上采样伪影）
        score += 0.30 * min(avg_autocorr / 0.3, 1.0)
        # 频谱周期性越强越可疑
        score += 0.25 * spectral_periodicity
        # 峰度偏离3越可疑（非高斯噪声）
        score += 0.20 * min(abs(kurtosis - 3.0) / 5.0, 1.0)
        # 偏度偏离0越可疑
        score += 0.15 * min(abs(skewness) / 1.0, 1.0)
        # 噪声能量过于均匀也可疑
        score += 0.10 * (1.0 - min(energy_uniformity, 1.0))

        fake_prob = float(np.clip(score, 0.0, 1.0))
        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "noise_mean": round(mean_val, 6),
                "noise_std": round(std_val, 6),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "avg_autocorrelation": round(avg_autocorr, 6),
                "spectral_periodicity": round(spectral_periodicity, 6),
                "energy_uniformity": round(energy_uniformity, 6),
            },
        )

    def _extract_noise_residual(self, img: Image.Image) -> np.ndarray:
        """提取噪声残差：灰度图 - 中值滤波图。"""
        import cv2
        gray = img.convert("L").resize(self.ANALYSIS_SIZE, Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)
        # 使用 OpenCV 中值滤波（比手动循环快 100 倍以上）
        filtered = cv2.medianBlur(arr.astype(np.uint8), self.MEDIAN_KERNEL).astype(np.float32)
        residual = arr - filtered
        return residual

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        flat = arr.flatten()
        m = flat.mean()
        s = flat.std()
        if s < 1e-8:
            return 0.0
        return float(np.mean(((flat - m) / s) ** 3))

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        flat = arr.flatten()
        m = flat.mean()
        s = flat.std()
        if s < 1e-8:
            return 3.0
        return float(np.mean(((flat - m) / s) ** 4))

    @staticmethod
    def _autocorrelation(arr: np.ndarray, axis: int) -> float:
        """计算 lag=1 的自相关系数。"""
        if axis == 1:
            a = arr[:, :-1].flatten()
            b = arr[:, 1:].flatten()
        else:
            a = arr[:-1, :].flatten()
            b = arr[1:, :].flatten()
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        denom = np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
        if denom < 1e-8:
            return 0.0
        return float((a_centered * b_centered).sum() / denom)

    @staticmethod
    def _spectral_periodicity(arr: np.ndarray) -> float:
        """检测噪声残差频谱中的周期性能量（上采样伪影特征）。"""
        fft = np.fft.fft2(arr)
        magnitude = np.abs(np.fft.fftshift(fft))
        # 排除直流分量
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        magnitude[cy, cx] = 0

        # 检测水平/垂直方向的强峰值（周期性伪影表现为线状频谱）
        row_energy = magnitude.sum(axis=1)
        col_energy = magnitude.sum(axis=0)

        # 中心线附近的能量占比（排除中心区域）
        center_band = 3
        row_center = row_energy[cy - center_band:cy + center_band + 1].sum()
        col_center = col_energy[cx - center_band:cx + center_band + 1].sum()
        total = magnitude.sum() + 1e-8

        periodicity = (row_center + col_center) / total
        # 归一化：真实噪声约 0.02-0.05，AI 伪影可达 0.1+
        return min((periodicity - 0.03) / 0.1, 1.0)

    @staticmethod
    def _energy_uniformity(arr: np.ndarray) -> float:
        """噪声能量空间均匀性（变异系数的倒数归一化）。"""
        block_size = 32
        h, w = arr.shape
        energies = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = arr[i:i + block_size, j:j + block_size]
                energies.append(float(np.var(block)))
        energies = np.array(energies)
        if energies.mean() < 1e-8:
            return 1.0
        cv = energies.std() / energies.mean()
        # 变异系数越小越均匀，越可疑；返回均匀度 [0,1]
        return float(1.0 - min(cv / 2.0, 1.0))
