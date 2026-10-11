"""
Mock 检测模型：无需权重，用于开发、演示、测试。

输出基于图片像素统计的伪概率，确保：
- 接口完全可用
- 不同图片返回不同结果（非恒定值）
- 延迟模拟真实模型（可配置）
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from .base import BaseDetector, DetectionResult
from .registry import register_model


@register_model("mock")
class MockDetector(BaseDetector):
    """基于像素统计的模拟检测器。"""

    model_name = "Mock Detector (Dev)"
    version = "1.0.0"
    default_weight = 0.5
    strong_postprocessing = ["jpeg", "resize", "none"]

    def __init__(self, weights_dir: str = "./models/weights", simulate_latency_ms: float = 15.0):
        super().__init__(weights_dir)
        self.simulate_latency_ms = simulate_latency_ms

    def _load(self) -> None:
        # Mock 模型无需加载
        self._loaded = True

    def _predict(self, img: Image.Image) -> DetectionResult:
        # 模拟推理延迟
        if self.simulate_latency_ms > 0:
            time.sleep(self.simulate_latency_ms / 1000.0)

        # 基于图像统计生成确定性伪概率
        arr = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        # 高频能量（差分）作为"伪造线索"模拟
        grad_x = np.diff(arr, axis=1)
        high_freq_energy = float(np.mean(np.abs(grad_x)))

        # 组合成 0-1 概率（纯模拟，无真实检测能力）
        raw = 0.3 * mean_val + 0.3 * std_val + 0.4 * high_freq_energy
        fake_prob = float(np.clip(raw, 0.0, 1.0))
        confidence = float(np.clip(abs(fake_prob - 0.5) * 2, 0.0, 1.0))
        label = "fake" if fake_prob > 0.5 else "real"

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label=label,
            confidence=confidence,
            details={
                "pixel_mean": round(mean_val, 4),
                "pixel_std": round(std_val, 4),
                "high_freq_energy": round(high_freq_energy, 4),
                "note": "mock model - for development only",
            },
        )
