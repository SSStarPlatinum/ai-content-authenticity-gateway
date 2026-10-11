"""
检测模型基类与统一数据结构。

所有开源检测模型必须实现 BaseDetector 接口，
确保上层融合引擎无需关心底层差异。
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    """
    单模型检测结果（统一输出格式）。

    Attributes:
        model_id: 模型唯一标识
        fake_prob: 伪造概率 [0, 1]，1.0 = 确定伪造
        label: 判定标签 "fake" | "real" | "uncertain"
        confidence: 模型对自身预测的置信度 [0, 1]
        latency_ms: 推理耗时（毫秒）
        details: 模型特有附加信息（如各子类别概率）
        error: 若推理失败，记录错误信息
    """
    model_id: str
    fake_prob: float = 0.5
    label: str = "uncertain"
    confidence: float = 0.0
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "fake_prob": round(self.fake_prob, 6),
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "latency_ms": round(self.latency_ms, 2),
            "details": self.details,
            "error": self.error,
        }


class BaseDetector(abc.ABC):
    """
    检测模型抽象基类。

    子类必须实现：
        - model_id / model_name / version 属性
        - _load()  加载权重
        - _predict(img) -> DetectionResult  单图推理
    可选重写：
        - preprocess(img) -> np.ndarray
        - batch_predict(images) -> List[DetectionResult]
    """

    # 子类必须设置
    model_id: str = "base"
    model_name: str = "Base Detector"
    version: str = "1.0.0"
    # 该模型在融合中的默认权重（可被配置覆盖）
    default_weight: float = 1.0
    # 模型擅长的后处理类型（用于动态加权）: "jpeg", "resize", "none", "crop"
    strong_postprocessing: List[str] = []

    def __init__(self, weights_dir: str = "./models/weights"):
        self.weights_dir = weights_dir
        self._loaded = False
        self._session = None  # ONNX Runtime InferenceSession

    # ----- 生命周期 -----

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """加载模型权重，幂等。"""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    @abc.abstractmethod
    def _load(self) -> None:
        """子类实现实际加载逻辑。"""

    # ----- 推理 -----

    def predict(self, img: Image.Image) -> DetectionResult:
        """
        单图推理，统一计时与异常捕获。
        子类不应重写此方法，应重写 _predict。
        """
        if not self._loaded:
            self.load()

        start = time.perf_counter()
        try:
            result = self._predict(img)
        except Exception as e:
            result = DetectionResult(
                model_id=self.model_id,
                fake_prob=0.5,
                label="uncertain",
                confidence=0.0,
                error=f"{type(e).__name__}: {e}",
            )
        result.latency_ms = (time.perf_counter() - start) * 1000
        result.model_id = self.model_id
        return result

    @abc.abstractmethod
    def _predict(self, img: Image.Image) -> DetectionResult:
        """子类实现实际推理逻辑。"""

    def batch_predict(self, images: List[Image.Image]) -> List[DetectionResult]:
        """
        批量推理默认实现：逐张调用。
        ONNX 模型可重写为真正的 batch 推理以提升吞吐。
        """
        return [self.predict(img) for img in images]

    # ----- 工具方法 -----

    def _softmax(self, logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的 softmax。"""
        exp = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.model_id} loaded={self._loaded}>"
