"""
CNNSpotter 适配器：基于 CNN 的通用 AI 生成图像检测。

参考开源项目：https://github.com/peterwang512/CNNDetection
原理：在 ImageNet 预训练的 ResNet 上微调，利用频域统计差异检测生成图像。
ONNX 导出后输入为 (1, 3, 224, 224)，输出为 (1, 2) logits [real, fake]。

权重文件：cnn_spotter.onnx（放置于 models/weights/）
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from ..utils.image import resize_to_tensor
from .base import BaseDetector, DetectionResult
from .registry import register_model

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@register_model("cnn_spotter")
class CNNSpotterDetector(BaseDetector):
    """CNN-based universal fake image detector."""

    model_name = "CNNSpotter (ResNet50)"
    version = "1.0.0"
    default_weight = 3.0
    strong_postprocessing = ["none", "jpeg"]

    WEIGHT_FILENAME = "cnn_spotter.onnx"
    INPUT_SIZE = (224, 224)
    # ImageNet 均值方差（该模型训练时使用）
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def _load(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        weight_path = os.path.join(self.weights_dir, self.WEIGHT_FILENAME)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"weight not found: {weight_path}. "
                f"Run scripts/download_weights.sh or place {self.WEIGHT_FILENAME} "
                f"in {self.weights_dir}/"
            )
        # 使用所有可用执行提供者
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            weight_path, sess_options=sess_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def _predict(self, img: Image.Image) -> DetectionResult:
        x = resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD)
        outputs = self._session.run(None, {self._input_name: x})
        logits = outputs[0][0]  # shape (2,)
        probs = self._softmax(logits)
        fake_prob = float(probs[1])
        confidence = float(abs(fake_prob - 0.5) * 2)
        label = "fake" if fake_prob > 0.5 else "real"
        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label=label,
            confidence=confidence,
            details={
                "real_prob": round(float(probs[0]), 6),
                "fake_prob_raw": round(fake_prob, 6),
                "input_size": f"{self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]}",
            },
        )

    def batch_predict(self, images):
        """真正的 batch 推理：堆叠后一次前向。"""
        if not self._loaded:
            self.load()
        if not images:
            return []
        batch = np.concatenate(
            [resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD) for img in images],
            axis=0,
        )
        outputs = self._session.run(None, {self._input_name: batch})
        logits_batch = outputs[0]  # (N, 2)
        probs_batch = self._softmax(logits_batch, axis=1)
        results = []
        for i, probs in enumerate(probs_batch):
            fake_prob = float(probs[1])
            results.append(DetectionResult(
                model_id=self.model_id,
                fake_prob=fake_prob,
                label="fake" if fake_prob > 0.5 else "real",
                confidence=float(abs(fake_prob - 0.5) * 2),
                details={"real_prob": round(float(probs[0]), 6)},
            ))
        return results
