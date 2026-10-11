"""
UniversalFakeDetect 适配器：基于 CLIP 的通用伪造检测。

参考论文："Universal Fake Image Detection" (Wang et al., CVPR 2023)
原理：利用 CLIP 视觉编码器提取特征，训练线性分类头区分真实/生成图像。
对未见生成器具有良好泛化性。

ONNX 模型输入：(1, 3, 224, 224)，CLIP 归一化
输出：(1, 2) logits [real, fake]

权重文件：universal_fake.onnx
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ..utils.image import resize_to_tensor
from .base import BaseDetector, DetectionResult
from .registry import register_model

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@register_model("universal_fake")
class UniversalFakeDetector(BaseDetector):
    """CLIP-based universal fake image detector."""

    model_name = "UniversalFakeDetect (CLIP)"
    version = "1.0.0"
    default_weight = 3.0
    strong_postprocessing = ["none", "jpeg", "resize"]

    WEIGHT_FILENAME = "universal_fake.onnx"
    INPUT_SIZE = (224, 224)
    # CLIP 专用归一化参数
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def _load(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        weight_path = os.path.join(self.weights_dir, self.WEIGHT_FILENAME)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"weight not found: {weight_path}. "
                f"Download from the UniversalFakeDetect repo and export to ONNX."
            )
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            weight_path, sess_options=sess_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name

    def _predict(self, img: Image.Image) -> DetectionResult:
        x = resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD)
        outputs = self._session.run(None, {self._input_name: x})
        logits = outputs[0][0]
        probs = self._softmax(logits)
        fake_prob = float(probs[1])
        confidence = float(abs(fake_prob - 0.5) * 2)
        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "real_prob": round(float(probs[0]), 6),
                "backbone": "CLIP ViT-B/16",
                "input_size": f"{self.INPUT_SIZE[0]}x{self.INPUT_SIZE[1]}",
            },
        )

    def batch_predict(self, images):
        if not self._loaded:
            self.load()
        if not images:
            return []
        batch = np.concatenate(
            [resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD) for img in images],
            axis=0,
        )
        outputs = self._session.run(None, {self._input_name: batch})
        probs_batch = self._softmax(outputs[0], axis=1)
        return [
            DetectionResult(
                model_id=self.model_id,
                fake_prob=float(p[1]),
                label="fake" if p[1] > 0.5 else "real",
                confidence=float(abs(p[1] - 0.5) * 2),
                details={"real_prob": round(float(p[0]), 6)},
            )
            for p in probs_batch
        ]
