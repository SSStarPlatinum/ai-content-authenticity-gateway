"""
AeroBlade 适配器：基于 VAE 重建误差的扩散模型生成图像检测。

原理（CVPR 2024）：
  AI 生成图像（尤其是扩散模型）通过 VAE 编码-解码后的重建误差
  显著低于真实图像。因为扩散模型本身就是在 VAE 潜空间中训练的，
  其生成结果与 VAE 的"流形"更契合，重建更精确。

实现：
  1. 将图像 resize 到 512x512，归一化到 [-1, 1]
  2. 通过 SD VAE 编码器得到潜空间均值
  3. 通过 SD VAE 解码器重建图像
  4. 计算输入与重建图像的 MSE（归一化后）
  5. 误差越低 -> fake_prob 越高

权重文件：
  aeroblade_encoder.onnx  (VAE 编码器)
  aeroblade_decoder.onnx  (VAE 解码器)
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


@register_model("aeroblade")
class AeroBladeDetector(BaseDetector):
    """VAE reconstruction error based diffusion image detector."""

    model_name = "AeroBlade (VAE Reconstruction Error)"
    version = "1.0.0"
    default_weight = 1.0
    strong_postprocessing = ["none", "resize"]

    ENCODER_FILENAME = "aeroblade_encoder.onnx"
    DECODER_FILENAME = "aeroblade_decoder.onnx"
    INPUT_SIZE = (128, 128)
    # VAE 输入归一化：[-1, 1]
    MEAN = (0.5, 0.5, 0.5)
    STD = (0.5, 0.5, 0.5)

    # 重建误差到 fake_prob 的映射参数
    # progan/face: real≈0.0020, fake≈0.0015 (误差低=fake)
    # Chameleon: 误差偏高，会被判为 real，由其他模型弥补
    ERROR_THRESHOLD = 0.0018
    ERROR_SCALE = 5000.0

    def _load(self) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")

        enc_path = os.path.join(self.weights_dir, self.ENCODER_FILENAME)
        dec_path = os.path.join(self.weights_dir, self.DECODER_FILENAME)

        if not os.path.exists(enc_path):
            raise FileNotFoundError(
                f"encoder weight not found: {enc_path}. "
                f"Run scripts/convert_aeroblade.py to download and convert."
            )
        if not os.path.exists(dec_path):
            raise FileNotFoundError(
                f"decoder weight not found: {dec_path}. "
                f"Run scripts/convert_aeroblade.py to download and convert."
            )

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._enc_session = ort.InferenceSession(
            enc_path, sess_options=sess_options, providers=providers
        )
        self._dec_session = ort.InferenceSession(
            dec_path, sess_options=sess_options, providers=providers
        )
        self._enc_input = self._enc_session.get_inputs()[0].name
        self._dec_input = self._dec_session.get_inputs()[0].name

    def _predict(self, img: Image.Image) -> DetectionResult:
        # 预处理：resize 到 512x512，归一化到 [-1, 1]
        x = resize_to_tensor(img, self.INPUT_SIZE, self.MEAN, self.STD)

        # 编码
        latent = self._enc_session.run(None, {self._enc_input: x})[0]
        # 解码
        recon = self._dec_session.run(None, {self._dec_input: latent})[0]

        # 计算重建误差（MSE，归一化到 [0,1] 空间）
        # x 和 recon 都在 [-1, 1] 空间，误差范围 [0, 4]
        mse = float(np.mean((x - recon) ** 2))

        # 将误差映射为 fake_prob
        # 误差低 -> fake 概率高；误差高 -> real 概率高
        # sigmoid(-(error - threshold) * scale)
        fake_prob = float(1.0 / (1.0 + np.exp((mse - self.ERROR_THRESHOLD) * self.ERROR_SCALE)))
        fake_prob = max(0.0, min(1.0, fake_prob))

        confidence = float(abs(fake_prob - 0.5) * 2)

        return DetectionResult(
            model_id=self.model_id,
            fake_prob=fake_prob,
            label="fake" if fake_prob > 0.5 else "real",
            confidence=confidence,
            details={
                "reconstruction_mse": round(mse, 6),
                "error_threshold": self.ERROR_THRESHOLD,
                "vae": "Stable Diffusion 1.5",
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
        latents = self._enc_session.run(None, {self._enc_input: batch})[0]
        recons = self._dec_session.run(None, {self._dec_input: latents})[0]

        mses = np.mean((batch - recons) ** 2, axis=(1, 2, 3))
        fake_probs = 1.0 / (1.0 + np.exp((mses - self.ERROR_THRESHOLD) * self.ERROR_SCALE))
        fake_probs = np.clip(fake_probs, 0.0, 1.0)

        return [
            DetectionResult(
                model_id=self.model_id,
                fake_prob=float(p),
                label="fake" if p > 0.5 else "real",
                confidence=float(abs(p - 0.5) * 2),
                details={"reconstruction_mse": round(float(mse), 6)},
            )
            for p, mse in zip(fake_probs, mses)
        ]
