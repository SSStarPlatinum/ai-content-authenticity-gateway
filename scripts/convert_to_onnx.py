#!/usr/bin/env python3
"""
PyTorch 模型 -> ONNX 转换脚本。

用法:
    python scripts/convert_to_onnx.py --model cnn_spotter --checkpoint model.pth --output models/weights/cnn_spotter.onnx

支持的模型类型:
    - cnn_spotter: ResNet50 二分类 (输入 224x224, 输出 2 logits)
    - universal_fake: CLIP + 线性头 (输入 224x224, 输出 2 logits)
    - aeroblade: 扩散检测 (输入 512x512, 输出 1 score)

注意: 此脚本需要 PyTorch 和对应模型的定义代码，
仅在有原始 PyTorch 权重时使用。转换后可用 quantize.py 进一步压缩。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def convert_cnn_spotter(checkpoint: str, output: str, opset: int = 17) -> None:
    """转换 CNNSpotter (ResNet50) 模型。"""
    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("ERROR: PyTorch and torchvision required. pip install torch torchvision")
        sys.exit(1)

    # 构建模型
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # 加载权重
    state_dict = torch.load(checkpoint, map_location="cpu")
    # 兼容不同保存格式
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    # 导出 ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"[OK] CNNSpotter exported to {output}")


def convert_universal_fake(checkpoint: str, output: str, opset: int = 17) -> None:
    """转换 UniversalFakeDetect (CLIP + 线性头)。"""
    try:
        import torch
        import clip
    except ImportError:
        print("ERROR: PyTorch and CLIP required. pip install torch git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    # 加载 CLIP 视觉编码器
    device = "cpu"
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    visual = clip_model.visual
    visual.eval()

    # 加载分类头
    classifier = torch.nn.Linear(visual.output_dim, 2)
    state = torch.load(checkpoint, map_location="cpu")
    if "classifier" in state:
        classifier.load_state_dict(state["classifier"])
    else:
        classifier.load_state_dict(state)
    classifier.eval()

    # 组合模型
    class CombinedModel(torch.nn.Module):
        def __init__(self, visual, classifier):
            super().__init__()
            self.visual = visual
            self.classifier = classifier

        def forward(self, x):
            features = self.visual(x)
            return self.classifier(features)

    model = CombinedModel(visual, classifier)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"[OK] UniversalFakeDetect exported to {output}")


def convert_aeroblade(checkpoint: str, output: str, opset: int = 17) -> None:
    """转换 AeroBlade 模型。"""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required.")
        sys.exit(1)

    # AeroBlade 模型结构需从官方仓库导入
    # 此处提供模板，实际使用时需替换为真实模型定义
    print("[WARN] AeroBlade conversion requires the official model definition.")
    print("       Refer to: https://github.com/EPFL-VILAB/AeroBlade")
    sys.exit(1)


def verify_onnx(output: str) -> None:
    """验证导出的 ONNX 模型。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] onnxruntime not installed, skipping verification")
        return

    sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    output_info = sess.get_outputs()[0]
    print(f"  Input:  {input_info.name} {input_info.shape}")
    print(f"  Output: {output_info.name} {output_info.shape}")

    # 运行一次推理验证
    dummy = np.random.randn(*[d if isinstance(d, int) else 1 for d in input_info.shape]).astype(np.float32)
    result = sess.run(None, {input_info.name: dummy})
    print(f"  Inference test: OK, output shape = {result[0].shape}")


CONVERTERS = {
    "cnn_spotter": convert_cnn_spotter,
    "universal_fake": convert_universal_fake,
    "aeroblade": convert_aeroblade,
}


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--model", required=True, choices=CONVERTERS.keys(), help="Model type")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pth/.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Converting {args.model} -> {args.output} (opset={args.opset})")
    CONVERTERS[args.model](args.checkpoint, args.output, args.opset)

    print("Verifying...")
    verify_onnx(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
