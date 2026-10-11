#!/usr/bin/env python3
"""
ONNX 模型量化压缩脚本。

将 FP32 ONNX 模型量化为 INT8，通常可减少 75% 体积并提升推理速度。
使用动态量化（无需校准数据集），适合快速部署。

用法:
    python scripts/quantize.py --input model.onnx --output model_quantized.onnx
    python scripts/quantize.py --input model.onnx --output model_int8.onnx --method static --calib-dir ./calib_images
"""
from __future__ import annotations

import argparse
import os
import sys


def dynamic_quantization(input_path: str, output_path: str) -> None:
    """动态量化：权重量化为 INT8，激活运行时量化。无需校准数据。"""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ERROR: onnxruntime required. pip install onnxruntime")
        sys.exit(1)

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    print(f"[OK] Dynamic quantization: {input_path} -> {output_path}")


def static_quantization(input_path: str, output_path: str, calib_dir: str) -> None:
    """静态量化：权重和激活均量化为 INT8，需要校准数据集。精度更高。"""
    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
        from PIL import Image
        import numpy as np
    except ImportError:
        print("ERROR: onnxruntime, Pillow, numpy required.")
        sys.exit(1)

    class ImageCalibrationReader(CalibrationDataReader):
        def __init__(self, image_dir, input_size=(224, 224)):
            self.images = []
            for f in os.listdir(image_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.images.append(os.path.join(image_dir, f))
            self.idx = 0
            self.input_size = input_size

        def get_next(self):
            if self.idx >= len(self.images):
                return None
            img = Image.open(self.images[self.idx]).convert("RGB").resize(self.input_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            arr = (arr - mean) / std
            self.idx += 1
            return {"input": np.expand_dims(arr, 0)}

    reader = ImageCalibrationReader(calib_dir)
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"[OK] Static quantization: {input_path} -> {output_path}")


def print_size_comparison(input_path: str, output_path: str) -> None:
    """打印体积对比。"""
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100
    print(f"\nSize comparison:")
    print(f"  Original:  {orig_size:.1f} MB")
    print(f"  Quantized: {quant_size:.1f} MB")
    print(f"  Reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model")
    parser.add_argument("--input", required=True, help="Input FP32 ONNX model")
    parser.add_argument("--output", required=True, help="Output quantized ONNX model")
    parser.add_argument(
        "--method",
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization method (default: dynamic)",
    )
    parser.add_argument("--calib-dir", help="Calibration images directory (for static method)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}")
        sys.exit(1)

    if args.method == "static" and not args.calib_dir:
        print("ERROR: --calib-dir required for static quantization")
        sys.exit(1)

    if args.method == "dynamic":
        dynamic_quantization(args.input, args.output)
    else:
        static_quantization(args.input, args.output, args.calib_dir)

    print_size_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
