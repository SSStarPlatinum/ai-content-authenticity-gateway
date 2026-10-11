#!/usr/bin/env python3
"""
权重下载与 ONNX 转换一体化脚本。

在有外网的环境（Google Colab / 海外服务器 / 本地代理）中运行：
    python scripts/download_and_convert.py

功能：
  1. 自动安装依赖（torch, torchvision, onnx, onnxruntime, clip）
  2. 下载 CNNDetection (ResNet50) 权重（Dropbox 直链）
  3. 下载 UniversalFakeDetect (CLIP) 分类头权重
  4. 转换为 ONNX 格式
  5. 验证 ONNX 推理正确性
  6. 打包为 onnx_weights.zip 方便下载回本地

输出：
  models/weights/cnn_spotter.onnx
  models/weights/universal_fake.onnx
  onnx_weights.zip
"""
from __future__ import annotations

import os
import sys
import subprocess
import zipfile
from pathlib import Path

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# 下载链接
CNNDETECTION_URL = "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"
# UniversalFakeDetect 分类头权重（尝试从仓库 raw 下载）
UFAKE_RAW_URL = "https://raw.githubusercontent.com/Yuheng-Li/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth"

CNN_PTH = WEIGHTS_DIR / "cnn_spotter.pth"
CNN_ONNX = WEIGHTS_DIR / "cnn_spotter.onnx"
UFAKE_PTH = WEIGHTS_DIR / "universal_fake.pth"
UFAKE_ONNX = WEIGHTS_DIR / "universal_fake.onnx"

ZIP_OUTPUT = PROJECT_ROOT / "onnx_weights.zip"


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def error(msg: str) -> None:
    print(f"[ERROR] {msg}", flush=True)


def run_cmd(cmd: list[str], desc: str) -> bool:
    """运行命令，返回是否成功。"""
    log(f"{desc}...")
    log(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error(f"{desc} failed")
        if result.stdout:
            print(result.stdout[-2000:])
        if result.stderr:
            print(result.stderr[-2000:])
        return False
    return True


def install_dependencies() -> bool:
    """安装必要依赖。"""
    packages = ["torch", "torchvision", "onnx", "onnxruntime", "ftfy", "regex", "tqdm"]
    # CLIP 需要从 git 安装
    log("Installing PyTorch and dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        error(f"pip install failed: {result.stderr[-500:]}")
        return False

    # 安装 CLIP
    log("Installing OpenAI CLIP...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "git+https://github.com/openai/CLIP.git"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        warn(f"CLIP install failed (UniversalFakeDetect may not work): {result.stderr[-300:]}")
    return True


def download_file(url: str, dest: Path, desc: str) -> bool:
    """下载文件，支持重定向。"""
    if dest.exists() and dest.stat().st_size > 1000:
        log(f"{desc} already exists: {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
        return True

    log(f"Downloading {desc}...")
    log(f"  URL: {url}")
    log(f"  -> {dest}")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        log(f"  Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        error(f"Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def convert_cnn_spotter() -> bool:
    """转换 CNNDetection (ResNet50) 为 ONNX。"""
    if CNN_ONNX.exists():
        log(f"CNNSpotter ONNX already exists: {CNN_ONNX}")
        return True

    if not CNN_PTH.exists():
        error(f"CNNSpotter weight not found: {CNN_PTH}")
        return False

    log("Converting CNNSpotter to ONNX...")
    try:
        import torch
        import torchvision.models as models

        # 构建 ResNet50，修改最后一层为二分类
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)

        # 加载权重
        state_dict = torch.load(str(CNN_PTH), map_location="cpu")
        # 兼容不同保存格式
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        # 导出 ONNX
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            str(CNN_ONNX),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
        log(f"CNNSpotter ONNX saved: {CNN_ONNX} ({CNN_ONNX.stat().st_size // 1024 // 1024} MB)")
        return True
    except Exception as e:
        error(f"CNNSpotter conversion failed: {e}")
        if CNN_ONNX.exists():
            CNN_ONNX.unlink()
        return False


def convert_universal_fake() -> bool:
    """转换 UniversalFakeDetect (CLIP ViT-L/14 + 线性头) 为 ONNX。"""
    if UFAKE_ONNX.exists():
        log(f"UniversalFake ONNX already exists: {UFAKE_ONNX}")
        return True

    if not UFAKE_PTH.exists():
        warn(f"UniversalFakeDetect weight not found: {UFAKE_PTH}")
        warn("Skipping UniversalFakeDetect conversion.")
        return False

    log("Converting UniversalFakeDetect to ONNX...")
    try:
        import torch
        import clip

        device = "cpu"
        clip_model, _ = clip.load("ViT-L/14", device=device)
        visual = clip_model.visual
        visual.eval()

        # 加载分类头
        classifier = torch.nn.Linear(visual.output_dim, 2)
        state = torch.load(str(UFAKE_PTH), map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            # 尝试只加载分类头权重
            classifier_state = {k.replace("fc.", "").replace("classifier.", ""): v
                                for k, v in state.items()
                                if "fc" in k or "classifier" in k}
            if classifier_state:
                classifier.load_state_dict(classifier_state)
            else:
                classifier.load_state_dict(state)
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
            str(UFAKE_ONNX),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
        log(f"UniversalFake ONNX saved: {UFAKE_ONNX} ({UFAKE_ONNX.stat().st_size // 1024 // 1024} MB)")
        return True
    except Exception as e:
        error(f"UniversalFake conversion failed: {e}")
        if UFAKE_ONNX.exists():
            UFAKE_ONNX.unlink()
        return False


def verify_onnx(onnx_path: Path, input_size: tuple = (224, 224)) -> bool:
    """验证 ONNX 模型可以正常推理。"""
    log(f"Verifying {onnx_path.name}...")
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        input_shape = sess.get_inputs()[0].shape

        # 构造 dummy 输入
        dummy = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
        result = sess.run([output_name], {input_name: dummy})
        log(f"  Input: {input_shape}, Output: {result[0].shape}")
        log(f"  Output sample: {result[0].flatten()[:4]}")
        return True
    except Exception as e:
        error(f"Verification failed: {e}")
        return False


def package_zip() -> None:
    """将所有 ONNX 文件打包为 zip。"""
    onnx_files = list(WEIGHTS_DIR.glob("*.onnx"))
    if not onnx_files:
        warn("No ONNX files found to package.")
        return

    log(f"Packaging {len(onnx_files)} ONNX files to {ZIP_OUTPUT}...")
    with zipfile.ZipFile(str(ZIP_OUTPUT), "w", zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(str(f), arcname=f.name)
            log(f"  + {f.name} ({f.stat().st_size // 1024 // 1024} MB)")
    log(f"Zip created: {ZIP_OUTPUT} ({ZIP_OUTPUT.stat().st_size // 1024 // 1024} MB)")


def main():
    print("=" * 60)
    print("  AI Content Authenticity - Weight Download & Convert")
    print("=" * 60)
    print(f"  Weights dir: {WEIGHTS_DIR}")
    print()

    # 1. 安装依赖
    if not install_dependencies():
        error("Dependency installation failed.")
        sys.exit(1)

    results = {}

    # 2. CNNSpotter
    print()
    print("-" * 40)
    print("[1/2] CNNSpotter (ResNet50)")
    print("-" * 40)
    if download_file(CNNDETECTION_URL, CNN_PTH, "CNNDetection weight"):
        if convert_cnn_spotter():
            results["cnn_spotter"] = verify_onnx(CNN_ONNX, (224, 224))
        else:
            results["cnn_spotter"] = False
    else:
        results["cnn_spotter"] = False

    # 3. UniversalFakeDetect
    print()
    print("-" * 40)
    print("[2/2] UniversalFakeDetect (CLIP ViT-L/14)")
    print("-" * 40)
    # 尝试从仓库 raw 下载分类头权重
    if not download_file(UFAKE_RAW_URL, UFAKE_PTH, "UniversalFakeDetect classifier weight"):
        warn("Could not download from GitHub raw.")
        warn("Please manually download fc_weights.pth from:")
        warn("  https://github.com/Yuheng-Li/UniversalFakeDetect")
        warn(f"And save as: {UFAKE_PTH}")
    if UFAKE_PTH.exists():
        if convert_universal_fake():
            results["universal_fake"] = verify_onnx(UFAKE_ONNX, (224, 224))
        else:
            results["universal_fake"] = False
    else:
        results["universal_fake"] = False

    # 4. 打包
    print()
    print("-" * 40)
    package_zip()

    # 5. 总结
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:20s} {status}")
    print()

    onnx_count = len(list(WEIGHTS_DIR.glob("*.onnx")))
    print(f"  ONNX files in weights dir: {onnx_count}")
    if ZIP_OUTPUT.exists():
        print(f"  Zip package: {ZIP_OUTPUT}")
    print()
    print("  Next steps:")
    print("  1. Copy .onnx files to local models/weights/")
    print("  2. Update .env: ENABLED_MODELS=...,cnn_spotter,universal_fake")
    print("  3. Restart service")
    print("=" * 60)


if __name__ == "__main__":
    main()
