"""
下载并转换深度学习模型权重为 ONNX 格式。
支持: cnn_spotter (CNNDetection), universal_fake (UniversalFakeDetect)
"""
import os
import sys
import urllib.request
import zipfile
import io

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import onnxruntime as ort

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def download_file(url, dest, desc=""):
    """下载文件，显示进度。"""
    print(f"[下载] {desc or url}")
    print(f"  目标: {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"  完成: {size/1024/1024:.2f} MB")
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False


def convert_cnn_spotter():
    """转换 CNNDetection (ResNet50) 为 ONNX."""
    print("\n" + "="*60)
    print("CNNSpotter (CNNDetection)")
    print("="*60)

    pth_path = os.path.join(WEIGHTS_DIR, "cnn_spotter.pth")
    onnx_path = os.path.join(WEIGHTS_DIR, "cnn_spotter.onnx")

    # 下载权重
    url = "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1"
    if not os.path.exists(pth_path) or os.path.getsize(pth_path) < 1000000:
        ok = download_file(url, pth_path, "CNNDetection blur_jpg_prob0.5.pth")
        if not ok:
            print("  跳过: 权重下载失败")
            return False

    # 构建模型
    print("[构建] ResNet50 二分类模型...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 加载权重
    print("[加载] 权重文件...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    print("  权重加载成功")

    # 导出 ONNX
    print("[导出] ONNX...")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"  导出成功: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

    # 验证
    print("[验证] ONNX 推理...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = sess.run(None, {"input": dummy_np})
    print(f"  输入: {dummy_np.shape}, 输出: {result[0].shape}")
    print(f"  logits: {result[0].flatten()}")
    print("  CNNSpotter 验证通过!")
    return True


def convert_universal_fake():
    """转换 UniversalFakeDetect (CLIP ViT-L/14 + 线性头) 为 ONNX."""
    print("\n" + "="*60)
    print("UniversalFakeDetect (CLIP ViT-L/14)")
    print("="*60)

    pth_path = os.path.join(WEIGHTS_DIR, "universal_fake.pth")
    onnx_path = os.path.join(WEIGHTS_DIR, "universal_fake.onnx")

    # 下载分类头权重 (尝试 GitHub raw)
    url = "https://raw.githubusercontent.com/Yuheng-Li/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth"
    if not os.path.exists(pth_path) or os.path.getsize(pth_path) < 1000:
        ok = download_file(url, pth_path, "UniversalFakeDetect fc_weights.pth")
        if not ok:
            print("  尝试备用地址...")
            # 备用: HuggingFace 镜像
            url2 = "https://huggingface.co/datasets/ohashi56225/UniversalFakeDetect/resolve/main/fc_weights.pth"
            ok = download_file(url2, pth_path, "UniversalFakeDetect fc_weights.pth (mirror)")
            if not ok:
                print("  跳过: 权重下载失败")
                return False

    # 安装 CLIP
    print("[依赖] 安装 CLIP...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git", "-q"])

    import clip

    # 加载 CLIP ViT-L/14
    print("[构建] CLIP ViT-L/14 视觉编码器...")
    clip_model, _ = clip.load("ViT-L/14", device="cpu")
    visual = clip_model.visual
    visual.eval()
    feat_dim = visual.output_dim  # 768 for ViT-L/14
    print(f"  特征维度: {feat_dim}")

    # 加载分类头
    print("[加载] 分类头权重...")
    classifier = nn.Linear(feat_dim, 2)
    state = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        # 尝试不同的 key 格式
        cls_state = {}
        for k, v in state.items():
            # 去掉前缀
            clean_k = k
            for prefix in ["fc.", "classifier.", "module.", "model."]:
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            cls_state[clean_k] = v
        try:
            classifier.load_state_dict(cls_state)
            print("  分类头加载成功 (dict format)")
        except Exception as e:
            print(f"  dict 加载失败: {e}")
            print(f"  可用 keys: {list(state.keys())[:5]}")
            # 尝试直接用 state
            try:
                classifier.load_state_dict(state)
                print("  分类头加载成功 (direct)")
            except Exception as e2:
                print(f"  直接加载也失败: {e2}")
                # 如果权重是单值（比如只是一个权重矩阵），手动设置
                if len(state) == 1:
                    v = list(state.values())[0]
                    if v.dim() == 2 and v.shape[0] == 2:
                        classifier.weight.data = v
                        classifier.bias.data.zero_()
                        print("  手动设置权重成功")
                    elif v.dim() == 2 and v.shape[1] == 2:
                        classifier.weight.data = v.T
                        classifier.bias.data.zero_()
                        print("  手动设置权重成功 (transposed)")
                    else:
                        print(f"  无法处理权重形状: {v.shape}")
                        return False
                else:
                    return False
    else:
        # state 直接是 tensor
        if state.dim() == 2 and state.shape[0] == 2:
            classifier.weight.data = state
            classifier.bias.data.zero_()
            print("  分类头加载成功 (tensor)")
        else:
            print(f"  无法处理权重形状: {state.shape}")
            return False

    classifier.eval()

    # 组合模型
    class CombinedModel(nn.Module):
        def __init__(self, visual, classifier):
            super().__init__()
            self.visual = visual
            self.classifier = classifier

        def forward(self, x):
            feat = self.visual(x)
            return self.classifier(feat)

    model = CombinedModel(visual, classifier)
    model.eval()

    # 导出 ONNX
    print("[导出] ONNX...")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"  导出成功: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

    # 验证
    print("[验证] ONNX 推理...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    result = sess.run(None, {"input": dummy_np})
    print(f"  输入: {dummy_np.shape}, 输出: {result[0].shape}")
    print(f"  logits: {result[0].flatten()}")
    print("  UniversalFakeDetect 验证通过!")
    return True


if __name__ == "__main__":
    print("AI 内容鉴真 - 深度学习模型权重下载与 ONNX 转换")
    print(f"输出目录: {WEIGHTS_DIR}")

    results = {}

    # CNNSpotter
    try:
        results["cnn_spotter"] = convert_cnn_spotter()
    except Exception as e:
        print(f"CNNSpotter 转换失败: {e}")
        import traceback
        traceback.print_exc()
        results["cnn_spotter"] = False

    # UniversalFakeDetect
    try:
        results["universal_fake"] = convert_universal_fake()
    except Exception as e:
        print(f"UniversalFakeDetect 转换失败: {e}")
        import traceback
        traceback.print_exc()
        results["universal_fake"] = False

    # 总结
    print("\n" + "="*60)
    print("转换总结")
    print("="*60)
    for name, ok in results.items():
        status = "成功" if ok else "失败"
        onnx_file = os.path.join(WEIGHTS_DIR, f"{name}.onnx")
        size = f"{os.path.getsize(onnx_file)/1024/1024:.2f} MB" if os.path.exists(onnx_file) else "N/A"
        print(f"  {name:20s} {status}  ({size})")

    # 清理 .pth 文件
    print("\n[清理] 删除临时 .pth 文件...")
    for f in ["cnn_spotter.pth", "universal_fake.pth"]:
        p = os.path.join(WEIGHTS_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"  已删除: {f}")

    print("\n完成! 将 .onnx 文件放入 models/weights/ 目录后重启服务即可。")
