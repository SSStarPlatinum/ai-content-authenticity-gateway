# 模型权重说明

本目录存放检测模型的 ONNX 权重文件。

## 目录结构

```
models/
├── README.md          # 本文件
└── weights/           # 权重文件目录（.onnx 文件不纳入版本控制）
    ├── cnn_spotter.onnx
    ├── universal_fake.onnx
    ├── aeroblade.onnx
    └── (freq_detector 无需权重)
```

## 模型清单

### 无权重模型（开箱即用，推荐默认启用）

| 模型 ID | 名称 | 原理 | 输入尺寸 | 权重文件 |
|---------|------|------|----------|----------|
| `freq_detector` | FreqDetector | 频域统计分析（对数幅度谱） | 256x256 | 无 |
| `ela_detector` | ELADetector | JPEG 压缩误差分析（Error Level Analysis） | 256x256 | 无 |
| `noise_detector` | NoiseResidualDetector | 噪声残差模式分析（中值滤波+自相关+频谱周期性） | 256x256 | 无 |
| `cfa_detector` | CFADetector | Bayer CFA 插值伪影检测（周期性频谱峰值） | 256x256 | 无 |
| `color_detector` | ColorDistributionDetector | HSV 色彩分布统计（饱和度/色相熵/色彩丰富度） | 256x256 | 无 |
| `mock` | Mock Detector | 像素统计模拟（仅开发测试用） | 任意 | 无 |

### 需权重模型（下载 ONNX 权重后启用）

| 模型 ID | 名称 | 原理 | 输入尺寸 | 权重文件 |
|---------|------|------|----------|----------|
| `cnn_spotter` | CNNSpotter | ResNet50 微调 | 224x224 | `cnn_spotter.onnx` |
| `universal_fake` | UniversalFakeDetect | CLIP + 线性分类头 | 224x224 | `universal_fake.onnx` |
| `aeroblade` | AeroBlade | 扩散模型专用检测 | 512x512 | `aeroblade.onnx` |

## 获取权重

### 方式一：使用下载脚本

```bash
# Linux/macOS
bash scripts/download_weights.sh

# Windows PowerShell
.\scripts\download_weights.ps1
```

### 方式二：手动下载并转换

1. 从对应开源仓库下载 PyTorch 权重（.pth/.pt）
2. 使用转换脚本导出为 ONNX：

```bash
# CNNSpotter
python scripts/convert_to_onnx.py \
  --model cnn_spotter \
  --checkpoint models/weights/cnn_spotter.pth \
  --output models/weights/cnn_spotter.onnx

# UniversalFakeDetect
python scripts/convert_to_onnx.py \
  --model universal_fake \
  --checkpoint models/weights/universal_fake.pth \
  --output models/weights/universal_fake.onnx
```

3. （可选）量化压缩以减小体积、提升速度：

```bash
python scripts/quantize.py \
  --input models/weights/cnn_spotter.onnx \
  --output models/weights/cnn_spotter_int8.onnx \
  --method dynamic
```

## 开源模型参考

- **CNNDetection**: https://github.com/peterwang512/CNNDetection
- **UniversalFakeDetect**: https://github.com/Yuheng-Li/UniversalFakeDetect
- **AeroBlade**: https://github.com/EPFL-VILAB/AeroBlade

## 启用模型

修改 `.env` 文件：

```env
# 推荐：5款无权重模型融合（开箱即用，多原理互补）
ENABLED_MODELS=freq_detector,ela_detector,noise_detector,cfa_detector,color_detector

# 开发测试
ENABLED_MODELS=mock

# 生产环境（含深度学习模型，需先下载权重）
ENABLED_MODELS=freq_detector,ela_detector,noise_detector,cfa_detector,color_detector,cnn_spotter,universal_fake,aeroblade
```

无权重模型基于完全不同的分析原理（频域、压缩误差、噪声模式、传感器伪影、色彩分布），
融合后可显著提升对各类 AI 生成图像的检测鲁棒性。权重加载失败的模型会被自动跳过，不影响其他模型正常工作。
