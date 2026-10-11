#!/bin/bash
# ============================================================
# 模型权重下载脚本 (Linux/macOS)
# 下载开源检测模型的预训练权重，并导出为 ONNX 格式
#
# 用法: bash scripts/download_weights.sh
# ============================================================
set -e

WEIGHTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/models/weights"
mkdir -p "$WEIGHTS_DIR"

echo "============================================"
echo " AI 内容鉴真 - 模型权重下载"
echo " 输出目录: $WEIGHTS_DIR"
echo "============================================"

# ----- 1. CNNSpotter (ResNet50) -----
echo ""
echo "[1/3] Downloading CNNSpotter weights..."
if [ ! -f "$WEIGHTS_DIR/cnn_spotter.pth" ]; then
    # 注意：以下为示例 URL，实际使用时替换为真实权重地址
    # 官方仓库: https://github.com/peterwang512/CNNDetection
    echo "  请从 https://github.com/peterwang512/CNNDetection 下载 model_epoch_last_100.pth"
    echo "  保存为 $WEIGHTS_DIR/cnn_spotter.pth"
    echo "  然后运行: python scripts/convert_to_onnx.py --model cnn_spotter --checkpoint $WEIGHTS_DIR/cnn_spotter.pth --output $WEIGHTS_DIR/cnn_spotter.onnx"
else
    echo "  已存在，跳过下载"
fi

# ----- 2. UniversalFakeDetect -----
echo ""
echo "[2/3] Downloading UniversalFakeDetect weights..."
if [ ! -f "$WEIGHTS_DIR/universal_fake.pth" ]; then
    # 官方仓库: https://github.com/Yuheng-Li/UniversalFakeDetect
    echo "  请从 https://github.com/Yuheng-Li/UniversalFakeDetect 下载权重"
    echo "  保存为 $WEIGHTS_DIR/universal_fake.pth"
    echo "  然后运行: python scripts/convert_to_onnx.py --model universal_fake --checkpoint $WEIGHTS_DIR/universal_fake.pth --output $WEIGHTS_DIR/universal_fake.onnx"
else
    echo "  已存在，跳过下载"
fi

# ----- 3. AeroBlade -----
echo ""
echo "[3/3] Downloading AeroBlade weights..."
if [ ! -f "$WEIGHTS_DIR/aeroblade.pth" ]; then
    # 官方仓库: https://github.com/EPFL-VILAB/AeroBlade
    echo "  请从 https://github.com/EPFL-VILAB/AeroBlade 下载权重"
    echo "  保存为 $WEIGHTS_DIR/aeroblade.pth"
else
    echo "  已存在，跳过下载"
fi

echo ""
echo "============================================"
echo " 完成！"
echo " FreqDetector 无需权重，开箱即用。"
echo " Mock 模型用于开发测试，无需下载。"
echo "============================================"
echo ""
echo "下载权重后，修改 .env 中的 ENABLED_MODELS："
echo "  ENABLED_MODELS=cnn_spotter,universal_fake,aeroblade,freq_detector"
echo ""
