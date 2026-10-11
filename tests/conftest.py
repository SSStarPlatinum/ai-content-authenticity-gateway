"""
pytest 配置与共享 fixtures。
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_image():
    """生成一张测试图片。"""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    return img


@pytest.fixture
def fake_image():
    """生成一张带噪点的图片（模拟 AI 生成特征）。"""
    import numpy as np
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def mock_models():
    """创建 mock 模型实例列表。"""
    from app.models.mock import MockDetector
    from app.models.freq_detector import FreqDetector

    m1 = MockDetector(simulate_latency_ms=0)
    m1.load()
    m2 = FreqDetector()
    m2.load()
    return [m1, m2]


@pytest.fixture
def fusion_engine():
    """创建融合引擎。"""
    from app.engine.fusion import FusionEngine
    return FusionEngine(strategy="weighted_vote")
