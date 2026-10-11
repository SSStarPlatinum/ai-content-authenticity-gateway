"""
新增无权重检测器的单元测试。
验证：ELA、噪声残差、CFA、色彩分布 4 款检测器的接口正确性与输出范围。
"""
import numpy as np
import pytest
from PIL import Image

from app.models.ela_detector import ELADetector
from app.models.noise_detector import NoiseResidualDetector
from app.models.cfa_detector import CFADetector
from app.models.color_detector import ColorDistributionDetector
from app.models.freq_detector import FreqDetector


@pytest.fixture
def real_like_image():
    """生成模拟真实照片的图片（有渐变+噪声+纹理）。"""
    np.random.seed(42)
    h, w = 256, 256
    # 渐变背景
    y, x = np.mgrid[0:h, 0:w]
    r = (x / w * 200 + 30).astype(np.uint8)
    g = (y / h * 180 + 40).astype(np.uint8)
    b = ((x + y) / (h + w) * 160 + 50).astype(np.uint8)
    # 添加传感器噪声
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    arr = np.stack([r, g, b], axis=2).astype(np.int16) + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def synthetic_image():
    """生成模拟 AI 生成的图片（平滑、低噪声、色彩均匀）。"""
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]
    # 非常平滑的渐变
    r = (x / w * 180 + 40).astype(np.uint8)
    g = (y / h * 160 + 50).astype(np.uint8)
    b = np.full((h, w), 120, dtype=np.uint8)
    arr = np.stack([r, g, b], axis=2)
    # 轻微高斯模糊效果（通过邻域平均模拟）
    from PIL import ImageFilter
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


class TestELADetector:
    def test_interface(self, real_like_image):
        model = ELADetector()
        model.load()
        result = model.predict(real_like_image)
        assert result.model_id == "ela_detector"
        assert 0 <= result.fake_prob <= 1
        assert result.label in ("fake", "real", "uncertain")
        assert result.latency_ms >= 0
        assert "ela_mean" in result.details
        assert "ela_std" in result.details

    def test_different_images_different_results(self, real_like_image, synthetic_image):
        model = ELADetector()
        model.load()
        r1 = model.predict(real_like_image)
        r2 = model.predict(synthetic_image)
        # 不同图片应返回不同概率
        assert r1.fake_prob != r2.fake_prob


class TestNoiseDetector:
    def test_interface(self, real_like_image):
        model = NoiseResidualDetector()
        model.load()
        result = model.predict(real_like_image)
        assert result.model_id == "noise_detector"
        assert 0 <= result.fake_prob <= 1
        assert "skewness" in result.details
        assert "kurtosis" in result.details
        assert "avg_autocorrelation" in result.details

    def test_noise_features(self, real_like_image, synthetic_image):
        model = NoiseResidualDetector()
        model.load()
        r_real = model.predict(real_like_image)
        r_synth = model.predict(synthetic_image)
        # 真实照片噪声 std 应大于合成平滑图
        assert r_real.details["noise_std"] > r_synth.details["noise_std"] * 0.5


class TestCFADetector:
    def test_interface(self, real_like_image):
        model = CFADetector()
        model.load()
        result = model.predict(real_like_image)
        assert result.model_id == "cfa_detector"
        assert 0 <= result.fake_prob <= 1
        assert "avg_periodicity" in result.details
        assert "cross_channel_corr" in result.details

    def test_periodicity_in_range(self, real_like_image):
        model = CFADetector()
        model.load()
        result = model.predict(real_like_image)
        assert 0 <= result.details["avg_periodicity"] <= 1


class TestColorDetector:
    def test_interface(self, real_like_image):
        model = ColorDistributionDetector()
        model.load()
        result = model.predict(real_like_image)
        assert result.model_id == "color_detector"
        assert 0 <= result.fake_prob <= 1
        assert "saturation_mean" in result.details
        assert "hue_entropy" in result.details
        assert "color_richness" in result.details

    def test_saturation_range(self, real_like_image):
        model = ColorDistributionDetector()
        model.load()
        result = model.predict(real_like_image)
        assert 0 <= result.details["saturation_mean"] <= 1


class TestAllWeightFreeDetectors:
    """所有无权重检测器的集成测试。"""

    def test_all_load_and_predict(self, real_like_image):
        detectors = [
            FreqDetector(),
            ELADetector(),
            NoiseResidualDetector(),
            CFADetector(),
            ColorDistributionDetector(),
        ]
        for d in detectors:
            d.load()
            assert d.is_loaded
            result = d.predict(real_like_image)
            assert 0 <= result.fake_prob <= 1
            assert result.error is None, f"{d.model_id} failed: {result.error}"

    def test_fusion_with_all_detectors(self, real_like_image):
        """5款无权重模型融合测试。"""
        from app.engine.fusion import FusionEngine

        detectors = [
            FreqDetector(),
            ELADetector(),
            NoiseResidualDetector(),
            CFADetector(),
            ColorDistributionDetector(),
        ]
        for d in detectors:
            d.load()

        results = [d.predict(real_like_image) for d in detectors]
        fusion = FusionEngine(strategy="weighted_vote")
        out = fusion.fuse(results, detectors)

        assert out.model_count == 5
        assert out.failed_models == []
        assert 0 <= out.fake_prob <= 1
        assert out.label in ("fake", "real", "uncertain")
