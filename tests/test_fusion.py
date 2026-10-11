"""
融合引擎单元测试。
"""
import pytest

from app.engine.fusion import FusionEngine
from app.models.base import DetectionResult


class TestFusionEngine:
    def test_weighted_vote_basic(self):
        """加权平均：两个模型概率应按权重平均。"""
        engine = FusionEngine(strategy="weighted_vote")
        r1 = DetectionResult(model_id="m1", fake_prob=0.8, confidence=0.6)
        r2 = DetectionResult(model_id="m2", fake_prob=0.4, confidence=0.8)
        from app.models.mock import MockDetector
        m1 = MockDetector(); m1.default_weight = 1.0
        m2 = MockDetector(); m2.default_weight = 1.0
        m1.model_id = "m1"; m2.model_id = "m2"

        out = engine.fuse([r1, r2], [m1, m2])
        assert abs(out.fake_prob - 0.6) < 0.01
        assert out.label == "fake"  # 0.6 >= 0.5

    def test_max_prob_strategy(self):
        """最大概率策略：取所有模型中最大的 fake_prob。"""
        engine = FusionEngine(strategy="max_prob")
        r1 = DetectionResult(model_id="m1", fake_prob=0.3)
        r2 = DetectionResult(model_id="m2", fake_prob=0.9)
        from app.models.mock import MockDetector
        m1 = MockDetector(); m1.model_id = "m1"
        m2 = MockDetector(); m2.model_id = "m2"

        out = engine.fuse([r1, r2], [m1, m2])
        assert abs(out.fake_prob - 0.9) < 0.01

    def test_confidence_weighted(self):
        """置信度加权：高置信度模型影响更大。"""
        engine = FusionEngine(strategy="confidence_weighted")
        r1 = DetectionResult(model_id="m1", fake_prob=0.9, confidence=0.9)
        r2 = DetectionResult(model_id="m2", fake_prob=0.1, confidence=0.1)
        from app.models.mock import MockDetector
        m1 = MockDetector(); m1.model_id = "m1"; m1.default_weight = 1.0
        m2 = MockDetector(); m2.model_id = "m2"; m2.default_weight = 1.0

        out = engine.fuse([r1, r2], [m1, m2])
        # 高置信度的 0.9 应占主导
        assert out.fake_prob > 0.7

    def test_all_models_fail(self):
        """所有模型失败时返回 uncertain。"""
        engine = FusionEngine()
        r1 = DetectionResult(model_id="m1", fake_prob=0.5, error="timeout")
        from app.models.mock import MockDetector
        m1 = MockDetector(); m1.model_id = "m1"

        out = engine.fuse([r1], [m1])
        assert out.label == "uncertain"
        assert out.model_count == 0
        assert "m1" in out.failed_models

    def test_partial_failure(self):
        """部分模型失败时，成功模型仍参与判决。"""
        engine = FusionEngine()
        r1 = DetectionResult(model_id="m1", fake_prob=0.9, confidence=0.8)
        r2 = DetectionResult(model_id="m2", fake_prob=0.5, error="crash")
        from app.models.mock import MockDetector
        m1 = MockDetector(); m1.model_id = "m1"
        m2 = MockDetector(); m2.model_id = "m2"

        out = engine.fuse([r1, r2], [m1, m2])
        assert out.model_count == 1
        assert abs(out.fake_prob - 0.9) < 0.01
        assert "m2" in out.failed_models

    def test_invalid_strategy(self):
        """无效策略应抛出异常。"""
        with pytest.raises(ValueError):
            FusionEngine(strategy="invalid")

    def test_dynamic_weight_postprocessing(self):
        """擅长该后处理的模型权重应被提升。"""
        engine = FusionEngine(strategy="weighted_vote")
        r1 = DetectionResult(model_id="m1", fake_prob=0.9)
        r2 = DetectionResult(model_id="m2", fake_prob=0.1)
        from app.models.mock import MockDetector
        m1 = MockDetector(); m1.model_id = "m1"; m1.default_weight = 1.0
        m1.strong_postprocessing = ["jpeg"]
        m2 = MockDetector(); m2.model_id = "m2"; m2.default_weight = 1.0
        m2.strong_postprocessing = []

        # jpeg 后处理时 m1 权重 *1.2
        out = engine.fuse([r1, r2], [m1, m2], postprocessing_type="jpeg")
        # (0.9*1.2 + 0.1*1.0) / 2.2 = 0.536
        assert out.fake_prob > 0.5
