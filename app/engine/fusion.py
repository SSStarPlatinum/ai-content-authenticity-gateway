"""
多模型融合判决引擎。

支持三种策略：
  - weighted_vote: 加权平均（默认）
  - max_prob: 取最大伪造概率（严格策略，降低漏报）
  - confidence_weighted: 以模型置信度为动态权重

同时支持基于后处理类型的动态权重调整：
  当检测到图片经过 JPEG 压缩/缩放时，提升对该后处理鲁棒的模型权重。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..models.base import BaseDetector, DetectionResult


@dataclass
class FusionOutput:
    """融合后的最终判决结果。"""
    fake_prob: float
    label: str
    confidence: float
    strategy: str
    per_model: List[Dict]
    model_count: int
    failed_models: List[str]

    def to_dict(self) -> Dict:
        return {
            "fake_prob": round(self.fake_prob, 6),
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "strategy": self.strategy,
            "model_count": self.model_count,
            "failed_models": self.failed_models,
            "per_model": self.per_model,
        }


class FusionEngine:
    """多模型融合判决器。"""

    VALID_STRATEGIES = {"weighted_vote", "max_prob", "confidence_weighted", "adaptive"}

    def __init__(
        self,
        strategy: str = "weighted_vote",
        model_weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        uncertain_low: float = 0.4,
        uncertain_high: float = 0.6,
    ):
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"invalid strategy '{strategy}', choose from {self.VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.model_weights = model_weights or {}
        self.threshold = threshold
        # 不确定区间：fake_prob 在 [uncertain_low, uncertain_high] 之间时判为 uncertain
        self.uncertain_low = uncertain_low
        self.uncertain_high = uncertain_high

    def _get_weight(self, model: BaseDetector) -> float:
        """获取模型权重：配置覆盖 > 默认权重。"""
        return self.model_weights.get(model.model_id, model.default_weight)

    def fuse(
        self,
        results: List[DetectionResult],
        models: List[BaseDetector],
        postprocessing_type: str = "none",
    ) -> FusionOutput:
        """
        执行融合判决。

        Args:
            results: 各模型的检测结果
            models: 对应的模型实例（用于取权重）
            postprocessing_type: 图片后处理类型，用于动态加权
        """
        # 分离成功/失败结果
        success = [(r, m) for r, m in zip(results, models) if r.error is None]
        failed = [r.model_id for r in results if r.error is not None]

        if not success:
            return FusionOutput(
                fake_prob=0.5,
                label="uncertain",
                confidence=0.0,
                strategy=self.strategy,
                per_model=[r.to_dict() for r in results],
                model_count=0,
                failed_models=failed,
            )

        # 动态加权：若模型擅长该后处理，权重 *1.2
        dynamic_weights = []
        for _, model in success:
            w = self._get_weight(model)
            if postprocessing_type in model.strong_postprocessing:
                w *= 1.2
            dynamic_weights.append(w)

        if self.strategy == "weighted_vote":
            fake_prob = self._weighted_vote(success, dynamic_weights)
        elif self.strategy == "max_prob":
            fake_prob = self._max_prob(success)
        elif self.strategy == "adaptive":
            fake_prob = self._adaptive_fusion(success, dynamic_weights)
        else:  # confidence_weighted
            fake_prob = self._confidence_weighted(success, dynamic_weights)

        confidence = self._compute_confidence(success, fake_prob)
        # 三态判决：低于下限=real，高于上限=fake，中间=uncertain
        if fake_prob >= self.uncertain_high:
            label = "fake"
        elif fake_prob <= self.uncertain_low:
            label = "real"
        else:
            label = "uncertain"

        return FusionOutput(
            fake_prob=fake_prob,
            label=label,
            confidence=confidence,
            strategy=self.strategy,
            per_model=[r.to_dict() for r in results],
            model_count=len(success),
            failed_models=failed,
        )

    # ----- 策略实现 -----

    @staticmethod
    def _weighted_vote(
        success: List[tuple], weights: List[float]
    ) -> float:
        total_w = sum(weights)
        if total_w == 0:
            return 0.5
        weighted_sum = sum(r.fake_prob * w for (r, _), w in zip(success, weights))
        return weighted_sum / total_w

    @staticmethod
    def _max_prob(success: List[tuple]) -> float:
        return max(r.fake_prob for r, _ in success)

    @staticmethod
    def _confidence_weighted(
        success: List[tuple], base_weights: List[float]
    ) -> float:
        total = 0.0
        weight_sum = 0.0
        for (r, _), bw in zip(success, base_weights):
            w = bw * max(r.confidence, 0.01)
            total += r.fake_prob * w
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.5

    @staticmethod
    def _adaptive_fusion(
        success: List[tuple], base_weights: List[float]
    ) -> float:
        """
        自适应融合：先加权投票，再用 ela_detector 作为仲裁信号。
        当深度学习模型一致判 real（prob<0.1），但 ela_detector 给出较高 fake_prob（>0.48）时，
        说明可能遇到了未见生成器，将结果调整为 uncertain，避免硬漏检。
        ela_detector 对 real 约 0.40-0.47，对未见生成器的 fake 约 0.48-0.55，有微弱区分度。
        """
        total_w = sum(base_weights)
        if total_w == 0:
            return 0.5
        base_prob = sum(r.fake_prob * w for (r, _), w in zip(success, base_weights)) / total_w

        # 收集各模型概率
        prob_map = {}
        weight_map = {}
        for (r, model), w in zip(success, base_weights):
            prob_map[model.model_id] = r.fake_prob
            weight_map[model.model_id] = w

        # 高权重（深度学习）模型
        high_ids = [mid for mid, w in weight_map.items() if w >= 2.0]
        if not high_ids:
            return base_prob

        high_avg = sum(prob_map[mid] for mid in high_ids) / len(high_ids)

        # 条件：深度学习模型一致判 real（平均 < 0.1，且都 < 0.3）
        deep_all_real = all(prob_map[mid] < 0.3 for mid in high_ids) and high_avg < 0.1

        if deep_all_real and "ela_detector" in prob_map:
            ela_prob = prob_map["ela_detector"]
            # ela > 0.48 是弱 fake 信号（real 通常 < 0.47）
            if ela_prob > 0.48:
                # 提升到 uncertain 区间，但不超过 0.6
                adjusted = max(base_prob, 0.42)
                adjusted = min(adjusted, 0.58)
                return adjusted

        return base_prob

    @staticmethod
    def _compute_confidence(success: List[tuple], final_prob: float) -> float:
        """
        融合置信度：基于模型间一致性 + 最终概率偏离 0.5 的程度。
        一致性越高、概率越极端，置信度越高。
        """
        probs = [r.fake_prob for r, _ in success]
        if len(probs) <= 1:
            return abs(final_prob - 0.5) * 2
        # 一致性：1 - 标准差（归一化）
        import numpy as np
        std = float(np.std(probs))
        consistency = max(0.0, 1.0 - std * 2)
        extremity = abs(final_prob - 0.5) * 2
        return min(1.0, consistency * 0.5 + extremity * 0.5)
