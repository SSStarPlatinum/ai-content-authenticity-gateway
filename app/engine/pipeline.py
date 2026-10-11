"""
推理流水线：端到端编排。

流程：
  1. 检查图片哈希缓存（命中则直接返回）
  2. 并行调用所有启用模型的推理（通过 BatchProcessor 攒批）
  3. 融合引擎综合判决
  4. 结果写入缓存
  5. 返回标准化响应

同时负责：
  - 后处理类型检测（JPEG 压缩痕迹、缩放比例估计）
  - 推理超时控制
  - 单模型失败降级（不影响整体判决）
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from ..cache.redis_cache import CacheBackend
from ..models.base import BaseDetector, DetectionResult
from ..observability.metrics import MetricsCollector
from .batch import BatchManager
from .fusion import FusionEngine, FusionOutput


@dataclass
class PipelineResult:
    """流水线最终输出。"""
    fusion: FusionOutput
    total_latency_ms: float
    cache_hit: bool
    image_meta: Dict
    postprocessing_type: str

    def to_dict(self) -> Dict:
        d = self.fusion.to_dict()
        d["total_latency_ms"] = round(self.total_latency_ms, 2)
        d["cache_hit"] = self.cache_hit
        d["image_meta"] = self.image_meta
        d["postprocessing_type"] = self.postprocessing_type
        return d


class InferencePipeline:
    """推理流水线主类。"""

    def __init__(
        self,
        models: List[BaseDetector],
        fusion: FusionEngine,
        cache: Optional[CacheBackend] = None,
        batch_manager: Optional[BatchManager] = None,
        metrics: Optional[MetricsCollector] = None,
        inference_timeout: int = 30,
    ):
        self.models = models
        self.fusion = fusion
        self.cache = cache
        self.batch_manager = batch_manager
        self.metrics = metrics
        self.inference_timeout = inference_timeout

    async def detect(
        self,
        image: Image.Image,
        cache_key: str,
        image_meta: Dict,
        raw_bytes: bytes,
    ) -> PipelineResult:
        """
        执行完整检测流水线。

        Args:
            image: 已加载的 PIL 图片
            cache_key: 图片缓存键
            image_meta: 图片元信息
            raw_bytes: 原始字节（用于后处理检测）
        """
        start = time.perf_counter()

        # 1. 缓存查询
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                latency = (time.perf_counter() - start) * 1000
                if self.metrics:
                    self.metrics.observe_cache_hit()
                return PipelineResult(
                    fusion=FusionOutput(**cached["fusion"]),
                    total_latency_ms=latency,
                    cache_hit=True,
                    image_meta=image_meta,
                    postprocessing_type=cached.get("postprocessing_type", "none"),
                )

        if self.metrics:
            self.metrics.observe_cache_miss()

        # 2. 检测后处理类型
        post_type = self._detect_postprocessing(raw_bytes, image_meta)

        # 3. 并行推理（带超时）
        results = await self._run_inference_parallel(image)

        # 4. 融合
        fusion_out = self.fusion.fuse(results, self.models, post_type)

        # 5. 写缓存
        if self.cache:
            await self.cache.set(
                cache_key,
                {"fusion": fusion_out.to_dict(), "postprocessing_type": post_type},
            )

        total_latency = (time.perf_counter() - start) * 1000

        # 6. 指标
        if self.metrics:
            self.metrics.observe_inference(
                label=fusion_out.label,
                latency_ms=total_latency,
                model_count=fusion_out.model_count,
            )

        return PipelineResult(
            fusion=fusion_out,
            total_latency_ms=total_latency,
            cache_hit=False,
            image_meta=image_meta,
            postprocessing_type=post_type,
        )

    async def _run_inference_parallel(
        self, image: Image.Image
    ) -> List[DetectionResult]:
        """并行调用所有模型推理，单个失败不影响其他。"""
        async def _infer_one(model: BaseDetector) -> DetectionResult:
            try:
                if self.batch_manager and model.model_id in self.batch_manager.processors:
                    return await asyncio.wait_for(
                        self.batch_manager.submit(model.model_id, image),
                        timeout=self.inference_timeout,
                    )
                return await asyncio.wait_for(
                    asyncio.to_thread(model.predict, image),
                    timeout=self.inference_timeout,
                )
            except asyncio.TimeoutError:
                return DetectionResult(
                    model_id=model.model_id,
                    fake_prob=0.5,
                    label="uncertain",
                    confidence=0.0,
                    error="inference timeout",
                )
            except Exception as e:
                return DetectionResult(
                    model_id=model.model_id,
                    fake_prob=0.5,
                    label="uncertain",
                    confidence=0.0,
                    error=f"{type(e).__name__}: {e}",
                )

        tasks = [_infer_one(m) for m in self.models]
        results = await asyncio.gather(*tasks)
        return list(results)

    @staticmethod
    def _detect_postprocessing(raw_bytes: bytes, meta: Dict) -> str:
        """
        简易后处理类型检测：
        - JPEG: 文件头为 FF D8 FF
        - 其他格式视为 none（resize/crop 难以从单图精确判断，预留）
        """
        if len(raw_bytes) >= 3 and raw_bytes[:3] == b"\xff\xd8\xff":
            return "jpeg"
        return "none"
