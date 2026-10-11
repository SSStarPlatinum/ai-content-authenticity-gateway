"""
Prometheus 指标收集器。

指标：
  - auth_gateway_requests_total{endpoint,status}  请求总数
  - auth_gateway_request_duration_seconds{endpoint}  请求耗时直方图
  - auth_gateway_inference_duration_seconds{model}  单模型推理耗时
  - auth_gateway_detections_total{label}  检测结果分布
  - auth_gateway_cache_hits_total / cache_misses_total  缓存命中
  - auth_gateway_rate_limited_total  限流拒绝数
  - auth_gateway_model_errors_total{model}  模型错误数
"""
from __future__ import annotations

from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


class MetricsCollector:
    """Prometheus 指标收集器（单例模式）。"""

    _instance: Optional["MetricsCollector"] = None

    def __new__(cls, enabled: bool = True) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, enabled: bool = True):
        if getattr(self, "_initialized", False):
            return
        self.enabled = enabled
        self._registry = CollectorRegistry()
        self._init_metrics()
        self._initialized = True

    def _init_metrics(self) -> None:
        if not self.enabled:
            return

        self.requests_total = Counter(
            "auth_gateway_requests_total",
            "Total API requests",
            ["endpoint", "status"],
            registry=self._registry,
        )
        self.request_duration = Histogram(
            "auth_gateway_request_duration_seconds",
            "API request duration in seconds",
            ["endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self._registry,
        )
        self.inference_duration = Histogram(
            "auth_gateway_inference_duration_seconds",
            "Model inference duration in seconds",
            ["model"],
            buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=self._registry,
        )
        self.detections_total = Counter(
            "auth_gateway_detections_total",
            "Detection results by label",
            ["label"],
            registry=self._registry,
        )
        self.cache_hits = Counter(
            "auth_gateway_cache_hits_total",
            "Cache hits",
            registry=self._registry,
        )
        self.cache_misses = Counter(
            "auth_gateway_cache_misses_total",
            "Cache misses",
            registry=self._registry,
        )
        self.rate_limited = Counter(
            "auth_gateway_rate_limited_total",
            "Rate-limited requests",
            registry=self._registry,
        )
        self.model_errors = Counter(
            "auth_gateway_model_errors_total",
            "Model inference errors",
            ["model"],
            registry=self._registry,
        )

    # ----- 观测方法 -----

    def observe_request(self, endpoint: str, status: str, duration_sec: float) -> None:
        if not self.enabled:
            return
        self.requests_total.labels(endpoint=endpoint, status=status).inc()
        self.request_duration.labels(endpoint=endpoint).observe(duration_sec)

    def observe_inference(self, label: str, latency_ms: float, model_count: int) -> None:
        if not self.enabled:
            return
        self.detections_total.labels(label=label).inc()

    def observe_model_inference(self, model_id: str, duration_sec: float) -> None:
        if not self.enabled:
            return
        self.inference_duration.labels(model=model_id).observe(duration_sec)

    def observe_model_error(self, model_id: str) -> None:
        if not self.enabled:
            return
        self.model_errors.labels(model=model_id).inc()

    def observe_cache_hit(self) -> None:
        if not self.enabled:
            return
        self.cache_hits.inc()

    def observe_cache_miss(self) -> None:
        if not self.enabled:
            return
        self.cache_misses.inc()

    def observe_rate_limited(self) -> None:
        if not self.enabled:
            return
        self.rate_limited.inc()

    def generate_latest(self) -> bytes:
        """生成 Prometheus 文本格式指标。"""
        if not self.enabled:
            return b""
        return generate_latest(self._registry)
