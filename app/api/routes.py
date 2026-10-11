"""
API 路由定义。

端点：
  POST /api/v1/detect        图片鉴真检测
  GET  /api/v1/models        列出已加载模型
  GET  /health               健康检查
  GET  /metrics              Prometheus 指标
  POST /api/v1/cache/clear   清空缓存（运维）
"""
from __future__ import annotations

import time
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from ..cache.redis_cache import CacheBackend
from ..config import settings
from ..engine.pipeline import InferencePipeline
from ..gateway.rate_limiter import RateLimiter
from ..gateway.router import get_request_id, request_context, verify_api_key
from ..models.base import BaseDetector
from ..observability.metrics import MetricsCollector
from ..utils.hash import combined_cache_key
from ..utils.image import load_image
from .schemas import (
    DetectionResponse,
    HealthResponse,
    ModelInfo,
    ModelScore,
    ModelsResponse,
)

router = APIRouter()

# 全局依赖（由 main.py 注入）
_pipeline: InferencePipeline | None = None
_models: List[BaseDetector] = []
_cache: CacheBackend | None = None
_rate_limiter: RateLimiter | None = None
_metrics: MetricsCollector | None = None
_start_time = time.time()


def setup_routes(
    pipeline: InferencePipeline,
    models: List[BaseDetector],
    cache: CacheBackend,
    rate_limiter: RateLimiter,
    metrics: MetricsCollector,
) -> None:
    """注入全局依赖。"""
    global _pipeline, _models, _cache, _rate_limiter, _metrics
    _pipeline = pipeline
    _models = models
    _cache = cache
    _rate_limiter = rate_limiter
    _metrics = metrics


# ----- 检测接口 -----

@router.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    dependencies=[Depends(verify_api_key), Depends(request_context)],
    summary="图片鉴真检测",
    description="上传一张图片，返回 AI 生成概率与多模型详细分数。",
)
async def detect_image(
    request: Request,
    file: UploadFile = File(..., description="图片文件 (JPEG/PNG/WEBP)"),
):
    # 1. 限流
    if _rate_limiter:
        client_ip = request.client.host if request.client else "unknown"
        allowed, info = await _rate_limiter.acquire(key=f"ip:{client_ip}")
        if not allowed:
            if _metrics:
                _metrics.observe_rate_limited()
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_ms": info["retry_after_ms"],
                },
                headers={"Retry-After": str(max(1, info["retry_after_ms"] // 1000))},
            )

    # 2. 读取并校验图片
    raw_bytes = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large, max {settings.max_upload_mb}MB",
        )

    try:
        image, meta = load_image(raw_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image: {e}",
        )

    # 3. 计算缓存键
    cache_key = combined_cache_key(image, raw_bytes)

    # 4. 执行推理流水线
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = await _pipeline.detect(
        image=image,
        cache_key=cache_key,
        image_meta=meta.__dict__,
        raw_bytes=raw_bytes,
    )

    # 5. 组装响应
    request_id = get_request_id(request)
    per_model = []
    for pm in result.fusion.per_model:
        per_model.append(ModelScore(
            model_id=pm["model_id"],
            fake_prob=pm["fake_prob"],
            label=pm["label"],
            confidence=pm["confidence"],
            latency_ms=pm["latency_ms"],
            details=pm.get("details", {}),
            error=pm.get("error"),
        ))

    return DetectionResponse(
        fake_prob=result.fusion.fake_prob,
        label=result.fusion.label,
        confidence=result.fusion.confidence,
        strategy=result.fusion.strategy,
        total_latency_ms=result.total_latency_ms,
        cache_hit=result.cache_hit,
        model_count=result.fusion.model_count,
        failed_models=result.fusion.failed_models,
        per_model=per_model,
        image_meta=result.image_meta,
        postprocessing_type=result.postprocessing_type,
        request_id=request_id,
    )


# ----- 模型信息 -----

@router.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    summary="列出已加载模型",
)
async def list_models():
    model_infos = []
    for m in _models:
        model_infos.append(ModelInfo(
            model_id=m.model_id,
            model_name=m.model_name,
            version=m.version,
            is_loaded=m.is_loaded,
            weight=m.default_weight,
            strong_postprocessing=m.strong_postprocessing,
        ))
    return ModelsResponse(
        models=model_infos,
        fusion_strategy=settings.fusion_strategy,
        enabled_count=len(_models),
    )


# ----- 健康检查 -----

@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health():
    loaded = sum(1 for m in _models if m.is_loaded)
    return HealthResponse(
        status="ok" if loaded > 0 else "degraded",
        version="1.0.0",
        models_loaded=loaded,
        models_total=len(_models),
        redis_available=_cache.is_redis_available if _cache else False,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ----- 缓存管理 -----

@router.post(
    "/api/v1/cache/clear",
    dependencies=[Depends(verify_api_key)],
    summary="清空检测结果缓存",
)
async def clear_cache():
    if _cache:
        await _cache.clear()
    return {"success": True, "message": "cache cleared"}


# ----- Prometheus 指标 -----

@router.get("/metrics", include_in_schema=False)
async def metrics():
    if _metrics:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=_metrics.generate_latest())
    return JSONResponse(content={"error": "metrics disabled"}, status_code=404)
