"""
FastAPI 应用入口：AI 内容鉴真私有化推理网关。

三层架构：
  接入网关层 (FastAPI + Nginx)
    -> 推理引擎层 (融合判决 + 批量推理 + 流水线)
      -> 模型层 (多模型适配器 + ONNX Runtime)

启动流程：
  1. 加载配置
  2. 初始化日志与指标
  3. 发现并加载检测模型
  4. 初始化缓存、限流、融合引擎、批量推理
  5. 组装推理流水线
  6. 注册路由与中间件
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router, setup_routes
from .cache.redis_cache import CacheBackend
from .config import settings
from .engine.batch import BatchManager
from .engine.fusion import FusionEngine
from .engine.pipeline import InferencePipeline
from .gateway.rate_limiter import RateLimiter
from .models.registry import auto_discover, registry
from .observability.logger import get_logger, setup_logging
from .observability.metrics import MetricsCollector
from .observability.middleware import setup_middleware

logger = get_logger("main")


def _load_models() -> list:
    """
    发现并实例化配置中启用的模型。
    权重加载失败的模型会被跳过，但不影响其他模型。
    """
    auto_discover()
    available = registry.list_models()
    logger.info("model_discovery", registered=available, requested=settings.enabled_model_list)

    models = []
    for model_id in settings.enabled_model_list:
        try:
            model = registry.create(model_id, weights_dir=settings.model_weights_dir)
            # 尝试加载权重，失败则跳过
            try:
                model.load()
                logger.info("model_loaded", model_id=model_id, name=model.model_name)
            except Exception as e:
                logger.warning(
                    "model_load_failed",
                    model_id=model_id,
                    error=str(e),
                    action="skipped",
                )
                continue
            models.append(model)
        except KeyError as e:
            logger.warning("model_not_found", model_id=model_id, error=str(e))

    if not models:
        logger.error("no_models_loaded", action="fallback_to_mock")
        # 兜底：至少加载 mock 模型保证服务可用
        mock = registry.create("mock", weights_dir=settings.model_weights_dir)
        mock.load()
        models.append(mock)

    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时清理。"""
    # ===== 启动 =====
    logger.info(
        "service_starting",
        version="1.0.0",
        env=settings.app_env,
        models=settings.enabled_model_list,
        fusion=settings.fusion_strategy,
    )

    # 1. 指标
    metrics = MetricsCollector(enabled=settings.prometheus_enabled)

    # 2. 模型加载
    models = _load_models()
    app.state.models = models

    # 3. 缓存
    cache = CacheBackend(
        redis_url=settings.redis_url,
        ttl=settings.cache_ttl,
    )
    await cache.connect()
    app.state.cache = cache

    # 4. 限流器（复用 Redis 连接）
    rate_limiter = RateLimiter(
        rate=settings.rate_limit_rate,
        capacity=settings.rate_limit_capacity,
        enabled=settings.rate_limit_enabled,
    )
    if cache.is_redis_available and cache._redis:
        rate_limiter.setup_redis(cache._redis)
    else:
        rate_limiter._redis_available = False
    app.state.rate_limiter = rate_limiter

    # 5. 融合引擎
    fusion = FusionEngine(
        strategy=settings.fusion_strategy,
        uncertain_low=settings.uncertain_low,
        uncertain_high=settings.uncertain_high,
    )

    # 6. 批量推理管理器
    batch_manager = BatchManager(
        models=models,
        max_size=settings.batch_max_size,
        timeout_ms=settings.batch_timeout_ms,
        enabled=True,
    )
    batch_manager.start_all()
    app.state.batch_manager = batch_manager

    # 7. 推理流水线
    pipeline = InferencePipeline(
        models=models,
        fusion=fusion,
        cache=cache,
        batch_manager=batch_manager,
        metrics=metrics,
        inference_timeout=settings.inference_timeout,
    )
    app.state.pipeline = pipeline

    # 8. 注入路由依赖
    setup_routes(pipeline, models, cache, rate_limiter, metrics)

    logger.info(
        "service_started",
        models_loaded=len(models),
        redis=cache.is_redis_available,
        batch_enabled=True,
    )

    yield

    # ===== 关闭 =====
    logger.info("service_shutting_down")
    batch_manager.stop_all()
    await cache.disconnect()
    logger.info("service_stopped")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    setup_logging(log_level=settings.app_log_level, log_format=settings.log_format)

    app = FastAPI(
        title="AI 内容鉴真私有化推理网关",
        description=(
            "企业级 AI 生成图像伪造检测推理服务。\n\n"
            "特性：多模型融合判决、ONNX 推理加速、图片哈希缓存、"
            "异步批量推理、分布式限流、全链路可观测。"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.debug = settings.app_env != "production"

    # 中间件（在路由注册前配置）
    metrics = MetricsCollector(enabled=settings.prometheus_enabled)
    setup_middleware(app, metrics)

    # 路由
    app.include_router(router)

    return app


# uvicorn 入口
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=settings.app_workers,
        log_level=settings.app_log_level,
    )
