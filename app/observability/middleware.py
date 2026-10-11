"""
FastAPI 中间件：请求计时、日志、指标、CORS。

- RequestTimeMiddleware: 记录每个请求的耗时并输出结构化日志
- MetricsMiddleware: 上报 Prometheus 指标
- 统一异常处理：未捕获异常返回标准 JSON 错误
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .logger import bind_request, get_logger, unbind_request
from .metrics import MetricsCollector

logger = get_logger("middleware")


class RequestObservabilityMiddleware(BaseHTTPMiddleware):
    """请求可观测性中间件：计时 + 日志 + 指标。"""

    def __init__(self, app: FastAPI, metrics: MetricsCollector):
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(self, request: Request, call_next):
        # 生成 request_id
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        bind_request(request_id=request_id)

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            logger.error("unhandled_exception", error=str(e), path=request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal Server Error",
                    "detail": str(e) if request.app.state.debug else None,
                    "request_id": request_id,
                },
            )
        finally:
            duration = time.perf_counter() - start
            status_label = f"{status_code // 100}xx"
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(duration * 1000, 2),
                client=request.client.host if request.client else "unknown",
            )
            self.metrics.observe_request(
                endpoint=request.url.path,
                status=status_label,
                duration_sec=duration,
            )
            unbind_request()


def setup_middleware(app: FastAPI, metrics: MetricsCollector) -> None:
    """配置所有中间件。"""
    # CORS（内网部署可放宽）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 可观测性中间件
    app.add_middleware(RequestObservabilityMiddleware, metrics=metrics)
