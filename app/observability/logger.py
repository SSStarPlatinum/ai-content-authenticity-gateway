"""
结构化日志配置。

输出 JSON 格式日志，包含 request_id、latency、status 等字段，
便于 ELK / Loki 采集。
同时支持控制台彩色输出（开发环境）。
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import structlog


def setup_logging(log_level: str = "info", log_format: str = "json") -> None:
    """
    配置全局日志。

    Args:
        log_level: debug / info / warning / error
        log_format: json（生产）或 console（开发，彩色）
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 配置标准 logging（uvicorn 等第三方库）
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    # structlog 处理器链
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer(ensure_ascii=False))
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "auth-gateway", **kwargs: Any) -> structlog.stdlib.BoundLogger:
    """获取绑定了名称的结构化日志器。"""
    return structlog.get_logger(name, **kwargs)


def bind_request(request_id: str, **extra: Dict[str, Any]) -> None:
    """将 request_id 绑定到当前上下文（中间件中调用）。"""
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def unbind_request() -> None:
    """清除请求上下文绑定。"""
    structlog.contextvars.clear_contextvars()
