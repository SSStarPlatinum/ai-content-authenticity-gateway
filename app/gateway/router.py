"""
网关路由层：API Key 校验、请求上下文。

作为 FastAPI 依赖注入使用，业务路由无需关心鉴权细节。
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from ..config import settings


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    API Key 校验依赖。
    若配置了 API_KEY，则请求头必须携带正确的 X-API-Key。
    未配置时跳过校验（内网部署场景）。
    """
    expected = settings.api_key
    if not expected:
        return "anonymous"

    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def request_context(request: Request) -> dict:
    """
    请求上下文：生成 request_id，记录开始时间。
    注入到请求 state，供日志和响应使用。
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return {"request_id": request_id}


def get_request_id(request: Request) -> str:
    """从请求 state 中获取 request_id。"""
    return getattr(request.state, "request_id", "unknown")
