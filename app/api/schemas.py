"""
API 请求/响应 Schema（Pydantic v2）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ----- 响应模型 -----

class ModelScore(BaseModel):
    """单模型分数。"""
    model_id: str
    model_name: Optional[str] = None
    fake_prob: float
    label: str
    confidence: float
    latency_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class DetectionResponse(BaseModel):
    """检测接口响应。"""
    success: bool = True
    fake_prob: float = Field(..., description="融合后的伪造概率 [0,1]")
    label: str = Field(..., description="最终判决: fake / real / uncertain")
    confidence: float = Field(..., description="判决置信度 [0,1]")
    strategy: str = Field(..., description="使用的融合策略")
    total_latency_ms: float
    cache_hit: bool
    model_count: int
    failed_models: List[str] = Field(default_factory=list)
    per_model: List[ModelScore] = Field(default_factory=list)
    image_meta: Dict[str, Any] = Field(default_factory=dict)
    postprocessing_type: str = "none"
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    version: str
    models_loaded: int
    models_total: int
    redis_available: bool
    uptime_seconds: float


class ModelInfo(BaseModel):
    """模型信息。"""
    model_id: str
    model_name: str
    version: str
    is_loaded: bool
    weight: float
    strong_postprocessing: List[str]


class ModelsResponse(BaseModel):
    """模型列表响应。"""
    models: List[ModelInfo]
    fusion_strategy: str
    enabled_count: int


class ErrorResponse(BaseModel):
    """统一错误响应。"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class RateLimitResponse(BaseModel):
    """限流响应。"""
    detail: str
    retry_after_ms: int
