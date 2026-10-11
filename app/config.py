"""
全局配置管理。
基于 pydantic-settings，支持 .env 文件与环境变量覆盖。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，字段名与 .env 一一对应（大写）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- 服务 -----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 2
    app_log_level: str = "info"
    app_env: str = "production"

    # ----- 安全 -----
    api_key: str = ""
    max_upload_mb: int = 20

    # ----- Redis -----
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    cache_ttl: int = 86400

    # ----- 限流 -----
    rate_limit_enabled: bool = True
    rate_limit_rate: float = 10.0
    rate_limit_capacity: int = 20

    # ----- 推理引擎 -----
    enabled_models: str = "mock"
    fusion_strategy: str = "weighted_vote"
    batch_max_size: int = 8
    batch_timeout_ms: int = 50
    inference_timeout: int = 30
    # 不确定判决区间：fake_prob <= uncertain_low 判 real，>= uncertain_high 判 fake，中间为 uncertain
    uncertain_low: float = 0.35
    uncertain_high: float = 0.7

    # ----- 模型 -----
    model_weights_dir: str = "./models/weights"

    # ----- 可观测性 -----
    prometheus_enabled: bool = True
    log_format: str = "json"

    @property
    def enabled_model_list(self) -> List[str]:
        """将逗号分隔的模型字符串解析为列表。"""
        return [m.strip() for m in self.enabled_models.split(",") if m.strip()]

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置，避免重复读取 .env。"""
    return Settings()


settings = get_settings()
