"""
Redis 缓存后端：图片检测结果缓存。

- 以图片 MD5 为键，缓存融合判决结果
- 支持 TTL 过期
- Redis 不可用时自动降级为内存缓存（不阻断推理）
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional

import redis.asyncio as aioredis


class CacheBackend:
    """
    缓存后端，优先 Redis，失败时降级为内存 LRU。
    所有方法均为异步，且永不抛出异常（缓存失败不应阻断主流程）。
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = 86400,
        memory_max_entries: int = 10000,
    ):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis: Optional[aioredis.Redis] = None
        self._memory: Dict[str, tuple] = {}  # key -> (value, expire_at)
        self._memory_max = memory_max_entries
        self._redis_available = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """连接 Redis，失败则仅使用内存缓存。"""
        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=2,
            )
            await self._redis.ping()
            self._redis_available = True
            print(f"[Cache] Redis connected: {self.redis_url}")
        except Exception as e:
            self._redis_available = False
            print(f"[Cache] Redis unavailable ({e}), using memory cache")

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """读取缓存，返回 None 表示未命中。"""
        # 1. 尝试 Redis
        if self._redis_available and self._redis:
            try:
                data = await self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                self._redis_available = False
        # 2. 内存缓存
        now = time.time()
        item = self._memory.get(key)
        if item and item[1] > now:
            return item[0]
        if key in self._memory:
            del self._memory[key]
        return None

    async def set(self, key: str, value: Dict[str, Any]) -> None:
        """写入缓存。"""
        # 1. Redis
        if self._redis_available and self._redis:
            try:
                await self._redis.setex(key, self.ttl, json.dumps(value, ensure_ascii=False))
                return
            except Exception:
                self._redis_available = False
        # 2. 内存
        async with self._lock:
            if len(self._memory) >= self._memory_max:
                # 简单淘汰：删除最早的条目
                oldest_key = next(iter(self._memory))
                del self._memory[oldest_key]
            self._memory[key] = (value, time.time() + self.ttl)

    async def clear(self) -> None:
        """清空缓存（运维用）。"""
        if self._redis_available and self._redis:
            try:
                await self._redis.flushdb()
            except Exception:
                pass
        self._memory.clear()

    @property
    def is_redis_available(self) -> bool:
        return self._redis_available
