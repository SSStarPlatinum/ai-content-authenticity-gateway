"""
分布式令牌桶限流器。

基于 Redis 实现，支持多实例共享限流状态。
Redis 不可用时降级为进程内令牌桶。

算法：
  - 桶容量 = capacity（突发上限）
  - 每秒补充 rate 个令牌
  - 每次请求消耗 1 个令牌，不足则拒绝（429）
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import redis.asyncio as aioredis


class RateLimiter:
    """令牌桶限流器。"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        rate: float = 10.0,
        capacity: int = 20,
        enabled: bool = True,
        key_prefix: str = "ratelimit",
    ):
        self.rate = rate
        self.capacity = capacity
        self.enabled = enabled
        self.key_prefix = key_prefix
        self._redis: Optional[aioredis.Redis] = None
        self._redis_available = False
        # 进程内降级状态
        self._local_tokens: dict = {}
        self._local_last_refill: dict = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if not self.enabled:
            return
        try:
            self._redis = aioredis.from_url(
                redis_url if hasattr(self, '_redis_url') else "redis://localhost:6379/0",
                socket_connect_timeout=3,
                socket_timeout=2,
            )
            await self._redis.ping()
            self._redis_available = True
        except Exception:
            self._redis_available = False

    def setup_redis(self, redis_client: aioredis.Redis) -> None:
        """复用已有的 Redis 连接。"""
        self._redis = redis_client
        self._redis_available = True

    async def acquire(self, key: str = "global") -> tuple[bool, dict]:
        """
        尝试获取一个令牌。

        Returns:
            (allowed, info)  info 包含 remaining_tokens 和 retry_after_ms
        """
        if not self.enabled:
            return True, {"remaining": self.capacity, "retry_after_ms": 0}

        if self._redis_available and self._redis:
            return await self._acquire_redis(key)
        return await self._acquire_local(key)

    async def _acquire_redis(self, key: str) -> tuple[bool, dict]:
        """基于 Redis 的令牌桶（Lua 脚本保证原子性）。"""
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local data = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(data[1])
        local last_refill = tonumber(data[2])

        if tokens == nil then
            tokens = capacity
            last_refill = now
        end

        -- 补充令牌
        local elapsed = now - last_refill
        local refill = elapsed * rate
        tokens = math.min(capacity, tokens + refill)
        last_refill = now

        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end

        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
        redis.call('EXPIRE', key, 60)

        local retry_after = 0
        if allowed == 0 then
            retry_after = math.ceil((1 - tokens) / rate * 1000)
        end

        return {allowed, tokens, retry_after}
        """
        try:
            full_key = f"{self.key_prefix}:{key}"
            result = await self._redis.eval(
                lua_script,
                1,
                full_key,
                self.rate,
                self.capacity,
                time.time(),
            )
            allowed = bool(result[0])
            remaining = float(result[1])
            retry_after = int(result[2])
            return allowed, {"remaining": remaining, "retry_after_ms": retry_after}
        except Exception:
            self._redis_available = False
            return await self._acquire_local(key)

    async def _acquire_local(self, key: str) -> tuple[bool, dict]:
        """进程内令牌桶（降级方案）。"""
        async with self._lock:
            now = time.time()
            if key not in self._local_tokens:
                self._local_tokens[key] = float(self.capacity)
                self._local_last_refill[key] = now

            elapsed = now - self._local_last_refill[key]
            refill = elapsed * self.rate
            tokens = min(self.capacity, self._local_tokens[key] + refill)
            self._local_last_refill[key] = now

            if tokens >= 1:
                tokens -= 1
                self._local_tokens[key] = tokens
                return True, {"remaining": tokens, "retry_after_ms": 0}
            else:
                self._local_tokens[key] = tokens
                retry_after = int((1 - tokens) / self.rate * 1000)
                return False, {"remaining": tokens, "retry_after_ms": retry_after}
