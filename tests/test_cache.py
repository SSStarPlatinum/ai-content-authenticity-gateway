"""
缓存后端单元测试（使用内存缓存，无需 Redis）。
"""
import pytest
import asyncio

from app.cache.redis_cache import CacheBackend


@pytest.fixture
def cache():
    """创建内存缓存（Redis 不可用时自动降级）。"""
    c = CacheBackend(redis_url="redis://localhost:6399/0", ttl=60)
    # 不连接 Redis，直接使用内存缓存
    c._redis_available = False
    return c


class TestMemoryCache:
    def test_set_and_get(self, cache):
        asyncio.run(cache.set("key1", {"value": 42}))
        result = asyncio.run(cache.get("key1"))
        assert result == {"value": 42}

    def test_get_miss(self, cache):
        result = asyncio.run(cache.get("nonexistent"))
        assert result is None

    def test_overwrite(self, cache):
        asyncio.run(cache.set("key", {"v": 1}))
        asyncio.run(cache.set("key", {"v": 2}))
        result = asyncio.run(cache.get("key"))
        assert result == {"v": 2}

    def test_clear(self, cache):
        asyncio.run(cache.set("k1", {"a": 1}))
        asyncio.run(cache.set("k2", {"b": 2}))
        asyncio.run(cache.clear())
        assert asyncio.run(cache.get("k1")) is None
        assert asyncio.run(cache.get("k2")) is None

    def test_lru_eviction(self):
        """测试内存缓存的 LRU 淘汰。"""
        c = CacheBackend(ttl=60, memory_max_entries=3)
        c._redis_available = False
        for i in range(5):
            asyncio.run(c.set(f"key{i}", {"i": i}))
        # 最多保留 3 条
        assert len(c._memory) <= 3
