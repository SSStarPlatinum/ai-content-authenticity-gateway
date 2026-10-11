"""
异步批量推理器。

将短时间内到达的多个请求攒成一个 batch，
调用模型的 batch_predict 一次性推理，显著提升高并发吞吐。

设计：
  - 使用 asyncio.Queue 收集请求
  - 达到 batch_max_size 或等待 batch_timeout_ms 后 flush
  - 每个请求通过 asyncio.Future 获取结果
  - 每个模型独立一个 BatchProcessor 实例
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..models.base import BaseDetector, DetectionResult


class _BatchItem:
    """单个待推理请求。"""
    __slots__ = ("image", "future", "arrival_time")

    def __init__(self, image: Image.Image, future: asyncio.Future):
        self.image = image
        self.future = future
        self.arrival_time = time.perf_counter()


class BatchProcessor:
    """
    单模型批量推理处理器。

    用法：
        processor = BatchProcessor(model, max_size=8, timeout_ms=50)
        asyncio.create_task(processor.run())
        result = await processor.submit(image)
    """

    def __init__(
        self,
        model: BaseDetector,
        max_size: int = 8,
        timeout_ms: int = 50,
    ):
        self.model = model
        self.max_size = max_size
        self.timeout_sec = timeout_ms / 1000.0
        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """启动后台批量处理循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        """停止后台循环。"""
        self._running = False
        if self._task:
            self._task.cancel()

    async def submit(self, image: Image.Image) -> DetectionResult:
        """提交一张图片，等待批量推理结果。"""
        if not self._running:
            # 未启动批量模式，退化为单图推理
            return await asyncio.to_thread(self.model.predict, image)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = _BatchItem(image, future)
        await self._queue.put(item)
        return await future

    async def _run(self) -> None:
        """后台循环：持续收集并 flush。"""
        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._flush(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # 批量推理失败时，逐个回退为单图推理
                print(f"[BatchProcessor:{self.model.model_id}] flush error: {e}")
                if batch:
                    await self._fallback_single(batch)

    async def _collect_batch(self) -> List[_BatchItem]:
        """收集一个批次：满员立即返回，否则等待超时。"""
        batch: List[_BatchItem] = []
        try:
            # 等待第一个元素
            first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            batch.append(first)
        except asyncio.TimeoutError:
            return []

        # 收集更多，直到满员或超时
        deadline = time.perf_counter() + self.timeout_sec
        while len(batch) < self.max_size and time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch

    async def _flush(self, batch: List[_BatchItem]) -> None:
        """执行一次批量推理并分发结果。"""
        images = [item.image for item in batch]
        start = time.perf_counter()
        try:
            results = await asyncio.to_thread(self.model.batch_predict, images)
            elapsed = (time.perf_counter() - start) * 1000
            # 批量推理均摊延迟到每个结果
            per_item_latency = elapsed / max(len(results), 1)
            for item, result in zip(batch, results):
                if result.latency_ms == 0.0:
                    result.latency_ms = per_item_latency
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as e:
            raise

    async def _fallback_single(self, batch: List[_BatchItem]) -> None:
        """批量失败时的降级：逐张单图推理。"""
        for item in batch:
            try:
                result = await asyncio.to_thread(self.model.predict, item.image)
                if not item.future.done():
                    item.future.set_result(result)
            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)


class BatchManager:
    """
    管理多个模型的 BatchProcessor，统一启停。
    """

    def __init__(
        self,
        models: List[BaseDetector],
        max_size: int = 8,
        timeout_ms: int = 50,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.processors: Dict[str, BatchProcessor] = {}
        if enabled:
            for model in models:
                self.processors[model.model_id] = BatchProcessor(
                    model, max_size=max_size, timeout_ms=timeout_ms
                )

    def start_all(self) -> None:
        for p in self.processors.values():
            p.start()

    def stop_all(self) -> None:
        for p in self.processors.values():
            p.stop()

    async def submit(self, model_id: str, image: Image.Image) -> DetectionResult:
        """提交到指定模型的批量处理器。"""
        if model_id in self.processors:
            return await self.processors[model_id].submit(image)
        # 未注册的模型直接单图推理
        raise KeyError(f"model '{model_id}' not in batch manager")
