"""
模型注册中心：集中管理所有检测模型的实例化、加载与查询。

采用装饰器注册模式，新增模型只需 @register_model 即可被发现。
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Type

from .base import BaseDetector


class ModelRegistry:
    """全局模型注册表（单例）。"""

    _instance: Optional["ModelRegistry"] = None
    _registry: Dict[str, Type[BaseDetector]] = {}

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, model_id: str, model_cls: Type[BaseDetector]) -> None:
        """注册一个模型类。"""
        if model_id in self._registry:
            raise ValueError(f"model_id '{model_id}' already registered")
        self._registry[model_id] = model_cls

    def get(self, model_id: str) -> Type[BaseDetector]:
        """按 ID 获取模型类。"""
        if model_id not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"model '{model_id}' not found. available: {available}")
        return self._registry[model_id]

    def list_models(self) -> List[str]:
        """列出所有已注册模型 ID。"""
        return sorted(self._registry.keys())

    def create(self, model_id: str, weights_dir: str = "./models/weights") -> BaseDetector:
        """实例化一个模型（不自动加载权重）。"""
        cls = self.get(model_id)
        return cls(weights_dir=weights_dir)

    def create_many(
        self, model_ids: List[str], weights_dir: str = "./models/weights"
    ) -> List[BaseDetector]:
        """批量实例化，跳过不存在的模型并记录。"""
        instances = []
        for mid in model_ids:
            try:
                instances.append(self.create(mid, weights_dir))
            except KeyError as e:
                print(f"[WARN] {e}, skipped")
        return instances


# 全局单例
registry = ModelRegistry()


def register_model(model_id: str):
    """类装饰器：将模型注册到全局注册表。"""
    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        registry.register(model_id, cls)
        cls.model_id = model_id
        return cls
    return decorator


def auto_discover() -> None:
    """
    自动导入所有模型模块，触发 @register_model 装饰器。
    在应用启动时调用一次。
    """
    modules = [
        "app.models.mock",
        "app.models.cnn_spotter",
        "app.models.universal_fake",
        "app.models.aeroblade",
        "app.models.freq_detector",
        "app.models.ela_detector",
        "app.models.noise_detector",
        "app.models.cfa_detector",
        "app.models.color_detector",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            print(f"[WARN] failed to import {mod}: {e}")
