# -*- coding: utf-8 -*-
"""ProviderRegistry：按 models.json 中 provider 字段路由到具体适配类。"""
from __future__ import annotations

from ..core.config import ModelSpec, Settings
from .base import BaseProvider
from .openai_image import OpenAIImageProvider
from .volcengine import VolcengineProvider

_PROVIDERS = {
    "volcengine": VolcengineProvider,
    "openai": OpenAIImageProvider,
}


def get_provider(model: ModelSpec, settings: Settings) -> BaseProvider:
    cls = _PROVIDERS.get(model.provider)
    if cls is None:
        raise ValueError(
            f"未知供应商 '{model.provider}'（模型 {model.display_name}）。"
            f"接入全新供应商需在 src/providers/ 新增适配类并在 registry 注册。")
    return cls(settings)
