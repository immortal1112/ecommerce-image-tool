# -*- coding: utf-8 -*-
"""配置管理：settings.json（用户可写） + models.json（模型能力声明，数据驱动 UI）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from .paths import config_dir, settings_file

IMAGE_TYPE_LABELS = {"main": "主图", "white": "白底", "selling": "卖点", "scene": "场景"}


@dataclass
class Settings:
    """用户设置（本地记住，不进 metadata / 不进日志）。"""
    work_dir: str = ""                      # 工作目录（首启选择并记住）
    volcengine_api_key: str = ""            # 火山方舟 API Key
    openai_api_key: str = ""                # OpenAI API Key
    text_llm_api_key: str = ""              # 提示词优化模型 Key（默认 DeepSeek）
    text_llm_base_url: str = "https://api.deepseek.com"
    text_llm_model: str = "deepseek-chat"
    volcengine_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    openai_base_url: str = ""               # 留空=官方默认
    concurrency: int = 3                    # 并发数，默认 3
    max_retries: int = 3                    # 限流/瞬时错误最大重试次数
    response_mode: str = "b64"              # b64=直接落盘(默认) / url=返回链接再下载(调试)
    watermark: bool = False                 # 火山水印开关，默认关闭
    last_model_key: str = ""                # 记住上次所选模型

    def save(self) -> None:
        settings_file().write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        p = settings_file()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in data.items() if k in known})
        except Exception:
            return cls()


@dataclass
class ModelSpec:
    """一个模型的能力声明（来自 models.json，驱动两级联动 UI 与参数翻译）。"""
    key: str
    display_name: str
    provider: str
    model_id: str
    supports_reference_images: bool
    max_reference_images: int
    supported_ratios: list[str]
    supported_tiers: list[str]
    output_formats: list[str]
    size_map: dict[str, dict[str, str]]
    quality_options: Optional[list[str]] = None
    default_quality: Optional[str] = None
    pricing: Optional[dict[str, Any]] = None

    def resolve_size(self, ratio: str, tier: str) -> str:
        """(比例, 档位) -> 精确像素串。这是尺寸归一化的唯一入口。"""
        try:
            return self.size_map[ratio][tier]
        except KeyError:
            raise ValueError(
                f"{self.display_name} 不支持 比例{ratio}+分辨率{tier} 组合，"
                f"可用组合见模型参数区提示。")

    def estimate_cost(self, tier: str, quality: Optional[str]) -> tuple[Optional[float], str]:
        """估算单张成本 -> (金额或None, 货币)。价格未配置返回 (None, '')。"""
        if not self.pricing:
            return None, ""
        cur = self.pricing.get("currency", "")
        by_tier = self.pricing.get("per_image_by_tier")
        if by_tier and tier in by_tier:
            return float(by_tier[tier]), cur
        by_q = self.pricing.get("per_image_by_quality_1024")
        if by_q and quality in by_q:
            return float(by_q[quality]), cur
        return None, cur


class ModelRegistry:
    """加载 models.json，向 GUI 提供能力查询。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (config_dir() / "models.json")
        self.models: dict[str, ModelSpec] = {}
        self.reload()

    def reload(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.models = {}
        for key, m in data["models"].items():
            self.models[key] = ModelSpec(
                key=key,
                display_name=m["display_name"],
                provider=m["provider"],
                model_id=m["model_id"],
                supports_reference_images=m.get("supports_reference_images", True),
                max_reference_images=m.get("max_reference_images", 10),
                supported_ratios=m["supported_ratios"],
                supported_tiers=m["supported_tiers"],
                output_formats=m["output_formats"],
                size_map=m["size_map"],
                quality_options=m.get("quality_options"),
                default_quality=m.get("default_quality"),
                pricing=m.get("pricing"),
            )

    def get(self, key: str) -> ModelSpec:
        return self.models[key]

    def keys(self) -> list[str]:
        return list(self.models.keys())
