# -*- coding: utf-8 -*-
"""OpenAI GPT Image Provider。

调用方式（官方文档核实，2026-07）：
- 无参考图：POST /v1/images/generations（images.generate）
- 有参考图：POST /v1/images/edits（images.edit，直接传文件字节，最多4张较稳）
- size 为精确像素串（16 的倍数、长短边比<=3:1，映射表已在 models.json 内保证合规）
- 支持 quality: low/medium/high；不支持透明背景（白底靠提示词实现）
- 返回恒为 b64_json。
"""
from __future__ import annotations

import base64

from openai import APIStatusError, OpenAI

from .base import (BaseProvider, GenOutcome, GenTask, ProviderError,
                   translate_http_error)


class OpenAIImageProvider(BaseProvider):
    name = "openai"

    def check_api_key(self):
        if not self.settings.openai_api_key.strip():
            return "尚未配置 OpenAI API Key，请到「设置」中填写。"
        return None

    def _client(self) -> OpenAI:
        kw = {"api_key": self.settings.openai_api_key, "timeout": 300.0}
        if self.settings.openai_base_url.strip():
            kw["base_url"] = self.settings.openai_base_url.strip()
        return OpenAI(**kw)

    def _do_generate(self, task: GenTask) -> GenOutcome:
        size = task.model.resolve_size(task.ratio, task.tier)
        quality = task.quality or task.model.default_quality or "medium"
        common = dict(model=task.model.model_id, prompt=task.prompt,
                      size=size, quality=quality,
                      output_format=task.output_format)
        files = []
        try:
            if task.reference_images:
                if len(task.reference_images) > task.model.max_reference_images:
                    raise ProviderError(
                        f"{task.model.display_name} 建议最多 "
                        f"{task.model.max_reference_images} 张参考图，请减少后重试。")
                files = [open(p, "rb") for p in task.reference_images]
                resp = self._client().images.edit(
                    image=files if len(files) > 1 else files[0], **common)
            else:
                resp = self._client().images.generate(**common)
        except APIStatusError as e:
            raise translate_http_error(e.status_code, getattr(e, "message", str(e)), "OpenAI")
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"连接 OpenAI 失败：{e}。请检查网络/代理设置后重试。",
                                retryable=True)
        finally:
            for f in files:
                try:
                    f.close()
                except Exception:
                    pass

        if not getattr(resp, "data", None):
            raise ProviderError("OpenAI 返回为空（可能被内容审核拦截），请调整提示词后重试。")
        img = base64.b64decode(resp.data[0].b64_json)
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = u.model_dump() if hasattr(u, "model_dump") else dict(u)
        return GenOutcome(ok=True, image_bytes=img, usage=usage, actual_size=size)
