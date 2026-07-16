# -*- coding: utf-8 -*-
"""火山方舟(Seedream) Provider。

调用方式（官方文档核实，2026-07）：
- 同步接口 POST {base_url}/images/generations，OpenAI SDK 兼容；
- 参考图经 extra_body["image"] 传入（URL 或 data URL，单图字符串/多图列表）；
- size 传精确像素串（由 ModelSpec.resolve_size 按官方比例映射表得出）——
  避免仅传档位时比例由模型猜测（已确认的决策3）；
- response_format 默认 b64_json 直接落盘，绕开图片URL仅保留24小时的限制（决策1）；
- watermark 默认 False；4.5 仅支持 jpeg（能力在 models.json 声明，UI已过滤）。
"""
from __future__ import annotations

import base64

import httpx
from openai import APIStatusError, OpenAI

from .base import (BaseProvider, GenOutcome, GenTask, ProviderError,
                   encode_image_data_url, translate_http_error)


class VolcengineProvider(BaseProvider):
    name = "volcengine"

    def check_api_key(self):
        if not self.settings.volcengine_api_key.strip():
            return "尚未配置火山方舟 API Key，请到「设置」中填写（控制台 ark 页面可创建）。"
        return None

    def _client(self) -> OpenAI:
        return OpenAI(base_url=self.settings.volcengine_base_url,
                      api_key=self.settings.volcengine_api_key,
                      timeout=300.0)

    def _do_generate(self, task: GenTask) -> GenOutcome:
        size = task.model.resolve_size(task.ratio, task.tier)
        use_b64 = self.settings.response_mode != "url"
        # 不传 sequential_image_generation：官方默认即单图（组图需显式传 auto），
        # 且 Seedream 5.0 pro 不支持组图，传该参数可能像 4.5 拒收 output_format 一样报 400。
        extra: dict = {"watermark": self.settings.watermark}
        if task.reference_images:
            if len(task.reference_images) > task.model.max_reference_images:
                raise ProviderError(
                    f"{task.model.display_name} 最多支持 "
                    f"{task.model.max_reference_images} 张参考图，"
                    f"当前选择了 {len(task.reference_images)} 张，请减少后重试。")
            urls = [encode_image_data_url(p) for p in task.reference_images]
            extra["image"] = urls[0] if len(urls) == 1 else urls

        kwargs: dict = dict(
            model=task.model.model_id,
            prompt=task.prompt,
            size=size,
            response_format="b64_json" if use_b64 else "url",
            extra_body=extra,
        )
        # 仅当模型支持多种输出格式时才传 output_format；
        # 单一格式的模型（如 Seedream 4.5 仅 jpeg）不接受该参数，传了会报 InvalidParameter (HTTP 400)。
        if len(task.model.output_formats) > 1:
            kwargs["output_format"] = task.output_format

        try:
            resp = self._client().images.generate(**kwargs)
        except APIStatusError as e:
            raise translate_http_error(e.status_code, getattr(e, "message", str(e)), "火山方舟")
        except Exception as e:
            raise ProviderError(f"连接火山方舟失败：{e}。请检查网络后重试。", retryable=True)

        if not getattr(resp, "data", None):
            raise ProviderError("火山方舟返回为空（可能被内容审核拦截），请调整提示词后重试。")
        item = resp.data[0]
        if use_b64:
            img = base64.b64decode(item.b64_json)
        else:  # url 模式：24h 内立刻下载
            r = httpx.get(item.url, timeout=120.0)
            if r.status_code != 200:
                raise ProviderError(f"图片下载失败（HTTP {r.status_code}），可切换为 b64 模式重试。",
                                    retryable=True)
            img = r.content
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = u.model_dump() if hasattr(u, "model_dump") else dict(u)
        return GenOutcome(ok=True, image_bytes=img, usage=usage,
                          actual_size=getattr(item, "size", "") or size)
