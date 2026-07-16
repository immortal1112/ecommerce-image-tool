# -*- coding: utf-8 -*-
"""供应商适配层基类与数据契约。

GUI 永远不直接碰任何一家的 API：
  GUI -> GenTask(统一任务) -> Provider.generate() -> GenOutcome(统一结果)
差异（参数名/尺寸表达/参考图传法/返回形态）全部收敛在各 Provider 内。
"""
from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.config import ModelSpec, Settings


@dataclass
class GenTask:
    """一次单图生成任务（'设张数'=N个GenTask循环，已确认的决策）。"""
    task_id: str
    model: ModelSpec
    prompt: str
    image_type: str                 # main/white/selling/scene
    ratio: str                      # "1:1" / "3:4"
    tier: str                       # "1K"/"2K"/...
    output_format: str              # png/jpeg/webp
    reference_images: list[Path] = field(default_factory=list)
    quality: Optional[str] = None   # 仅 OpenAI 有效
    parent_generation_id: Optional[str] = None  # 单图重生时指向旧图


@dataclass
class GenOutcome:
    """统一生成结果。"""
    ok: bool
    image_bytes: Optional[bytes] = None
    error_message: str = ""         # 已翻译成中文、含下一步建议
    retryable: bool = False         # True=限流/瞬时错误，可退避重试
    usage: Optional[dict] = None    # 原样记录供应商 usage（token 等）
    elapsed_sec: float = 0.0
    actual_size: str = ""           # 供应商回报的实际尺寸（若有）


def encode_image_data_url(path: Path) -> str:
    """本地图片 -> data URL（火山要求 data:image/<小写格式>;base64,<b64>）。"""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        ext = path.suffix.lower().lstrip(".")
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime.lower()};base64,{b64}"


class ProviderError(Exception):
    def __init__(self, message_cn: str, retryable: bool = False):
        super().__init__(message_cn)
        self.message_cn = message_cn
        self.retryable = retryable


class BaseProvider:
    """Provider 抽象：接入全新供应商时继承本类，实现 _do_generate。"""

    name = "base"

    def __init__(self, settings: Settings):
        self.settings = settings

    # -- 子类必须实现 --------------------------------------------------
    def _do_generate(self, task: GenTask) -> GenOutcome:
        raise NotImplementedError

    def check_api_key(self) -> Optional[str]:
        """返回缺 Key 的中文提示；配置齐全返回 None。"""
        raise NotImplementedError

    # -- 公共入口 ------------------------------------------------------
    def generate(self, task: GenTask) -> GenOutcome:
        missing = self.check_api_key()
        if missing:
            return GenOutcome(ok=False, error_message=missing, retryable=False)
        t0 = time.time()
        try:
            outcome = self._do_generate(task)
        except ProviderError as e:
            outcome = GenOutcome(ok=False, error_message=e.message_cn,
                                 retryable=e.retryable)
        except Exception as e:  # 兜底：未预料错误也给中文外壳
            outcome = GenOutcome(
                ok=False, retryable=False,
                error_message=f"生成失败（未预料的错误）：{type(e).__name__}: {e}。"
                              f"可尝试重新生成该张；若反复出现请检查网络或反馈日志。")
        outcome.elapsed_sec = round(time.time() - t0, 2)
        return outcome


def translate_http_error(status: int, body_text: str, provider_cn: str) -> ProviderError:
    """把常见 HTTP/API 错误翻译成中文 + 建议（§13 错误处理）。"""
    if status == 401:
        return ProviderError(f"{provider_cn} API Key 无效或已过期，请到「设置」中重新填写。")
    if status == 402:
        return ProviderError(f"{provider_cn} 账户余额不足，请充值后再生成。")
    if status == 429:
        return ProviderError(f"{provider_cn} 触发限流（请求过于频繁），正在自动退避重试…", retryable=True)
    if status >= 500:
        return ProviderError(f"{provider_cn} 服务端暂时异常（{status}），正在自动重试…", retryable=True)
    low = (body_text or "").lower()
    if "moderation" in low or "sensitive" in low or "审核" in body_text:
        return ProviderError("内容未通过平台审核，请调整提示词（避免敏感词/人物侵权描述）后重试。")
    return ProviderError(f"{provider_cn} 返回错误（HTTP {status}）：{body_text[:200]}。"
                         f"请检查参数后重试；若持续失败请查看 logs/app.log。")
