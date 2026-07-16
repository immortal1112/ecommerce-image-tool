# -*- coding: utf-8 -*-
"""提示词系统（§7）。

1) 内置默认提示词：解析 通用.md（版本B 四段）。解析契约（prompts_source.json 声明）：
   - 按 Markdown 标题行(以#开头)切分；标题含关键词 主图/白底/卖点/场景 即认领该段；
   - 标题下正文（到下一个标题前）整体作为该类型默认提示词；
   - 文件缺失/某段缺失 -> 使用 fallback（占位提示词），并记录提示。
2) 优先级：用户手输 > 优化确认稿 > 内置默认（由 GUI 侧按此取用）。
3) 优化：默认 DeepSeek（OpenAI 兼容 chat 接口），只整理文本不生图，结果交用户确认。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from openai import APIStatusError, OpenAI

from .core.config import Settings
from .core.paths import config_dir

_OPTIMIZE_SYSTEM = (
    "你是电商产品图提示词整理助手。把用户给的生图提示词整理为一条更清晰可控的中文提示词，"
    "必须明确：画面主体、背景、构图/比例意图、'保持参考图产品外观颜色材质不变'的不变量、"
    "以及禁止项（无文字、无水印、不改变产品）。不得编造用户未提供的卖点或参数。"
    "只输出整理后的提示词正文，不要任何解释或前后缀。")


class PromptEngine:
    def __init__(self, settings: Settings, work_dir: Optional[Path] = None):
        self.settings = settings
        self.work_dir = work_dir
        cfg_path = config_dir() / "prompts_source.json"
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.defaults: dict[str, str] = {}
        self.load_notes: list[str] = []
        self._load_defaults()

    # ---------------- 内置默认提示词 ----------------
    def _source_path(self) -> Optional[Path]:
        raw = self.cfg.get("source_file", "")
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute() and self.work_dir:
            p = Path(self.work_dir) / raw
        return p

    def _load_defaults(self) -> None:
        fb = self.cfg["fallback_prompts"]
        keywords = self.cfg["section_keywords"]
        parsed: dict[str, str] = {}
        src = self._source_path()
        if src and src.exists():
            parsed = self._parse_md(src.read_text(encoding="utf-8"), keywords)
            missing = [k for k in keywords if k not in parsed]
            if missing:
                self.load_notes.append(
                    "通用.md 缺少段落: " + ",".join(keywords[m] for m in missing)
                    + "，这些类型暂用占位提示词。")
        else:
            self.load_notes.append(
                f"未找到内置提示词文件（{src}），全部类型暂用占位提示词。"
                f"把 通用.md 放到该路径后重启即可自动接入。")
        for k in keywords:
            self.defaults[k] = parsed.get(k, fb[k])

    @staticmethod
    def _parse_md(text: str, keywords: dict[str, str]) -> dict[str, str]:
        """按标题行切分并用关键词认领段落。"""
        result: dict[str, str] = {}
        # 找出所有标题行位置
        heads = [(m.start(), m.group(0)) for m in re.finditer(r"(?m)^#{1,6}\s.*$", text)]
        for i, (pos, head) in enumerate(heads):
            end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
            body = text[pos + len(head):end].strip()
            if not body:
                continue
            for key, kw in keywords.items():
                if kw in head and key not in result:
                    result[key] = body
        return result

    def default_for(self, image_type: str) -> str:
        return self.defaults.get(image_type, "")

    # ---------------- 提示词优化（DeepSeek 默认） ----------------
    def optimize(self, prompt: str) -> str:
        """调文本模型整理提示词；异常抛 RuntimeError(中文)。不自动覆盖原文，由GUI确认。"""
        if not self.settings.text_llm_api_key.strip():
            raise RuntimeError("尚未配置提示词优化模型的 API Key（默认 DeepSeek），请到「设置」中填写。")
        client = OpenAI(base_url=self.settings.text_llm_base_url,
                        api_key=self.settings.text_llm_api_key, timeout=120.0)
        try:
            resp = client.chat.completions.create(
                model=self.settings.text_llm_model,
                messages=[{"role": "system", "content": _OPTIMIZE_SYSTEM},
                          {"role": "user", "content": prompt}],
                temperature=0.3)
        except APIStatusError as e:
            raise RuntimeError(f"提示词优化服务返回错误（HTTP {e.status_code}），请稍后重试或检查 Key。")
        except Exception as e:
            raise RuntimeError(f"连接提示词优化服务失败：{e}")
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("优化服务返回为空，请重试。")
        return text
