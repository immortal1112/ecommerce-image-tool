# -*- coding: utf-8 -*-
"""云端后端（Hugging Face Spaces / 任意容器平台）。

与本地版差异：
- 无 settings.json：API Key 必须由前端请求头提供（x-ark-key / x-openai-key / x-llm-key），
  服务器不保存任何 Key；
- 不落盘出图：火山走 response_format=url，把官方图片链接直接回给浏览器
  （浏览器直连北京下载，不经本服务器；链接 24h 有效）；
  OpenAI 只支持 b64，转成 data URL 返回；
- 上传的参考图存 /tmp（容器重启即清），仅供本批生成与重生成使用；
- CORS 全开，供 Vercel 前端跨域调用。
"""
from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import APIStatusError, OpenAI

from src.core.config import IMAGE_TYPE_LABELS, ModelRegistry, ModelSpec, Settings
from src.prompt_engine import PromptEngine
from src.providers.base import encode_image_data_url, translate_http_error, ProviderError

log = logging.getLogger("cloud")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
UPLOAD_DIR = Path("/tmp/uploads") if sys.platform != "win32" else Path(__file__).parent / "tmp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

registry = ModelRegistry(ROOT / "config" / "models.json")
prompt_engine = PromptEngine(Settings(), None)   # 云端仅用 fallback 默认提示词

STYLE_REF_PROMPT = (
    "图1为产品图，图2为风格参考图。将图2中的主体替换为图1的产品，"
    "保持图2的风格、构图、光影和场景不变；产品外观、颜色、材质与图1完全一致。"
    "不要改变其他任何内容。")

app = FastAPI(title="电商图片生成工具 · 云端")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

TASKS: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=4)


# ---------------- 供应商直调（url 优先，避免大图过境本服务器） ----------------
def _gen_volcengine(m: ModelSpec, key: str, prompt: str, size: str,
                    fmt: str, refs: list[Path]) -> str:
    client = OpenAI(base_url=ARK_BASE, api_key=key, timeout=300.0)
    kwargs: dict = dict(model=m.model_id, prompt=prompt, size=size,
                        response_format="url",
                        extra_body={"watermark": False})
    if len(m.output_formats) > 1:
        kwargs["output_format"] = fmt
    if refs:
        urls = [encode_image_data_url(p) for p in refs]
        kwargs["extra_body"]["image"] = urls[0] if len(urls) == 1 else urls
    try:
        resp = client.images.generate(**kwargs)
    except APIStatusError as e:
        raise translate_http_error(e.status_code, getattr(e, "message", str(e)), "火山方舟")
    except Exception as e:
        raise ProviderError(f"连接火山方舟失败:{e}", retryable=True)
    if not getattr(resp, "data", None):
        raise ProviderError("火山方舟返回为空(可能被内容审核拦截),请调整提示词。")
    return resp.data[0].url


def _gen_openai(m: ModelSpec, key: str, prompt: str, size: str,
                fmt: str, quality: str, refs: list[Path]) -> str:
    client = OpenAI(api_key=key, timeout=300.0)
    common = dict(model=m.model_id, prompt=prompt, size=size,
                  quality=quality or m.default_quality or "medium",
                  output_format=fmt)
    files = []
    try:
        if refs:
            files = [open(p, "rb") for p in refs]
            resp = client.images.edit(image=files if len(files) > 1 else files[0],
                                      **common)
        else:
            resp = client.images.generate(**common)
    except APIStatusError as e:
        raise translate_http_error(e.status_code, getattr(e, "message", str(e)), "OpenAI")
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"连接 OpenAI 失败:{e}", retryable=True)
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass
    if not getattr(resp, "data", None):
        raise ProviderError("OpenAI 返回为空(可能被内容审核拦截),请调整提示词。")
    return f"data:image/{fmt};base64,{resp.data[0].b64_json}"


def _run_job(tid: str) -> None:
    t = TASKS[tid]
    t["status"] = "running"
    t0 = time.time()
    m = registry.get(t["model"])
    try:
        size = m.resolve_size(t["ratio"], t["tier"])
        refs = [Path(p) for p in t["refs"] if Path(p).exists()]
        if m.provider == "volcengine":
            url = _gen_volcengine(m, t["ark_key"], t["prompt"], size, t["fmt"], refs)
        else:
            url = _gen_openai(m, t["oai_key"], t["prompt"], size, t["fmt"],
                              t["quality"], refs)
        t.update(status="done", url=url, elapsed=round(time.time() - t0, 1))
        log.info("task %s done (%.1fs)", tid, time.time() - t0)
    except ProviderError as e:
        t.update(status="failed", message=e.message_cn)
        log.warning("task %s failed: %s", tid, e.message_cn)
    except Exception as e:
        t.update(status="failed", message=f"未预料的错误:{type(e).__name__}: {e}")
        log.exception("task %s crashed", tid)


# ---------------- API ----------------
@app.get("/api/config")
def api_config():
    models = {}
    for key in registry.keys():
        m = registry.get(key)
        models[key] = {"display_name": m.display_name,
                       "provider": m.provider,
                       "supported_ratios": m.supported_ratios,
                       "supported_tiers": m.supported_tiers,
                       "output_formats": m.output_formats,
                       "quality_options": m.quality_options,
                       "default_quality": m.default_quality,
                       "size_map": m.size_map,
                       "max_reference_images": m.max_reference_images}
    return {"models": models, "image_types": IMAGE_TYPE_LABELS,
            "default_prompts": {k: prompt_engine.default_for(k)
                                for k in IMAGE_TYPE_LABELS},
            "style_ref_prompt": STYLE_REF_PROMPT,
            "cloud": True,
            "has_ark_key": False, "has_openai_key": False, "has_llm_key": False}


@app.post("/api/generate")
async def api_generate(
    products: list[UploadFile] = File(...),
    style: UploadFile | None = File(None),
    model: str = Form(...), ratio: str = Form(...), tier: str = Form(...),
    fmt: str = Form(""), quality: str = Form(""), prompt: str = Form(""),
    types: str = Form("{}"),
    x_ark_key: str = Header(default=""), x_openai_key: str = Header(default=""),
):
    if model not in registry.keys():
        raise HTTPException(400, f"未知模型 {model}")
    m = registry.get(model)
    if m.provider == "volcengine" and not x_ark_key.strip():
        raise HTTPException(401, "请先在「API KEY」中填写火山方舟 Key")
    if m.provider == "openai" and not x_openai_key.strip():
        raise HTTPException(401, "请先在「API KEY」中填写 OpenAI Key")
    try:
        type_counts = {k: int(v) for k, v in json.loads(types).items()
                       if k in IMAGE_TYPE_LABELS and int(v) > 0}
    except Exception:
        raise HTTPException(400, "types 参数格式错误")
    if not type_counts:
        raise HTTPException(400, "请至少选择一种图片类型")
    if ratio not in m.supported_ratios or tier not in m.size_map.get(ratio, {}):
        raise HTTPException(400, f"{m.display_name} 不支持 {ratio}+{tier}")

    def _save(f: UploadFile) -> Path:
        dst = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}.jpg"
        dst.write_bytes(f.file.read())
        return dst

    style_path = _save(style) if style is not None else None
    if style_path and m.max_reference_images < 2:
        raise HTTPException(400, f"{m.display_name} 不支持产品图+风格图同传")
    fmt = fmt or m.output_formats[0]
    created = []
    for pf in products:
        img = _save(pf)
        name = Path(pf.filename or "img").stem or "img"
        refs = [str(img)] + ([str(style_path)] if style_path else [])
        n_seq = 0
        for tkey, n in type_counts.items():
            p = prompt.strip() or (STYLE_REF_PROMPT if style_path
                                   else prompt_engine.default_for(tkey))
            for _ in range(n):
                n_seq += 1
                tid = f"c_{uuid.uuid4().hex[:8]}"
                TASKS[tid] = {"status": "queued", "message": "", "url": "",
                              "type": IMAGE_TYPE_LABELS[tkey], "type_key": tkey,
                              "seq": n_seq, "version": 1,
                              "input_name": name, "refs": refs,
                              "model": model, "ratio": ratio, "tier": tier,
                              "fmt": fmt, "quality": quality or "",
                              "prompt": p,
                              "ark_key": x_ark_key.strip(),
                              "oai_key": x_openai_key.strip()}
                executor.submit(_run_job, tid)
                created.append(tid)
    return {"task_ids": created}


@app.get("/api/tasks")
def api_tasks(ids: str):
    out = {}
    for tid in ids.split(","):
        t = TASKS.get(tid)
        if t:
            out[tid] = {k: t.get(k) for k in
                        ("status", "message", "type", "seq", "version",
                         "input_name", "elapsed", "prompt", "url")}
            out[tid]["file"] = t.get("url", "")   # 与本地版前端字段兼容
    return out


@app.post("/api/regen")
async def api_regen(payload: dict, x_ark_key: str = Header(default=""),
                    x_openai_key: str = Header(default="")):
    old = TASKS.get(payload.get("task_id", ""))
    if not old:
        raise HTTPException(404, "找不到原任务（服务器可能已重启，请重新生成一批）")
    refs = [p for p in old["refs"] if Path(p).exists()]
    if not refs:
        raise HTTPException(400, "原参考图已过期，请重新上传生成。")
    tid = f"c_{uuid.uuid4().hex[:8]}"
    TASKS[tid] = dict(old, status="queued", message="", url="",
                      version=old["version"] + 1,
                      prompt=(payload.get("prompt") or "").strip() or old["prompt"],
                      ark_key=x_ark_key.strip() or old["ark_key"],
                      oai_key=x_openai_key.strip() or old["oai_key"])
    executor.submit(_run_job, tid)
    return {"task_id": tid, "version": TASKS[tid]["version"]}


@app.post("/api/optimize")
async def api_optimize(payload: dict, x_llm_key: str = Header(default="")):
    raw = (payload.get("prompt") or "").strip()
    if not raw:
        raise HTTPException(400, "提示词为空")
    if not x_llm_key.strip():
        raise HTTPException(401, "请先在「API KEY」中填写 DeepSeek Key（提示词优化用）")
    s = Settings()
    s.text_llm_api_key = x_llm_key.strip()
    try:
        return {"result": PromptEngine(s, None).optimize(raw)}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.get("/")
def root():
    return {"service": "ecommerce-image-tool-cloud", "status": "ok",
            "made_by": "immortal"}
