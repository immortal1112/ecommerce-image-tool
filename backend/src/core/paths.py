# -*- coding: utf-8 -*-
"""跨平台路径管理。

设计要点（对应确认过的决策）：
- 程序以 `python app.py` 直接运行，配置文件 config/ 跟随程序目录（只读数据）。
- 用户可写数据（settings.json、工作目录下 inputs/outputs/logs）与程序目录分离：
  settings.json 存放在各平台标准用户配置目录，避免程序目录只读导致写入失败。
- 全部使用 pathlib + UTF-8，兼容中文用户名 / 中文路径。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_NAME = "ecommerce_image_tool"


def program_dir() -> Path:
    """程序根目录（app.py 所在目录）。"""
    return Path(__file__).resolve().parent.parent.parent


def config_dir() -> Path:
    """程序自带配置目录（models.json / prompts_source.json）。"""
    return program_dir() / "config"


def user_config_dir() -> Path:
    """跨平台用户配置目录（存 settings.json，可写）。"""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings_file() -> Path:
    return user_config_dir() / "settings.json"


def open_in_file_manager(path: Path) -> None:
    """在系统文件管理器中打开目录/文件所在位置（三平台）。"""
    path = Path(path)
    target = path if path.is_dir() else path.parent
    if sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
