"""配置加载。

优先级：进程环境变量 > .env > 代码默认值。

库代码不在 import 时自动读 .env；只有显式调用 load_settings() 才会加载。
这样测试可以完全控制配置来源，也不会因为某个目录下的 .env 改变行为。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

# mini_agent/mini_agent/config.py -> mini_agent/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROVIDER = "deepseek"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_STEPS = 8
DEFAULT_TEMPERATURE = 0.3
SUPPORTED_PROVIDERS = ("deepseek", "fake")


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = DEFAULT_PROVIDER
    # repr=False：避免日志 / 异常信息里意外打印密钥。
    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float = DEFAULT_TEMPERATURE

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ConfigError(
                f"不支持的 provider: {self.provider!r}，可选 {SUPPORTED_PROVIDERS}"
            )
        if not self.base_url:
            raise ConfigError("MINI_AGENT_BASE_URL 不能为空")
        if not self.model:
            raise ConfigError("MINI_AGENT_MODEL 不能为空")
        if self.max_steps < 1:
            raise ConfigError(f"MINI_AGENT_MAX_STEPS 必须 >= 1，当前为 {self.max_steps}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(
                f"MINI_AGENT_TEMPERATURE 必须在 [0, 2] 之间，当前为 {self.temperature}"
            )


def _read_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _read_int(name: str, default: int) -> int:
    raw = _read_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前为 {raw!r}") from exc


def _read_float(name: str, default: float) -> float:
    raw = _read_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是数字，当前为 {raw!r}") from exc


def load_settings(
    env_file: Path | None = None,
    *,
    load_env: bool = True,
) -> Settings:
    """从环境变量（以及可选的 .env）构造 Settings。

    参数：
        env_file: 显式指定 .env 路径；默认使用项目根目录下的 .env。
        load_env: 是否读取 .env。测试通常传 False 以完全隔离。

    只允许 .env 补充进程里不存在的变量（override=False），
    这样临时覆盖和 CI 环境变量总是优先。
    """
    if load_env:
        path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        if path.exists():
            # utf-8-sig 兼容 Windows 记事本写出的 UTF-8 BOM；
            # 否则第一个键会变成 "﻿MINI_AGENT_API_KEY"，表现为「明明写了却读不到」。
            load_dotenv(dotenv_path=path, override=False, encoding="utf-8-sig")

    provider = (_read_str("MINI_AGENT_PROVIDER", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).lower()

    return Settings(
        provider=provider,
        api_key=_read_str("MINI_AGENT_API_KEY"),
        base_url=_read_str("MINI_AGENT_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL,
        model=_read_str("MINI_AGENT_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        max_steps=_read_int("MINI_AGENT_MAX_STEPS", DEFAULT_MAX_STEPS),
        temperature=_read_float("MINI_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE),
    )
