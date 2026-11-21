"""Provider 工厂。

真实 provider 采用延迟导入：
- 纯离线测试即使没有安装 openai 也能 import 本包；
- openai SDK 只出现在 openai_compat.py 中。
"""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import LLMProvider
from .fake import FakeProvider, RecordedCall, ScriptItem

__all__ = [
    "LLMProvider",
    "FakeProvider",
    "RecordedCall",
    "ScriptItem",
    "create_provider",
]


def create_provider(settings: Settings) -> LLMProvider:
    """根据配置创建对应的 Provider。

    参数：
        settings: 已校验的配置；provider 字段决定具体实现。

    返回值：
        FakeProvider 或 OpenAICompatibleProvider 实例。

    异常：
        ConfigError: 真实 provider 缺少 API key，或 provider 不受支持时抛出。
    """
    if settings.provider == "fake":
        return FakeProvider()

    if settings.provider == "deepseek":
        if not settings.api_key:
            raise ConfigError(
                "缺少 MINI_AGENT_API_KEY。请在 .env 中填写，"
                "或在进程环境变量中设置后重试。"
            )
        from .openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
        )

    raise ConfigError(
        f"不支持的 provider: {settings.provider!r}，可选 ('deepseek', 'fake')"
    )
