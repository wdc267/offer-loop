"""MiniAgent 的异常层级。

设计原则：
- 模型语义上的「正常停止」（答完、空回复、到达步数上限）用 AgentResult 表达，不抛异常；
- 配置错误、输入错误、网络/鉴权/响应格式等基础设施故障才抛异常，
  让调用方可以决定重试、降级或提示用户。

这样调用方不会把「模型没话说」和「网络挂了」混为一谈。
"""
from __future__ import annotations


class MiniAgentError(Exception):
    """MiniAgent 所有自定义异常的基类。"""


# ---------------------------------------------------------------------------
# 调用方可以修正的错误
# ---------------------------------------------------------------------------
class ConfigError(MiniAgentError):
    """配置缺失或非法，例如真实 provider 缺少 API key。"""


class InvalidInputError(MiniAgentError):
    """调用方传入了非法输入，例如空白用户消息。"""


class UnsupportedToolCallError(MiniAgentError):
    """s01 不支持工具调用；收到 tool_calls 时抛出，提示升级到 s02。"""


class ScriptExhaustedError(MiniAgentError):
    """FakeProvider 脚本耗尽且未开启回声。"""


# ---------------------------------------------------------------------------
# 模型调用故障
# ---------------------------------------------------------------------------
class ProviderError(MiniAgentError):
    """模型调用失败的基类。"""


class ProviderAuthError(ProviderError):
    """API key 无效或没有权限。"""


class ProviderConnectionError(ProviderError):
    """无法连接模型服务。"""


class ProviderTimeoutError(ProviderConnectionError):
    """模型请求超时。

    继承 ProviderConnectionError：调用方只关心「连接类问题」时，
    捕获父类即可同时覆盖超时。
    """


class ProviderRateLimitError(ProviderError):
    """触发供应商限流。"""


class ProviderResponseError(ProviderError):
    """供应商返回了无法解析或不符合预期的响应。"""
