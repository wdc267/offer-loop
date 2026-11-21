"""OpenAI 兼容适配器测试。

全部注入假 client，不联网、不需要真实 key。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import APIConnectionError, APITimeoutError

from mini_agent.errors import (
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from mini_agent.models import Message, ToolSpec, Usage
from mini_agent.providers.openai_compat import OpenAICompatibleProvider


class _Completions:
    def __init__(self, *, completion=None, exc: Exception | None = None) -> None:
        """创建假的 completions 端点。

        参数：
            completion: create() 正常返回时使用的响应对象。
            exc: 非空时 create() 直接抛出该异常。
        """
        self.completion = completion
        self.exc = exc
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        """记录请求参数并返回预设响应或抛出预设异常。

        参数：
            **kwargs: Provider 传给 SDK 的请求参数。

        返回值：
            构造时传入的 completion 对象。

        异常：
            构造时传入了 exc 则原样抛出。
        """
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.completion


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        """模拟 SDK 的 chat 资源。

        参数：
            completions: 假的 completions 端点。
        """
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        """模拟 AsyncOpenAI 客户端。

        参数：
            completions: 假的 completions 端点。
        """
        self.chat = _Chat(completions)

    async def close(self) -> None:
        """模拟关闭客户端，本测试替身不持有资源。"""
        return None


def _completion(
    *,
    content: str = "hi",
    finish_reason: str = "stop",
    tool_calls=None,
    usage=(1, 2, 3),
):
    """构造一个形状与 SDK completion 相同的测试响应。

    参数：
        content: 模型正文。
        finish_reason: 结束原因。
        tool_calls: 原始 tool_calls 列表。
        usage: (prompt, completion, total) 三元组。

    返回值：
        可用 SimpleNamespace 模拟的 completion 对象。
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            total_tokens=usage[2],
        ),
    )


def _provider(completions: _Completions, model: str = "test-model"):
    """创建注入了假 client 的 Provider。

    参数：
        completions: 假的 completions 端点。
        model: 传给 Provider 的模型名。

    返回值：
        不访问网络的 OpenAICompatibleProvider。
    """
    return OpenAICompatibleProvider(
        api_key="dummy",
        client=_Client(completions),
        model=model,
    )


def test_maps_completion_to_turn(run_async):
    """验证 SDK completion 被映射为 AssistantTurn，包括文本、结束原因和用量。"""
    completions = _Completions(completion=_completion())
    provider = _provider(completions)

    turn = run_async(provider.generate([Message.user("hi")], temperature=0.2))

    assert turn.text == "hi"
    assert turn.finish_reason == "stop"
    assert turn.usage == Usage(1, 2, 3)
    assert turn.tool_calls == ()


def test_maps_tool_calls(run_async):
    """验证响应中的 tool_calls 被映射为 ToolCall（id、名称、原始参数）。"""
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_resume", arguments='{"x":1}'),
    )
    completions = _Completions(
        completion=_completion(
            content="",
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )
    )
    provider = _provider(completions)

    turn = run_async(provider.generate([Message.user("hi")]))

    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "get_resume"
    assert turn.tool_calls[0].arguments == '{"x":1}'


def test_request_payload(run_async):
    """验证请求 payload 的 model、messages、temperature、stream 等字段正确。"""
    completions = _Completions(completion=_completion())
    provider = _provider(completions)

    run_async(
        provider.generate(
            [
                Message.system("sys"),
                Message.user("hello"),
            ],
            temperature=0.2,
        )
    )

    assert completions.kwargs is not None
    assert completions.kwargs["model"] == "test-model"
    assert completions.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert completions.kwargs["temperature"] == 0.2
    assert completions.kwargs["stream"] is False
    # max_tokens=None 时不传该参数，避免兼容端点把 null 当非法值。
    assert "max_tokens" not in completions.kwargs


def test_tools_are_mapped_to_function_schema(run_async):
    """验证 ToolSpec 被转换成 OpenAI 的 function 工具声明。"""
    completions = _Completions(completion=_completion())
    provider = _provider(completions)
    spec = ToolSpec(
        name="read_resume",
        description="读取简历",
        parameters={"type": "object", "properties": {}},
    )

    run_async(provider.generate([Message.user("hi")], tools=[spec]))

    assert completions.kwargs is not None
    assert completions.kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_resume",
                "description": "读取简历",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_no_tools_field_when_tools_empty(run_async):
    """验证没有工具时请求中不携带空的 tools 字段。"""
    completions = _Completions(completion=_completion())
    provider = _provider(completions)

    run_async(provider.generate([Message.user("hi")], tools=[]))

    assert completions.kwargs is not None
    assert "tools" not in completions.kwargs


def test_max_tokens_is_forwarded_when_set(run_async):
    """验证设置了 max_tokens 时会原样传给接口。"""
    completions = _Completions(completion=_completion())
    provider = _provider(completions)

    run_async(provider.generate([Message.user("hi")], max_tokens=64))

    assert completions.kwargs is not None
    assert completions.kwargs["max_tokens"] == 64


def test_empty_choices_raises(run_async):
    """验证响应缺少 choices 时抛 ProviderResponseError。"""
    completion = SimpleNamespace(choices=[], usage=None)
    provider = _provider(_Completions(completion=completion))

    with pytest.raises(ProviderResponseError):
        run_async(provider.generate([Message.user("hi")]))


def test_translates_timeout(run_async):
    """验证 SDK 超时异常被翻译成 ProviderTimeoutError。"""
    # SDK 3.x 内部使用 httpx2；这里只关心异常类型，不需要真实 request。
    request = object()
    completions = _Completions(exc=APITimeoutError(request=request))
    provider = _provider(completions)

    with pytest.raises(ProviderTimeoutError):
        run_async(provider.generate([Message.user("hi")]))


def test_translates_connection_error(run_async):
    """验证 SDK 连接异常被翻译成 ProviderConnectionError。"""
    request = object()
    completions = _Completions(exc=APIConnectionError(request=request))
    provider = _provider(completions)

    with pytest.raises(ProviderConnectionError):
        run_async(provider.generate([Message.user("hi")]))


def test_programming_error_is_not_swallowed(run_async):
    """验证非 SDK 异常（如自身 Bug）不会被包装成网络错误。"""
    completions = _Completions(exc=ValueError("bug"))
    provider = _provider(completions)

    with pytest.raises(ValueError):
        run_async(provider.generate([Message.user("hi")]))
