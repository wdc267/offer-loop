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
from mini_agent.models import Message, Usage
from mini_agent.providers.openai_compat import OpenAICompatibleProvider


class _Completions:
    def __init__(self, *, completion=None, exc: Exception | None = None) -> None:
        self.completion = completion
        self.exc = exc
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.completion


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)

    async def close(self) -> None:
        return None


def _completion(
    *,
    content: str = "hi",
    finish_reason: str = "stop",
    tool_calls=None,
    usage=(1, 2, 3),
):
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
    return OpenAICompatibleProvider(
        api_key="dummy",
        client=_Client(completions),
        model=model,
    )


def test_maps_completion_to_turn(run_async):
    completions = _Completions(completion=_completion())
    provider = _provider(completions)

    turn = run_async(provider.generate([Message.user("hi")], temperature=0.2))

    assert turn.text == "hi"
    assert turn.finish_reason == "stop"
    assert turn.usage == Usage(1, 2, 3)
    assert turn.tool_calls == ()


def test_maps_tool_calls(run_async):
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


def test_max_tokens_is_forwarded_when_set(run_async):
    completions = _Completions(completion=_completion())
    provider = _provider(completions)

    run_async(provider.generate([Message.user("hi")], max_tokens=64))

    assert completions.kwargs is not None
    assert completions.kwargs["max_tokens"] == 64


def test_empty_choices_raises(run_async):
    completion = SimpleNamespace(choices=[], usage=None)
    provider = _provider(_Completions(completion=completion))

    with pytest.raises(ProviderResponseError):
        run_async(provider.generate([Message.user("hi")]))


def test_translates_timeout(run_async):
    # SDK 3.x 内部使用 httpx2；这里只关心异常类型，不需要真实 request。
    request = object()
    completions = _Completions(exc=APITimeoutError(request=request))
    provider = _provider(completions)

    with pytest.raises(ProviderTimeoutError):
        run_async(provider.generate([Message.user("hi")]))


def test_translates_connection_error(run_async):
    request = object()
    completions = _Completions(exc=APIConnectionError(request=request))
    provider = _provider(completions)

    with pytest.raises(ProviderConnectionError):
        run_async(provider.generate([Message.user("hi")]))


def test_programming_error_is_not_swallowed(run_async):
    completions = _Completions(exc=ValueError("bug"))
    provider = _provider(completions)

    with pytest.raises(ValueError):
        run_async(provider.generate([Message.user("hi")]))
