"""FakeProvider 的行为测试。"""
from __future__ import annotations

import pytest

from mini_agent.errors import (
    ProviderConnectionError,
    ScriptExhaustedError,
)
from mini_agent.models import AssistantTurn, Message
from mini_agent.providers.fake import FakeProvider


def test_script_consumed_in_order(run_async):
    provider = FakeProvider(
        [
            "first",
            AssistantTurn("second", "length"),
            ProviderConnectionError("boom"),
        ]
    )

    first = run_async(provider.generate([Message.user("q1")]))
    second = run_async(provider.generate([Message.user("q2")]))

    assert first.text == "first"
    assert first.finish_reason == "stop"
    assert second.text == "second"
    assert second.finish_reason == "length"

    with pytest.raises(ProviderConnectionError):
        run_async(provider.generate([Message.user("q3")]))


def test_records_calls(run_async):
    provider = FakeProvider(["a"])

    run_async(
        provider.generate(
            [Message.user("hello")],
            temperature=0.7,
            max_tokens=128,
        )
    )

    call = provider.calls[0]
    assert call.messages[0].content == "hello"
    assert call.temperature == 0.7
    assert call.max_tokens == 128


def test_echo_fallback(run_async):
    provider = FakeProvider()

    turn = run_async(provider.generate([Message.user("你好")]))

    assert "你好" in turn.text
    assert turn.finish_reason == "stop"


def test_script_exhausted_raises(run_async):
    provider = FakeProvider([], echo=False)

    with pytest.raises(ScriptExhaustedError):
        run_async(provider.generate([Message.user("q")]))


def test_scripted_exception_raised(run_async):
    error = ProviderConnectionError("down")
    provider = FakeProvider([error])

    with pytest.raises(ProviderConnectionError) as excinfo:
        run_async(provider.generate([Message.user("q")]))

    assert excinfo.value is error
