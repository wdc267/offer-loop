"""FakeProvider 的行为测试。"""

from __future__ import annotations

import pytest

from mini_agent.errors import (
    ProviderConnectionError,
    ScriptExhaustedError,
)
from mini_agent.models import AssistantTurn, Message, ToolSpec
from mini_agent.providers.fake import FakeProvider


def test_script_consumed_in_order(run_async):
    """验证 FakeProvider 按脚本顺序返回文本、指定 turn 或异常。"""
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
    """验证每次 generate 的参数（消息、温度、max_tokens、工具）都被记录下来。"""
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
    assert call.tools == ()


def test_records_tools(run_async):
    """验证传入的工具声明会被记录，便于在测试中断言工具透传。"""
    provider = FakeProvider(["a"])
    spec = ToolSpec(name="read_resume", description="读取简历")

    run_async(provider.generate([Message.user("hello")], tools=[spec]))

    assert provider.calls[0].tools == (spec,)


def test_echo_fallback(run_async):
    """验证脚本耗尽后默认回显最后一条用户消息。"""
    provider = FakeProvider()

    turn = run_async(provider.generate([Message.user("你好")]))

    assert "你好" in turn.text
    assert turn.finish_reason == "stop"


def test_script_exhausted_raises(run_async):
    """验证关闭回声后脚本耗尽会抛 ScriptExhaustedError。"""
    provider = FakeProvider([], echo=False)

    with pytest.raises(ScriptExhaustedError):
        run_async(provider.generate([Message.user("q")]))


def test_scripted_exception_raised(run_async):
    """验证脚本中的异常对象会被原样抛出，用于测试故障路径。"""
    error = ProviderConnectionError("down")
    provider = FakeProvider([error])

    with pytest.raises(ProviderConnectionError) as excinfo:
        run_async(provider.generate([Message.user("q")]))

    assert excinfo.value is error
