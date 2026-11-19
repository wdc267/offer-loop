"""Agent Loop 的行为测试。

全部使用 FakeProvider，不联网、不需要 API key。
"""
from __future__ import annotations

import pytest

from mini_agent.agent import MiniAgent
from mini_agent.errors import (
    InvalidInputError,
    ProviderConnectionError,
    UnsupportedToolCallError,
)
from mini_agent.models import (
    AssistantTurn,
    Message,
    StopReason,
    ToolCall,
    Usage,
)
from mini_agent.providers.fake import FakeProvider


class AlwaysContinueAgent(MiniAgent):
    """测试专用：模拟「每一步都要求再调一次模型」的策略。

    s01 的正常路径永远在第 1 步结束，无法触发 max_steps；
    这个子类专门用来验证步数保护，也顺带证明扩展缝确实可替换。
    """

    async def _on_assistant_turn(self, turn: AssistantTurn) -> bool:
        self._history.append(Message.user("(test) continue"))
        return True


def test_single_turn_completed(run_async):
    provider = FakeProvider(["你好，我是 MiniAgent"])
    agent = MiniAgent(provider, system_prompt="你是助手")

    result = run_async(agent.run("你好"))

    assert result.text == "你好，我是 MiniAgent"
    assert result.stop_reason is StopReason.COMPLETED
    assert result.steps == 1
    assert len(provider.calls) == 1


def test_system_prompt_first_and_not_duplicated(run_async):
    provider = FakeProvider(["a", "b"])
    agent = MiniAgent(provider, system_prompt="sys")

    run_async(agent.run("q1"))
    run_async(agent.run("q2"))

    assert provider.calls[0].messages[0] == Message.system("sys")
    assert sum(1 for m in provider.calls[1].messages if m.role == "system") == 1


def test_multi_turn_history_accumulates(run_async):
    provider = FakeProvider(["a1", "a2"])
    agent = MiniAgent(provider, system_prompt="sys")

    run_async(agent.run("q1"))
    run_async(agent.run("q2"))

    second_call = provider.calls[1].messages
    assert [m.role for m in second_call] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_call[1].content == "q1"
    assert second_call[2].content == "a1"
    assert second_call[3].content == "q2"


def test_history_snapshot_is_immutable(run_async):
    seed = [Message.user("构造时的消息")]
    provider = FakeProvider(["a", "b"])
    agent = MiniAgent(provider, initial_messages=seed)

    # 外部修改原列表不应影响 Agent。
    seed.append(Message.user("后来追加的消息"))
    assert isinstance(agent.history, tuple)
    assert all(m.content != "后来追加的消息" for m in agent.history)

    run_async(agent.run("q1"))
    first_snapshot = provider.calls[0].messages
    run_async(agent.run("q2"))

    # 后续调用不应改变已经记录的调用内容。
    assert provider.calls[0].messages == first_snapshot


def test_max_steps_guard(run_async):
    provider = FakeProvider(["a"] * 10)
    agent = AlwaysContinueAgent(provider, max_steps=3, system_prompt="sys")

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 3
    assert len(provider.calls) == 3


def test_provider_exception_propagates(run_async):
    provider = FakeProvider([ProviderConnectionError("boom")])
    agent = MiniAgent(provider)

    with pytest.raises(ProviderConnectionError):
        run_async(agent.run("q"))

    assert len(provider.calls) == 1


def test_empty_response_stops(run_async):
    provider = FakeProvider([AssistantTurn(text="", finish_reason="stop")])
    agent = MiniAgent(provider)

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.EMPTY_RESPONSE
    assert result.steps == 1
    # 空响应不重试：只调用一次。
    assert len(provider.calls) == 1


def test_tool_calls_rejected_in_s01(run_async):
    turn = AssistantTurn(
        text="",
        finish_reason="tool_calls",
        tool_calls=(ToolCall(id="call_1", name="get_resume", arguments="{}"),),
    )
    provider = FakeProvider([turn])
    agent = MiniAgent(provider)

    with pytest.raises(UnsupportedToolCallError, match="s02"):
        run_async(agent.run("q"))


def test_usage_aggregated_across_steps(run_async):
    script = [AssistantTurn("a", "stop", Usage(1, 2, 3)) for _ in range(3)]
    provider = FakeProvider(script)
    agent = AlwaysContinueAgent(provider, max_steps=3)

    result = run_async(agent.run("q"))

    assert result.usage == Usage(3, 6, 9)


def test_finish_reason_length_is_surfaced(run_async):
    provider = FakeProvider([AssistantTurn("partial", "length")])
    agent = MiniAgent(provider)

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.COMPLETED
    assert result.finish_reason == "length"


def test_blank_input_rejected(run_async):
    provider = FakeProvider()
    agent = MiniAgent(provider)

    with pytest.raises(InvalidInputError):
        run_async(agent.run("   "))

    assert provider.calls == ()


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError):
        MiniAgent(FakeProvider(), max_steps=0)


def test_message_to_openai_plain():
    assert Message.assistant("hi").to_openai() == {
        "role": "assistant",
        "content": "hi",
    }


def test_message_to_openai_with_tool_calls():
    message = Message.assistant(
        "",
        (ToolCall(id="call_1", name="get_resume", arguments='{"x":1}'),),
    )

    payload = message.to_openai()

    assert payload["role"] == "assistant"
    assert payload["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_resume", "arguments": '{"x":1}'},
        }
    ]
