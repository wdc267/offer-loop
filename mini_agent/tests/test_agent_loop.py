"""Agent Loop 的行为测试。

全部使用 FakeProvider，不联网、不需要 API key。
"""

from __future__ import annotations

import pytest

from mini_agent.agent import MiniAgent
from mini_agent.errors import (
    InvalidInputError,
    ProviderConnectionError,
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
        """忽略模型输出，每步都追加一条消息并要求继续。

        参数：
            turn: 模型本次输出，本测试子类不使用。

        返回值：
            始终返回 True，用于触发 max_steps 保护。
        """
        self._history.append(Message.user("(test) continue"))
        return True


def test_single_turn_completed(run_async):
    """验证单轮问答：模型返回文本后本轮结束，步数为 1。"""
    provider = FakeProvider(["你好，我是 MiniAgent"])
    agent = MiniAgent(provider, system_prompt="你是助手")

    result = run_async(agent.run("你好"))

    assert result.text == "你好，我是 MiniAgent"
    assert result.stop_reason is StopReason.COMPLETED
    assert result.steps == 1
    assert len(provider.calls) == 1


def test_system_prompt_first_and_not_duplicated(run_async):
    """验证 system prompt 始终位于历史首位，多轮对话中不会重复追加。"""
    provider = FakeProvider(["a", "b"])
    agent = MiniAgent(provider, system_prompt="sys")

    run_async(agent.run("q1"))
    run_async(agent.run("q2"))

    assert provider.calls[0].messages[0] == Message.system("sys")
    assert sum(1 for m in provider.calls[1].messages if m.role == "system") == 1


def test_multi_turn_history_accumulates(run_async):
    """验证两次 run() 的消息按顺序累积，第二次调用能看到完整历史。"""
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
    """验证外部消息列表被复制，事后修改不影响 Agent；Provider 拿到的是快照。"""
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
    """验证步数保护：模型不断要求继续时，最多调用 max_steps 次后停止。"""
    provider = FakeProvider(["a"] * 10)
    agent = AlwaysContinueAgent(provider, max_steps=3, system_prompt="sys")

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 3
    assert len(provider.calls) == 3


def test_provider_exception_propagates(run_async):
    """验证 Provider 异常原样向上传播，且只调用一次不重试。"""
    provider = FakeProvider([ProviderConnectionError("boom")])
    agent = MiniAgent(provider)

    with pytest.raises(ProviderConnectionError):
        run_async(agent.run("q"))

    assert len(provider.calls) == 1


def test_empty_response_stops(run_async):
    """验证模型返回空内容时立即结束本轮，不重试。"""
    provider = FakeProvider([AssistantTurn(text="", finish_reason="stop")])
    agent = MiniAgent(provider)

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.EMPTY_RESPONSE
    assert result.steps == 1
    # 空响应不重试：只调用一次。
    assert len(provider.calls) == 1


def test_usage_aggregated_across_steps(run_async):
    """验证多步调用的 token 用量会累加汇总到 AgentResult。"""
    script = [AssistantTurn("a", "stop", Usage(1, 2, 3)) for _ in range(3)]
    provider = FakeProvider(script)
    agent = AlwaysContinueAgent(provider, max_steps=3)

    result = run_async(agent.run("q"))

    assert result.usage == Usage(3, 6, 9)


def test_finish_reason_length_is_surfaced(run_async):
    """验证 finish_reason=length 被保留在结果里，供调用方提示回答可能被截断。"""
    provider = FakeProvider([AssistantTurn("partial", "length")])
    agent = MiniAgent(provider)

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.COMPLETED
    assert result.finish_reason == "length"


def test_blank_input_rejected(run_async):
    """验证空白用户输入被拒绝，且不会触发任何模型调用。"""
    provider = FakeProvider()
    agent = MiniAgent(provider)

    with pytest.raises(InvalidInputError):
        run_async(agent.run("   "))

    assert provider.calls == ()


def test_max_steps_must_be_positive():
    """验证 max_steps 小于 1 时构造 Agent 直接报错。"""
    with pytest.raises(ValueError):
        MiniAgent(FakeProvider(), max_steps=0)


def test_message_to_openai_plain():
    """验证普通 assistant 消息转换成 OpenAI 格式时只包含 role 和 content。"""
    assert Message.assistant("hi").to_openai() == {
        "role": "assistant",
        "content": "hi",
    }


def test_message_to_openai_with_tool_calls():
    """验证 assistant 消息里的 tool_calls 被转换成 OpenAI 的 function 调用结构。"""
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
