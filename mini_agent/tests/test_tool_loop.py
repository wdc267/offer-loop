"""Agent 与工具调用的集成测试（FakeProvider，全程离线）。"""

from __future__ import annotations

from mini_agent.agent import MiniAgent
from mini_agent.errors import ToolError
from mini_agent.models import (
    AssistantTurn,
    StopReason,
    ToolCall,
    ToolSpec,
)
from mini_agent.providers.fake import FakeProvider
from mini_agent.tools import ToolRegistry, create_default_registry


class _EchoTool:
    spec = ToolSpec(name="echo", description="回显 value")

    async def run(self, **kwargs):
        """回显 value 参数，便于断言工具结果被回填。

        参数：
            **kwargs: 模型解析出的工具参数，需包含 value。

        返回值：
            形如 ``echo:<value>`` 的字符串。
        """
        return f"echo:{kwargs.get('value')}"


class _FailingTool:
    spec = ToolSpec(name="fail", description="总是失败")

    async def run(self, **kwargs):
        """模拟工具业务失败，始终抛出 ToolError。

        参数：
            **kwargs: 工具参数，本替身忽略。

        异常：
            ToolError: 恒定抛出，用于验证失败回填路径。
        """
        raise ToolError("文件不存在")


def _tool_turn(*calls: ToolCall) -> AssistantTurn:
    """构造一条带工具调用的模型输出，供 FakeProvider 按脚本返回。

    参数：
        *calls: 本轮模型发起的一个或多个工具调用。

    返回值：
        finish_reason 为 ``tool_calls`` 的 AssistantTurn。
    """
    return AssistantTurn(
        text="",
        finish_reason="tool_calls",
        tool_calls=calls,
    )


def test_tool_call_round_trip(run_async):
    """验证完整闭环：模型发起工具调用、Agent 执行并回填 tool 消息、再次调用模型得到最终回答。"""
    provider = FakeProvider(
        [
            _tool_turn(ToolCall(id="call_1", name="echo", arguments='{"value":"hi"}')),
            AssistantTurn(text="最终回答", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider, tools=ToolRegistry([_EchoTool()]))

    result = run_async(agent.run("读一下"))

    assert result.text == "最终回答"
    assert result.steps == 2
    assert [m.role for m in agent.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tool_message = agent.history[2]
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.content == "echo:hi"

    # 第二次模型调用必须能看到完整的 assistant(tool_calls) + tool 消息。
    second_call = provider.calls[1].messages
    assert second_call[1].tool_calls[0].id == "call_1"
    assert second_call[2].role == "tool"


def test_multiple_tool_calls_are_all_answered_in_order(run_async):
    """验证一轮内多个 tool_calls 会按顺序逐个执行并全部配平。"""
    provider = FakeProvider(
        [
            _tool_turn(
                ToolCall(id="call_1", name="echo", arguments='{"value":"a"}'),
                ToolCall(id="call_2", name="echo", arguments='{"value":"b"}'),
            ),
            AssistantTurn(text="done", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider, tools=ToolRegistry([_EchoTool()]))

    run_async(agent.run("q"))

    tool_messages = [m for m in agent.history if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["call_1", "call_2"]
    assert [m.content for m in tool_messages] == ["echo:a", "echo:b"]


def test_unknown_tool_is_reported_and_loop_continues(run_async):
    """验证未知工具的错误被回填给模型，循环继续到最终回答。"""
    provider = FakeProvider(
        [
            _tool_turn(ToolCall(id="call_1", name="nope", arguments="{}")),
            AssistantTurn(text="工具不可用", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider, tools=ToolRegistry([_EchoTool()]))

    result = run_async(agent.run("q"))

    assert result.text == "工具不可用"
    tool_message = next(m for m in agent.history if m.role == "tool")
    assert "未知工具" in tool_message.content
    assert "echo" in tool_message.content


def test_invalid_json_arguments_are_reported(run_async):
    """验证非法 JSON 参数被回填为 tool 错误，模型仍能给出回答。"""
    provider = FakeProvider(
        [
            _tool_turn(ToolCall(id="call_1", name="echo", arguments="{bad")),
            AssistantTurn(text="参数错误", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider, tools=ToolRegistry([_EchoTool()]))

    result = run_async(agent.run("q"))

    assert result.text == "参数错误"
    tool_message = next(m for m in agent.history if m.role == "tool")
    assert "不是合法 JSON" in tool_message.content


def test_tool_error_is_reported(run_async):
    """验证工具业务失败（如文件不存在）会以错误文本回填给模型。"""
    provider = FakeProvider(
        [
            _tool_turn(ToolCall(id="call_1", name="fail", arguments="{}")),
            AssistantTurn(text="读取失败", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider, tools=ToolRegistry([_FailingTool()]))

    result = run_async(agent.run("q"))

    assert result.text == "读取失败"
    tool_message = next(m for m in agent.history if m.role == "tool")
    assert tool_message.content == "工具执行失败：文件不存在"


def test_tool_call_without_registered_tools_is_paired(run_async):
    """验证 Agent 未注册工具时，仍会为 tool_calls 补上错误消息，保持协议配对。"""
    provider = FakeProvider(
        [
            _tool_turn(ToolCall(id="call_1", name="echo", arguments="{}")),
            AssistantTurn(text="没有工具", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider)

    result = run_async(agent.run("q"))

    assert result.text == "没有工具"
    tool_message = next(m for m in agent.history if m.role == "tool")
    assert "未注册任何工具" in tool_message.content
    assert provider.calls[0].tools == ()


def test_tool_specs_are_forwarded_every_step(run_async):
    """验证每一步模型调用都会携带完整的工具声明。"""
    provider = FakeProvider(
        [
            _tool_turn(
                ToolCall(id="call_1", name="read_resume", arguments='{"path":"x"}')
            ),
            AssistantTurn(text="done", finish_reason="stop"),
        ]
    )
    agent = MiniAgent(provider, tools=create_default_registry())

    run_async(agent.run("q"))

    expected = ["read_resume", "read_jd", "analyze_resume"]
    assert [spec.name for spec in provider.calls[0].tools] == expected
    assert [spec.name for spec in provider.calls[1].tools] == expected


def test_tool_loop_respects_max_steps(run_async):
    """验证模型反复请求工具时，max_steps 仍能终止循环。"""
    turn = _tool_turn(ToolCall(id="call_1", name="echo", arguments='{"value":"x"}'))
    provider = FakeProvider([turn, turn, turn, turn], echo=False)
    agent = MiniAgent(
        provider,
        max_steps=3,
        tools=ToolRegistry([_EchoTool()]),
    )

    result = run_async(agent.run("q"))

    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 3
    assert len(provider.calls) == 3
    assert sum(1 for m in agent.history if m.role == "tool") == 3
