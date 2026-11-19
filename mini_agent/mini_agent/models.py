"""MiniAgent 的领域数据结构。

这一层刻意不依赖模型 SDK、不做 I/O：
- Agent 与 Provider 之间只传递这些对象；
- 只有 providers/openai_compat.py 负责把它们转换成 SDK 需要的 wire format。

三个核心概念要分清：
- Message:        历史中的一条消息（可回放、可发给模型）
- AssistantTurn:  模型一次调用的原始输出（可能带 tool_calls / finish_reason）
- AgentResult:    一次 run() 的整体结论（为什么停、跑了几步、花了多少 token）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Sequence

Role = Literal["system", "user", "assistant", "tool"]


class StopReason(StrEnum):
    """一次 run() 结束的原因。

    COMPLETED:       模型给出了非空最终回答
    EMPTY_RESPONSE:  模型返回空内容（不是故障，s01 不重试）
    MAX_STEPS:       达到循环上限（s01 主要作为 s02 的护栏）
    """

    COMPLETED = "completed"
    EMPTY_RESPONSE = "empty_response"
    MAX_STEPS = "max_steps"


@dataclass(frozen=True, slots=True)
class Usage:
    """一次或多次模型调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: object) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求调用一次工具。

    s01 只检测、不分发；arguments 保留原始 JSON 字符串，
    等 s02 再做解析和参数校验，避免提前引入 s01 用不到的失败路径。
    """

    id: str
    name: str
    arguments: str = "{}"


@dataclass(frozen=True, slots=True)
class Message:
    """历史中的一条消息。

    frozen 是有意的：历史是 Agent 的状态，应当只能追加，不能被随意改写。
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"非法 role: {self.role!r}")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("只有 assistant 消息可以携带 tool_calls")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool 消息必须提供 tool_call_id")
        # 允许调用方传 list，但内部统一成 tuple，保证不可变。
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: Sequence[ToolCall] = ()
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(
        cls, tool_call_id: str, content: str, name: str | None = None
    ) -> Message:
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )

    def to_openai(self) -> dict[str, Any]:
        """转换成 OpenAI 兼容接口需要的 dict。

        这里是「领域对象 -> wire format」的唯一转换点。
        """
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """模型一次调用的输出。

    text 可能为空但 tool_calls 非空（模型只想调工具），
    所以不能简单用「text 是否为空」判断这次调用是否成功。
    """

    text: str = ""
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def to_message(self) -> Message:
        """把这一次输出记进历史。"""
        return Message.assistant(self.text, self.tool_calls)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """一次 run() 的结论。"""

    text: str
    stop_reason: StopReason
    steps: int
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
