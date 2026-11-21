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
        """累加两份 token 用量，支持 `usage_a + usage_b`。

        参数：
            other: 另一份 Usage；不是 Usage 时返回 NotImplemented。

        返回值：
            三个字段逐项相加后的新 Usage。
        """
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """工具对模型暴露的声明。

    parameters 是 JSON Schema；只有 openai_compat.py 负责把它转成
    wire format，其余代码不关心协议细节。
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        """转换成 OpenAI 的 function 工具声明。

        返回值：
            可直接放进请求 tools 列表的字典。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求调用一次工具。

    arguments 保留模型返回的原始 JSON 字符串，
    由 s02 的 ToolDispatcher 负责解析和参数校验。
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
        """校验 role 与工具字段的约束，并把 tool_calls 统一成 tuple。

        异常：
            ValueError: role 非法、非 assistant 消息携带 tool_calls，
                或 tool 消息缺少 tool_call_id 时抛出。
        """
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
        """构造 system 消息。

        参数：
            content: system prompt 文本。

        返回值：
            role="system" 的 Message。
        """
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """构造 user 消息。

        参数：
            content: 用户输入文本。

        返回值：
            role="user" 的 Message。
        """
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str = "", tool_calls: Sequence[ToolCall] = ()
    ) -> Message:
        """构造 assistant 消息。

        参数：
            content: 模型输出的文本，可能为空。
            tool_calls: 模型请求调用的工具列表。

        返回值：
            role="assistant" 的 Message。
        """
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str | None = None) -> Message:
        """构造 tool 消息，用于回填工具执行结果。

        参数：
            tool_call_id: 对应的 ToolCall.id，必须与 assistant 消息配对。
            content: 工具返回的纯文本或错误信息。
            name: 可选的工具名，仅用于调试展示。

        返回值：
            role="tool" 的 Message。
        """
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )

    def to_openai(self) -> dict[str, Any]:
        """转换成 OpenAI 兼容接口需要的 dict。

        这里是「领域对象 -> wire format」的唯一转换点。

        返回值：
            可直接放进 messages 列表的字典；tool 字段只在有值时出现。
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
        """把 tool_calls 统一成 tuple，保证输出对象不可变。"""
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def to_message(self) -> Message:
        """把这一次模型输出转换成可追加到历史的 assistant 消息。

        返回值：
            携带相同文本和 tool_calls 的 Message。
        """
        return Message.assistant(self.text, self.tool_calls)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """一次 run() 的结论。"""

    text: str
    stop_reason: StopReason
    steps: int
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
