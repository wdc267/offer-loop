"""脚本化的离线 Provider。

用途：
- 单元测试不联网、不需要 API key；
- CLI 的 --provider fake 模式可以演示完整循环；
- 可以精确控制「模型返回什么」，从而测试空响应、异常、max_steps 等边界。

行为是确定性的：不睡眠、不随机、不访问网络。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..errors import ScriptExhaustedError
from ..models import AssistantTurn, Message, Usage

ScriptItem = str | AssistantTurn | Exception


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """一次 generate() 的入参快照，供测试断言。"""

    messages: tuple[Message, ...]
    temperature: float
    max_tokens: int | None


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        echo: bool = True,
    ) -> None:
        """
        参数：
            script: 按顺序消费的脚本项。
                - str: 返回一条 finish_reason="stop" 的文本
                - AssistantTurn: 原样返回
                - Exception: 立即抛出（用于测试故障路径）
            echo: 脚本耗尽后的行为。
                - True:  返回 "[fake] 收到：<最后一条 user 消息>"
                - False: 抛 ScriptExhaustedError，用于断言调用次数
        """
        self._script = list(script or [])
        self._index = 0
        self._echo = echo
        self._calls: list[RecordedCall] = []

    @property
    def calls(self) -> tuple[RecordedCall, ...]:
        """已经发生的调用记录（不可变快照）。"""
        return tuple(self._calls)

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        self._calls.append(
            RecordedCall(
                messages=tuple(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        if self._index < len(self._script):
            item = self._script[self._index]
            self._index += 1
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):
                return AssistantTurn(text=item, finish_reason="stop")
            return item

        if not self._echo:
            raise ScriptExhaustedError(
                f"FakeProvider 脚本已耗尽（已调用 {len(self._calls)} 次）"
            )

        last_user = next(
            (m for m in reversed(messages) if m.role == "user"),
            None,
        )
        content = last_user.content if last_user is not None else "(无用户消息)"
        return AssistantTurn(
            text=f"[fake] 收到：{content}",
            finish_reason="stop",
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def aclose(self) -> None:
        return None
