"""模型调用的统一边界。

Agent 只依赖这个 Protocol，不知道背后是 DeepSeek、通义还是 FakeProvider。
这样换供应商、写离线测试、后续按任务路由模型，都不需要改业务代码。
"""
from __future__ import annotations

from typing import Protocol, Sequence

from ..models import AssistantTurn, Message


class LLMProvider(Protocol):
    """最小模型接口。"""

    name: str
    model: str

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        """根据完整消息历史生成一次 assistant 输出。"""
        ...

    async def aclose(self) -> None:
        """释放底层资源。FakeProvider 为空实现。"""
        ...
