"""模型调用的统一边界。

Agent 只依赖这个 Protocol，不知道背后是 DeepSeek、通义还是 FakeProvider。
这样换供应商、写离线测试、后续按任务路由模型，都不需要改业务代码。
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..models import AssistantTurn, Message, ToolSpec


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
        tools: Sequence[ToolSpec] | None = None,
    ) -> AssistantTurn:
        """根据完整消息历史生成一次 assistant 输出。

        参数：
            messages: 完整消息历史，按时间顺序排列。
            temperature: 采样温度。
            max_tokens: 单次生成上限；None 表示不限制。
            tools: 可选的工具声明；为空或 None 时请求中不携带。

        返回值：
            模型本次输出的 AssistantTurn。

        异常：
            ProviderError: 网络、鉴权、限流或响应格式异常时抛出。
        """
        ...

    async def aclose(self) -> None:
        """释放底层资源；FakeProvider 为空实现。"""
