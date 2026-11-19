"""Agent Loop 本体（s01）。

主循环只做三件事：调模型、记历史、判断是否继续。
「判断是否继续」被抽到 _on_assistant_turn，作为 s02 接入工具的唯一扩展缝。
主循环本身不感知任何具体工具。
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

from .errors import InvalidInputError, UnsupportedToolCallError
from .models import AgentResult, AssistantTurn, Message, StopReason, Usage
from .providers.base import LLMProvider

logger = logging.getLogger(__name__)


class MiniAgent:
    """最小 Agent：持有消息历史，负责一次用户输入的完整处理。"""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        system_prompt: str | None = None,
        max_steps: int = 8,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        initial_messages: Sequence[Message] | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")

        self._provider = provider
        self._max_steps = max_steps
        self._temperature = temperature
        self._max_tokens = max_tokens

        seed: list[Message] = []
        if system_prompt:
            seed.append(Message.system(system_prompt))
        if initial_messages:
            # 复制一份：调用方后续修改原列表不应影响 Agent。
            seed.extend(initial_messages)

        self._history = list(seed)
        # reset() 后保留的固定前缀。只保留 system 消息：
        # reset 的语义是「开始一段新对话」，而不是「回到构造时的任意快照」。
        self._seed = tuple(m for m in seed if m.role == "system")

    @property
    def history(self) -> tuple[Message, ...]:
        """当前历史（不可变快照）。"""
        return tuple(self._history)

    def reset(self) -> None:
        """清空对话历史，保留 system prompt。"""
        self._history = list(self._seed)

    async def run(self, user_input: str) -> AgentResult:
        """处理一次用户输入，直到模型给出最终回答或用完步数。"""
        if not isinstance(user_input, str) or not user_input.strip():
            raise InvalidInputError("用户输入不能为空")

        self._history.append(Message.user(user_input.strip()))
        total_usage = Usage()
        last_text = ""
        last_finish_reason: str | None = None

        for step in range(1, self._max_steps + 1):
            # 传快照而不是内部 list：Provider 无法反向修改 Agent 的状态。
            snapshot = tuple(self._history)
            started = time.perf_counter()
            turn = await self._provider.generate(
                snapshot,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            total_usage += turn.usage
            self._history.append(turn.to_message())
            last_text = turn.text
            last_finish_reason = turn.finish_reason

            # 不打印完整用户内容：后续会处理简历和面试回答，先养成隐私习惯。
            logger.info(
                "step=%d/%d provider=%s model=%s messages=%d "
                "finish_reason=%s text_chars=%d tool_calls=%d "
                "usage=%s elapsed_ms=%.0f",
                step,
                self._max_steps,
                getattr(self._provider, "name", "?"),
                getattr(self._provider, "model", "?"),
                len(snapshot),
                turn.finish_reason,
                len(turn.text),
                len(turn.tool_calls),
                turn.usage,
                elapsed_ms,
            )

            # s02 的工具分发入口：返回 True 表示「需要再调一次模型」。
            if await self._on_assistant_turn(turn):
                continue

            if not turn.text.strip():
                logger.warning("模型返回空内容，结束本轮")
                return AgentResult(
                    text="",
                    stop_reason=StopReason.EMPTY_RESPONSE,
                    steps=step,
                    finish_reason=turn.finish_reason,
                    usage=total_usage,
                )

            if turn.finish_reason == "length":
                logger.warning("finish_reason=length，回答可能被截断")

            return AgentResult(
                text=turn.text,
                stop_reason=StopReason.COMPLETED,
                steps=step,
                finish_reason=turn.finish_reason,
                usage=total_usage,
            )

        logger.warning("达到最大步数 %d，返回最后一次输出", self._max_steps)
        return AgentResult(
            text=last_text,
            stop_reason=StopReason.MAX_STEPS,
            steps=self._max_steps,
            finish_reason=last_finish_reason,
            usage=total_usage,
        )

    async def _on_assistant_turn(self, turn: AssistantTurn) -> bool:
        """决定这一步之后是否继续循环。

        返回值：
            True  = 已经向历史补好下一步需要的内容（s02 是 tool 结果），
                    请再调用一次模型。
            False = 本轮结束，使用 turn.text 作为最终回答。

        s01 约定：
            - 收到 tool_calls -> 抛 UnsupportedToolCallError（工具属于 s02）；
            - 其余情况       -> 返回 False。

        s02 约定：
            - 解析 turn.tool_calls 的参数；
            - 调用 ToolDispatcher 执行；
            - 为每个调用追加 Message.tool(tool_call_id=..., content=...)；
            - 返回 True。
        主循环结构不需要任何修改。
        """
        if turn.tool_calls:
            raise UnsupportedToolCallError(
                "s01 不支持工具调用；工具分发将在 s02 的 "
                "_on_assistant_turn() 中实现。"
            )
        return False
