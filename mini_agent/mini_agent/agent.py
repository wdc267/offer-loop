"""Agent Loop 本体（s01 循环 + s02 工具分发）。

主循环只做三件事：调模型、记历史、判断是否继续。
「判断是否继续」被抽到 _on_assistant_turn，是接入工具的唯一扩展缝。
主循环本身不感知任何具体工具。
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from .errors import InvalidInputError
from .models import AgentResult, AssistantTurn, Message, StopReason, Usage
from .providers.base import LLMProvider
from .tools import ToolDispatcher, ToolRegistry

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
        tools: ToolRegistry | None = None,
    ) -> None:
        """创建 Agent 并准备消息历史。

        参数：
            provider: 模型调用实现（真实 API 或 FakeProvider）。
            system_prompt: 系统提示词；为空时不写入历史。
            max_steps: 单次 run() 最多调用模型的次数，必须 >= 1。
            temperature: 采样温度，透传给 Provider。
            max_tokens: 单次生成上限；None 表示不传该参数。
            initial_messages: 构造时预置的历史消息，会被复制一份。
            tools: 工具注册表；为空或 None 时 Agent 不具备任何工具。

        异常：
            ValueError: max_steps 小于 1 时抛出。
        """
        if max_steps < 1:
            raise ValueError("max_steps 必须 >= 1")

        self._provider = provider
        self._max_steps = max_steps
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._registry = tools if tools and len(tools) else None
        self._dispatcher = (
            ToolDispatcher(self._registry) if self._registry is not None else None
        )

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
        """返回当前消息历史的不可变快照。

        返回值：
            Message 组成的 tuple；调用方无法通过它修改 Agent 内部状态。
        """
        return tuple(self._history)

    def reset(self) -> None:
        """清空对话历史，只保留构造时的 system prompt。"""
        self._history = list(self._seed)

    async def run(self, user_input: str) -> AgentResult:
        """处理一次用户输入，直到模型给出最终回答或用完步数。

        参数：
            user_input: 用户输入文本；会自动去掉首尾空白。

        返回值：
            AgentResult，包含最终文本、停止原因、步数和累计 token 用量。

        异常：
            InvalidInputError: 输入为空或不是字符串时抛出。
            ProviderError: 模型调用失败时向上传播。
        """
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
                tools=self._registry.specs() if self._registry is not None else None,
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
        """处理模型的一次输出，决定是否继续循环。

        参数：
            turn: 模型本次调用的输出；若带 tool_calls 则逐个执行并回填。

        返回值：
            True  = 已经向历史补好下一步需要的内容（这里就是 tool 结果），
                    请再调用一次模型。
            False = 本轮结束，使用 turn.text 作为最终回答。

        工具调用必须逐个配平：每个 tool_call_id 都要有一条 tool 消息，
        否则下一次请求会被 OpenAI 兼容端点拒绝。
        """
        if not turn.tool_calls:
            return False

        if self._dispatcher is None:
            # 模型要求工具但 Agent 没注册：回填错误让模型改口，
            # 不能直接结束，否则历史里会留下没有配对的 tool_calls。
            for call in turn.tool_calls:
                self._history.append(
                    Message.tool(call.id, "工具执行失败：本 Agent 未注册任何工具")
                )
            return True

        for call in turn.tool_calls:
            self._history.append(await self._dispatcher.dispatch(call))
        return True
