"""OpenAI 兼容接口适配器。

这是整个项目里唯一接触 openai SDK 的文件。
上层只看到 AssistantTurn / Message，不看到任何 SDK 对象。

适配范围：DeepSeek、通义千问、智谱等提供 OpenAI 兼容
/chat/completions 端点的服务。切换供应商只需改 base_url 和 model。
"""

from __future__ import annotations

from typing import Any, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from ..errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from ..models import AssistantTurn, Message, ToolCall, ToolSpec, Usage


class OpenAICompatibleProvider:
    name = "openai-compat"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """创建 OpenAI 兼容 Provider。

        参数：
            api_key: 服务商 API key。
            base_url: OpenAI 兼容端点地址。
            model: 模型名。
            timeout: 单次请求超时秒数。
            client: 可注入的 AsyncOpenAI 客户端，主要用于测试；
                注入时本类不负责关闭它。
        """
        self.model = model
        self._base_url = base_url
        # 注入 client 是为了测试；只有自己创建的 client 才由本类负责关闭。
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            # 显式关闭 SDK 默认重试（默认 2 次）。
            # s01 明确不做重试；否则一次网络抖动会变成 3 次请求，
            # 错误测试也无法断言调用次数。
            max_retries=0,
        )

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> AssistantTurn:
        """调用 OpenAI 兼容的 /chat/completions 端点。

        参数：
            messages: 完整消息历史。
            temperature: 采样温度。
            max_tokens: 生成上限；None 时不传该字段。
            tools: 工具声明；为空时不传 tools 字段。

        返回值：
            模型输出映射后的 AssistantTurn。

        异常：
            ProviderError: SDK 异常翻译后的统一领域异常。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        # 某些兼容端点不接受 max_tokens=None，因此为 None 时干脆不传。
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # 没有工具时不传 tools 字段，避免部分端点拒绝空数组。
        if tools:
            kwargs["tools"] = [tool.to_openai() for tool in tools]

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - 需要按 SDK 异常类型翻译
            translated = self._translate_error(exc)
            if translated is exc:
                # 不是 SDK 错误（例如我们自己的编程错误），原样冒泡，
                # 避免把 Bug 伪装成网络问题。
                raise
            raise translated from exc

        return self._to_turn(completion)

    async def aclose(self) -> None:
        """关闭本类自己创建的 HTTP 客户端；注入的客户端不处理。"""
        if self._owns_client:
            await self._client.close()

    # ------------------------------------------------------------------
    # SDK 响应 -> 领域对象
    # ------------------------------------------------------------------
    def _to_turn(self, completion: Any) -> AssistantTurn:
        """把 SDK 响应对象转换成领域对象 AssistantTurn。

        参数：
            completion: openai SDK 返回的 chat completion 对象。

        返回值：
            包含文本、结束原因、用量和 tool_calls 的 AssistantTurn。

        异常：
            ProviderResponseError: 响应缺少 choices 或 message 时抛出。
        """
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ProviderResponseError("模型响应中没有 choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ProviderResponseError("模型响应中没有 message")

        text = getattr(message, "content", None) or ""
        raw_tool_calls = getattr(message, "tool_calls", None) or ()
        tool_calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for call in raw_tool_calls
        )

        raw_usage = getattr(completion, "usage", None)
        usage = (
            Usage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
            )
            if raw_usage is not None
            else Usage()
        )

        return AssistantTurn(
            text=text,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # SDK 异常 -> 领域异常
    # ------------------------------------------------------------------
    def _translate_error(self, exc: Exception) -> Exception:
        """把 SDK 异常翻译成 MiniAgent 的领域异常。

        参数：
            exc: 调用 SDK 时捕获到的异常。

        返回值：
            对应的 ProviderError；不是 SDK 异常时原样返回 exc，
            以便上层区分「供应商故障」和「自身 Bug」。
        """
        # 顺序很重要：子类必须先于父类判断。
        # APITimeoutError 继承 APIConnectionError；
        # AuthenticationError / RateLimitError 继承 APIStatusError。
        if isinstance(exc, APITimeoutError):
            return ProviderTimeoutError(f"模型请求超时：{exc}")
        if isinstance(exc, AuthenticationError):
            return ProviderAuthError(f"模型鉴权失败，请检查 MINI_AGENT_API_KEY：{exc}")
        if isinstance(exc, RateLimitError):
            return ProviderRateLimitError(f"模型触发限流：{exc}")
        if isinstance(exc, APIConnectionError):
            return ProviderConnectionError(
                f"无法连接模型服务，请检查 MINI_AGENT_BASE_URL 和网络：{exc}"
            )
        if isinstance(exc, APIStatusError):
            return ProviderResponseError(f"模型服务返回 HTTP {exc.status_code}：{exc}")
        if isinstance(exc, OpenAIError):
            return ProviderError(str(exc))
        return exc
