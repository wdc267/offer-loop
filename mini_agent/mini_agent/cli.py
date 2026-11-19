"""命令行入口。

两种模式：
- 单次：python -m mini_agent "你好"
- REPL：python -m mini_agent

关键约束：整个进程只调用一次 asyncio.run()。
每轮都新建事件循环会关闭 AsyncOpenAI 绑定的 HTTP 客户端，
第二轮会报 "Event loop is closed"。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from typing import Sequence

from .agent import MiniAgent
from .config import Settings, load_settings
from .errors import ConfigError, ProviderError, UnsupportedToolCallError
from .models import AgentResult, Message, StopReason
from .providers import create_provider

DEFAULT_SYSTEM_PROMPT = (
    "你是 MiniAgent，一个用于学习 Agent Loop 的最小助手。回答简洁、准确。"
)

_HELP = """可用命令：
  /history  查看当前消息历史
  /reset    清空历史（保留 system prompt）
  /help     显示本帮助
  /exit     退出（也可用 /quit 或 Ctrl+D）"""

logger = logging.getLogger("mini_agent.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="MiniAgent s01：最小 Agent Loop",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="单次模式：直接提问；省略则进入 REPL",
    )
    parser.add_argument(
        "--provider",
        choices=("deepseek", "fake"),
        default=None,
        help="覆盖 MINI_AGENT_PROVIDER",
    )
    parser.add_argument("--model", default=None, help="覆盖 MINI_AGENT_MODEL")
    parser.add_argument("--base-url", default=None, help="覆盖 MINI_AGENT_BASE_URL")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="覆盖 MINI_AGENT_MAX_STEPS"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="覆盖 MINI_AGENT_TEMPERATURE"
    )
    parser.add_argument("--system", default=None, help="覆盖默认 system prompt")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="输出 DEBUG 日志"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        settings = load_settings()
        overrides: dict[str, object] = {}
        if args.provider is not None:
            overrides["provider"] = args.provider
        if args.model is not None:
            overrides["model"] = args.model
        if args.base_url is not None:
            overrides["base_url"] = args.base_url
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps
        if args.temperature is not None:
            overrides["temperature"] = args.temperature
        if overrides:
            settings = replace(settings, **overrides)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    try:
        return asyncio.run(_amain(settings, args))
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


async def _amain(settings: Settings, args: argparse.Namespace) -> int:
    try:
        provider = create_provider(settings)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    # Provider 在运行中的事件循环内创建并持有，直到 aclose()。
    agent = MiniAgent(
        provider,
        system_prompt=args.system or DEFAULT_SYSTEM_PROMPT,
        max_steps=settings.max_steps,
        temperature=settings.temperature,
    )

    logger.info(
        "provider=%s model=%s base_url=%s max_steps=%d temperature=%.2f",
        getattr(provider, "name", settings.provider),
        getattr(provider, "model", settings.model),
        settings.base_url if settings.provider != "fake" else "-",
        settings.max_steps,
        settings.temperature,
    )

    try:
        if args.prompt is not None:
            return await _run_once(agent, args.prompt)
        return await _repl(agent)
    finally:
        await provider.aclose()


async def _run_once(agent: MiniAgent, prompt: str) -> int:
    try:
        result = await agent.run(prompt)
    except ProviderError as exc:
        print(f"模型调用失败：{exc}", file=sys.stderr)
        return 2
    except UnsupportedToolCallError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _print_result(result)
    return 0


async def _repl(agent: MiniAgent) -> int:
    print("MiniAgent s01 REPL。输入 /help 查看命令，/exit 退出。")
    while True:
        try:
            line = input("you> ")
        except EOFError:
            print(file=sys.stderr)
            return 0
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 130

        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/reset":
            agent.reset()
            print("已清空历史（保留 system prompt）。")
            continue
        if text == "/history":
            _print_history(agent.history)
            continue
        if text == "/help":
            print(_HELP)
            continue

        try:
            result = await agent.run(text)
        except ProviderError as exc:
            print(f"模型调用失败：{exc}", file=sys.stderr)
            continue
        except UnsupportedToolCallError as exc:
            print(f"{exc} 本轮历史已重置。", file=sys.stderr)
            agent.reset()
            continue
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 130

        print(f"assistant> {result.text}")
        if result.stop_reason is StopReason.EMPTY_RESPONSE:
            print("（模型返回空内容）", file=sys.stderr)
        elif result.stop_reason is StopReason.MAX_STEPS:
            print("（达到最大步数）", file=sys.stderr)
        if result.finish_reason == "length":
            print("（回答可能被截断：finish_reason=length）", file=sys.stderr)


def _print_result(result: AgentResult) -> None:
    if result.text:
        print(result.text)
    if result.stop_reason is StopReason.EMPTY_RESPONSE:
        print("（模型返回空内容）", file=sys.stderr)
    elif result.stop_reason is StopReason.MAX_STEPS:
        print("（达到最大步数，返回最后一次输出）", file=sys.stderr)
    if result.finish_reason == "length":
        print("（回答可能被截断：finish_reason=length）", file=sys.stderr)


def _print_history(history: Sequence[Message]) -> None:
    if not history:
        print("（历史为空）")
        return
    for index, message in enumerate(history, start=1):
        preview = message.content.replace("\n", " ")[:60]
        suffix = f" tool_calls={len(message.tool_calls)}" if message.tool_calls else ""
        print(f"{index:02d} {message.role:9s} {preview!r}{suffix}")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def _ensure_utf8_stdio() -> None:
    """Windows 重定向/管道场景下按 UTF-8 读写，避免中文乱码。

    PEP 528 之后，直接连控制台时 stdin/stdout 已经是 UTF-8；
    但管道和文件重定向仍按 locale（通常是 cp936）解码。这里：
    - stdout/stderr 总是重设为 UTF-8；
    - stdin 只在非交互时重设，避免改变终端本身的输入行为。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # pytest/IDE 包装的流可能不支持 reconfigure，忽略即可。
            pass

    if sys.stdin.isatty():
        return
    reconfigure = getattr(sys.stdin, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass
