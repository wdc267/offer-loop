"""工具调用前的权限检查（s03）。

两张表、三道门，按顺序执行：
1. 拒绝表 ``DENY_LIST``：明确的危险命令，命中即拒绝，不询问用户；
2. 确认规则表 ``PERMISSION_RULES``：上下文相关的风险操作（越界写、修改/删除简历），
   命中后进入下一道门；
3. 用户确认 ``ask_user``：在终端逐次询问，只有明确同意才放行。

规则命中表示「需要用户确认」，不是自动拒绝；只有拒绝表是硬阻断。
拒绝结果由 ToolDispatcher 写成 tool 消息，模型可以解释原因或换方案。
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, TextIO

logger = logging.getLogger(__name__)

# 写 / 删操作的默认边界：进程启动目录。
# 规则命中只触发询问，不会直接拒绝；需要更严格时修改本值即可。
WORKDIR = Path.cwd().resolve()

# Gate 1：命中即拒绝，不询问。
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

_DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)

# 未来接入 bash 工具时，这些关键字也会触发确认。
_DESTRUCTIVE_KEYWORDS = ("rm ", "> /etc/", "chmod 777")


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    """一次需要用户确认的工具操作。

    由工具在 ``permission_request()`` 中生成，只包含确定性摘要，
    不能携带简历 / JD 正文。

    属性：
        tool_name: 发起操作的工具名。
        action: 操作类型，当前为 ``modify`` 或 ``delete``。
        path: 被操作文件的目标绝对路径。
        summary: 工具生成的说明，如写入模式、文件大小和内容预览。
    """

    tool_name: str
    action: str
    path: str
    summary: str


Approver = Callable[
    [str, dict[str, Any], str, PermissionRequest | None],
    Awaitable[bool],
]


def check_deny_list(command: str) -> str | None:
    """检查命令是否命中拒绝表。

    参数：
        command: 待执行的命令文本；其他工具的路径参数不会传到这里。

    返回值：
        命中时返回拒绝原因，否则返回 None。
    """
    for pattern in DENY_LIST:
        if pattern in command:
            return f"命令包含禁止项 {pattern!r}"
    return None


def contains_destructive_command(command: str) -> bool:
    """判断命令中是否包含 rm / del 等破坏性命令单词。

    参数：
        command: 待检查的命令文本。

    返回值：
        True 表示出现独立的 rm / del 命令。
    """
    return bool(_DESTRUCTIVE_COMMAND_WORD.search(command))


def _inside_workspace(path_value: str) -> bool:
    """判断路径是否位于 WORKDIR 之内。

    参数：
        path_value: 模型给出的路径；相对路径按 WORKDIR 解析。

    返回值：
        True 表示路径解析后仍在工作区内；空值或无法解析时为 False。
    """
    if not path_value:
        return False
    try:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = WORKDIR / path
        return path.resolve().is_relative_to(WORKDIR)
    except (OSError, RuntimeError):
        return False


def _outside_workspace(args: dict[str, Any]) -> bool:
    """规则检查：写 / 删目标是否越过工作区。

    参数：
        args: 工具参数，读取其中的 path 字段。

    返回值：
        True 表示路径在 WORKDIR 之外，需要用户确认。
    """
    raw = args.get("path")
    return not isinstance(raw, str) or not _inside_workspace(raw)


def _destructive_command(args: dict[str, Any]) -> bool:
    """规则检查：命令是否具有破坏性。

    参数：
        args: 工具参数，读取其中的 command 字段。

    返回值：
        True 表示命令包含 rm / del 或危险关键字，需要用户确认。
    """
    command = args.get("command")
    if not isinstance(command, str):
        return False
    return contains_destructive_command(command) or any(
        keyword in command for keyword in _DESTRUCTIVE_KEYWORDS
    )


def _always_confirm(args: dict[str, Any]) -> bool:
    """规则检查：修改 / 删除操作一律需要确认。

    参数：
        args: 工具参数；本检查不使用。

    返回值：
        始终为 True。
    """
    _ = args  # 规则签名统一，本检查不需要参数。
    return True


# Gate 2：上下文相关的确认规则。按列表顺序匹配，返回第一条命中的原因。
PERMISSION_RULES: list[dict[str, Any]] = [
    {
        "tools": ["update_resume", "delete_resume"],
        "check": _outside_workspace,
        "message": "路径不在工作区内",
    },
    {
        "tools": ["bash"],
        "check": _destructive_command,
        "message": "可能具有破坏性的命令",
    },
    {
        "tools": ["update_resume", "delete_resume"],
        "check": _always_confirm,
        "message": "修改 / 删除简历文件须由用户确认",
    },
]


def check_rules(tool_name: str, args: dict[str, Any]) -> str | None:
    """按顺序检查确认规则表。

    参数：
        tool_name: 即将执行的工具名。
        args: 模型传入的 JSON 参数。

    返回值：
        第一条命中规则的说明；没有命中时返回 None。
    """
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


async def check_permission(
    tool_name: str,
    args: dict[str, Any],
    request: PermissionRequest | None = None,
    *,
    approver: Approver | None = None,
) -> str | None:
    """依次通过拒绝表、确认规则和用户确认，决定是否允许执行。

    参数：
        tool_name: 即将执行的工具名。
        args: 模型传入的 JSON 参数。
        request: 工具生成的安全摘要，用于终端提示；可为 None。
        approver: 用户确认实现；None 时使用交互式 ``ask_user``。

    返回值：
        None 表示允许执行；字符串是拒绝原因，由分发器回填给模型。
    """
    command = args.get("command")
    if isinstance(command, str):
        blocked = check_deny_list(command)
        if blocked is not None:
            logger.warning("命中拒绝表 name=%s reason=%s", tool_name, blocked)
            return f"操作被拒绝：{blocked}"

    reason = check_rules(tool_name, args)
    if reason is None:
        return None

    approve = approver if approver is not None else ask_user
    try:
        allowed = await approve(tool_name, args, reason, request)
    except Exception:  # noqa: BLE001 - 询问失败按拒绝处理，不能放行
        logger.exception("权限询问异常 name=%s", tool_name)
        return f"操作未执行：权限确认过程出错（{reason}）"

    if not allowed:
        logger.info("用户拒绝工具调用 name=%s reason=%s", tool_name, reason)
        return f"操作被用户拒绝，未执行（{reason}）"
    return None


def format_request(
    tool_name: str,
    reason: str,
    request: PermissionRequest | None = None,
) -> str:
    """把一次待确认操作渲染成终端提示文本。

    参数：
        tool_name: 即将执行的工具名。
        reason: 命中的规则说明。
        request: 工具生成的安全摘要；None 时只展示原因和工具名。

    返回值：
        多行提示文本；不包含完整正文。
    """
    lines = ["⚠ 工具调用需要确认", f"原因：{reason}", f"工具：{tool_name}"]
    if request is not None:
        if request.path:
            lines.append(f"路径：{request.path}")
        if request.summary:
            lines.append(request.summary)
    return "\n".join(lines)


async def ask_user(
    tool_name: str,
    args: dict[str, Any],
    reason: str,
    request: PermissionRequest | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    stream: TextIO | None = None,
) -> bool:
    """在终端显示操作摘要并等待用户确认。

    参数：
        tool_name: 即将执行的工具名。
        args: 模型传入的参数；本函数不直接展示，避免泄露简历正文。
        reason: 命中的规则说明。
        request: 工具生成的安全摘要。
        input_fn: 读取用户输入的函数，测试时可替换。
        stream: 提示输出的目标流；None 时使用 stderr。

    返回值：
        用户输入 y / yes / 是 时为 True；其余输入或 EOF 为 False。
    """
    text = format_request(tool_name, reason, request)
    target = stream if stream is not None else sys.stderr
    # args 里可能有整份简历正文，只使用工具生成的安全摘要。
    _ = args
    # input() 是阻塞调用，放线程里避免卡住事件循环。
    return await asyncio.to_thread(_ask, text, input_fn, target)


def _ask(text: str, input_fn: Callable[[str], str], stream: TextIO) -> bool:
    """同步完成一次终端询问。

    参数：
        text: 已渲染好的操作说明。
        input_fn: 读取用户输入的函数。
        stream: 提示输出的目标流。

    返回值：
        True 表示用户确认，False 表示拒绝或输入结束。
    """
    print(text, file=stream)
    try:
        answer = input_fn("允许执行？[y/N] ")
    except EOFError:
        print(file=stream)
        return False
    return answer.strip().lower() in {"y", "yes", "是"}
