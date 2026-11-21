"""工具定义、注册与分发（s02）。

分工：
- Tool          只做确定性工作（读文件、结构统计），不调用模型；
- ToolRegistry  管理工具名 -> 实例，并向模型暴露 ToolSpec；
- ToolDispatcher 解析模型参数、执行工具、把结果 / 错误转换成 tool 消息。

工具层失败不会抛出到 Agent 循环：Dispatcher 统一转成一条 tool 消息，
让模型自己决定重试、换文件还是如实告知用户。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Protocol, Sequence

from .documents import load_document
from .errors import ToolError
from .models import Message, ToolCall, ToolSpec

logger = logging.getLogger(__name__)

_PATH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "文件的绝对路径，或相对于当前工作目录的路径",
        }
    },
    "required": ["path"],
    "additionalProperties": False,
}


class Tool(Protocol):
    """一个可被模型调用的工具。"""

    spec: ToolSpec

    async def run(self, **kwargs: Any) -> str:
        """执行工具并返回给模型看的纯文本。

        参数：
            **kwargs: 模型给出的 JSON 参数，键名与 ToolSpec.parameters 对应。

        返回值：
            可作为 tool 消息内容的纯文本。

        异常：
            ToolError: 参数非法或工具执行失败时抛出。
        """
        ...


# ---------------------------------------------------------------------------
# 三个业务工具
# ---------------------------------------------------------------------------
class _ReadDocumentTool:
    """读取类工具的公共实现：加载文档 -> 渲染成模型可读文本。"""

    label = "文档"
    spec: ToolSpec

    async def run(self, **kwargs: Any) -> str:
        """读取文档并渲染成模型可读文本。

        参数：
            **kwargs: 必须包含 path 字段，指向待读取的文件。

        返回值：
            带文件元信息（类型、字符数）的纯文本。

        异常：
            ToolError: 缺少 path、传入多余参数或文档读取失败时抛出。
        """
        path = _require_path(kwargs)
        # 文件解析是阻塞 I/O / CPU 操作，放线程里避免卡住事件循环。
        document = await asyncio.to_thread(load_document, path)
        return document.to_model_text(self.label)


class ReadResumeTool(_ReadDocumentTool):
    label = "简历"
    spec = ToolSpec(
        name="read_resume",
        description=(
            "读取简历文件并返回纯文本。支持 PDF、DOCX、Markdown、TXT。"
            "用户提供简历路径时先调用本工具，不要臆造简历内容。"
        ),
        parameters=_PATH_PARAMETERS,
    )


class ReadJDTool(_ReadDocumentTool):
    label = "JD"
    spec = ToolSpec(
        name="read_jd",
        description=(
            "读取岗位 JD 文件并返回纯文本。支持 PDF、DOCX、Markdown、TXT。"
            "用户提供 JD 路径时先调用本工具，不要臆造 JD 内容。"
        ),
        parameters=_PATH_PARAMETERS,
    )


class AnalyzeResumeTool:
    """规则版简历结构分析：只陈述事实，不做主观评价。"""

    spec = ToolSpec(
        name="analyze_resume",
        description=(
            "对简历做确定性的结构分析，返回联系方式、章节完整性、"
            "量化指标、日期区间和字数等事实报告。"
            "不做主观评价；是否匹配 JD 由你结合原文判断。"
        ),
        parameters=_PATH_PARAMETERS,
    )

    async def run(self, **kwargs: Any) -> str:
        """读取简历并输出规则版结构分析。

        参数：
            **kwargs: 必须包含 path 字段，指向简历文件。

        返回值：
            文件名 + 结构分析报告的多行纯文本。

        异常：
            ToolError: 参数非法或简历读取 / 分析失败时抛出。
        """
        path = _require_path(kwargs)
        document = await asyncio.to_thread(load_document, path)
        return f"简历文件：{document.path}\n{analyze_resume_text(document.text)}"


def _require_path(arguments: dict[str, Any]) -> str:
    """校验并取出工具参数中的 path。

    参数：
        arguments: 模型传入的 JSON 参数（已解析成 dict）。

    返回值：
        去掉首尾空白后的文件路径。

    异常：
        ToolError: 出现未知参数、path 缺失或不是非空字符串时抛出。
    """
    extra = set(arguments) - {"path"}
    if extra:
        raise ToolError(f"不支持的参数：{', '.join(sorted(extra))}")
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ToolError("参数 path 必须是非空字符串")
    return path.strip()


# ---------------------------------------------------------------------------
# 规则版分析
# ---------------------------------------------------------------------------
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_QUANTIFIED = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|万|千|亿|人|倍|次|项|个|篇)")
_DATE_RANGE = re.compile(
    r"(?:19|20)\d{2}\s*(?:[./年-]\s*\d{1,2}\s*月?)?\s*"
    r"(?:-|–|—|~|～|至|到)\s*"
    r"(?:(?:19|20)\d{2}\s*(?:[./年-]\s*\d{1,2}\s*月?)?|至今|今|present|now)",
    re.IGNORECASE,
)

_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "教育": ("教育", "学历", "学校", "education"),
    "实习/工作": ("实习", "工作经历", "工作经验", "任职", "experience", "employment"),
    "项目": ("项目", "project"),
    "技能": ("技能", "技术栈", "skills"),
    "自我评价": ("自我评价", "个人评价", "个人总结", "summary", "about me"),
    "获奖": ("获奖", "荣誉", "奖项", "award"),
    "校园": ("校园", "社团", "学生会", "campus"),
}


def analyze_resume_text(text: str) -> str:
    """对简历正文做规则检查，返回紧凑的纯文本报告。

    参数：
        text: 从简历文档中提取的纯文本。

    返回值：
        包含联系方式、章节、量化指标、日期区间和提示的多行报告。

    异常：
        ToolError: 正文为空时抛出。
    """
    if not text.strip():
        raise ToolError("简历内容为空，无法分析")

    lowered = text.lower()
    sections = [
        (name, any(keyword in lowered for keyword in keywords))
        for name, keywords in _SECTION_KEYWORDS.items()
    ]
    section_map = dict(sections)

    has_email = bool(_EMAIL.search(text))
    has_phone = bool(_PHONE.search(text))
    quantified = len(_QUANTIFIED.findall(text))
    date_ranges = len(_DATE_RANGE.findall(text))
    chars = len(re.sub(r"\s", "", text))

    hints: list[str] = []
    if not section_map["教育"]:
        hints.append("未检测到教育经历")
    if not section_map["实习/工作"] and not section_map["项目"]:
        hints.append("未检测到实习/工作或项目经历")
    if not section_map["技能"]:
        hints.append("未检测到技能章节")
    if not section_map["自我评价"]:
        hints.append("未检测到自我评价 / 个人总结（可选）")
    if (section_map["实习/工作"] or section_map["项目"]) and date_ranges == 0:
        hints.append("经历中未检测到明确的日期区间")
    if quantified == 0:
        hints.append("未检测到量化指标（如 30%、10 万用户）")
    if chars < 300:
        hints.append("正文较短，信息可能不完整")

    lines = [
        "简历结构分析（规则版，仅陈述可检测到的事实，不做主观评价）",
        f"- 联系方式：邮箱 {'✓' if has_email else '✗'} / 电话 {'✓' if has_phone else '✗'}",
        "- 章节："
        + " | ".join(f"{name} {'✓' if ok else '✗'}" for name, ok in sections),
        f"- 量化指标：{quantified} 处",
        f"- 日期区间：{date_ranges} 段",
        f"- 正文字数（不含空白）：{chars}",
    ]
    if hints:
        lines.append("- 提示：")
        lines.extend(f"  - {hint}" for hint in hints)
    else:
        lines.append("- 提示：未发现明显的结构缺失（规则检查）")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 注册表与分发器
# ---------------------------------------------------------------------------
class ToolRegistry:
    """工具名 -> 工具实例。名称重复时构造阶段就失败。"""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        """创建注册表并建立工具名到实例的映射。

        参数：
            tools: 初始工具序列；按传入顺序保留。

        异常：
            ValueError: 两个工具使用同一个 spec.name 时抛出。
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            name = tool.spec.name
            if name in self._tools:
                raise ValueError(f"重复的工具名：{name}")
            self._tools[name] = tool

    def __len__(self) -> int:
        """返回已注册工具的数量。

        返回值：
            工具数量。
        """
        return len(self._tools)

    @property
    def names(self) -> tuple[str, ...]:
        """按注册顺序返回所有工具名。

        返回值：
            工具名组成的 tuple。
        """
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        """返回所有工具的声明，用于传给模型。

        返回值：
            ToolSpec 组成的 tuple。
        """
        return tuple(tool.spec for tool in self._tools.values())

    def get(self, name: str) -> Tool | None:
        """按名称查找工具。

        参数：
            name: 工具名。

        返回值：
            对应的 Tool；未注册时返回 None。
        """
        return self._tools.get(name)


def create_default_registry() -> ToolRegistry:
    """创建 s02 默认工具集。

    返回值：
        包含 read_resume、read_jd、analyze_resume 三个工具的注册表。
    """
    return ToolRegistry([ReadResumeTool(), ReadJDTool(), AnalyzeResumeTool()])


class ToolDispatcher:
    """把一次 ToolCall 转换成一条 tool 消息。"""

    def __init__(self, registry: ToolRegistry) -> None:
        """绑定一个工具注册表。

        参数：
            registry: 分发器执行工具时使用的注册表。
        """
        self._registry = registry

    async def dispatch(self, call: ToolCall) -> Message:
        """把一次工具调用转换成一条 tool 消息。

        参数：
            call: 模型返回的工具调用，包含 id、名称和原始参数。

        返回值：
            role="tool" 的 Message；失败时 content 是错误说明。
        """
        content = await self._execute(call)
        return Message.tool(call.id, content)

    async def _execute(self, call: ToolCall) -> str:
        """执行工具并把所有失败转换成可回填的文本。

        参数：
            call: 模型返回的工具调用。

        返回值：
            工具结果或错误说明的纯文本；不会向调用方抛异常。
        """
        tool = self._registry.get(call.name)
        if tool is None:
            available = ", ".join(self._registry.names) or "(无)"
            logger.warning("未知工具 name=%s", call.name)
            return f"工具执行失败：未知工具 {call.name!r}。可用工具：{available}"

        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as exc:
            logger.warning("工具参数不是合法 JSON name=%s", call.name)
            return f"工具执行失败：参数不是合法 JSON（{exc.msg}）"
        if not isinstance(arguments, dict):
            return "工具执行失败：参数必须是 JSON 对象"

        try:
            result = await tool.run(**arguments)
        except ToolError as exc:
            logger.warning("工具失败 name=%s error=%s", call.name, exc)
            return f"工具执行失败：{exc}"
        except Exception as exc:  # noqa: BLE001 - 工具异常不能打断 Agent 循环
            logger.exception("工具异常 name=%s", call.name)
            return f"工具执行失败：{type(exc).__name__}: {exc}"

        if not isinstance(result, str):
            return f"工具执行失败：{call.name} 返回了非文本结果"

        # 只记录名称和长度：简历 / JD 正文属于隐私，不进日志。
        logger.info("工具完成 name=%s chars=%d", call.name, len(result))
        return result
