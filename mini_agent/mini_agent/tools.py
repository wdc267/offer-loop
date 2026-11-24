"""工具定义、注册与分发（s03）。

分工：
- Tool          只做确定性工作（读文件、结构统计、写文件），不调用模型；
- ToolRegistry  管理工具名 -> 实例，并向模型暴露 ToolSpec；
- ToolDispatcher 解析模型参数、通过三道权限门、执行工具、
                 把结果 / 错误转换成 tool 消息。

工具层失败不会抛出到 Agent 循环：Dispatcher 统一转成一条 tool 消息，
让模型自己决定重试、换文件还是如实告知用户。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from .documents import MAX_TEXT_CHARS, SUPPORTED_SUFFIXES, load_document
from .errors import ToolError
from .models import Message, ToolCall, ToolSpec
from .permissions import Approver, PermissionRequest, check_permission

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

_UPDATE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "原简历文件的绝对路径，或相对于当前工作目录的路径",
        },
        "content": {
            "type": "string",
            "description": "修改后的完整简历正文（不是 diff），纯文本或 Markdown",
        },
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

# 可以原地覆盖的文本格式；PDF / DOCX 一律生成 _new 新文件。
_TEXT_WRITE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


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
# 五个业务工具
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
            "需要简历原文时使用；不要臆造简历内容。"
        ),
        parameters=_PATH_PARAMETERS,
    )


class ReadJDTool(_ReadDocumentTool):
    label = "JD"
    spec = ToolSpec(
        name="read_jd",
        description=(
            "读取岗位 JD 文件并返回纯文本。支持 PDF、DOCX、Markdown、TXT。"
            "需要 JD 原文时使用；不要臆造 JD 内容。"
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
            "需要这些结构事实时使用；不做主观评价，是否匹配 JD 由你结合原文判断。"
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


@dataclass(frozen=True, slots=True)
class _UpdatePlan:
    """一次简历写入的源文件、目标文件与正文。

    属性：
        source: 原始简历文件。
        target: 实际写入路径；原地替换时与 source 相同。
        content: 已归一化的完整正文。
        in_place: True 表示覆盖原文件（写入前备份）。
    """

    source: Path
    target: Path
    content: str
    in_place: bool


class UpdateResumeTool:
    """把模型改写后的完整简历写入文件。

    文本格式原地替换并备份；PDF / DOCX 生成 ``_new`` 新文件，原文件保留。
    """

    spec = ToolSpec(
        name="update_resume",
        description=(
            "把修改后的完整简历写入文件。content 必须是完整正文，不能只给 diff。"
            ".md/.markdown/.txt 原地更新并自动生成 .bak 备份；"
            ".pdf 生成 _new.md，.docx 生成 _new.docx，原文件保持不变。"
            "写入前会请求用户确认。"
        ),
        parameters=_UPDATE_PARAMETERS,
    )

    async def permission_request(self, arguments: dict[str, Any]) -> PermissionRequest:
        """生成修改操作的确认请求。

        参数：
            arguments: 模型传入的 JSON 参数，包含 path 和 content。

        返回值：
            包含目标路径、写入模式和正文预览的 PermissionRequest。

        异常：
            ToolError: 参数非法、源文件不存在或目标新文件已存在时抛出。
        """
        plan = await asyncio.to_thread(_plan_update, arguments)
        size = plan.source.stat().st_size
        mode = (
            "原地替换（写入前创建 .bak 备份）"
            if plan.in_place
            else "生成 _new 新文件（原文件保留）"
        )
        summary = (
            f"模式：{mode}\n"
            f"源文件：{plan.source}（{size} 字节）\n"
            f"新正文：{len(plan.content)} 字符\n"
            f"预览（前 8 行）：\n{_preview(plan.content)}"
        )
        return PermissionRequest(
            tool_name=self.spec.name,
            action="modify",
            path=str(plan.target),
            summary=summary,
        )

    async def run(self, **kwargs: Any) -> str:
        """执行写入并返回变更说明。

        参数：
            **kwargs: 必须包含 path（源文件）和 content（完整新正文）。

        返回值：
            写入目标、备份路径（原地替换时）和正文字符数。

        异常：
            ToolError: 参数非法、写入失败或目标新文件已存在时抛出。
        """
        plan = await asyncio.to_thread(_plan_update, kwargs)
        size = plan.source.stat().st_size

        if plan.in_place:
            backup = await asyncio.to_thread(_backup_file, plan.source)
            await asyncio.to_thread(_write_text, plan.target, plan.content)
            return (
                f"已更新简历：{plan.target}\n"
                f"备份：{backup}\n"
                f"原文件 {size} 字节，新正文 {len(plan.content)} 字符"
            )

        if plan.target.suffix.lower() == ".docx":
            await asyncio.to_thread(_write_docx, plan.target, plan.content)
        else:
            await asyncio.to_thread(_write_text, plan.target, plan.content)
        return (
            f"已生成新简历：{plan.target}\n"
            f"原文件未修改：{plan.source}（{size} 字节）\n"
            f"新正文 {len(plan.content)} 字符"
        )


class DeleteResumeTool:
    """永久删除受支持的简历 / JD 文件。"""

    spec = ToolSpec(
        name="delete_resume",
        description=(
            "永久删除简历文件，删除后无法恢复。支持 PDF、DOCX、Markdown、TXT。"
            "执行前会请求用户确认。"
        ),
        parameters=_PATH_PARAMETERS,
    )

    async def permission_request(self, arguments: dict[str, Any]) -> PermissionRequest:
        """生成永久删除操作的确认请求。

        参数：
            arguments: 模型传入的 JSON 参数，只包含 path。

        返回值：
            包含文件大小和不可恢复提示的 PermissionRequest。

        异常：
            ToolError: 参数非法、文件不存在或后缀不支持时抛出。
        """
        path = await asyncio.to_thread(_resolve_document_path, _require_path(arguments))
        size = path.stat().st_size
        summary = f"文件大小：{size} 字节\n此操作将永久删除文件，无法恢复。"
        return PermissionRequest(
            tool_name=self.spec.name,
            action="delete",
            path=str(path),
            summary=summary,
        )

    async def run(self, **kwargs: Any) -> str:
        """永久删除目标文件。

        参数：
            **kwargs: 必须包含 path，指向受支持的文档文件。

        返回值：
            确认删除的纯文本。

        异常：
            ToolError: 参数非法、文件不存在或删除失败时抛出。
        """
        raw = _require_path(kwargs)
        path = await asyncio.to_thread(_resolve_document_path, raw)
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError as exc:
            raise ToolError(f"文件不存在：{raw}") from exc
        except OSError as exc:
            raise ToolError(f"删除失败：{exc}") from exc
        return f"已永久删除：{path}"


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
# 写入与删除辅助
# ---------------------------------------------------------------------------
def _resolve_document_path(raw: str) -> Path:
    """把路径解析成已存在的受支持文档。

    参数：
        raw: 用户或模型给出的文件路径。

    返回值：
        解析后的绝对路径。

    异常：
        ToolError: 文件不存在、路径不是文件或后缀不支持时抛出。
    """
    path = Path(raw).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ToolError(f"文件不存在：{raw}") from exc
    if not resolved.is_file():
        raise ToolError(f"路径不是文件：{raw}")
    if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = "、".join(sorted(SUPPORTED_SUFFIXES))
        suffix = resolved.suffix or "(无后缀)"
        raise ToolError(f"不支持的文件类型：{suffix}，支持：{allowed}")
    return resolved


def _plan_update(arguments: dict[str, Any]) -> _UpdatePlan:
    """校验 update_resume 参数并计算实际写入目标。

    参数：
        arguments: 模型传入的 JSON 参数，包含 path 和 content。

    返回值：
        源文件、目标文件、归一化正文和是否原地替换。

    异常：
        ToolError: 参数非法、源文件不存在或目标新文件已存在时抛出。
    """
    extra = set(arguments) - {"path", "content"}
    if extra:
        raise ToolError(f"不支持的参数：{', '.join(sorted(extra))}")

    raw_path = arguments.get("path")
    content = arguments.get("content")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ToolError("参数 path 必须是非空字符串")
    if not isinstance(content, str) or not content.strip():
        raise ToolError("参数 content 必须是非空字符串")
    if len(content) > MAX_TEXT_CHARS:
        raise ToolError(f"简历正文过长：{len(content)} 字符，上限 {MAX_TEXT_CHARS}")

    source = _resolve_document_path(raw_path)
    normalized = _normalize_text(content)
    if source.suffix.lower() in _TEXT_WRITE_SUFFIXES:
        return _UpdatePlan(
            source=source,
            target=source,
            content=normalized,
            in_place=True,
        )

    # PDF 不能直接重写，DOCX 重写会丢失原格式：统一生成新文件。
    new_suffix = ".docx" if source.suffix.lower() == ".docx" else ".md"
    target = source.with_name(f"{source.stem}_new{new_suffix}")
    if target.exists():
        raise ToolError(f"目标文件已存在：{target}")
    return _UpdatePlan(
        source=source,
        target=target,
        content=normalized,
        in_place=False,
    )


def _normalize_text(content: str) -> str:
    """统一换行并保证正文以单个换行结尾。

    参数：
        content: 模型给出的完整正文。

    返回值：
        使用 ``\\n`` 换行、末尾恰好一个换行的正文。
    """
    text = content.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return text + "\n"


def _preview(content: str, max_lines: int = 8) -> str:
    """截取正文前若干非空行，用于终端确认提示。

    参数：
        content: 待预览正文。
        max_lines: 最多显示的行数。

    返回值：
        缩进后的预览文本；过滤控制字符，超出部分以省略号结尾。
    """
    lines = []
    for line in content.splitlines():
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", line).strip()
        if cleaned:
            lines.append(cleaned[:80])
    shown = lines[:max_lines]
    rendered = "\n".join(f"  {line}" for line in shown)
    if len(lines) > max_lines:
        rendered += "\n  …"
    return rendered or "  （空）"


def _backup_file(source: Path) -> Path:
    """复制原文件到带时间戳的 .bak 路径。

    参数：
        source: 要备份的原文件。

    返回值：
        新创建的备份文件路径。

    异常：
        OSError: 复制失败时向上抛出。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = source.with_name(f"{source.name}.bak-{stamp}")
    counter = 2
    while candidate.exists():
        candidate = source.with_name(f"{source.name}.bak-{stamp}-{counter}")
        counter += 1
    shutil.copy2(source, candidate)
    return candidate


def _write_text(target: Path, content: str) -> None:
    """以 UTF-8 原子写入文本文件。

    参数：
        target: 目标路径。
        content: 已归一化的正文。

    异常：
        OSError: 写入或替换失败时向上抛出。
    """
    fd, temp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_docx(target: Path, content: str) -> None:
    """把纯文本按段落写入新的 DOCX 文件。

    参数：
        target: 目标 .docx 路径。
        content: 已归一化的正文。

    异常：
        OSError: 写入或替换失败时向上抛出。
        ImportError: 未安装 python-docx 时抛出。
    """
    from docx import Document as DocxDocument

    document = DocxDocument()
    for line in content.splitlines():
        document.add_paragraph(line)

    fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".docx")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        document.save(temp_path)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


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
    """创建 s03 默认工具集。

    返回值：
        包含读取、分析、修改、删除五个工具的注册表。
    """
    return ToolRegistry(
        [
            ReadResumeTool(),
            ReadJDTool(),
            AnalyzeResumeTool(),
            UpdateResumeTool(),
            DeleteResumeTool(),
        ]
    )


class ToolDispatcher:
    """把一次 ToolCall 转换成一条 tool 消息。"""

    def __init__(
        self,
        registry: ToolRegistry,
        approver: Approver | None = None,
    ) -> None:
        """绑定工具注册表与用户确认实现。

        参数：
            registry: 分发器执行工具时使用的注册表。
            approver: 规则命中后的确认函数；None 时使用交互式 ask_user。
        """
        self._registry = registry
        self._approver = approver

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

        # 1. 工具生成用户可读的安全摘要（目标路径、模式、大小、预览）。
        request: PermissionRequest | None = None
        describe = getattr(tool, "permission_request", None)
        if describe is not None:
            try:
                request = await describe(arguments)
            except ToolError as exc:
                logger.warning("权限摘要失败 name=%s error=%s", call.name, exc)
                return f"工具执行失败：{exc}"
            except Exception as exc:  # noqa: BLE001 - 摘要失败不能打断循环
                logger.exception("权限摘要异常 name=%s", call.name)
                return f"工具执行失败：{type(exc).__name__}: {exc}"

            if not isinstance(request, PermissionRequest):
                return f"工具执行失败：{call.name} 返回了无效的权限摘要"

        # 2. 拒绝表 -> 确认规则 -> 用户确认；任一门拒绝都不执行。
        try:
            denial = await check_permission(
                call.name,
                arguments,
                request,
                approver=self._approver,
            )
        except Exception:  # noqa: BLE001 - 权限检查失败按拒绝处理
            logger.exception("权限检查异常 name=%s", call.name)
            return "操作未执行：权限检查过程出错"

        if denial is not None:
            # 只记录工具名和拒绝原因；不记录工具参数正文。
            logger.info("工具调用被拒绝 name=%s reason=%s", call.name, denial)
            return denial

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
