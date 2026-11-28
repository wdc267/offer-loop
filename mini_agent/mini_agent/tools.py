"""工作区内的四个简历工具，不依赖模型或 Hook。"""

import re
from pathlib import Path

RESUME_SUFFIXES = {".md", ".txt", ".pdf"}
SECTIONS = ("教育经历", "工作经历", "实习经历", "项目经历", "专业技能")


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string"} for field in fields},
                "required": fields,
                "additionalProperties": False,
            },
        },
    }
    for name, description, fields in [
        ("read_resume", "读取工作区内的 .md、.txt 或有文本层的 .pdf 简历。path 是相对路径。", ["path"]),
        ("analyze_resume", "本地检查简历章节、联系方式和量化表述。path 是相对路径。", ["path"]),
        ("update_resume", "用完整的 content 更新简历；PDF 生成 _new.md。需要用户确认。", ["path", "content"]),
        ("delete_resume", "永久删除相对路径 path 指定的简历。需要用户确认。", ["path"]),
    ]
]

TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "read_tool_result",
        "description": "按 result_id 分页读取当前会话的归档工具结果，不接受文件路径。offset 按字符计。",
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8000, "default": 4000},
            },
            "required": ["result_id"],
            "additionalProperties": False,
        },
    },
})


def workspace_path(workspace, path):
    """解析文件路径，并确保它指向工作区下的文件位置。

    参数：
        workspace: 工作区根目录。
        path: 模型给出的非空相对路径。

    返回值：
        解析了符号链接和相对路径段的绝对路径。

    异常：
        ValueError: 路径非法，或解析后不在工作区内。
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path 必须是非空字符串")
    relative = Path(path)
    if relative.anchor:
        raise ValueError("只允许工作区内的相对路径")
    if any(":" in part or Path(part).is_reserved() for part in relative.parts):
        raise ValueError("不允许使用设备名或带冒号的路径")
    root = workspace.resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("路径位于工作区外或未指向文件")
    return target


def resume_path(workspace, path):
    """校验相对路径是否指向支持的简历文件。

    参数：
        workspace: 工作区根目录。
        path: 简历的相对路径。

    返回值：
        简历文件的绝对路径。

    异常：
        ValueError: 路径越界、格式不支持或文件不存在。
    """
    target = workspace_path(workspace, path)
    if target.suffix.lower() not in RESUME_SUFFIXES:
        raise ValueError("只支持 .md、.txt、.pdf 简历")
    if not target.is_file():
        raise ValueError("简历文件不存在或不是普通文件")
    return target


def make_tools(workspace, *, artifacts=None):
    """创建绑定到指定工作区的工具映射。

    参数：
        workspace: 简历工具可访问的根目录（Path）。
        artifacts: 当前会话的归档存储；未提供时不装配分页读取工具。

    返回值：
        工具名到文件读写函数的映射。
    """

    def read_resume(path):
        """读取 UTF-8 文本简历或提取 PDF 的文本层。

        参数：
            path: 简历的相对路径。

        返回值：
            简历正文。

        异常：
            ValueError: 路径非法，或 PDF 不可解析、没有文本层或缺少依赖。
            OSError: 文件无法读取。
        """
        source = resume_path(workspace, path)
        if source.suffix.lower() != ".pdf":
            return source.read_text(encoding="utf-8-sig")
        try:
            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
        except ImportError as exc:
            raise ValueError('读取 PDF 需要安装可选依赖：python -m pip install -e ".[pdf]"') from exc
        try:
            with source.open("rb") as stream:
                text = "\n".join(page.extract_text() or "" for page in PdfReader(stream).pages)
        except PyPdfError as exc:
            raise ValueError("PDF 无法解析") from exc
        if not text.strip():
            raise ValueError("PDF 没有可提取的文本，不支持 OCR")
        return text

    def analyze_resume(path):
        """在本地检查章节、联系方式和量化表述，不调用模型。

        参数：
            path: 简历的相对路径。

        返回值：
            简单的规则检查摘要，不包含联系方式原文。

        异常：
            ValueError: 简历路径或文档格式不合法。
            OSError: 文件无法读取。
        """
        text = read_resume(path)
        sections = "、".join(section for section in SECTIONS if section in text) or "未发现"
        email = bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))
        phone = bool(re.search(r"(?<!\d)1[3-9]\d{9}(?!\d)", text))
        metrics = re.findall(r"\d+(?:\.\d+)?\s*(?:%|％|倍|万|小时|毫秒)", text)
        return (
            f"章节：{sections}\n邮箱：{'有' if email else '未发现'}"
            f"\n手机：{'有' if phone else '未发现'}\n量化表述：{len(metrics)}"
        )

    def update_resume(path, content):
        """覆盖文本简历，或从 PDF 派生新 Markdown 而不覆盖任何文件。

        参数：
            path: 原简历的相对路径。
            content: 完整的新简历正文。

        返回值：
            更新或生成文件的说明。

        异常：
            ValueError: 路径非法或正文为空。
            OSError: 写入失败或 PDF 的输出文件已存在。
        """
        source = resume_path(workspace, path)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须是非空正文")
        if source.suffix.lower() == ".pdf":
            target = source.with_name(f"{source.stem}_new.md")
            try:
                with target.open("x", encoding="utf-8") as stream:
                    stream.write(content)
            except FileExistsError as exc:
                raise ValueError(f"{target.name} 已存在，不会覆盖") from exc
            return f"已生成 {target.relative_to(workspace.resolve())}，原 PDF 保持不变。"
        source.write_text(content, encoding="utf-8")
        return f"已更新 {source.relative_to(workspace.resolve())}。"

    def delete_resume(path):
        """删除工作区内的简历文件；用户审批由 PreToolUse 负责。

        参数：
            path: 简历的相对路径。

        返回值：
            删除结果。

        异常：
            ValueError: 路径非法或不是简历文件。
            OSError: 删除失败。
        """
        source = resume_path(workspace, path)
        source.unlink()
        return f"已删除 {source.relative_to(workspace.resolve())}。"

    tools = {
        "read_resume": read_resume,
        "analyze_resume": analyze_resume,
        "update_resume": update_resume,
        "delete_resume": delete_resume,
    }
    if artifacts is not None:
        tools["read_tool_result"] = artifacts.read_tool_result
    return tools
