"""文档读取：把简历 / JD 文件统一转换成纯文本。

工具层只关心「给我文本」，格式差异全部收敛在这里：
- PDF   -> pypdf
- DOCX  -> python-docx
- MD/TXT -> 编码回退读取

所有失败都抛 DocumentError（ToolError 的子类），由 ToolDispatcher
转换成 tool 消息，不打断 Agent 循环。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import DocumentError

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".md", ".markdown", ".txt"})
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 50_000
# PDF 提取不到足够文本时，大概率是扫描件（图片），需要 OCR。
MIN_PDF_TEXT_CHARS = 20
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")

_KIND_BY_SUFFIX = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}


@dataclass(frozen=True, slots=True)
class Document:
    """一份已经转成纯文本的文档。"""

    path: Path
    kind: str
    text: str
    original_chars: int
    truncated: bool

    def to_model_text(self, label: str = "文档") -> str:
        """渲染成给模型看的文本：一行元信息 + 正文。

        参数：
            label: 文档类型标签，例如「简历」或「JD」。

        返回值：
            可直接作为 tool 消息内容的纯文本。
        """
        meta = f"{self.kind}，{self.original_chars} 字符"
        if self.truncated:
            meta += f"，已截断到 {len(self.text)} 字符"
        return f"{label}文件：{self.path}（{meta}）\n---\n{self.text}"


def load_document(path: str | Path) -> Document:
    """按扩展名读取文件并返回规范化后的纯文本。

    参数：
        path: 文件路径，支持绝对路径或相对于当前工作目录的路径。

    返回值：
        包含正文、文件类型、原始字符数和截断标记的 Document。

    异常：
        DocumentError: 路径为空、文件不存在、类型不支持、
            文件过大、解析失败或没有可提取文本时抛出。
    """
    raw = str(path).strip()
    if not raw:
        raise DocumentError("文件路径不能为空")

    resolved = Path(raw).expanduser().resolve()
    if not resolved.exists():
        raise DocumentError(f"文件不存在：{resolved}")
    if not resolved.is_file():
        raise DocumentError(f"路径不是文件：{resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".doc":
        raise DocumentError("暂不支持旧版 .doc，请另存为 .docx 后重试")
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(
            f"不支持的文件类型 {suffix or '(无扩展名)'}，"
            f"支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    size = resolved.stat().st_size
    if size > MAX_FILE_BYTES:
        raise DocumentError(
            f"文件过大（{size / 1024 / 1024:.1f} MB），上限 "
            f"{MAX_FILE_BYTES / 1024 / 1024:.0f} MB"
        )

    kind = _KIND_BY_SUFFIX[suffix]
    if suffix == ".pdf":
        text = _read_pdf(resolved)
    elif suffix == ".docx":
        text = _read_docx(resolved)
    else:
        text = _read_plain(resolved)

    text = _normalize(text)
    if not text:
        raise DocumentError("未能从文件中提取到文本内容")

    original_chars = len(text)
    truncated = original_chars > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]

    return Document(
        path=resolved,
        kind=kind,
        text=text,
        original_chars=original_chars,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# 各格式的读取实现
# ---------------------------------------------------------------------------
def _read_pdf(path: Path) -> str:
    """用 pypdf 提取 PDF 每一页的文本。

    参数：
        path: PDF 文件路径。

    返回值：
        按页拼接的纯文本。

    异常：
        DocumentError: PDF 加密、解析失败或没有文本层（疑似扫描件）时抛出。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
        raise DocumentError("读取 PDF 需要 pypdf，请先安装") from exc

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            # 空密码可以打开「只有权限密码」的 PDF；真加密会在下面失败。
            if reader.decrypt("") == 0:
                raise DocumentError("PDF 已加密，无法读取")
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf 异常类型不稳定
        raise DocumentError(f"PDF 解析失败：{exc}") from exc

    text = "\n".join(pages)
    if len(text.strip()) < MIN_PDF_TEXT_CHARS:
        raise DocumentError("无法从 PDF 提取文本（可能是扫描件，暂不支持 OCR）")
    return text


def _read_docx(path: Path) -> str:
    """用 python-docx 提取 DOCX 的段落、表格和页眉页脚。

    参数：
        path: DOCX 文件路径。

    返回值：
        各文本块以换行拼接的纯文本。

    异常：
        DocumentError: python-docx 解析失败时抛出。
    """
    try:
        from docx import Document as DocxReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
        raise DocumentError("读取 DOCX 需要 python-docx，请先安装") from exc

    try:
        document = DocxReader(str(path))
    except Exception as exc:  # noqa: BLE001 - python-docx 异常类型不稳定
        raise DocumentError(f"DOCX 解析失败：{exc}") from exc

    parts: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)

    # 简历经常用表格排版；表格内容不在 paragraphs 里。
    for table in document.tables:
        for row in table.rows:
            cells: list[str] = []
            for cell in row.cells:
                value = cell.text.strip()
                # 合并单元格会让同一个值重复出现，去掉相邻重复。
                if value and (not cells or cells[-1] != value):
                    cells.append(value)
            if cells:
                parts.append(" | ".join(cells))

    # 联系方式有时写在页眉 / 页脚。
    for section in document.sections:
        for container in (section.header, section.footer):
            for paragraph in container.paragraphs:
                if paragraph.text.strip():
                    parts.append(paragraph.text)

    return "\n".join(parts)


def _read_plain(path: Path) -> str:
    """按 UTF-8 → gb18030 的顺序尝试解码纯文本文件。

    参数：
        path: TXT 或 Markdown 文件路径。

    返回值：
        解码后的文本。

    异常：
        DocumentError: 所有候选编码都无法解码时抛出。
    """
    data = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentError("无法识别文本编码（已尝试 utf-8 / gb18030）")


def _normalize(text: str) -> str:
    """统一换行、去掉行尾空格和多余空行。

    参数：
        text: 原始提取文本。

    返回值：
        清理后的文本；连续 3 个以上换行会压缩成 1 个空行。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
