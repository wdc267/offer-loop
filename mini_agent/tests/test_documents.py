"""文档加载测试：全部离线，文件在 tmp_path 中现场生成。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.documents import load_document
from mini_agent.errors import DocumentError


def _write_pdf(path: Path, text: str) -> None:
    """生成一个只含一行文本的最小 PDF，避免测试依赖 PDF 生成库。

    参数：
        path: 输出 PDF 文件路径。
        text: 写入 PDF 的 ASCII 文本。
    """
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_at = len(payload)
    payload += f"xref\n0 {len(objects) + 1}\n".encode()
    payload += b"0000000000 65535 f \n"
    for offset in offsets:
        payload += f"{offset:010d} 00000 n \n".encode()
    payload += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(payload))


def test_reads_plain_text_and_normalizes(tmp_path):
    """验证 TXT 能按 UTF-8 读取，并统一换行、压缩多余空行。"""
    path = tmp_path / "resume.txt"
    path.write_text("张三\r\n\r\n\r\n后端开发", encoding="utf-8")

    document = load_document(path)

    assert document.kind == "text"
    assert document.text == "张三\n\n后端开发"
    assert document.original_chars == len(document.text)
    assert not document.truncated


def test_reads_markdown_and_strips_bom(tmp_path):
    """验证 Markdown 正常读取，且 UTF-8 BOM 不会残留到正文。"""
    path = tmp_path / "resume.md"
    path.write_text("# 张三\n\n技能：Python", encoding="utf-8-sig")

    document = load_document(path)

    assert document.kind == "markdown"
    assert document.text.startswith("# 张三")


def test_falls_back_to_gb18030(tmp_path):
    """验证 UTF-8 解码失败时会回退到 gb18030，兼容 Windows 中文文本。"""
    path = tmp_path / "resume.txt"
    path.write_bytes("姓名：张三".encode("gb18030"))

    document = load_document(path)

    assert document.text == "姓名：张三"


def test_reads_docx_paragraphs_and_tables(tmp_path):
    """验证 DOCX 的段落和表格内容都会被提取。"""
    from docx import Document as DocxDocument

    path = tmp_path / "resume.docx"
    docx = DocxDocument()
    docx.add_paragraph("张三")
    docx.add_paragraph("邮箱：zhang@example.com")
    table = docx.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "项目A"
    table.cell(0, 1).text = "提升 30%"
    docx.save(path)

    document = load_document(path)

    assert document.kind == "docx"
    assert "张三" in document.text
    assert "zhang@example.com" in document.text
    assert "项目A | 提升 30%" in document.text


def test_reads_pdf_text(tmp_path):
    """验证带文本层的 PDF 能提取出正文。"""
    path = tmp_path / "resume.pdf"
    _write_pdf(path, "Hello Resume from mini_agent")

    document = load_document(path)

    assert document.kind == "pdf"
    assert "Hello Resume from mini_agent" in document.text


def test_pdf_without_text_reports_scanned_file(tmp_path):
    """验证没有文本层的 PDF 被判定为扫描件并给出明确错误。"""
    from pypdf import PdfWriter

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(DocumentError, match="扫描件"):
        load_document(path)


def test_legacy_doc_has_clear_error(tmp_path):
    """验证旧版 .doc 会提示另存为 .docx，而不是抛出底层解析错误。"""
    path = tmp_path / "resume.doc"
    path.write_bytes(b"old word")

    with pytest.raises(DocumentError, match="docx"):
        load_document(path)


def test_unsupported_suffix_rejected(tmp_path):
    """验证不在白名单内的扩展名会被拒绝。"""
    path = tmp_path / "resume.exe"
    path.write_bytes(b"MZ")

    with pytest.raises(DocumentError, match="不支持的文件类型"):
        load_document(path)


def test_missing_file(tmp_path):
    """验证文件不存在时返回清晰的错误信息。"""
    with pytest.raises(DocumentError, match="文件不存在"):
        load_document(tmp_path / "nope.txt")


def test_directory_rejected(tmp_path):
    """验证传入目录而不是文件时会被拒绝。"""
    with pytest.raises(DocumentError, match="路径不是文件"):
        load_document(tmp_path)


def test_empty_path_rejected():
    """验证空白路径会被拒绝。"""
    with pytest.raises(DocumentError, match="不能为空"):
        load_document("   ")


def test_empty_file_rejected(tmp_path):
    """验证只有空白的文件在规整后视为没有文本并报错。"""
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(DocumentError, match="提取到文本"):
        load_document(path)


def test_file_size_limit(tmp_path, monkeypatch):
    """验证超过大小上限的文件会被拒绝。"""
    monkeypatch.setattr("mini_agent.documents.MAX_FILE_BYTES", 4)
    path = tmp_path / "big.txt"
    path.write_text("12345", encoding="utf-8")

    with pytest.raises(DocumentError, match="文件过大"):
        load_document(path)


def test_long_text_is_truncated(tmp_path, monkeypatch):
    """验证超长正文会被截断，并记录原始字符数和截断标记。"""
    monkeypatch.setattr("mini_agent.documents.MAX_TEXT_CHARS", 5)
    path = tmp_path / "long.txt"
    path.write_text("一二三四五六七八九十", encoding="utf-8")

    document = load_document(path)

    assert document.text == "一二三四五"
    assert document.original_chars == 10
    assert document.truncated
    assert "已截断" in document.to_model_text("简历")


def test_to_model_text_contains_metadata(tmp_path):
    """验证给模型看的文本包含标签、文件名、类型、字符数和正文。"""
    path = tmp_path / "resume.txt"
    path.write_text("内容", encoding="utf-8")

    rendered = load_document(path).to_model_text("简历")

    assert rendered.startswith("简历文件：")
    assert "text，2 字符" in rendered
    assert rendered.endswith("内容")
