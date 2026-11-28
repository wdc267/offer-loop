"""验证会话归档的精确存储、分页读取及路径边界。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

from mini_agent import artifacts
from mini_agent.artifacts import ArtifactStore


def test_directories_are_created_lazily_and_output_is_exact(tmp_path):
    """验证构造时不创建目录，写入时按 UTF-8 精确保留原文且不创建快照目录。"""
    root = tmp_path / "session"
    store = ArtifactStore(root)
    assert store.root == root
    assert not root.exists()
    content = "中文😀\r\n第二行\n末行\r"

    result_id = store.save_output(content, "read_resume")

    assert (root / "outputs" / f"{result_id}.txt").read_bytes() == content.encode("utf-8")
    assert not (root / "transcripts").exists()
    assert store.description(result_id) == {
        "result_id": result_id, "source": "read_resume", "total_chars": len(content),
    }
    metadata = store.description(result_id)
    metadata["source"] = "changed"
    assert store.description(result_id)["source"] == "read_resume"


@pytest.mark.parametrize(
    ("content", "offset", "limit", "expected", "next_offset"),
    [
        ("甲😀乙\r\n丙丁", 1, 3, "😀乙\r", 4),
        ("甲😀乙\r\n丙丁", 5, 20, "丙丁", None),
        ("甲😀乙\r\n丙丁", 7, 2, "", None),
        ("", 0, 4000, "", None),
    ],
)
def test_pagination_uses_characters_and_preserves_newlines(tmp_path, content, offset, limit, expected, next_offset):
    """验证中段、末段、EOF 和空文件分页按字符计数，且不改变换行。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output(content, "read_jd")
    assert store.validate_read(result_id, offset, limit) is None

    assert json.loads(store.read_tool_result(result_id, offset, limit)) == {
        "result_id": result_id,
        "source": "read_jd",
        "total_chars": len(content),
        "offset": offset,
        "end": offset + len(expected),
        "next_offset": next_offset,
        "content": expected,
    }


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (4, 1), (True, 1), (1.0, 1), ("1", 1), (None, 1),
     (0, 0), (0, -1), (0, 8001), (0, True), (0, 1.0), (0, "1"), (0, None)],
)
def test_invalid_pagination_is_rejected_in_both_entry_points(tmp_path, offset, limit):
    """验证非法分页参数在预校验和实际读取中都被拒绝。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output("abc", "tool")
    with pytest.raises(ValueError):
        store.validate_read(result_id, offset, limit)
    with pytest.raises(ValueError):
        store.read_tool_result(result_id, offset, limit)


def test_default_and_maximum_page_sizes(tmp_path):
    """验证默认读取 4000 字符，允许的最大单页为 8000 字符。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output("字" * 9000, "tool")
    assert len(json.loads(store.read_tool_result(result_id))["content"]) == 4000
    assert len(json.loads(store.read_tool_result(result_id, limit=8000))["content"]) == 8000


@pytest.mark.parametrize("result_id", ["../outside", "/absolute", "call-tool-123", "f" * 32, "", None, []])
def test_unknown_and_traversal_ids_are_rejected(tmp_path, result_id):
    """验证伪造 ID、工具调用 ID 和目录遍历不能读取或查询归档。"""
    store = ArtifactStore(tmp_path)
    for operation in (store.validate_read, store.read_tool_result, store.description):
        with pytest.raises(ValueError):
            operation(result_id)
    assert not (tmp_path / "outputs").exists()


def test_outputs_are_registered_only_in_the_creating_instance(tmp_path):
    """验证跨会话以及指向同一目录的新实例都不能使用另一实例登记的 ID。"""
    store = ArtifactStore(tmp_path / "first")
    result_id = store.save_output("secret", "tool")
    for other in (ArtifactStore(tmp_path / "second"), ArtifactStore(store.root)):
        with pytest.raises(ValueError):
            other.read_tool_result(result_id)


def test_missing_output_is_rejected_on_every_read(tmp_path, monkeypatch):
    """模拟已登记文件丢失，验证预校验及实际读取均报告缺失且不重建文件。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output("content", "tool")
    output = tmp_path / "outputs" / f"{result_id}.txt"
    original = Path.is_file

    def is_file(path):
        """模拟指定输出已丢失。

        参数：
            path: 待检查的路径。
        返回值：
            文件是否存在。
        """
        return False if path == output else original(path)

    monkeypatch.setattr(Path, "is_file", is_file)
    with pytest.raises(FileNotFoundError):
        store.validate_read(result_id)
    with pytest.raises(FileNotFoundError):
        store.read_tool_result(result_id)


@pytest.mark.parametrize("location", ["root", "parent", "outputs", "transcripts", "output_file", "snapshot_file"])
def test_symlinks_are_rejected_without_writing(tmp_path, monkeypatch, location):
    """模拟根、祖先、子目录和目标文件链接，验证拒绝归档且不登记输出。"""
    root = tmp_path / "session"
    store = ArtifactStore(root)
    fixed_id = "a" * 32
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    targets = {
        "root": root, "parent": tmp_path, "outputs": root / "outputs",
        "transcripts": root / "transcripts",
        "output_file": root / "outputs" / f"{fixed_id}.txt",
        "snapshot_file": root / "transcripts" / f"{fixed_id}.json",
    }
    original = Path.is_symlink

    def is_symlink(path):
        """模拟指定路径是符号链接。

        参数：
            path: 待检查的路径。
        返回值：
            是否为符号链接。
        """
        return path == targets[location] or original(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ValueError):
        if location in {"transcripts", "snapshot_file"}:
            store.save_snapshot([])
        else:
            store.save_output("secret", "tool")
    with pytest.raises(ValueError):
        store.description(fixed_id)
    assert not targets[location].is_file()


@pytest.mark.parametrize("location", ["root", "outputs", "output_file", "transcripts"])
def test_resolve_escape_is_rejected(tmp_path, monkeypatch, location):
    """模拟路径解析越界，验证即使链接检测未命中也不会写入外部目录。"""
    root = tmp_path / "session"
    store = ArtifactStore(root)
    fixed_id = "b" * 32
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    targets = {
        "root": root, "outputs": root / "outputs", "transcripts": root / "transcripts",
        "output_file": root / "outputs" / f"{fixed_id}.txt",
    }
    original = Path.resolve
    outside = tmp_path / "outside"

    def resolve(path, *args, **kwargs):
        """模拟指定路径解析到归档根目录之外。

        参数：
            path: 待解析的路径。
            *args: 原方法的位置参数。
            **kwargs: 原方法的关键字参数。
        返回值：
            模拟或实际解析后的路径。
        """
        return outside if path == targets[location] else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError):
        if location == "transcripts":
            store.save_snapshot([])
        else:
            store.save_output("secret", "tool")
    assert not outside.exists()


def test_read_rechecks_links_after_prevalidation(tmp_path, monkeypatch):
    """验证预校验通过后目标被替换为链接时，实际读取仍拒绝访问。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output("secret", "tool")
    store.validate_read(result_id)
    target = tmp_path / "outputs" / f"{result_id}.txt"
    original = Path.is_symlink

    def is_symlink(path):
        """模拟输出文件在预校验后变成符号链接。

        参数：
            path: 待检查路径。
        返回值：
            是否为符号链接。
        """
        return path == target or original(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    with pytest.raises(ValueError):
        store.read_tool_result(result_id)


def test_snapshot_preserves_messages_and_output_metadata_without_logging(tmp_path, capsys):
    """验证快照保留完整消息及输出来源元数据，不把正文打印到日志。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output("机密输出😀\r\n", "read_resume")
    messages = [
        {"role": "user", "content": "机密问题\r\n"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "../../unsafe", "type": "function", "function": {"name": "read_resume", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "../../unsafe", "content": "精确消息😀"},
    ]

    snapshot_id = store.save_snapshot(messages)

    assert snapshot_id != result_id
    snapshot = json.loads((tmp_path / "transcripts" / f"{snapshot_id}.json").read_text(encoding="utf-8"))
    assert snapshot == {"version": 1, "messages": messages, "outputs": {result_id: store.description(result_id)}}
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_empty_snapshot_creates_only_transcripts(tmp_path):
    """验证无输出时也能保存快照，且不提前创建 outputs 目录。"""
    store = ArtifactStore(tmp_path / "session")
    snapshot_id = store.save_snapshot([])
    assert json.loads((store.root / "transcripts" / f"{snapshot_id}.json").read_text(encoding="utf-8")) == {
        "version": 1, "messages": [], "outputs": {},
    }
    assert not (store.root / "outputs").exists()


@pytest.mark.parametrize("snapshot", [False, True])
def test_id_collision_never_overwrites_existing_file(tmp_path, monkeypatch, snapshot):
    """验证宿主生成的 ID 碰撞时独占创建失败，已有文件不变且不登记输出。"""
    fixed_id = "c" * 32
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    store = ArtifactStore(tmp_path)
    directory, suffix = ("transcripts", "json") if snapshot else ("outputs", "txt")
    target = tmp_path / directory / f"{fixed_id}.{suffix}"
    target.parent.mkdir()
    target.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        if snapshot:
            store.save_snapshot([])
        else:
            store.save_output("new", "tool")

    assert target.read_bytes() == b"existing"
    with pytest.raises(ValueError):
        store.description(fixed_id)


@pytest.mark.parametrize("failure", ["write", "close"])
def test_failed_output_write_is_not_registered(tmp_path, monkeypatch, failure):
    """模拟写入或关闭失败，验证失败 ID 不进入登记表且不删除任何文件。"""
    fixed_id = "d" * 32
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    store = ArtifactStore(tmp_path)
    mocked = mock_open()
    if failure == "write":
        mocked.return_value.write.side_effect = OSError("disk full")
    else:
        mocked.return_value.__exit__.side_effect = OSError("close failed")
    monkeypatch.setattr(Path, "open", mocked)

    with pytest.raises(OSError):
        store.save_output("content", "tool")
    with pytest.raises(ValueError):
        store.description(fixed_id)
    assert mocked.call_args.kwargs["mode"] == "x"
    assert mocked.call_args.kwargs["newline"] == ""


@pytest.mark.parametrize("location", ["root", "parent"])
@pytest.mark.parametrize("link_method", ["is_symlink", "is_junction"])
def test_constructor_rejects_existing_root_and_ancestor_links(tmp_path, monkeypatch, location, link_method):
    """验证构造阶段就拒绝根目录及祖先中的符号链接或 Windows 联接。"""
    root = tmp_path / "session"
    target = root if location == "root" else tmp_path
    original = getattr(Path, link_method)

    def is_link(path):
        """模拟预先存在的目录链接。

        参数：
            path: 待检查的路径。
        返回值：
            指定路径或原有路径是否为链接。
        """
        return path == target or original(path)

    monkeypatch.setattr(Path, link_method, is_link)
    with pytest.raises(ValueError):
        ArtifactStore(root)
    assert not root.exists()


@pytest.mark.parametrize("snapshot", [False, True])
def test_partial_failure_keeps_file_and_does_not_register(tmp_path, monkeypatch, snapshot):
    """验证实际产生的部分文件在失败后保留，ID 不登记且后续碰撞不会覆盖残留。"""
    fixed_id = "e" * 32
    monkeypatch.setattr(artifacts.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed_id))
    store = ArtifactStore(tmp_path)
    original = Path.open

    def partial_open(path, *args, **kwargs):
        """在独占创建后模拟磁盘写入失败并留下部分文件。

        参数：
            path: 待写入的文件路径。
            *args: 文件打开的位置参数。
            **kwargs: 文件打开的关键字参数。
        异常：
            OSError: 模拟的磁盘写入失败。
        """
        with original(path, *args, **kwargs) as stream:
            stream.write("partial")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", partial_open)
    with pytest.raises(OSError):
        if snapshot:
            store.save_snapshot([])
        else:
            store.save_output("original", "tool")
    monkeypatch.setattr(Path, "open", original)
    directory, suffix = ("transcripts", "json") if snapshot else ("outputs", "txt")
    target = tmp_path / directory / f"{fixed_id}.{suffix}"
    assert target.read_bytes() == b"partial"
    with pytest.raises(ValueError):
        store.description(fixed_id)
    with pytest.raises(FileExistsError):
        if snapshot:
            store.save_snapshot([])
        else:
            store.save_output("replacement", "tool")
    assert target.read_bytes() == b"partial"


def test_large_offsets_are_skipped_in_bounded_chunks(tmp_path, monkeypatch):
    """验证大偏移量通过有界分块跳过，不一次读取整个前缀或文件。"""
    store = ArtifactStore(tmp_path)
    result_id = store.save_output("中" * 50000 + "😀尾", "tool")
    original = Path.open
    sizes = []

    class TrackedReader:
        """记录读取尺寸并透传文件上下文。"""

        def __init__(self, stream):
            """保存真实文件流。

            参数：
                stream: 被包装的文本流。
            """
            self.stream = stream

        def __enter__(self):
            """进入真实文件流的上下文。

            返回值：
                当前包装器。
            """
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            """退出真实文件流上下文。

            参数：
                *args: 上下文异常信息。
            返回值：
                真实文件流的退出结果。
            """
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            """记录尺寸并读取文本。

            参数：
                size: 本次最多读取的字符数。
            返回值：
                读取到的文本。
            """
            sizes.append(size)
            assert 0 <= size <= 8192
            return self.stream.read(size)

    def open_tracked(path, *args, **kwargs):
        """为读取操作包装文件流。

        参数：
            path: 文件路径。
            *args: 文件打开的位置参数。
            **kwargs: 文件打开的关键字参数。
        返回值：
            记录读取尺寸的文件包装器。
        """
        return TrackedReader(original(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", open_tracked)
    page = json.loads(store.read_tool_result(result_id, 50000, 2))
    assert page["content"] == "😀尾"
    assert page["next_offset"] is None
    assert len(sizes) > 2
