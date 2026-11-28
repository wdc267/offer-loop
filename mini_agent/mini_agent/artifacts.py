"""按会话归档工具原文与消息快照，不在日志中输出正文。"""

import json
import uuid
from pathlib import Path


class ArtifactStore:
    """管理调用方指定的唯一会话目录中的归档和实例内输出登记。"""

    def __init__(self, root: Path):
        """绑定会话归档根目录，不提前创建任何目录。

        参数：
            root: 调用方提供的唯一会话归档目录。
        异常：
            ValueError: 根目录或其祖先包含链接，或解析位置发生重定向。
        """
        self.root = Path(root).absolute()
        self._outputs: dict[str, dict] = {}
        self._check_path(self.root)

    def _check_path(self, path: Path) -> None:
        """检查路径及其祖先不是链接，且解析结果未离开固定归档位置。

        参数：
            path: 待检查的根目录、子目录或文件路径。
        异常：
            ValueError: 存在符号链接、Windows 联接或路径解析越界。
        """
        for candidate in (path, *path.parents):
            if candidate.is_symlink() or candidate.is_junction():
                raise ValueError("归档路径不能包含链接")
        resolved_root = self.root.resolve()
        resolved_path = path.resolve()
        if resolved_root != self.root or not resolved_path.is_relative_to(self.root):
            raise ValueError("归档路径解析超出固定根目录")
        if resolved_path != path:
            raise ValueError("归档路径不能重定向")

    def _directory(self, name: str) -> Path:
        """安全地惰性创建归档子目录。

        参数：
            name: 宿主指定的 outputs 或 transcripts 子目录名。
        返回值：
            已创建并重新校验的子目录路径。
        异常：
            ValueError: 目录或其祖先包含链接或解析越界。
            OSError: 创建目录失败。
        """
        directory = self.root / name
        self._check_path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._check_path(directory)
        return directory

    def save_output(self, content: str, source: str) -> str:
        """独占保存工具原文，并在文件成功关闭后登记元数据。

        参数：
            content: 需要按 UTF-8 原样保存的工具输出。
            source: 产生输出的工具或其他来源名称。
        返回值：
            宿主生成且仅当前实例可读取的输出 ID。
        异常：
            ValueError: 参数不是字符串，或归档路径不安全。
            OSError: 独占创建、写入或关闭失败；失败文件不会被删除或覆盖。
        """
        if not isinstance(content, str) or not isinstance(source, str):
            raise ValueError("content 和 source 必须是字符串")
        result_id = uuid.uuid4().hex
        path = self._directory("outputs") / f"{result_id}.txt"
        self._check_path(path)
        with path.open(mode="x", encoding="utf-8", newline="") as stream:
            stream.write(content)
        self._outputs[result_id] = {
            "result_id": result_id, "source": source, "total_chars": len(content),
        }
        return result_id

    def description(self, result_id: str) -> dict:
        """返回当前实例已登记输出的元数据副本，不读取正文。

        参数：
            result_id: 当前实例成功保存过的输出 ID。
        返回值：
            包含 result_id、source、total_chars 的独立字典。
        异常：
            ValueError: ID 类型非法或不属于当前实例。
        """
        if not isinstance(result_id, str) or result_id not in self._outputs:
            raise ValueError("输出 ID 未在当前实例登记")
        return dict(self._outputs[result_id])

    def validate_read(self, result_id: str, offset: int = 0, limit: int = 4000) -> None:
        """校验输出归属、分页参数和文件安全性，可供 PreToolUse 调用。

        参数：
            result_id: 当前实例登记的输出 ID。
            offset: 从零开始的字符偏移，允许等于原文长度。
            limit: 单页字符上限，取值 1 至 8000。
        异常：
            ValueError: ID 不属于当前实例、参数非法或路径不安全。
            FileNotFoundError: 已登记的输出文件不存在或不再是普通文件。
        """
        metadata = self.description(result_id)
        if type(offset) is not int or type(limit) is not int:
            raise ValueError("offset 和 limit 必须是非 bool 的整数")
        if not 0 <= offset <= metadata["total_chars"]:
            raise ValueError("offset 超出输出字符范围")
        if not 1 <= limit <= 8000:
            raise ValueError("limit 必须在 1 至 8000 之间")
        path = self.root / "outputs" / f"{result_id}.txt"
        self._check_path(path)
        if not path.is_file():
            raise FileNotFoundError("已登记的输出文件不存在")

    def read_tool_result(self, result_id: str, offset: int = 0, limit: int = 4000) -> str:
        """重新校验并按字符分页读取原文，以 JSON 字符串返回结果。

        参数：
            result_id: 当前实例登记的输出 ID。
            offset: 从零开始的字符偏移。
            limit: 本页最多返回的字符数，默认 4000，最大 8000。
        返回值：
            包含元数据、offset、end、next_offset 及 content 的 JSON 字符串；EOF 的 next_offset 为 null。
        异常：
            ValueError: ID、分页参数或路径非法。
            OSError: 文件缺失或读取失败。
        """
        self.validate_read(result_id, offset, limit)
        metadata = self.description(result_id)
        path = self.root / "outputs" / f"{result_id}.txt"
        with path.open(mode="r", encoding="utf-8", newline="") as stream:
            remaining = offset
            while remaining:
                skipped = stream.read(min(remaining, 8192))
                if not skipped:
                    raise ValueError("输出文件长度已变化")
                remaining -= len(skipped)
            content = stream.read(limit)
        if len(content) != min(limit, metadata["total_chars"] - offset):
            raise ValueError("输出文件长度已变化")
        end = offset + len(content)
        return json.dumps({
            **metadata,
            "offset": offset,
            "end": end,
            "next_offset": end if end < metadata["total_chars"] else None,
            "content": content,
        }, ensure_ascii=False)

    def save_snapshot(self, messages) -> str:
        """独占保存版本化消息快照及当前输出登记元数据，不输出正文日志。

        参数：
            messages: 需要精确保留的、可 JSON 序列化的消息历史。
        返回值：
            宿主生成的快照 ID，对应 transcripts 下的 JSON 文件。
        异常：
            ValueError: 路径不安全或消息包含循环引用。
            TypeError: 消息不能被 JSON 序列化。
            OSError: 独占创建、写入或关闭失败；不会删除或覆盖文件。
        """
        snapshot_id = uuid.uuid4().hex
        payload = json.dumps({
            "version": 1,
            "messages": messages,
            "outputs": {result_id: self.description(result_id) for result_id in self._outputs},
        }, ensure_ascii=False)
        path = self._directory("transcripts") / f"{snapshot_id}.json"
        self._check_path(path)
        with path.open(mode="x", encoding="utf-8", newline="") as stream:
            stream.write(payload)
        return snapshot_id
