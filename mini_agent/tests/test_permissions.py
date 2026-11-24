"""Permission System 测试：拒绝表、确认规则、用户询问，全程离线。"""

from __future__ import annotations

import io

import pytest

from mini_agent import permissions
from mini_agent.permissions import (
    PermissionRequest,
    ask_user,
    check_deny_list,
    check_permission,
    check_rules,
    contains_destructive_command,
    format_request,
)


def _request() -> PermissionRequest:
    """构造一个用于测试的权限摘要。

    返回值：
        工具名、动作、路径和摘要固定的 PermissionRequest。
    """
    return PermissionRequest(
        tool_name="update_resume",
        action="modify",
        path=r"D:\resume.md",
        summary="模式：原地替换",
    )


class _RecordingApprover:
    """记录每次询问，并按预设结果放行、拒绝或抛错。"""

    def __init__(self, *, allow: bool = True, raises: bool = False) -> None:
        """创建记录型确认函数。

        参数：
            allow: 返回给 check_permission 的固定结果。
            raises: 为 True 时每次询问都抛 RuntimeError。
        """
        self.allow = allow
        self.raises = raises
        self.calls: list[tuple[str, str, PermissionRequest | None]] = []

    async def __call__(
        self,
        tool_name: str,
        args: dict,
        reason: str,
        request: PermissionRequest | None,
    ) -> bool:
        """记录询问参数并返回预设结果。

        参数：
            tool_name: 即将执行的工具名。
            args: 工具参数；本实现只记录不使用。
            reason: 命中的规则说明。
            request: 工具生成的安全摘要。

        返回值：
            构造时指定的 allow。

        异常：
            RuntimeError: 构造时 raises=True 时抛出。
        """
        self.calls.append((tool_name, reason, request))
        if self.raises:
            raise RuntimeError("询问失败")
        return self.allow


@pytest.mark.parametrize(
    "command",
    ["sudo rm -rf /tmp/x", "shutdown -h now", "mkfs.ext4 /dev/sda1"],
)
def test_check_deny_list_blocks_forbidden_patterns(command):
    """验证拒绝表命中危险命令时返回原因。"""
    assert check_deny_list(command) is not None


@pytest.mark.parametrize("command", ["ls -la", "python app.py", "git status"])
def test_check_deny_list_allows_safe_commands(command):
    """验证普通命令不会命中拒绝表。"""
    assert check_deny_list(command) is None


def test_contains_destructive_command_only_matches_command_words():
    """验证 rm/del 仅在作为独立命令时命中，普通单词不误判。"""
    assert contains_destructive_command("rm -rf build") is True
    assert contains_destructive_command("echo x; del a.txt") is True
    assert contains_destructive_command("format_disk.py") is False


def test_check_rules_flags_path_outside_workspace(monkeypatch, tmp_path):
    """验证写操作指向工作区外时命中越界规则。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())

    reason = check_rules(
        "update_resume", {"path": str(tmp_path.parent / "outside" / "resume.md")}
    )

    assert reason == "路径不在工作区内"


def test_check_rules_confirms_modify_inside_workspace(monkeypatch, tmp_path):
    """验证工作区内的修改仍会命中「需要确认」规则。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())

    reason = check_rules("update_resume", {"path": str(tmp_path / "resume.md")})

    assert reason == "修改 / 删除简历文件须由用户确认"


def test_check_rules_ignores_read_tools(monkeypatch, tmp_path):
    """验证只读工具不会命中任何确认规则。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())

    assert check_rules("read_resume", {"path": "/outside/resume.pdf"}) is None


def test_check_rules_flags_destructive_bash_command():
    """验证 bash 工具收到危险命令时命中确认规则。"""
    reason = check_rules("bash", {"command": "rm -rf build"})

    assert reason == "可能具有破坏性的命令"


@pytest.mark.parametrize(
    "tool_name",
    ["update_resume", "delete_resume"],
)
def test_confirmation_rule_applies_to_all_write_tools(tool_name):
    """验证修改和删除两类工具都被「每次确认」规则覆盖。"""
    assert check_rules(tool_name, {"path": "resume.md"}) is not None


def test_check_permission_deny_list_skips_approver(run_async):
    """验证拒绝表直接拒绝，不进入用户确认。"""
    approver = _RecordingApprover(allow=True)

    denial = run_async(
        check_permission("bash", {"command": "sudo reboot"}, approver=approver)
    )

    assert denial is not None
    assert "操作被拒绝" in denial
    assert approver.calls == []


def test_check_permission_asks_and_allows(run_async, monkeypatch, tmp_path):
    """验证规则命中后询问用户，用户同意时允许执行。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())
    approver = _RecordingApprover(allow=True)

    denial = run_async(
        check_permission(
            "update_resume",
            {"path": str(tmp_path / "resume.md")},
            _request(),
            approver=approver,
        )
    )

    assert denial is None
    assert len(approver.calls) == 1
    assert approver.calls[0][0] == "update_resume"
    assert approver.calls[0][2] is not None


def test_check_permission_user_denial_is_reported(run_async, monkeypatch, tmp_path):
    """验证用户拒绝时返回拒绝原因，不执行操作。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())
    approver = _RecordingApprover(allow=False)

    denial = run_async(
        check_permission(
            "delete_resume",
            {"path": str(tmp_path / "resume.md")},
            _request(),
            approver=approver,
        )
    )

    assert denial is not None
    assert "用户拒绝" in denial


def test_check_permission_without_rule_match_does_not_ask(
    run_async, monkeypatch, tmp_path
):
    """验证只读工具不触发用户询问并直接放行。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())
    approver = _RecordingApprover(allow=False)

    denial = run_async(
        check_permission(
            "read_jd", {"path": str(tmp_path / "jd.txt")}, approver=approver
        )
    )

    assert denial is None
    assert approver.calls == []


def test_check_permission_approver_error_denies(run_async, monkeypatch, tmp_path):
    """验证询问过程抛异常时按拒绝处理，不会放行。"""
    monkeypatch.setattr(permissions, "WORKDIR", tmp_path.resolve())
    approver = _RecordingApprover(raises=True)

    denial = run_async(
        check_permission(
            "update_resume",
            {"path": str(tmp_path / "resume.md")},
            _request(),
            approver=approver,
        )
    )

    assert denial is not None
    assert "权限确认过程出错" in denial


def test_format_request_shows_reason_tool_and_summary():
    """验证提示文本包含原因、工具名、路径和摘要。"""
    text = format_request("update_resume", "路径不在工作区内", _request())

    assert "路径不在工作区内" in text
    assert "update_resume" in text
    assert r"D:\resume.md" in text
    assert "模式：原地替换" in text


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES ", "是"])
def test_ask_user_accepts_yes(run_async, answer):
    """验证 y/yes/是 等肯定输入被识别为允许。"""
    allowed = run_async(
        ask_user(
            "update_resume",
            {},
            "修改简历",
            _request(),
            input_fn=lambda _prompt: answer,
            stream=io.StringIO(),
        )
    )

    assert allowed is True


@pytest.mark.parametrize("answer", ["", "n", "no", "随便"])
def test_ask_user_rejects_other_input(run_async, answer):
    """验证空输入、否定和其他文本都按拒绝处理。"""
    allowed = run_async(
        ask_user(
            "update_resume",
            {},
            "修改简历",
            _request(),
            input_fn=lambda _prompt: answer,
            stream=io.StringIO(),
        )
    )

    assert allowed is False


def test_ask_user_denies_on_eof(run_async):
    """验证标准输入结束时按拒绝处理，不会抛出异常。"""

    def _eof(_prompt: str) -> str:
        """模拟读取到 EOF 的输入函数。

        参数：
            _prompt: 提示文本，本函数不使用。

        异常：
            EOFError: 恒定抛出，模拟输入结束。
        """
        raise EOFError

    output = io.StringIO()

    allowed = run_async(
        ask_user(
            "delete_resume",
            {},
            "永久删除",
            _request(),
            input_fn=_eof,
            stream=output,
        )
    )

    assert allowed is False
    assert "delete_resume" in output.getvalue()
