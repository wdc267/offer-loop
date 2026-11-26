"""离线验证多轮会话和简历工具的 Hook 链。"""

import json
import sys
from pathlib import Path

import pytest

from mini_agent import __main__ as cli
from mini_agent.__main__ import demo_model, run_session
from mini_agent.agent import run_turn
from mini_agent.harness import default_hooks, run_tool
from mini_agent.tools import make_tools


def call_tool(name, path="resume.md", **kwargs):
    """构造与模型输出结构相同的简历工具调用。

    参数：
        name: 工具名。
        path: 简历的相对路径。
        **kwargs: path 以外的工具参数。

    返回值：
        带固定调用 ID 的工具调用字典。
    """
    return {
        "id": "call-resume", "type": "function",
        "function": {"name": name, "arguments": json.dumps({"path": path, **kwargs})},
    }


@pytest.mark.parametrize("live", [False, True])
def test_cli_uses_persistent_project_workspace(tmp_path, monkeypatch, capsys, live):
    """验证两种模式都使用仓库根目录下的 workspace，不受启动目录影响，并且不会覆盖已有文件。"""
    project = tmp_path / "project"
    monkeypatch.setattr(cli, "__file__", str(project / "mini_agent" / "mini_agent" / "__main__.py"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "live_model", lambda: demo_model)
    args = ["mini_agent", "--live"] if live else ["mini_agent"]
    monkeypatch.setattr(sys, "argv", [*args, "读简历"])

    cli.main()

    workspace = project / "workspace"
    demo = workspace / "demo_resume.md"
    assert workspace.is_dir()
    assert not (tmp_path / "workspace").exists()
    if live:
        assert not demo.exists()
    else:
        assert "示例简历" in demo.read_text(encoding="utf-8")
    (workspace / "resume.md").write_text("另一份简历", encoding="utf-8")
    demo.write_text("保留已有内容", encoding="utf-8")
    capsys.readouterr()

    cli.main()

    assert demo.read_text(encoding="utf-8") == "保留已有内容"
    assert "保留已有内容" in capsys.readouterr().out
    assert (workspace / "resume.md").read_text(encoding="utf-8") == "另一份简历"


def test_cli_rejects_workspace_redirected_outside_project(tmp_path, monkeypatch, capsys):
    """验证固定工作区解析到项目外时，启动失败且不会在外部目录写入示例。"""
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        """模拟工作区通过链接指向项目外。

        参数：
            path: 待解析的路径。
            *args: 传给原方法的位置参数。
            **kwargs: 传给原方法的关键字参数。

        返回值：
            解析后的绝对路径。
        """
        if path == project / "workspace":
            return outside
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(cli, "__file__", str(project / "mini_agent" / "mini_agent" / "__main__.py"))
    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(sys, "argv", ["mini_agent", "读简历"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
    assert "项目目录外" in capsys.readouterr().err
    assert not outside.exists()


def test_session_reuses_history_between_user_turns(tmp_path, monkeypatch, capsys):
    """验证第一轮 Stop 后会话继续，下一次模型请求保留之前的对话和工具结果。"""
    (tmp_path / "resume.md").write_text("示例简历", encoding="utf-8")
    inputs = iter(["读简历", "刚才读了什么？", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    requests = []

    def model(messages):
        """在第一轮请求工具，其他时候直接回答。

        参数：
            messages: 当前会话的消息历史。

        返回值：
            模拟模型的下一条 assistant 消息。
        """
        requests.append(list(messages))
        if len(requests) == 1:
            return {"role": "assistant", "tool_calls": [call_tool("read_resume")]}
        return {"role": "assistant", "content": "已回答"}

    run_session(model, tmp_path)

    assert len(requests) == 3
    followup = requests[-1]
    assert [m["role"] for m in followup] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert followup[1]["content"] == "读简历"
    assert followup[3] == {"role": "tool", "tool_call_id": "call-resume", "content": "示例简历"}
    assert followup[-1]["content"] == "刚才读了什么？"
    assert capsys.readouterr().err.count("[Stop]") == 2


@pytest.mark.parametrize("name", ["update_resume", "delete_resume"])
def test_rejected_mutation_is_returned_without_running_tool(tmp_path, name, capsys):
    """验证拒绝修改或删除时文件不变，拒绝原因按调用 ID 回填且不触发执行后 Hook。"""
    resume = tmp_path / "resume.md"
    resume.write_text("旧简历", encoding="utf-8")
    prompts = []

    def deny(prompt):
        """记录审批提示并拒绝操作。

        参数：
            prompt: 传给用户的审批提示。

        返回值：
            表示拒绝的输入。
        """
        prompts.append(prompt)
        return "n"

    fields = {"content": "新简历"} if name == "update_resume" else {}
    call = call_tool(name, **fields)
    hooks = default_hooks(tmp_path, confirm=deny)
    history = []

    def model(messages):
        """请求工具并在下一次调用中复述结果。

        参数：
            messages: 当前用户输入与工具结果。

        返回值：
            模拟模型的下一条 assistant 消息。
        """
        last = messages[-1]
        if last["role"] == "tool":
            return {"role": "assistant", "content": last["content"]}
        return {"role": "assistant", "tool_calls": [call]}

    answer = run_turn("操作简历", history, model, make_tools(tmp_path), hooks)

    assert "用户拒绝" in answer
    assert history[-2]["tool_call_id"] == call["id"]
    assert resume.read_text(encoding="utf-8") == "旧简历"
    assert resume.as_posix() in prompts[0]
    if name == "delete_resume":
        assert "不可恢复" in prompts[0]
    trace = capsys.readouterr().err
    assert "[PreToolUse]" not in trace
    assert "[PostToolUse]" not in trace


@pytest.mark.parametrize("name", ["update_resume", "delete_resume"])
def test_approved_mutation_runs_and_triggers_post_hook(tmp_path, capsys, name):
    """验证审批通过后工具真正执行，注册的观察 Hook 能看到结果。"""
    resume = tmp_path / "resume.md"
    resume.write_text("旧简历", encoding="utf-8")
    hooks = default_hooks(tmp_path, confirm=lambda _: "yes")
    observed = []
    hooks["PostToolUse"].append(lambda *args: observed.append(args))
    fields = {"content": "新简历"} if name == "update_resume" else {}
    result = run_tool(call_tool(name, **fields), make_tools(tmp_path), hooks)

    assert observed == [(name, "ok", result["content"])]
    assert [line.split()[0] for line in capsys.readouterr().err.splitlines()] == ["[PreToolUse]", "[PostToolUse]"]
    if name == "update_resume":
        assert resume.read_text(encoding="utf-8") == "新简历"
    else:
        assert not resume.exists()


def test_failed_approval_does_not_write(tmp_path):
    """验证确认输入中断时不执行修改。"""
    resume = tmp_path / "resume.md"
    resume.write_text("旧简历", encoding="utf-8")

    def unavailable(prompt):
        """模拟审批无法读取输入。

        参数：
            prompt: 审批提示，本替身不使用。

        异常：
            EOFError: 模拟终端输入结束。
        """
        raise EOFError

    hooks = default_hooks(tmp_path, confirm=unavailable)
    result = run_tool(call_tool("update_resume", content="新简历"), make_tools(tmp_path), hooks)

    assert "检查失败" in result["content"]
    assert resume.read_text(encoding="utf-8") == "旧简历"


def test_outside_resume_is_rejected_before_approval(tmp_path):
    """验证工作区外的简历不能被删除，也不会触发用户确认。"""
    outside = tmp_path / "outside.md"
    outside.write_text("保持不变", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompts = []
    hooks = default_hooks(workspace, confirm=prompts.append)

    result = run_tool(call_tool("delete_resume", "../outside.md"), make_tools(workspace), hooks)

    assert "工作区外" in result["content"]
    assert prompts == []
    assert outside.read_text(encoding="utf-8") == "保持不变"


def test_analysis_uses_local_rules_without_approval(tmp_path):
    """验证分析简历返回章节、联系方式和量化信息，不需要审批。"""
    (tmp_path / "resume.txt").write_text("教育经历\n项目经历\n减少耗时 30%\nuser@example.com", encoding="utf-8")
    prompts = []
    result = run_tool(
        call_tool("analyze_resume", "resume.txt"), make_tools(tmp_path),
        default_hooks(tmp_path, confirm=prompts.append),
    )["content"]

    assert "教育经历" in result and "项目经历" in result
    assert "邮箱：有" in result
    assert "量化表述：1" in result
    assert "user@example.com" not in result
    assert prompts == []


def test_step_limit_still_returns_tool_result(tmp_path):
    """验证达到步数上限时，已发出的工具调用仍有对应结果。"""
    (tmp_path / "demo_resume.md").write_text("示例简历", encoding="utf-8")
    history = []
    answer = run_turn(
        "读简历", history, demo_model, make_tools(tmp_path), default_hooks(tmp_path), max_steps=1,
    )

    assert "步数上限" in answer
    assert history[-1] == {
        "role": "tool",
        "tool_call_id": history[-2]["tool_calls"][0]["id"],
        "content": "示例简历",
    }


def test_pdf_read_extracts_text(tmp_path):
    """验证读取有文本层的 PDF 会提取正文，不需要用户审批。"""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 150 Td (Demo resume) Tj ET")
    page.replace_contents(stream)
    writer.write(tmp_path / "resume.pdf")
    prompts = []
    result = run_tool(
        call_tool("read_resume", "resume.pdf"), make_tools(tmp_path),
        default_hooks(tmp_path, confirm=prompts.append),
    )

    assert result["content"].strip() == "Demo resume"
    assert prompts == []


def test_pdf_update_creates_markdown_without_overwriting(tmp_path):
    """验证修改 PDF 只生成新 Markdown，再次修改不会覆盖同名文件或更改原件。"""
    source = tmp_path / "resume.pdf"
    source.write_bytes(b"demo source")
    tools = make_tools(tmp_path)
    hooks = default_hooks(tmp_path, confirm=lambda _: "y")
    call = call_tool("update_resume", "resume.pdf", content="# 新简历")

    assert "已生成" in run_tool(call, tools, hooks)["content"]
    assert "已存在" in run_tool(call, tools, hooks)["content"]
    assert source.read_bytes() == b"demo source"
    assert (tmp_path / "resume_new.md").read_text(encoding="utf-8") == "# 新简历"
