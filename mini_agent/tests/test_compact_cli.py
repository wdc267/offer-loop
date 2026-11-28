"""从 CLI 与 SDK 边界验证归档、摘要和预算错误。"""

import json
import sys
import types

import pytest

from mini_agent import __main__ as cli
from mini_agent.context import ContextOverflow
from mini_agent.harness import default_hooks, run_tool
from mini_agent.tools import TOOL_SCHEMAS, make_tools


def test_archive_tool_schema_and_permission_are_not_resume_paths(tmp_path):
    """归档工具使用整数字段和会话 ID 校验，不走简历路径审批，也不能访问任意文件。"""
    from mini_agent.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "archive")
    result_id = store.save_output("abcdefgh", "read_resume:resume.md")
    tools = make_tools(tmp_path, artifacts=store)
    prompts = []
    hooks = default_hooks(tmp_path, confirm=prompts.append, artifacts=store)
    schema = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == "read_tool_result")
    assert schema["properties"]["offset"]["type"] == "integer"
    call = {"id": "page", "function": {"name": "read_tool_result", "arguments": json.dumps({
        "result_id": result_id, "offset": 2, "limit": 3,
    })}}
    result = run_tool(call, tools, hooks)
    assert json.loads(result["content"])["content"] == "cde"
    assert prompts == []
    call["function"]["arguments"] = json.dumps({"result_id": "../.env"})
    assert "content" in run_tool(call, tools, hooks)
    assert "cde" not in run_tool(call, tools, hooks)["content"]


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_cli_rejects_invalid_context_budget_before_startup(monkeypatch, capsys, value):
    """非法预算在启动与文件写入前被 argparse 拒绝。"""
    monkeypatch.setattr(sys, "argv", ["mini_agent", "--context-chars", value, "任务"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "context-chars" in capsys.readouterr().err


def test_cli_large_read_persists_snapshot_output_without_network(tmp_path, monkeypatch, capsys):
    """CLI 读取长简历后输出归档预览，磁盘完整正文可恢复且无需真实端点。"""
    project = tmp_path / "project"
    workspace = project / "workspace"
    workspace.mkdir(parents=True)
    text = "开头" + "面试资料" * 6000 + "尾部"
    (workspace / "demo_resume.md").write_text(text, encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(project / "mini_agent" / "mini_agent" / "__main__.py"))
    monkeypatch.setattr(sys, "argv", ["mini_agent", "--context-chars", "10000", "读简历"])
    cli.main()
    files = list((project / ".mini_agent" / "sessions").glob("*/outputs/*.txt"))
    assert len(files) == 1 and files[0].read_text(encoding="utf-8") == text
    output = capsys.readouterr()
    assert "不完整" in output.out
    assert "明文" in output.out
    assert text not in output.err


def test_repl_remains_interactive_after_budget_stop(tmp_path, monkeypatch, capsys):
    """超限停止当前轮次但不退出 REPL，下一条输入仍可获得答复。"""
    inputs = iter(["过长" * 4000, "上一条指令", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    cli.run_session(cli.demo_model, tmp_path, context_chars=4000)
    output = capsys.readouterr()
    assert "上下文预算不足" in output.out
    assert output.err.count("[Stop]") == 2


def test_cli_session_summarizes_then_continues_with_source_request(tmp_path, monkeypatch, capsys):
    """CLI 会话超限后保存快照并摘要，后续模型仍收到本轮原始问题。"""
    inputs = iter(["旧问题", "当前问题", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    requests = []
    summaries = []

    def summarize(messages):
        """返回不含执行指令的历史参考。

        参数：
            messages: 有界摘要输入。
        返回值：
            历史摘要。
        """
        summaries.append(messages)
        return "之前讨论过项目，结论待核查。"

    def model(messages):
        """第一次产生长响应，第二次检查压缩后的当前请求。

        参数：
            messages: 执行请求。
        返回值：
            长响应或完成响应。
        """
        requests.append(messages)
        return {"role": "assistant", "content": "旧响应" * 2000 if len(requests) == 1 else "完成当前问题"}

    model.summarize = summarize
    cli.run_session(model, tmp_path / "workspace", context_chars=4000)
    assert len(summaries) == 1
    assert requests[-1][-1] == {"role": "user", "content": "当前问题"}
    assert list((tmp_path / ".mini_agent" / "sessions").glob("*/transcripts/*.json"))
    assert "完成当前问题" in capsys.readouterr().out


class FakeBadRequest(Exception):
    """提供与 SDK 请求错误一致的测试字段。"""

    def __init__(self, body):
        """保存端点错误详情。

        参数：
            body: 错误对象。
        """
        super().__init__("HTTP 400")
        self.body = body


def fake_sdk(monkeypatch, responses):
    """安装记录请求的 OpenAI SDK 替身。

    参数：
        monkeypatch: pytest 修改依赖工具。
        responses: 响应或异常序列。
    返回值：
        已记录的请求列表。
    """
    requests = []
    items = iter(responses)

    class Client:
        """模拟客户端，不访问网络。"""

        def __init__(self, **kwargs):
            """提供 chat.completions 入口。

            参数：
                **kwargs: 兼容客户端构造参数。
            """
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, **kwargs):
            """返回脚本化响应。

            参数：
                **kwargs: 模型请求参数。
            返回值：
                SDK 响应替身。
            异常：
                Exception: 脚本指定的错误。
            """
            requests.append(kwargs)
            item = next(items)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client, BadRequestError=FakeBadRequest))
    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=lambda *a, **k: None))
    monkeypatch.setenv("MINI_AGENT_API_KEY", "test-not-a-secret")
    return requests


@pytest.mark.parametrize("body,recognized", [
    ({"code": "context_length_exceeded", "message": "too long"}, True),
    ({"error": {"code": "context_length_exceeded"}}, True),
    ({"code": "invalid_request_error", "message": "prompt is too long"}, False),
    ({"code": "invalid_tool", "message": "context length"}, False),
])
def test_live_adapter_only_recognizes_explicit_context_error(monkeypatch, body, recognized):
    """只有明确上下文错误码转换为可重试异常，一般请求错误仍原样传播。"""
    fake_sdk(monkeypatch, [FakeBadRequest(body)])
    model = cli.live_model()
    with pytest.raises(ContextOverflow if recognized else FakeBadRequest):
        model([{"role": "user", "content": "任务"}])


def test_live_summary_has_no_tools_and_rejects_truncated_output(monkeypatch):
    """真实适配器的摘要请求不带工具，并拒绝截断输出。"""
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(
        finish_reason="length", message=types.SimpleNamespace(content="半截摘要"),
    )])
    requests = fake_sdk(monkeypatch, [response])
    model = cli.live_model()
    with pytest.raises(ValueError, match="摘要"):
        model.summarize([{"role": "user", "content": "历史数据"}])
    assert "tools" not in requests[0]
    assert requests[0]["model"] == "deepseek-chat"
    assert requests[0]["max_tokens"] > 0
