"""工具实现、注册表与分发器测试。"""

from __future__ import annotations

import pytest

from mini_agent.errors import ToolError
from mini_agent.models import ToolCall, ToolSpec
from mini_agent.tools import (
    AnalyzeResumeTool,
    ReadJDTool,
    ReadResumeTool,
    ToolDispatcher,
    ToolRegistry,
    analyze_resume_text,
    create_default_registry,
)

SAMPLE_RESUME = """张三
邮箱：zhang@example.com 手机：13800138000
教育背景
2018.09-2022.06 某大学 计算机科学与技术
工作经历
2022.07-至今 某公司 后端开发
项目经历
统一网关：接口延迟下降 30%，支撑 10 万用户
专业技能
Python / SQL / Redis
"""


class _EchoTool:
    spec = ToolSpec(name="echo", description="回显参数")

    async def run(self, **kwargs):
        """回显全部参数，用于断言分发器透传了解析后的 kwargs。

        参数：
            **kwargs: 分发器解析出的工具参数。

        返回值：
            把参数字典拼进字符串的结果。
        """
        return f"echo:{kwargs}"


class _FailingTool:
    spec = ToolSpec(name="fail", description="总是失败")

    async def run(self, **_kwargs):
        """模拟工具业务失败，始终抛出 ToolError。

        参数：
            **_kwargs: 工具参数，本替身忽略。

        异常：
            ToolError: 恒定抛出，用于验证错误转换。
        """
        raise ToolError("boom")


class _CrashingTool:
    spec = ToolSpec(name="crash", description="抛出非工具异常")

    async def run(self, **_kwargs):
        """模拟编程错误，抛出非工具异常。

        参数：
            **_kwargs: 工具参数，本替身忽略。

        异常：
            RuntimeError: 恒定抛出，用于验证兜底捕获。
        """
        raise RuntimeError("bug")


class _NonTextTool:
    spec = ToolSpec(name="non_text", description="返回非文本")

    async def run(self, **_kwargs):
        """返回非文本结果，用于验证分发器的类型校验。

        参数：
            **_kwargs: 工具参数，本替身忽略。

        返回值：
            整数 42，故意不是字符串。
        """
        return 42


# ---------------------------------------------------------------------------
# 读取工具
# ---------------------------------------------------------------------------
def test_read_resume_tool_returns_rendered_text(run_async, tmp_path):
    """验证 read_resume 返回带文件头信息的简历正文。"""
    path = tmp_path / "resume.txt"
    path.write_text("张三\nPython", encoding="utf-8")

    text = run_async(ReadResumeTool().run(path=str(path)))

    assert text.startswith("简历文件：")
    assert "张三" in text
    assert "Python" in text


def test_read_jd_tool_returns_rendered_text(run_async, tmp_path):
    """验证 read_jd 返回带文件头信息的 JD 正文。"""
    path = tmp_path / "jd.md"
    path.write_text("# 后端工程师\n熟悉 Python", encoding="utf-8")

    text = run_async(ReadJDTool().run(path=str(path)))

    assert text.startswith("JD文件：")
    assert "后端工程师" in text


def test_tool_rejects_missing_path(run_async):
    """验证读取工具缺少 path 参数时抛 ToolError。"""
    with pytest.raises(ToolError, match="path"):
        run_async(ReadResumeTool().run())


def test_tool_rejects_unknown_arguments(run_async, tmp_path):
    """验证传入了 schema 之外的参数时会被拒绝。"""
    path = tmp_path / "resume.txt"
    path.write_text("内容", encoding="utf-8")

    with pytest.raises(ToolError, match="不支持的参数"):
        run_async(ReadResumeTool().run(path=str(path), extra=1))


def test_analyze_tool_reads_file(run_async, tmp_path):
    """验证 analyze_resume 会读取文件并输出结构分析报告。"""
    path = tmp_path / "resume.txt"
    path.write_text(SAMPLE_RESUME, encoding="utf-8")

    text = run_async(AnalyzeResumeTool().run(path=str(path)))

    assert text.startswith("简历文件：")
    assert "简历结构分析" in text
    assert "量化指标：2 处" in text


# ---------------------------------------------------------------------------
# 规则分析
# ---------------------------------------------------------------------------
def test_analyze_resume_reports_structure():
    """验证规则分析能识别联系方式、各章节、量化指标和日期区间。"""
    report = analyze_resume_text(SAMPLE_RESUME)

    assert "邮箱 ✓" in report
    assert "电话 ✓" in report
    assert "教育 ✓" in report
    assert "实习/工作 ✓" in report
    assert "项目 ✓" in report
    assert "技能 ✓" in report
    assert "自我评价 ✗" in report
    assert "量化指标：2 处" in report
    assert "日期区间：2 段" in report
    assert "未检测到自我评价" in report


def test_analyze_resume_rejects_empty_text():
    """验证空简历文本无法分析并抛 ToolError。"""
    with pytest.raises(ToolError, match="为空"):
        analyze_resume_text("   ")


def test_analyze_resume_clean_text_has_no_missing_hints():
    """验证结构完整的简历不会产生缺失提示。"""
    filler = "负责后端接口开发与性能优化，" * 20
    text = (
        "教育背景\n2018.09-2022.06 某大学\n"
        f"工作经历\n2022.07-至今 某公司，性能提升 30%\n{filler}\n"
        "项目经历\n项目A 支撑 10 万用户\n"
        "专业技能\nPython\n"
        "自我评价\n认真负责"
    )

    report = analyze_resume_text(text)

    assert "未发现明显的结构缺失" in report


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
def test_default_registry_names():
    """验证默认注册表包含三个工具且顺序固定。"""
    registry = create_default_registry()

    assert registry.names == ("read_resume", "read_jd", "analyze_resume")
    assert len(registry) == 3
    assert [spec.name for spec in registry.specs()] == list(registry.names)


def test_registry_rejects_duplicate_names():
    """验证同名工具在注册阶段直接报错。"""
    with pytest.raises(ValueError, match="重复的工具名"):
        ToolRegistry([ReadResumeTool(), ReadResumeTool()])


def test_registry_get_unknown_returns_none():
    """验证查询未注册的工具名返回 None。"""
    assert create_default_registry().get("nope") is None


# ---------------------------------------------------------------------------
# 分发器
# ---------------------------------------------------------------------------
def _dispatcher(*tools):
    """用给定工具临时构造一个分发器。

    参数：
        *tools: 要注册的工具实例。

    返回值：
        基于这些工具构建的 ToolDispatcher。
    """
    return ToolDispatcher(ToolRegistry(tools))


def test_dispatcher_success(run_async):
    """验证分发器执行工具并把结果包装成带 tool_call_id 的 tool 消息。"""
    message = run_async(
        _dispatcher(_EchoTool()).dispatch(
            ToolCall(id="call_1", name="echo", arguments='{"x": 1}')
        )
    )

    assert message.role == "tool"
    assert message.tool_call_id == "call_1"
    assert message.content == "echo:{'x': 1}"


def test_dispatcher_unknown_tool(run_async):
    """验证未知工具会回填错误消息并列出可用工具。"""
    message = run_async(
        _dispatcher(_EchoTool()).dispatch(
            ToolCall(id="call_1", name="nope", arguments="{}")
        )
    )

    assert message.role == "tool"
    assert "未知工具" in message.content
    assert "echo" in message.content


def test_dispatcher_invalid_json(run_async):
    """验证参数不是合法 JSON 时回填错误而不是抛异常。"""
    message = run_async(
        _dispatcher(_EchoTool()).dispatch(
            ToolCall(id="call_1", name="echo", arguments="{bad")
        )
    )

    assert "不是合法 JSON" in message.content


def test_dispatcher_non_object_arguments(run_async):
    """验证 JSON 参数不是对象时被拒绝。"""
    message = run_async(
        _dispatcher(_EchoTool()).dispatch(
            ToolCall(id="call_1", name="echo", arguments="[1, 2]")
        )
    )

    assert "必须是 JSON 对象" in message.content


def test_dispatcher_catches_tool_error(run_async):
    """验证工具抛 ToolError 时被转换成统一的失败消息。"""
    message = run_async(
        _dispatcher(_FailingTool()).dispatch(
            ToolCall(id="call_1", name="fail", arguments="{}")
        )
    )

    assert message.content == "工具执行失败：boom"


def test_dispatcher_catches_unexpected_exception(run_async):
    """验证工具抛出的非预期异常也不会打断 Agent 循环。"""
    message = run_async(
        _dispatcher(_CrashingTool()).dispatch(
            ToolCall(id="call_1", name="crash", arguments="{}")
        )
    )

    assert "RuntimeError" in message.content
    assert "bug" in message.content


def test_dispatcher_rejects_non_text_result(run_async):
    """验证工具返回非字符串结果时按失败处理。"""
    message = run_async(
        _dispatcher(_NonTextTool()).dispatch(
            ToolCall(id="call_1", name="non_text", arguments="{}")
        )
    )

    assert "非文本结果" in message.content
