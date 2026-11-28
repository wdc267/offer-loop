"""验证有界上下文、归档引用和一次性强制压缩。"""

import copy
import json

import pytest

from mini_agent.agent import run_turn
from mini_agent.context import ContextBudgetExceeded, ContextManager, ContextOverflow, request_chars


class MemoryStore:
    """模拟可恢复归档，避免纯逻辑测试依赖磁盘。"""

    def __init__(self):
        """初始化输出和快照登记。"""
        self.outputs = {}
        self.snapshots = []

    def save_output(self, content, source):
        """保存模拟结果。

        参数：
            content: 完整正文。
            source: 来源描述。
        返回值：
            新结果 ID。
        """
        key = f"output-{len(self.outputs)}"
        self.outputs[key] = {"content": content, "source": source}
        return key

    def description(self, result_id):
        """返回模拟结果元数据。

        参数：
            result_id: 结果 ID。
        返回值：
            来源与长度。
        """
        value = self.outputs[result_id]
        return {"result_id": result_id, "source": value["source"], "total_chars": len(value["content"])}

    def save_snapshot(self, messages):
        """保存独立历史快照。

        参数：
            messages: 历史列表。
        返回值：
            快照 ID。
        """
        self.snapshots.append(copy.deepcopy(messages))
        return f"snapshot-{len(self.snapshots)}"


def batch(name="read_resume", content="正文", key="read", **args):
    """构造闭合工具批次。

    参数：
        name: 工具名称。
        content: 结果正文。
        key: 调用 ID。
        **args: 工具参数。
    返回值：
        assistant 与 tool 消息。
    """
    return [
        {"role": "assistant", "tool_calls": [{"id": key, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args, ensure_ascii=False),
        }}]},
        {"role": "tool", "tool_call_id": key, "content": content},
    ]


def test_short_context_uses_independent_request_copy():
    """预算内模型可修改其输入副本，不会污染真实历史的嵌套字段。"""
    history = [{"role": "system", "content": "规则"}]

    def model(messages):
        """修改输入副本并返回答复。

        参数：
            messages: 模型请求副本。
        返回值：
            完成响应。
        """
        messages[0]["content"] = "污染"
        return {"role": "assistant", "content": "完成"}

    assert run_turn("任务", history, model, {}, {}) == "完成"
    assert history[0]["content"] == "规则"


def test_large_read_is_saved_before_preview_and_not_summarized():
    """大读取正文完整归档，低成本预览足够时不调用摘要器。"""
    store = MemoryStore()
    context = ContextManager(store=store, schemas=[], summarize=lambda _: pytest.fail("不应摘要"))
    user = {"role": "user", "content": "读资料"}
    history = [user, *batch(content="开头" + "x" * 20000 + "结尾")]
    original = history[-1]["content"]
    context.register_result(history[-1], "read_resume", {"path": "resume.md"}, "ok")
    context.prepare(history, user)
    assert store.outputs["output-0"]["content"] == original
    assert "output-0" in history[-1]["content"]
    assert "不完整" in history[-1]["content"]
    assert "开头" in history[-1]["content"] and "结尾" in history[-1]["content"]
    assert request_chars(history, []) <= context.limit


def test_only_successful_reads_are_shortened_and_arguments_are_unchanged():
    """低成本处理不改修改参数、拒绝结果或错误读取，完整批次保持配对。"""
    store = MemoryStore()
    context = ContextManager(store=store, schemas=[], limit=100000)
    user = {"role": "user", "content": "任务"}
    history = [user, *batch("update_resume", "已更新", content_arg="x" * 14000),
               *batch(content="错误" * 8000, key="bad")]
    original = copy.deepcopy(history)
    context.register_result(history[-1], "read_resume", {}, "error")
    context.prepare(history, user)
    assert history == original
    assert store.outputs == {}


def test_summary_replaces_history_after_snapshot_and_preserves_required_batches():
    """超限时摘要较早历史，但当前请求、完整写入与最新工具批次保持原样。"""
    store = MemoryStore()
    summary_inputs = []

    def summarize(messages):
        """记录有界摘要输入并返回固定摘要。

        参数：
            messages: 独立摘要请求。
        返回值：
            有损事实摘要。
        """
        summary_inputs.append(messages)
        return "过去讨论了项目，仍需核查。"

    context = ContextManager(store=store, schemas=[], limit=2200, summarize=summarize)
    user = {"role": "user", "content": "当前任务原文"}
    write = batch("update_resume", "操作未执行：用户拒绝。", key="write", path="resume.md")
    write[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "resume.md", "content": "新简历"})
    recent = batch(key="latest", content="最新结果")
    history = [{"role": "system", "content": "不虚构"}, {"role": "user", "content": "旧问题"},
               {"role": "assistant", "content": "旧内容" * 1500}, user, *write, *recent]
    context.prepare(history, user)
    assert store.snapshots
    assert summary_inputs and request_chars(summary_inputs[0], []) <= context.summary_input_limit
    assert history[0]["content"] == "不虚构"
    assert user in history and write[0] in history and write[1] in history
    assert history[-2:] == recent
    assert any("过去讨论" in str(m.get("content")) for m in history)
    assert request_chars(history, []) <= 2200


def test_snapshot_failure_does_not_drop_history():
    """历史归档失败时不替换消息，也不发送无法恢复的摘要视图。"""
    class BrokenStore(MemoryStore):
        """模拟磁盘不可写。"""

        def save_snapshot(self, messages):
            """拒绝保存快照。

            参数：
                messages: 原始历史。
            异常：
                OSError: 模拟磁盘故障。
            """
            raise OSError("disk full")

    context = ContextManager(store=BrokenStore(), schemas=[], limit=100, summarize=lambda _: "摘要")
    user = {"role": "user", "content": "当前"}
    history = [{"role": "user", "content": "旧" * 500}, user]
    original = copy.deepcopy(history)
    with pytest.raises(ContextBudgetExceeded):
        context.prepare(history, user)
    assert history == original


def test_failed_summary_falls_back_to_archived_complete_old_turns():
    """摘要失败后只裁剪已归档的完整旧轮次，不冒充成功摘要。"""
    context = ContextManager(store=MemoryStore(), schemas=[], limit=600, summarize=lambda _: "")
    user = {"role": "user", "content": "当前"}
    history = [{"role": "system", "content": "规则"}, {"role": "user", "content": "旧" * 1000},
               *batch(key="old"), user]
    context.prepare(history, user)
    assert history[-1] == user
    assert not any(m.get("role") == "tool" for m in history)
    assert request_chars(history, []) <= 600


def test_summary_calls_and_inputs_are_bounded_for_huge_history():
    """过大的摘要输入有限分段处理，调用和每个请求都不能无限增长。"""
    requests = []

    def summarize(messages):
        """记录请求并返回简短摘要。

        参数：
            messages: 有界摘要请求。
        返回值：
            固定摘要。
        """
        requests.append(messages)
        return "已归档旧信息。"

    context = ContextManager(store=MemoryStore(), schemas=[], limit=1500, summarize=summarize)
    user = {"role": "user", "content": "现在"}
    history = [{"role": "user", "content": "长文本" * 50000}, user]
    try:
        context.prepare(history, user)
    except ContextBudgetExceeded:
        pass
    assert 0 < len(requests) <= 4
    assert all(request_chars(m, []) <= context.summary_input_limit for m in requests)


def test_schemas_and_json_escaping_count_towards_budget():
    """字符预算包含 schema 和 JSON 转义，必要输入超限时明确失败。"""
    history = [{"role": "user", "content": '中文😀\n"\\'}]
    schemas = [{"description": "schema" * 100}]
    size = request_chars(history, schemas)
    context = ContextManager(schemas=schemas, limit=size)
    context.prepare(history, history[0])
    with pytest.raises(ContextBudgetExceeded):
        ContextManager(schemas=schemas, limit=size - 1).prepare(history, history[0])


@pytest.mark.parametrize("failures", [1, 2])
def test_context_overflow_retries_same_request_at_most_once_without_replaying_tools(failures):
    """明确超限只重试当前模型调用一次，已执行的修改不会被重新调用。"""
    context = ContextManager(store=MemoryStore(), schemas=[], summarize=lambda _: "历史参考")
    history = []
    calls = []
    writes = []
    stops = []

    def write():
        """记录一次副作用。

        返回值：
            写入说明。
        """
        writes.append("written")
        return "已更新"

    def model(messages):
        """先请求写入，再模拟执行请求超限。

        参数：
            messages: 当前工作视图。
        返回值：
            工具调用或完成响应。
        异常：
            ContextOverflow: 模拟端点明确超限。
        """
        calls.append(copy.deepcopy(messages))
        if len(calls) == 1:
            return batch("update_resume", key="write")[0]
        if len(calls) <= 1 + failures:
            raise ContextOverflow("context_length_exceeded")
        return {"role": "assistant", "content": "完成"}

    answer = run_turn("更新", history, model, {"update_resume": write},
                      {"Stop": [stops.append]}, context=context)
    assert writes == ["written"]
    assert len(calls) == 3
    assert len(stops) == 1
    assert ("完成" in answer) if failures == 1 else ("上下文" in answer)
    assert any(m.get("tool_call_id") == "write" for m in history)


def test_unrelated_model_error_is_not_retried():
    """一般模型错误保持传播，不被当成上下文超限重试。"""
    calls = []

    def model(messages):
        """模拟非上下文错误。

        参数：
            messages: 请求历史。
        异常：
            ValueError: 一般模型错误。
        """
        calls.append(messages)
        raise ValueError("invalid tool protocol")

    with pytest.raises(ValueError, match="protocol"):
        run_turn("任务", [], model, {}, {})
    assert len(calls) == 1
