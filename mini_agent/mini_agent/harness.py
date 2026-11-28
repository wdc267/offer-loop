"""四个生命周期事件：权限 Hook 可以阻断，日志 Hook 只观察。"""

import json
import sys

from .tools import resume_path


def run_hooks(hooks, event, *args):
    """按注册顺序触发 Hook；执行前检查失败时阻止工具执行。

    参数：
        hooks: 事件名到回调列表的映射。
        event: 当前生命周期事件名。
        *args: 传给该事件回调的参数。

    返回值：
        PreToolUse 的首个拒绝原因；没有阻断时返回 None。
    """
    for hook in hooks.get(event, []):
        try:
            result = hook(*args)
        except Exception as exc:
            print(f"[{event}] Hook 异常：{type(exc).__name__}", file=sys.stderr)
            if event == "PreToolUse":
                return "操作未执行：执行前检查失败。"
            continue
        if event == "PreToolUse" and result is not None:
            return result
    return None


def run_tool(call, tools, hooks, *, on_result=None):
    """执行一次工具调用；错误和拒绝也作为工具结果回填。

    参数：
        call: 带 id、工具名和 JSON 参数的模型调用。
        tools: 工具名到函数的映射。
        hooks: 事件名到回调列表的映射。
        on_result: 可选的宿主结果登记函数，接收消息、工具名、参数与状态。

    返回值：
        带原始 tool_call_id 的 tool 消息。
    """
    name = call["function"]["name"]
    message = {"role": "tool", "tool_call_id": call["id"]}
    try:
        args = json.loads(call["function"]["arguments"])
        if not isinstance(args, dict):
            raise ValueError("参数必须是 JSON 对象")
        if name not in tools:
            raise ValueError("未知工具")
    except (TypeError, ValueError) as exc:
        return {**message, "content": f"工具失败：{exc}"}

    reason = run_hooks(hooks, "PreToolUse", name, args)
    if reason is not None:
        return {**message, "content": str(reason)}
    try:
        status, result = "ok", tools[name](**args)
    except (OSError, TypeError, ValueError) as exc:
        status, result = "error", f"工具失败：{exc}"

    run_hooks(hooks, "PostToolUse", name, status, result)
    message["content"] = result
    if on_result is not None:
        on_result(message, name, args, status)
    return message


def default_hooks(workspace, *, confirm=input, artifacts=None):
    """把路径约束、修改与删除审批、日志挂到生命周期事件上。

    参数：
        workspace: 文件工具可访问的工作区根目录。
        confirm: 读取审批结果的函数；默认读取终端输入。
        artifacts: 可选的当前会话归档存储，用于分页读取边界校验。

    返回值：
        默认事件和回调的映射。
    """

    def log(event, message):
        """把不含正文的事件信息打印到 stderr。

        参数：
            event: 生命周期事件名。
            message: 本次事件的日志元数据。
        """
        print(f"[{event}] {message}", file=sys.stderr)

    def permission_hook(name, args):
        """检查简历路径；修改和永久删除必须经用户明确同意。

        参数：
            name: 模型请求的工具名。
            args: 已解析的工具参数；此 Hook 不改写参数。

        返回值：
            拒绝原因；允许执行时返回 None。
        """
        if name == "read_tool_result":
            if artifacts is None:
                return "操作未执行：会话归档不可用。"
            try:
                artifacts.validate_read(**args)
            except (ValueError, TypeError, OSError):
                return "操作未执行：归档 ID、分页参数或文件不可用。"
            return None
        if name not in {"read_resume", "analyze_resume", "update_resume", "delete_resume"}:
            return "操作未执行：未知工具类别。"
        try:
            path = resume_path(workspace, args.get("path"))
        except ValueError as exc:
            return f"操作未执行：{exc}"
        if name == "update_resume":
            content = args.get("content")
            if not isinstance(content, str) or not content.strip():
                return "操作未执行：content 必须是非空正文。"
            if path.suffix.lower() == ".pdf":
                target = path.with_name(f"{path.stem}_new.md")
                action = f"从 {path.as_posix()!r} 生成 {target.as_posix()!r}（原 PDF 保留）"
            else:
                action = f"覆盖 {path.as_posix()!r}"
        elif name == "delete_resume":
            action = f"永久删除 {path.as_posix()!r}（不可恢复）"
        else:
            return None

        if confirm(f"允许{action}？[y/N] ").strip().lower() not in {"y", "yes", "是"}:
            return "操作未执行：用户拒绝。"
        return None

    return {
        "UserPromptSubmit": [lambda prompt: log("UserPromptSubmit", f"{len(prompt)} 字符")],
        "PreToolUse": [permission_hook, lambda name, args: log("PreToolUse", name)],
        "PostToolUse": [lambda name, status, result: log("PostToolUse", f"{name}: {status}，{len(result)} 字符")],
        "Stop": [lambda reason: log("Stop", reason)],
    }
