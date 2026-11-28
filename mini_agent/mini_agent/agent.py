"""模型提出工具调用；Harness 执行并回填结果。"""

import copy

from .context import CONTEXT_CHAR_LIMIT, ContextBudgetExceeded, ContextManager, ContextOverflow
from .harness import run_hooks, run_tool
from .tools import TOOL_SCHEMAS


def run_turn(prompt, messages, model, tools, hooks, *, max_steps=8,
             context=None, context_chars=CONTEXT_CHAR_LIMIT):
    """运行一轮对话，直到模型不再请求工具或达到步数上限。

    参数：
        prompt: 本轮用户输入。
        messages: 对话历史；原地追加用户、模型和工具消息。
        model: 接收消息列表并返回下一条 assistant 消息的函数。
        tools: 工具名到函数的映射。
        hooks: 事件名到回调列表的映射。
        max_steps: 本轮最多成功调用执行模型的次数。
        context: 可选的会话压缩管理器。
        context_chars: 未传管理器时使用的字符预算。

    返回值：
        模型最终答复，或达到步数上限时的提示。
    """
    context = context or ContextManager(schemas=TOOL_SCHEMAS, limit=context_chars)
    run_hooks(hooks, "UserPromptSubmit", prompt)
    active = {"role": "user", "content": prompt}
    messages.append(active)

    for _ in range(max_steps):
        try:
            forced = context.prepare(messages, active)
            while True:
                try:
                    reply = model(copy.deepcopy(messages))
                    break
                except ContextOverflow as exc:
                    if forced:
                        raise ContextBudgetExceeded("重试仍被模型端判定为上下文超限") from exc
                    context.prepare(messages, active, force=True, continuation=True)
                    forced = True # 记录已经用过强制压缩
        except ContextBudgetExceeded as exc:
            run_hooks(hooks, "Stop", "上下文预算不足")
            return (f"本轮停止：上下文预算不足（{exc}）。"
                    "已返回的工具操作不会回滚，未自动重放；请先核查结果。")
        context.mark_seen(messages)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            run_hooks(hooks, "Stop", "完成")
            return reply.get("content") or "（模型没有返回文本）"
        for call in calls:
            messages.append(run_tool(call, tools, hooks, on_result=context.register_result))

    run_hooks(hooks, "Stop", "达到步数上限")
    return "本轮达到步数上限。"
