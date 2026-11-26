"""模型提出工具调用；Harness 执行并回填结果。"""

from .harness import run_hooks, run_tool


def run_turn(prompt, messages, model, tools, hooks, *, max_steps=8):
    """运行一轮对话，直到模型不再请求工具或达到步数上限。

    参数：
        prompt: 本轮用户输入。
        messages: 对话历史；原地追加用户、模型和工具消息。
        model: 接收消息列表并返回下一条 assistant 消息的函数。
        tools: 工具名到函数的映射。
        hooks: 事件名到回调列表的映射。
        max_steps: 本轮最多调用模型的次数。

    返回值：
        模型最终答复，或达到步数上限时的提示。
    """
    run_hooks(hooks, "UserPromptSubmit", prompt)
    messages.append({"role": "user", "content": prompt})

    for _ in range(max_steps):
        reply = model(messages)
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            run_hooks(hooks, "Stop", "完成")
            return reply.get("content") or "（模型没有返回文本）"
        for call in calls:
            messages.append(run_tool(call, tools, hooks))

    run_hooks(hooks, "Stop", "达到步数上限")
    return "本轮达到步数上限。"
