# MiniAgent：简历 Harness Demo

**模型提出工具调用；Harness 决定能不能执行，并把结果送回模型。** 这个教学 demo 对应 `参考文档/code.py` 的 s04 Hooks。不用 Agent 框架，不包含 shell 工具、DOCX 或会话历史持久化。

## 先跑离线多轮对话

在仓库根目录执行（Python 3.12+，不安装依赖、不用 API key）：

```bash
cd mini_agent
python -m mini_agent
```

依次输入 `读简历`、`上一条指令`、`分析简历`、`修改简历`。最后一条会请求确认，输入 `n` 看拒绝，再试一次输入 `y` 看修改。也可以试 `删除简历`，输入 `/exit` 结束会话。离线模型只识别这些固定指令，且只操作 `workspace/demo_resume.md`。启动时文件不存在才创建虚构示例，不覆盖已有内容。

离线和真实模式共用 **OfferLoop 仓库根目录下的 `workspace/`**，首次运行自动创建，不随命令的启动目录变化。工作区文件在退出后保留，Git 默认忽略整个目录。

## 按这个顺序读代码

1. [`__main__.py`](mini_agent/__main__.py)：`run_session()` 是外层输入循环。历史、工具和 Hook 只创建一次，每轮复用。
2. [`agent.py`](mini_agent/agent.py)：`run_turn()` 是内层模型循环。模型请求工具时，把调用交给 Harness 并追加结果；没有工具调用时结束本轮。最多调用模型 8 次。
3. [`harness.py`](mini_agent/harness.py)：`run_tool()` 实现「检查 → 执行 → 回填」。审批和日志挂在 Hook 上，新增观察回调不需要改内层循环。
4. [`tools.py`](mini_agent/tools.py)：四个简历工具和 JSON Schema。工具自己校验相对路径是否仍在工作区内，这不是系统级沙箱。

| 事件 | 触发时机 | 默认行为 |
|---|---|---|
| `UserPromptSubmit` | 每轮输入进入历史前 | 记录输入长度 |
| `PreToolUse` | 解析工具参数之后、执行之前 | 先检查路径并审批，再记录允许执行的工具名 |
| `PostToolUse` | 工具执行或抛出工具错误之后 | 记录状态和结果长度，不打印简历正文 |
| `Stop` | 模型结束本轮或达到步数上限 | 记录停止原因；REPL 继续等待输入 |

`PreToolUse` 返回非 `None` 或检查抛异常时，工具不执行，也不触发 `PostToolUse`。拒绝原因与执行结果一样，按原始工具调用 ID 回填给模型。

## 简历工具

| 工具 | 行为 | 确认 |
|---|---|---|
| `read_resume` | 读取 UTF-8 Markdown / TXT，或提取 PDF 文本层（需 `pdf` 依赖，不支持 OCR） | 否 |
| `analyze_resume` | 本地规则检查：常见中文章节、联系方式是否存在、量化表述数量 | 否 |
| `update_resume` | 用完整正文覆盖 Markdown / TXT；PDF 则生成 `_new.md`，保留原件，不覆盖已有输出 | 是 |
| `delete_resume` | 永久删除工作区内指定的简历 | 是 |

## 用真实模型多轮对话

在 `mini_agent/` 目录执行：

```bash
python -m pip install -e ".[live,pdf]"
python -m mini_agent --live
```

从环境变量或模块 `.env` 读取 `MINI_AGENT_API_KEY`。`MINI_AGENT_BASE_URL` 和 `MINI_AGENT_MODEL` 可覆盖 DeepSeek 默认值。不读 PDF 时只需要 `.[live]`。

把简历放在仓库根目录的 `workspace/`，在对话中使用相对于工作区的路径，例如「读取 `resume.pdf`」，再问「分析刚才那份简历」。也可在命令末尾加一个带引号的指令，单次运行后退出。**真实模式下，输入和工具结果（含读取到的简历正文）会发送给配置的模型端点。** 建议使用脱敏副本；修改、删除会真实影响工作区文件。

## 离线测试

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

测试不访问模型端点，覆盖多轮历史、Hook 顺序、许可与拒绝、越界拦截和四个简历工具。
