# MiniAgent s01：Agent Loop

这是 OfferLoop 的第一个学习模块：不依赖 LangChain / LangGraph，手写一个最小 Agent 循环，理解「Agent = Model + Harness」中 Model 边界和循环控制各自负责什么。

## 这一模块解决什么问题

- 消息（system / user / assistant）如何累积成上下文；
- 模型返回什么，如何判断本轮是否结束；
- 停止条件、最大步数保护、异常边界；
- 如何用 Provider 隔离模型供应商，让离线测试成为可能。

## 明确不做（留给后续模块）

- **s02**：工具调用、Tool Registry / Dispatcher；
- **s03 / s04**：权限、Hook；
- **s08 / s09**：上下文压缩、记忆；
- 流式输出、重试/退避、预算控制、持久化、结构化输出、RAG、FastAPI / 数据库 / 前端。

## 架构

```text
CLI / config
    ↓
MiniAgent（循环 + 历史 + 停止条件）
    ├── _on_assistant_turn   控制缝：继续 or 结束（s02 只改这里）
    └── LLMProvider          模型边界：上层只认 AssistantTurn
            ├── OpenAICompatible（DeepSeek 等兼容端点）
            └── Fake（离线脚本）
    ↓
models / errors（不可变数据 + 异常层级）
```

三条原则：

1. **一个控制缝**：`_on_assistant_turn` 决定这一步之后是否继续，s02 不需要改主循环。
2. **一个模型边界**：业务代码不接触 SDK 对象。
3. **历史只增不改**：消息 `frozen`，Provider 只拿到 tuple 快照。

## 目录

```text
mini_agent/
├── mini_agent/
│   ├── models.py            # Message / AssistantTurn / AgentResult / StopReason
│   ├── errors.py            # 异常层级
│   ├── config.py            # Settings 与 .env 加载
│   ├── agent.py             # Agent Loop 本体
│   ├── cli.py               # 单次模式与 REPL
│   └── providers/
│       ├── base.py          # LLMProvider Protocol
│       ├── openai_compat.py # 唯一接触 openai SDK 的文件
│       └── fake.py          # 离线脚本 Provider
└── tests/                   # 37 个离线测试
```

## 快速开始

```bash
cd mini_agent

python -m venv .venv
source .venv/Scripts/activate        # Git Bash
# PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install "openai>=1.0" "python-dotenv>=1.0" "pytest>=8.0"
```

### 离线模式（不需要 API key）

```bash
python -m mini_agent --provider fake "你好"
```

REPL：

```bash
python -m mini_agent --provider fake -v
```

可用命令：`/history`、`/reset`、`/help`、`/exit`。

### 真实模型

复制 `.env.example` 为 `.env`，填写 `MINI_AGENT_API_KEY`：

```dotenv
MINI_AGENT_PROVIDER=deepseek
MINI_AGENT_API_KEY=sk-你的key
MINI_AGENT_BASE_URL=https://api.deepseek.com
MINI_AGENT_MODEL=deepseek-chat
```

然后：

```bash
python -m mini_agent --provider deepseek "用一句话介绍你自己"
```

> `.env` 已被 `.gitignore` 忽略。程序不会打印 key，也不提供 `--api-key` 参数。

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MINI_AGENT_PROVIDER` | `deepseek` | `deepseek` 或 `fake` |
| `MINI_AGENT_API_KEY` | 无 | 真实模式下必填 |
| `MINI_AGENT_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容端点 |
| `MINI_AGENT_MODEL` | `deepseek-chat` | 模型名 |
| `MINI_AGENT_MAX_STEPS` | `8` | 单次 `run()` 的最大模型调用次数 |
| `MINI_AGENT_TEMPERATURE` | `0.3` | 采样温度，范围 0–2 |

优先级：进程环境变量 > `.env` > 代码默认值。

## 测试

```bash
python -m pytest -q
```

全部离线，不读取真实 key。覆盖：

- 单轮 / 多轮历史累积、system 不重复；
- `max_steps` 保护、空响应停止、`finish_reason=length`；
- Provider 异常传播、`tool_calls` 被明确拒绝；
- token 用量跨步汇总；
- FakeProvider 脚本顺序、回声兜底、异常注入；
- OpenAI 适配器的请求 payload、响应映射、超时/连接错误翻译；
- 配置优先级、BOM `.env`、缺 key 快速失败、CLI 冒烟。

## 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 正常结束 |
| 1 | 配置/用法错误 |
| 2 | 单次模式下的模型调用故障 |
| 130 | Ctrl+C |

## 验收状态

| # | 标准 | 状态 |
|---|---|---|
| A1 | `--provider fake "你好"` 正常输出，退出 0 | ✅ |
| A2 | `pytest` 全绿，离线 | ✅ 37 passed |
| A3 | fake REPL 两轮后历史完整累积 | ✅ |
| A4 | 真实 API 至少 3 轮对话 | ✅ |
| A5 | 缺 key 在请求前快速失败 | ✅ |
| A6 | Provider 恰好调用 `max_steps` 次 | ✅ |
| A7 | 空响应立即停止、不重试 | ✅ |
| A8 | `tool_calls` 抛 `UnsupportedToolCallError` 并提示 s02 | ✅ |
| A9 | 历史不可变、外部列表不影响 Agent | ✅ |
| A10 | 依赖仅 openai / python-dotenv / pytest | ✅ |
| A11 | 关键设计有「为什么」注释与 README | ✅ |
| A12 | `compileall` 通过、中文无乱码 | ✅ |

## s02 接入点

| s01 位置 | s02 变化 |
|---|---|
| `providers/base.py::generate` | 增加 `tools` 关键字参数 |
| `models.py` | 新增 `ToolSpec` 与参数解析辅助 |
| `agent.py::_on_assistant_turn` | 解析参数、调用 Dispatcher、追加 `Message.tool(...)`、返回 `True` |
| `openai_compat.py` | 把 `ToolSpec` 映射为 function schema |
| 新增 `mini_agent/tools.py` | `ToolRegistry`、`ToolDispatcher`、三个示例工具 |
| `agent.py::run` | 循环结构不变 |

## 已知边界

- 同一个 `MiniAgent` 不支持并发 `run()`；s11 再引入任务管理。
- `deepseek-reasoner` 的正文可能在 `reasoning_content`，默认的 `deepseek-chat` 不受影响。
- s01 的 `max_steps` 在正常路径不会触发（没有工具），由测试子类验证，主要作为 s02 的护栏。
