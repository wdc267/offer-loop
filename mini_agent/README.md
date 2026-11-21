# MiniAgent s02：Agent Loop + Tool Use

这是 OfferLoop 的第二个学习模块：在 s01 手写 Agent Loop 的基础上加入工具调用，理解「模型发起调用 → Harness 执行 → 结果回填 → 模型继续」的完整闭环。

## 这一模块解决什么问题

- 工具如何以 JSON Schema 声明给模型；
- 模型返回 `tool_calls` 后，如何解析参数、执行并回填 tool 消息；
- 工具失败（文件不存在、格式不支持、参数错误）如何不打断 Agent 循环；
- 一轮多个工具调用如何逐个配平；
- 文档读取（PDF / DOCX / Markdown / TXT）如何统一成纯文本。

## 工具

| 工具 | 作用 |
|---|---|
| `read_resume` | 读取简历文件，返回纯文本 |
| `read_jd` | 读取岗位 JD 文件，返回纯文本 |
| `analyze_resume` | 规则版简历结构分析：联系方式、章节、量化指标、日期区间、字数 |

`analyze_resume` 只陈述规则可检测的事实，不做主观评价；「是否匹配 JD」由模型结合原文判断。工具内不调用模型，后续 s10 再引入子 Agent。

支持的文档格式：

| 格式 | 解析方式 | 说明 |
|---|---|---|
| `.pdf` | pypdf | 扫描件（无文本层）会明确报错，暂不支持 OCR |
| `.docx` | python-docx | 读取段落、表格、页眉页脚；不支持旧版 `.doc` |
| `.md` / `.markdown` | 编码回退读取 | — |
| `.txt` | 编码回退读取 | `utf-8-sig` → `utf-8` → `gb18030` |

限制：单文件 ≤ 10 MB，正文 ≤ 50,000 字符（超出截断并标注）。日志只记录工具名和结果长度，不记录正文。

## 明确不做（留给后续模块）

- **s03 / s04**：权限、沙箱、Hook；
- **s08 / s09**：上下文压缩、记忆；
- **s10**：工具内调用 LLM 的子 Agent；
- OCR / 扫描件识别、旧版 `.doc`；
- 流式输出、重试/退避、预算控制、持久化、RAG、FastAPI / 数据库 / 前端。

## 架构

```text
CLI / config
    ↓
MiniAgent（循环 + 历史 + 停止条件）
    ├── _on_assistant_turn   工具分发入口
    │       └── ToolDispatcher → ToolRegistry → Tool
    ├── tools.py             三个业务工具 + 注册表 + 分发器
    ├── documents.py         PDF / DOCX / MD / TXT → 纯文本
    └── LLMProvider          模型边界：上层只认 AssistantTurn
            ├── OpenAICompatible（DeepSeek 等兼容端点）
            └── Fake（离线脚本）
    ↓
models / errors（不可变数据 + 异常层级）
```

四条原则：

1. **一个控制缝**：`_on_assistant_turn` 决定这一步之后是否继续，主循环不感知具体工具。
2. **一个模型边界**：业务代码不接触 SDK 对象。
3. **历史只增不改**：消息 `frozen`，Provider 只拿到 tuple 快照。
4. **工具失败不打断循环**：Dispatcher 把异常转成 tool 消息，让模型自我纠正。

## 目录

```text
mini_agent/
├── mini_agent/
│   ├── models.py            # Message / AssistantTurn / ToolCall / ToolSpec ...
│   ├── errors.py            # 异常层级（ToolError / DocumentError）
│   ├── config.py            # Settings 与 .env 加载
│   ├── documents.py         # 文档加载与文本规范化
│   ├── tools.py             # 工具、注册表、分发器、规则分析
│   ├── agent.py             # Agent Loop + 工具分发
│   ├── cli.py               # 单次模式与 REPL
│   └── providers/
│       ├── base.py          # LLMProvider Protocol
│       ├── openai_compat.py # 唯一接触 openai SDK 的文件
│       └── fake.py          # 离线脚本 Provider
└── tests/                   # 80 个离线测试
```

## 快速开始

```bash
cd mini_agent

python -m venv .venv
source .venv/Scripts/activate        # Git Bash
# PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 离线模式（不需要 API key）

```bash
python -m mini_agent --provider fake "你好"
```

REPL：

```bash
python -m mini_agent --provider fake -v
```

可用命令：`/history`、`/reset`、`/help`、`/exit`。`/history` 对 tool 消息只显示字符数，不打印简历 / JD 正文。

### 真实模型

复制 `.env.example` 为 `.env`，填写 `MINI_AGENT_API_KEY`：

```dotenv
MINI_AGENT_PROVIDER=deepseek
MINI_AGENT_API_KEY=sk-你的key
MINI_AGENT_BASE_URL=https://api.deepseek.com
MINI_AGENT_MODEL=deepseek-chat
```

读取并对比简历 / JD：

```bash
python -m mini_agent "帮我看看 resume.pdf 和 jd.txt 的匹配度"
```

模型会自动调用 `read_resume`、`read_jd`，需要结构事实时调用 `analyze_resume`。

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
- 工具调用闭环：参数解析、执行、tool 消息回填、多调用配平；
- 未知工具、非法 JSON、工具异常、未注册工具时的错误回填；
- TXT / Markdown / DOCX / PDF 读取、编码回退、扫描件拦截、大小与截断；
- 规则版简历分析；
- OpenAI 适配器的请求 payload（含 tools schema）、响应映射、错误翻译；
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
| S1 | `--provider fake "你好"` 正常输出，退出 0 | ✅ |
| S2 | `pytest` 全绿，离线 | ✅ 80 passed |
| S3 | fake 脚本驱动 `tool_call → tool 消息 → 最终回答` | ✅ |
| S4 | TXT / MD / DOCX / PDF 四类文档读取 | ✅ |
| S5 | 扫描件 PDF、文件不存在、扩展名不支持均回填错误 | ✅ |
| S6 | 未知工具 / 非法 JSON 不打断循环 | ✅ |
| S7 | 一轮多个 `tool_calls` 全部配平 | ✅ |
| S8 | 工具反复调用时 `max_steps` 生效 | ✅ |
| S9 | 日志与 `/history` 不泄露简历 / JD 正文 | ✅ |
| S10 | `compileall` 通过、中文无乱码 | ✅ |
| S11 | 真实 API 完成「读简历 + 读 JD + 对比」 | ⏳ 待真实 key 验收 |

## s03 接入点

| s02 位置 | s03 变化 |
|---|---|
| `tools.py::ToolDispatcher` | 执行前增加权限检查（允许 / 询问 / 拒绝） |
| `tools.py::Tool` | 增加 `needs_permission` 或风险等级声明 |
| `documents.py::load_document` | 增加允许目录（工作区根）限制 |
| `cli.py` | 增加交互式授权提示 |

## 已知边界

- 同一个 `MiniAgent` 不支持并发 `run()`；s11 再引入任务管理。
- 本地文件读取在 `asyncio.to_thread` 中执行，但没有工具级超时。
- DOCX 文本框里的文字无法提取（python-docx 限制）；扫描件 PDF 需要 OCR。
- PDF 多栏排版的阅读顺序可能不理想。
- 工具结果直接进入上下文；单份文档已截断到 50k 字符，长文档压缩留给 s08。
