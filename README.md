# OfferLoop

> 面向计算机类应届毕业生的 AI 求职与智能面试教练  
> 当前阶段：**Phase 1 — 从 0 到 1 手写 MiniAgent**

OfferLoop 的目标不是做一个功能堆砌的 AI 求职平台，而是完成一条可验证的闭环：

```text
简历 → 结构化事实 → 岗位匹配 → 个性化面试题
     → 模拟面试（文本 / 语音）→ 动态追问
     → 证据化评估 → 薄弱点训练 → 下一轮面试
```

产品最终形态：用户上传简历、填写目标岗位 JD，系统分析匹配度、生成个性化面试题，通过语音模拟真实面试，并基于真实回答生成有证据支撑的评估报告和训练建议。

当前仓库先从 **MiniAgent** 做起：不依赖 LangChain / LangGraph / Agent SDK，自己实现 Agent Loop、工具调用、权限控制等 Harness 机制，为后续 OfferLoop 业务开发打基础。

---

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | MiniAgent：Agent Loop + Tool Use + Permission | ✅ M1~M3 已完成 |
| Phase 1 | Hooks & Sandbox、Context、Memory、Task、Integrated Harness | ⏳ 进行中 |
| Phase 2 | OfferLoop 业务产品 | 待启动 |

当前测试基线：**125 个离线测试全部通过**（2026-11-24）。

---

## 已实现能力

### Agent Loop

- 单轮 / 多轮对话，完整消息历史（system / user / assistant / tool）；
- 正常结束、空响应、达到最大步数三种停止原因；
- `max_steps` 保护，避免工具循环失控；
- 模型调用异常统一翻译为领域异常。

### Tool Use

当前提供 5 个业务工具：

| 工具 | 作用 | 需要用户确认 |
|---|---|---|
| `read_resume` | 读取 PDF / DOCX / Markdown / TXT 简历 | 否 |
| `read_jd` | 读取岗位 JD 文件 | 否 |
| `analyze_resume` | 规则版简历结构分析 | 否 |
| `update_resume` | 写入修改后的完整简历 | 是 |
| `delete_resume` | 永久删除简历文件 | 是 |

工具失败不会打断 Agent 循环：错误会转成 `tool` 消息回填给模型，让模型自行纠正或告知用户。

### Permission

执行工具前依次通过三道门：

```text
拒绝表（危险命令直接拒绝）
  → 确认规则表（越界 / 修改 / 删除需要确认）
  → 用户确认（终端询问，y / yes / 是 才放行）
```

- 修改 / 删除简历每次都需要确认；
- 确认提示只展示目标路径、写入模式和正文预览，不打印完整简历；
- 用户拒绝后文件不变，模型收到拒绝原因并继续对话。

### 模型与文档

- 统一 `LLMProvider` 接口，当前支持：
  - `OpenAICompatibleProvider`：DeepSeek / 通义千问 / 智谱等 OpenAI 兼容端点；
  - `FakeProvider`：离线脚本化模型，用于测试和演示。
- 文档读取支持 PDF、DOCX、Markdown、TXT，包含编码回退、大小限制、截断和扫描件检测。

---

## 目录结构

```text
offer-loop/
├── README.md                 # 本文件
├── CLAUDE.md                 # 项目开发约定（本地）
├── mini_agent/               # Phase 1：MiniAgent
│   ├── mini_agent/
│   │   ├── agent.py          # Agent Loop
│   │   ├── tools.py          # 工具、注册表、分发器
│   │   ├── permissions.py    # 权限门
│   │   ├── documents.py      # 文档读取
│   │   ├── models.py         # 领域模型
│   │   ├── errors.py         # 异常层级
│   │   ├── config.py         # 配置
│   │   ├── cli.py            # CLI / REPL
│   │   └── providers/        # LLM Provider
│   ├── tests/                # 125 个离线测试
│   ├── pyproject.toml
│   └── README.md             # MiniAgent 详细说明
├── docs/                     # 本地设计文档（当前不纳入 git）
└── 参考文档/                 # 本地参考资料（不纳入 git）
```

---

## 快速开始

### 环境要求

- Python 3.12+
- 可选：真实模型 API key（默认 DeepSeek，任意 OpenAI 兼容端点均可）

### 安装

```bash
cd mini_agent

python -m venv .venv
source .venv/Scripts/activate        # Git Bash
# PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

### 离线运行（不需要 API key）

```bash
python -m mini_agent --provider fake "你好"
```

进入 REPL：

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

读取并对比简历 / JD：

```bash
python -m mini_agent "帮我看看 resume.pdf 和 jd.txt 的匹配度"
```

修改简历（写入前会请求确认）：

```bash
python -m mini_agent "读一下 resume.md，按分析结果优化后再写回"
```

> `.env` 已被 `.gitignore` 忽略。程序不会打印 key，也不提供 `--api-key` 参数。

### 运行测试

```bash
cd mini_agent
python -m pytest
```

全部测试离线执行，不读取真实 API key。

---

## 设计原则

1. **Agent = Model + Harness**：模型负责推理，Harness 提供工具、权限、上下文、状态和执行环境；
2. **一个控制缝**：主循环只负责调模型、记历史、判断是否继续；工具执行通过 `_on_assistant_turn` 接入；
3. **一个模型边界**：业务代码只依赖 `LLMProvider`，不接触任何 SDK 对象；
4. **历史只增不改**：消息不可变，Provider 只拿到快照；
5. **工具失败不打断循环**：异常转成 `tool` 消息，让模型自我纠正；
6. **危险操作先问再做**：修改 / 删除默认需要确认，拒绝表命中直接阻断；
7. **离线可验证**：核心行为都能用 Fake Provider 和临时文件测试；
8. **隐私优先**：日志和历史展示不泄露简历 / JD 正文。

---

## 开发路线

### Phase 1：MiniAgent

```text
M1 Agent Loop        ✅
M2 Tool Use          ✅
M3 Permission        ✅
M4 Hooks & Sandbox   ⏳ 下一步
M5 Context Compact
M6 Memory
M7 Task & State
M8 Integrated Harness
M9 Workflow Runtime  （可选）
M10 Goal Loop        （可选）
```

阶段一完成后应能：

- 从零解释并实现 Agent Loop、Tool Use、Permission、Hook、Context、Memory、State；
- 用测试证明每个机制有效；
- 完成「读简历 + 读 JD + 对比」「修改简历被拒绝」「多步骤任务中断恢复」等端到端场景；
- 把可复用的设计带入 Phase 2。

### Phase 2：OfferLoop

```text
O0 脚手架与测试基线
O1 简历上传与结构化
O2 JD 分析与事实约束
O3 个性化面试题生成
O4 文本模拟面试与状态机
O5 语音问答（Turn-based）
O6 证据化评估与报告
O7 能力画像与训练闭环
O8 工程化与可观测性
O9 打磨与交付
```

目标技术栈：

- 前端：Vue 3 + TypeScript + Vite + Pinia + Element Plus；
- 后端：Python + FastAPI + Pydantic + SQLAlchemy + Alembic；
- 数据库：开发 SQLite，目标 PostgreSQL；
- LLM：统一 Provider，默认 DeepSeek；
- 语音：云端 STT / TTS，Turn-based，失败降级文本；
- 部署：Docker Compose；
- 暂不引入：Redis、MongoDB、Kafka、LangGraph、向量数据库、MCP。

---

## 文档说明

- `docs/` 目录包含 PRD、技术架构、Roadmap 和开发规则，目前**只在本地保留，不纳入 git**；
- `参考文档/` 为本地参考资料，同样不纳入 git；
- 开发规则要求 **先写单元测试再写代码**，并且每个功能至少有一条覆盖端到端输入输出的测试；
- MiniAgent 的详细使用说明见 [`mini_agent/README.md`](mini_agent/README.md)。

---

## 已知限制

- 同一个 `MiniAgent` 不支持并发 `run()`；
- 工具执行暂无工具级超时；
- DOCX 文本框内容无法提取，扫描件 PDF 需要 OCR；
- PDF 多栏排版的阅读顺序可能不理想；
- 长文档会截断到 50k 字符，上下文压缩尚未实现；
- 写回 DOCX 只保留纯文本段落，原样式、图片、表格会丢失；
- 备份和 `_new` 文件不会自动清理；
- Phase 1 暂不提供 Web UI、数据库、持久化会话。

---

## 状态

项目正在按模块推进：一次只做一个模块，先给设计说明和验收标准，确认后再写测试和实现。

当前下一步：**M4 Hooks & Sandbox**。
