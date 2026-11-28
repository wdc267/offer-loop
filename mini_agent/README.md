# MiniAgent

**一个命令行简历助手，支持通过对话读取、分析和修改简历，并在修改或删除文件前请求用户确认。**

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)
![Project version 0.1.0](https://img.shields.io/badge/project_version-0.1.0-informational)

MiniAgent 是 [OfferLoop](../README.md) 的学习用演示项目，不是完整的求职产品。它不依赖 Agent 框架，通过简历操作展示模型调用工具、宿主程序检查权限并回传结果的过程。离线模式无需 API key，可直接体验；接入真实模型后支持自然语言对话。

## 目录

- [功能特性](#功能特性)
- [环境与安装](#环境与安装)
- [快速开始](#快速开始)
- [使用与配置](#使用与配置)
- [安全与限制](#安全与限制)
- [开发与贡献](#开发与贡献)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 功能特性

- **简历读取**：支持 UTF-8 Markdown、TXT；安装可选依赖后可提取 PDF 文本层。
- **规则分析**：在本地检查常见中文章节、联系方式是否存在，以及量化表述数量。
- **简历修改**：覆盖 Markdown / TXT；PDF 则生成新的 `_new.md` 文件，保留原件。
- **操作审批**：修改和永久删除前请求确认；拒绝原因也会回传给模型。
- **多轮对话**：在当前会话内保留历史，可结合上一轮继续提问。
- **上下文管理**：大读取结果保存为归档并支持分页；真实模型模式可摘要较早的对话，控制请求大小。

> 离线模式仅识别固定演示指令，不会理解任意自然语言，也不会调用模型生成摘要。

## 环境与安装

需要 **Python 3.12+**。以下命令从已获取的 OfferLoop 仓库根目录开始；如果已在 `mini_agent/`，无需再次 `cd`。

**只体验离线模式：** 无需安装任何依赖，直接进入下一节。

**需要真实模型、PDF 或开发测试：** 建议先创建虚拟环境。

```bash
cd mini_agent
python -m venv .venv
```

激活环境，按终端选择一条命令：

| 终端 | 命令 |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Git Bash | `source .venv/Scripts/activate` |
| macOS / Linux Bash、Zsh | `source .venv/bin/activate` |

然后在 `mini_agent/` 中按需安装：

```bash
python -m pip install -e ".[live,pdf]"  # 真实模型 + PDF 读取
# 只需模型接入用 ".[live]"；只需 PDF 读取用 ".[pdf]"
```

以上是源码安装方式；激活命令按各平台惯例列出，不代表已完成跨平台验证。

## 快速开始

### 1. 启动离线对话

从仓库根目录执行：

```bash
cd mini_agent
python -m mini_agent
```

程序会在仓库根目录准备 `workspace/`，仅当 `workspace/demo_resume.md` 不存在时创建虚构示例，不覆盖已有文件。

启动输出示意，绝对路径和会话 ID 因环境而异：

```text
简历工作区：<仓库根目录>/workspace
离线指令：读简历、分析简历、修改简历、删除简历、上一条指令。
归档位置：<仓库根目录>/.mini_agent/sessions/<会话ID>（大结果和压缩历史将以明文保存，不自动清理）
输入 /exit 退出。
mini_agent >>
```

### 2. 读取、分析并体验审批

在提示符后依次输入：

```text
读简历
上一条指令
分析简历
修改简历
```

预期行为：

- `读简历`：显示示例简历正文。
- `上一条指令`：回答 `上一条指令：读简历`。
- `分析简历`：返回章节、邮箱、手机和量化表述检查结果。
- `修改简历`：询问是否允许覆盖 `demo_resume.md`；**输入 `n` 体验拒绝，文件不会被修改**。

终端同时会显示工具调用等事件日志。若改为输入 `y`，会用固定的演示正文真实覆盖示例文件，并非智能润色；演示中的经历和数字均为虚构。

### 3. 退出

输入 `/exit`，也可使用 `exit`、`q` 或 `Ctrl+C`。工作区与已写入的归档不会随退出删除。

## 使用与配置

### 接入真实模型

先在 `mini_agent/` 安装 `live` 可选依赖。**真实模式会将输入和工具结果（包括读取到的简历正文）发送给配置的模型端点；建议只使用脱敏副本。**

配置当前终端的 API key，按终端选择一种写法，并将占位符换成自己的 key：

```bash
# Bash / Zsh / Git Bash
export MINI_AGENT_API_KEY='your-api-key'
```

```powershell
# Windows PowerShell
$env:MINI_AGENT_API_KEY = 'your-api-key'
```

然后在 `mini_agent/` 执行：

```bash
python -m mini_agent --live
```

把简历放在仓库根目录 `workspace/` 中，在对话中使用相对于工作区的路径，例如：

```text
读取 resume.pdf
分析刚才那份简历
根据分析建议修改简历，不要虚构经历或成果
```

修改前会请求确认；对 Markdown / TXT 的确认意味着允许覆盖全文。PDF 修改会生成 `resume_new.md`，若该输出已存在则拒绝覆盖。

### 配置项

真实模式从环境变量和 `mini_agent/.env` 读取配置；已有环境变量优先。不要将真实密钥提交到 Git。

| 环境变量 | 是否必填 | 默认值 / 用途 |
|---|---|---|
| `MINI_AGENT_API_KEY` | 真实模式必填 | 模型服务 API key |
| `MINI_AGENT_BASE_URL` | 否 | `https://api.deepseek.com`，OpenAI 兼容端点 |
| `MINI_AGENT_MODEL` | 否 | `deepseek-chat`，需支持工具调用 |

### 命令行参数

以下命令均在 `mini_agent/` 执行：

```bash
python -m mini_agent --help
python -m mini_agent "读简历"                       # 离线单次运行
python -m mini_agent --live "读取 resume.md"        # 真实模型单次运行
python -m mini_agent --live --context-chars 10000  # 调整请求大小预算
```

| 参数 | 说明 |
|---|---|
| `--live` | 使用真实模型；省略则运行离线演示 |
| `--context-chars` | 消息与工具定义的 JSON 字符预算，默认 `50000`，必须为正整数；不是 token 上限 |
| `prompt` | 可选的单条指令；省略则进入多轮对话。单次模式也可能请求操作确认 |

### 工具与文件行为

工具由模型请求调用，不是终端命令。

| 工具 | 行为 | 需要确认 |
|---|---|---|
| `read_resume` | 读取 `.md`、`.txt` 或有文本层的 `.pdf` | 否 |
| `analyze_resume` | 本地规则检查，不等于全面的简历质量评估 | 否 |
| `update_resume` | 覆盖文本简历，或从 PDF 生成 `_new.md` | 是 |
| `delete_resume` | 永久删除指定简历，不进入回收站 | 是 |
| `read_tool_result` | 按归档 ID 分页读取当前会话已登记结果，不接受任意文件路径 | 否 |

## 安全与限制

- **文件会真实变化**：修改和删除均作用于工作区文件，请使用副本。离线模式只操作 `demo_resume.md`，真实模式可操作工作区内支持格式的文件。
- **归档是明文**：大结果与压缩前历史保存在仓库根 `.mini_agent/sessions/`，可能包含完整简历和提示词，不自动清理。程序启动时显示具体位置。
- **日志不等于脱敏输出**：事件日志仅记录元数据，但终端工具答复和磁盘归档仍可能包含正文。
- **路径检查不是 OS 沙箱**：简历工具限制在工作区内；归档读取校验会话 ID 与路径边界，拒绝符号链接、Windows junction 和解析重定向。不要据此运行不可信代码。
- **暂无会话恢复**：退出后文件保留，但不支持重启恢复对话或跨会话读取归档。
- **范围有限**：不包含 shell 工具、DOCX、OCR、Web UI、持久化用户记忆或知识库检索。当前已实现上下文压缩，持久化记忆仍是后续计划。

<details>
<summary>上下文压缩与停止策略（进阶）</summary>

- 成功读取结果超过 12,000 字符时，原文保存至 `outputs/<id>.txt`，模型先收到 ID 与前后预览；正文预览合计默认 2000 字符。分页偏移按字符计，默认每页 4000、最大 8000 字符。
- 仍超预算时先保存 `transcripts/<快照ID>.json`，再在真实模式独立请求同一模型生成无工具摘要；候选历史验证成功才替换工作历史。摘要是参考消息，不是系统指令或已确认事实。
- 保留 system、当前用户原文、最新闭合工具批次和本轮非只读批次。摘要失败时，只在归档成功后裁剪完整旧轮次；必要内容仍放不下则停止本轮，交互会话继续等待输入。
- 摘要单次输入含结构最多 40,000 字符、输出最多 6000 字符；同一次逻辑执行请求最多 4 次摘要调用，累计输入最多 160,000 字符。离线模式只能归档、整理和裁剪，不会生成模型摘要。
- 普通压缩不足，或服务端明确返回 `context_length_exceeded` 时，共享一次更强压缩机会；重试不重放工具、不重复审批、不重置步数。失败不会回滚已执行的操作。

</details>

## 开发与贡献

在 `mini_agent/` 中安装测试依赖并运行：

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

测试不访问真实模型端点，覆盖多轮对话、审批与拒绝、Hook 顺序、路径边界，以及归档、上下文预算、摘要替身和超限重试。只检查上下文相关行为可运行：

```bash
python -m pytest tests/test_artifacts.py tests/test_context.py tests/test_context_edges.py tests/test_compact_cli.py
```

### 代码入口

| 文件 | 职责 |
|---|---|
| [`__main__.py`](mini_agent/__main__.py) | CLI、离线模型、真实模型接入与多轮输入循环 |
| [`agent.py`](mini_agent/agent.py) | 单轮模型与工具调用循环 |
| [`harness.py`](mini_agent/harness.py) | 工具执行前检查、结果回填与生命周期 Hook |
| [`tools.py`](mini_agent/tools.py) | 简历工具、路径约束与工具定义 |
| [`context.py`](mini_agent/context.py) | 请求计量、上下文整理与压缩 |
| [`artifacts.py`](mini_agent/artifacts.py) | 大结果、历史快照与归档分页 |
| [`tests/`](tests/) | 离线测试与模型替身 |

Hook 包括 `UserPromptSubmit`、`PreToolUse`、`PostToolUse` 和 `Stop`。执行前检查拒绝时工具不会执行，也不触发 `PostToolUse`；结果或拒绝原因均按原工具调用 ID 回传给模型。

### 提交问题与改进

- 在 [Issues](https://github.com/wdc267/offer-loop/issues) 提供复现步骤、运行环境和脱敏错误输出，不附带真实简历或密钥。
- 提交 PR 前说明修改范围，补充相关测试并运行上述测试命令。
- 遵循现有注释规范：每个函数和方法写 docstring，按需说明参数、返回值与业务异常；每个 `test_*` 函数用 1–2 句中文描述所测行为与预期结果。

## 常见问题

**为什么离线模式不理解我的问题？**

离线模型是固定规则替身，只识别快速开始中列出的五条指令。自由对话需要安装 `live` 依赖并使用 `--live`。

**为什么 PDF 读不出来？**

先确认已安装 `pdf` 依赖。仅支持可提取文本层的 PDF，不支持扫描图片 OCR，也不保证保留原版面。

**退出后数据在哪里？能恢复对话吗？**

简历在仓库根 `workspace/`，归档在 `.mini_agent/sessions/`；位置不随命令启动目录改变。文件会保留，但归档不是可恢复会话，重启后不会加载旧对话。

## 许可证

仓库尚未提供项目许可证。请勿将公开可见理解为已获自由使用、修改或再分发授权；相关授权需向维护者确认。
