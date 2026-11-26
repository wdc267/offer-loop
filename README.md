# OfferLoop

> 面向计算机类应届毕业生的 AI 求职与智能面试教练  
> 当前阶段：**Phase 1 — 用 MiniAgent 学习 Harness**

OfferLoop 的长期目标是一条可验证的求职训练闭环：

```text
简历 → 结构化事实 → 岗位匹配 → 个性化面试题
     → 模拟面试（文本 / 语音）→ 动态追问
     → 证据化评估 → 薄弱点训练 → 下一轮面试
```

当前仓库实现的是学习用的 **MiniAgent Harness demo**，不是上述业务产品。它不依赖 LangChain、LangGraph 或 Agent SDK，重点是看懂一条调用链：**模型提出工具调用 → Harness 记录和审批 → 工具执行或被阻止 → 结果回填给模型**。

## 快速体验

```bash
cd mini_agent
python -m mini_agent
```

仅需 Python 3.12+。在同一个会话里依次输入 `读简历`、`上一条指令`、`分析简历`、`修改简历`，体验历史复用和 Hook 审批。也可以输入 `删除简历`，修改和删除都需确认。离线和真实模式共用仓库根目录下的 `workspace/`；离线只操作其中的 `demo_resume.md`，文件在输入 `/exit` 后仍会保留。整个工作区默认被 Git 忽略。

学习顺序、真实模型接入和离线测试见 [`mini_agent/README.md`](mini_agent/README.md)。简历工具支持 Markdown、TXT 和可选的 PDF 文本提取，不包含 DOCX、shell 工具、数据库或 Web UI。

## 开发路线

### Phase 1：MiniAgent

- M1–M4：Agent Loop、Tool Use、Permission、Hooks 与文件路径约束，已在 demo 中演示。
- 下一步：M5 Context Compact。再往后学习 Memory、Task & State、Integrated Harness；Workflow Runtime、Goal Loop 可选。

### Phase 2：OfferLoop（未启动）

1. 简历上传与结构化、JD 分析与事实约束；
2. 个性化面试题生成、文本模拟面试与状态机；
3. 语音问答（Turn-based）、证据化评估与报告；
4. 能力画像与训练闭环、可观测性、部署与交付。

目标技术栈：Vue 3 + TypeScript + Vite + Pinia + Element Plus；Python + FastAPI + Pydantic + SQLAlchemy + Alembic；开发用 SQLite，目标 PostgreSQL；模型默认 DeepSeek，语音用云端 STT / TTS，Docker Compose 部署。暂不引入 Redis、MongoDB、LangGraph、向量数据库或 MCP。

`docs/` 和 `参考文档/` 是本地设计与参考资料，不纳入 Git。项目按模块推进：先定设计和验收标准，确认后再写测试和实现。
