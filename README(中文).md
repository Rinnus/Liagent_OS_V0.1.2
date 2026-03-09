# LiAgent OS 中文说明

[返回首页](README.md) | [English README](<README(EN).md>)

> 让 AI 不止会回答，还能在你的边界内持续工作。

LiAgent OS 是一个面向本地模型与本地/云端混合模型服务的本地优先智能体运行时。它把对话、工具调用、多智能体编排、任务调度、Heartbeat、人工确认和审计治理放进同一个长期运行闭环里。

它的重点不是做一个“聊天壳子”，而是把真正会在真实场景里暴露出来的问题提前放进运行时层解决: 长时执行、失败恢复、可控自治、权限边界、预算约束和多终端协作。

它的长期目标也不只是“更强一点的聊天机器人”，而是逐步打造一个真正意义上的全能个人助理型 AI Agent: 能理解上下文、持续跟进任务、监控重要信息、帮助组织工作，并且在隐私、安全和控制权上始终站在用户这一边。

## 它解决什么问题

大多数 AI 应用擅长“一次回答”，但不擅长下面这些事情:

- 一件事不是一轮对话完成，而是要持续跟进、复盘、再执行。
- 某些动作有风险，不能让模型在背后静默执行。
- 系统需要在本地模型、云端模型和混合模型服务之间做现实可用的切换。
- 需要知道它为什么这么做、用了什么工具、花了多少预算、是否可审计。

LiAgent OS 的目标，就是让“长期运行的智能体”不再只是一个概念演示，而是逐步演化成一个可信的个人 AI 助理操作层。

## 长期产品方向

LiAgent OS 不想停留在“自动化工具集合”或“聊天壳子”这个层面。它的远期方向，是一个真正可长期使用的个人 AI Agent，能够：

- 做研究、做总结，并给出可追溯依据
- 持续监控你关心的主题、公司、市场与任务
- 记住你的偏好和长期工作模式
- 不是回答完就结束，而是能继续推进后续事项
- 在明确的确认、隐私和安全边界内行动

用更产品化的话说，它想成为一个**会越来越懂你、越来越能持续帮你做事，但不会越来越失控的个人 AI 助理**。

## 现在已经能做什么

- 通过 CLI、Web 或 Discord 与 Agent 交互。
- 调用工具、拆解复杂任务、并行子任务并逐步合成结果。
- 运行定时任务、延迟任务、文件变更触发任务和 Heartbeat 周期检查。
- 在 Web / Discord 中接收高风险动作确认卡片，手动 approve / reject。
- 在本地模型缺失时自动切到 API bootstrap 模式，降低首次启动门槛。
- 记录执行事件、路由轨迹、缓存/成本、任务结果和审计信息。
- 基于行为信号、兴趣监控、目标存储与反思循环，进入“受治理的半自治”阶段。

## 项目阶段与愿景（2026-03）

LiAgent OS 目前已经越过“玩具 Demo”阶段，但也没有假装自己已经变成一键式消费产品。更准确地说，它正处于从“能真实跑起来”走向“能长期稳定工作”的过渡期。

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 对话 + 工具调用 | 已完成基础闭环 | 已具备多端交互、工具调用、策略路由和事件流。 |
| 长时执行与恢复 | 已接入主链路 | 任务队列、失败分类、checkpoint 匹配、恢复继续执行都已接通。 |
| 受治理的半自治 | 持续推进 | Heartbeat、兴趣监控、目标存储、反思循环已经落地，并继续强化稳定性。 |
| 本地模型与混合模型服务优化 | 下一重点 | 会持续强化本地模型体验、混合推理路由、成本/延迟优化、隐私保护与安全治理能力。 |

一句话概括: **LiAgent OS 已经具备真实运行能力，接下来会持续把它打磨成一个更强调本地模型、更重视隐私与安全、也更像真正个人助理的长期执行系统。**

## 最新公开更新

这次公开更新主要聚焦在让 LiAgent OS 在长任务链里变得更可观测、更一致，也更可靠。

### 主要改进

- 更清晰的运行时可见性、状态提示与执行反馈
- 更一致的取消语义与运行态处理
- 更透明的工具执行链路与 fallback 路径
- 更稳健的中断 / preempted 运行恢复能力
- Web 与 Discord 共享统一的运行态表达
- 更强的长任务链稳定性

## 适合谁

- 想搭建本地优先或混合式 Agent Runtime 的开发者。
- 想研究“可持续执行 + 可确认 + 可审计”这类 Agent 基础设施的人。
- 想把研究、监控、整理、持续跟进这类工作交给 Agent，但又不想失控的人。
- 相信未来最有价值的个人 AI 产品，应该天然重视本地模型、隐私和安全的人。

如果你期待的是零配置、零模型、零环境差异的一键 SaaS，LiAgent OS 现在还不是这个定位。

## 5 分钟上手（推荐路径）

第一次体验，建议先走 **API bootstrap 模式**。这是当前对陌生用户最友好的路径，不需要先准备本地模型资产。

### 1. 安装

```bash
cd /path/to/liagent_git
python3 -m venv venv
./venv/bin/pip install -e .
```

可选扩展:

```bash
./venv/bin/pip install -e '.[browser]'
./venv/bin/pip install -e '.[discord]'
./venv/bin/pip install -e '.[mcp]'
```

### 2. 复制配置模板

```bash
cp config.example.json config.json
cp .env.example .env
```

### 3. 只填最少的 API 配置

在 `.env` 中至少填入:

```bash
LLM_API_KEY=your_api_key
# 可选:
# LLM_API_BASE_URL=https://api.openai.com/v1
# LLM_API_MODEL=gpt-4o
```

### 4. 启动 Web

```bash
./venv/bin/liagent --web --host 127.0.0.1 --port 8080
```

然后打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)。

### 5. 为什么这条路径最容易

`config.example.json` 默认仍保留本地优先配置，但 LiAgent 启动时会检查本地模型路径。如果路径不存在，而你又提供了 `LLM_API_KEY`，它会自动切到 API bootstrap 模式，并把可运行配置保存下来。

这意味着第一次不必先解决 MLX、本地模型目录和 TTS/STT 资产，先把系统跑起来更重要。

## 本地优先进阶路径

如果你想完全走本地或混合模式，再补这些配置:

- 在 `config.json` 中填写 `llm.local_model_path`、`tts.local_model_path`、`stt.model`。
- 将 `runtime_mode` 调整为 `local_private`，或保留 `hybrid_balanced`。
- Apple Silicon + MLX 是当前最合适的本地体验路径。

常见关键项:

| 配置项 | 用途 |
| --- | --- |
| `runtime_mode` | `local_private` / `hybrid_balanced` / `cloud_performance` |
| `llm.backend` | `local` 或 `api` |
| `tasks.enabled` | 是否启用长期任务系统 |
| `tasks.cwork_dir` | 允许 Agent 访问的工作目录根 |
| `routing.*` | 路由、代码委托、长上下文阈值 |
| `budget.*` | token / 美元预算上限 |
| `sandbox.*` | 沙箱执行配置 |

## 启动后你会看到什么

- Web 聊天界面: 不只是对话，还能看到运行时事件。
- Settings 面板: 可以切换运行模式、LLM/TTS/STT、工具策略和 MCP 配置。
- 任务系统: 能创建任务、查看运行记录、暂停/恢复、接收 Heartbeat 确认。
- 时间线与运行数据: 包括 dispatch、sub agent、tool、synthesis、budget、artifact 等链路信息。

如果你更偏命令行，也可以直接启动 CLI:

```bash
./venv/bin/liagent
# 或
./venv/bin/python -m liagent
```

如果你需要把确认动作送到 Discord:

```bash
LIAGENT_DISCORD_TOKEN=<your_discord_token> \
LIAGENT_WS_URL=ws://localhost:8080/ws/chat \
LIAGENT_WEB_TOKEN=<your_web_token> \
./venv/bin/liagent --discord
```

当 `--host` 不是 loopback 地址时，请务必设置 `LIAGENT_WEB_TOKEN`。

## 可以怎么用它

如果你第一次打开项目，不知道该问什么，可以直接试这些:

- `帮我整理这周 AI Agent 领域值得继续追踪的发布。`
- `每天早上检查 AAPL 和 TSLA 的新闻，有明显变化再提醒我。`
- `把这个需求拆成执行计划，先研究再给我最终方案。`
- `如果涉及写文件、执行命令或高风险操作，先问我。`

LiAgent OS 的重点不是“花哨对话”，而是把研究、监控、整理、持续跟进这类任务放进同一个运行时里。

## 核心能力地图

### 1. 模型与路由

- 本地优先推理，支持 API 回退与混合路由。
- Provider 注册与协议路由，兼容 OpenAI 风格和本地路径。
- 提示词缓存与分层 TTL。
- token / 美元双预算与缓存命中可观测。

### 2. 执行与编排

- 多智能体研究流: 拆解、并行、渐进合成、最终整合。
- 编码委托流: `coder -> verifier -> synthesis`。
- 失败分类、重试预算、checkpoint 匹配与恢复继续执行。

### 3. 长期任务与半自治

- 内置任务队列: cron、延迟一次性任务、文件变更触发。
- Heartbeat 闭环: 候选动作结构化、去重、预算传递、过期确认清扫。
- 目标与反思回路: `GoalStore`、行为信号、兴趣监控、目标复盘与建议投递。

### 4. 多端交互与可观测性

- CLI / Web / Discord 三端协同。
- Web 时间线、运行态事件、artifact 抽屉、任务结果推送。
- REST + WebSocket API: 配置、任务、兴趣、健康检查、推送通道都已接通。

### 5. 安全与治理

- 高风险工具确认、审计日志、输出脱敏。
- 工具档位: `minimal | research | full`。
- 工具信任注册表: 未信任 MCP server 不会静默接入。
- 文件访问边界限制在 `tasks.cwork_dir`，并带有预算、防线和 grounding gate。

## 仓库结构

```text
src/liagent/
  agent/          # 推理主循环、目标/任务/heartbeat、执行恢复
  engine/         # LLM / TTS / STT 路由与引擎管理
  knowledge/      # 检索与知识注入
  orchestrator/   # 多智能体编排
  tools/          # 工具、策略、信任注册表、执行器
  ui/             # CLI / Web / Discord / API routes
  voice/          # 语音相关能力
  skills/         # 技能路由与技能资产
  utils/          # 通用工具
tests/            # 单测与集成测试
docs/architecture.md  # 面向公开仓库的架构总览
docs/roadmap.md       # 面向公开仓库的路线图与版本方向
```

## 开发验证

与当前 CI 对齐的最小验证:

```bash
PYTHONPATH=src ./venv/bin/python -m compileall -q src
PYTHONPATH=src ./venv/bin/python -m pytest -q \
  tests/test_events.py \
  tests/test_skill_router_simplified.py \
  tests/test_health.py \
  tests/test_brain_layers.py \
  tests/test_web_event_envelope.py \
  tests/test_tool_parsing.py
```

完整回归:

```bash
PYTHONPATH=src ./venv/bin/python -m pytest -q
```

## 公开文档与治理

- 许可证: 见 [LICENSE](LICENSE)
- 贡献指南: 见 [CONTRIBUTING.md](CONTRIBUTING.md)
- 安全策略: 见 [SECURITY.md](SECURITY.md)
- 行为准则: 见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- 架构总览: 见 [docs/architecture.md](docs/architecture.md)
- 路线图: 见 [docs/roadmap.md](docs/roadmap.md)

## 故障排查

- 启动时报模型路径错误
  - 检查 `config.json` 中 `llm / tts / stt` 的路径是否存在，或改走 API bootstrap 路径。
- Web 外网访问失败
  - 确认已设置 `LIAGENT_WEB_TOKEN`，并检查 host / port 与防火墙。
- `web_fetch` 相关能力不可用
  - 安装 `.[browser]` 扩展，并完成 Playwright 依赖安装。
- Discord 无法收到确认卡片
  - 检查 `LIAGENT_DISCORD_TOKEN`、`LIAGENT_WS_URL`、`LIAGENT_WEB_TOKEN` 是否一致。
