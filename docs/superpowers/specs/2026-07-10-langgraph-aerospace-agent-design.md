# LangGraph 航天领域 Agent 框架设计

**状态：** 已由用户于 2026-07-10 确认

**范围：** `aerospace_agent/langgraph_agent` 及其配置、协议、知识和持久化资产

**架构路线：** 单进程领域服务架构；LangGraph 负责编排，领域服务负责存储和副作用

## 1. 目标与事实边界

本次交付把现有实验性 LangGraph 目录改造成可运行、可测试、可恢复的航天领域 Agent。必须交付：

1. 强类型、可序列化、可版本化的输入、图状态、节点决策、工具结果、证据和最终输出协议。
2. LangGraph 状态图，包含上下文管理、RAG、MCP 工具执行、循环检测、业务步数限制和运行时递归限制。
3. SQLite checkpointer，实现线程级对话记忆、多轮追问、状态历史和故障后从检查点继续。
4. 使用现有 6 条英文轨道动力学种子文本的持久化 RAG，返回可追踪的页面、块、分数和来源。
5. 可运行的 Markdown Wiki：主题页面、`index.md`、只追加的 `log.md`、交叉引用和幂等更新。
6. 与 Wiki 同步的知识图谱，以及可在浏览器打开、节点可跳转到 Wiki 页面的 HTML 导出。
7. 可回滚自主进化后端：隔离复盘、候选变更、白名单、备份、验证、事务日志、提交、冲突检测和回滚。
8. 标准 MCP stdio 协议冒烟，以及 Agent 经 `MCPGateway` 访问工具的统一边界。
9. 接入并真实测试 `http://127.0.0.1:8000/v1` 的本地 `qwythos` 模型。
10. 完整终端命令与测试证据。

已核验的仓库事实：

- 新目录已有 `agent.py`、`state.py`、`schema.py`、`graph.py`、`nodes.py`、`router.py`、`checkpointer.py`、`cycle_detector.py` 和 `evolution.py`，但尚未纳入 Git，且没有专属测试。
- 当前启动器实际使用 6 条内存种子和 `MockRAG`，没有使用仓库已有的持久化 `AerospaceRAG`、混合检索或知识图谱。
- 当前循环检测器被图闭包共享，跨线程污染计数；当前 `resume()` 只检查记录是否存在，不构成恢复执行。
- 当前 Python 3.13 默认环境缺少 LangGraph、SQLite checkpointer、MCP 和 PDF 依赖；依赖清单也未声明 LangGraph。
- 本地 `qwythos` 的 `/v1/models` 已返回成功。连通成功不构成内容正确性证据。
- 工作区已有大量用户改动和未跟踪数据；实现不得覆盖、清理或混入无关改动。

明确假设：本阶段只以 6 条种子构建知识库，不声称已导入《轨道动力学》书籍。后续 PDF/EPUB 导入沿同一 `KnowledgeSource` 协议扩展。

## 2. CTO 技术章程与关键决策

| 字段 | 决策 |
| --- | --- |
| 技术目标 | 本地优先、可恢复、可审计的航天领域 LangGraph Agent |
| 优化边界 | 可靠性、可维护性、数据可追踪和本地隐私优先于分布式扩展 |
| 架构假设 | 单进程领域服务足以支持当前单用户终端场景；MCP 保持标准协议边界 |
| 不可接受 | 跨线程状态污染、不可逆自修改、无来源 RAG、静默工具失败、伪造测试结论 |
| 交付目标 | 代码、配置、JSON Schema、Markdown Wiki、知识图谱导出、测试和终端报告 |

按“错误决策对成本、风险和维护性的影响”排序，Top-K 为：

1. **持久化状态边界：** 图状态只能保存可序列化的小对象和外部对象 ID；数据库连接、服务实例、大型工具结果不得进入 checkpoint。
2. **进化写入边界：** 所有自主修改先暂存和验证，内置源码/技能受保护，提交可追踪且默认可安全回滚。
3. **知识源与派生索引边界：** Markdown 是用户可审计的知识源；向量、关键词和图谱索引是可重建派生物。
4. **MCP 边界：** Agent 依赖 `MCPGateway`，不依赖工具注册表实现；标准 stdio 是真实协议路径，进程内实现只是测试/明确降级适配器。
5. **终止边界：** 循环指纹与业务预算属于线程状态，同时使用 LangGraph `recursion_limit` 做运行时硬限制。

## 3. 方案比较与选择

### 方案 A：继续扩充现有大文件

改动少，但 `nodes.py` 和 `evolution.py` 会同时持有编排、文件系统、索引和回滚职责。该方案不能清晰隔离副作用，也无法可靠测试故障恢复。

### 方案 B：单进程领域服务架构（已选）

LangGraph 节点只读取状态、调用显式服务接口并返回状态增量。知识、MCP、上下文、持久化和进化由独立服务负责。它复用现有 RAG/图谱代码，不引入部署系统。

### 方案 C：多进程服务化

隔离最强，但当前单机终端场景会新增端口、进程管理、健康检查和序列化故障面。只有并发、多用户或远程部署成为真实需求时才复审。

## 4. 目录与模块边界

```text
aerospace_agent/langgraph_agent/
├── agent.py                  # 公共门面、生命周期、同步/流式/恢复/进化 API
├── config.py                 # YAML + 环境变量配置加载和路径解析
├── schema.py                 # 所有版本化 Pydantic 协议
├── state.py                  # 只含可 checkpoint 的 LangGraph 状态
├── graph.py                  # 图拓扑、条件边、编译与 recursion_limit 配置
├── nodes.py                  # 薄节点：调用服务并返回状态增量
├── router.py                 # 规则优先、LLM 可选的领域意图/动作路由
├── cycle_detector.py         # 无共享可变计数的状态指纹和终止决策
├── checkpointer.py           # SQLite/内存 checkpointer 与线程历史 API
├── evolution.py              # 向后兼容导出；委托 EvolutionService
└── services/
    ├── __init__.py
    ├── context.py            # essential/summary/recent/artifact_ref 装配
    ├── knowledge.py          # Wiki、RAG、图谱和可视化统一服务
    ├── mcp_gateway.py        # stdio 与进程内 MCP 适配器
    └── evolution.py          # 提案、暂存、验证、提交和回滚事务

config/langgraph_agent.yaml   # 限制、端点、路径、进化策略
schemas/langgraph_agent/      # 导出的 Agent/Evidence/Tool/Evolution JSON Schema
knowledge/
├── index.md
├── log.md
└── orbital-dynamics/*.md     # 6 条种子主题页
evolved_skills/               # 进化生成的 Skill，内置技能不在此目录
workflows/evolved/            # 经验证的沉淀工作流
data/langgraph/
├── checkpoints.sqlite
├── rag/                      # 可重建向量/关键词/图谱索引
├── artifacts/                # 上下文卸载的大结果
└── evolution/<evolution_id>/ # proposal、staging、backup、manifest、report
tests/langgraph_agent/        # 独立测试套件
```

Python 是框架的实现语言，不是全部框架资产。YAML 配置、JSON Schema、Markdown Wiki、SQLite 状态和进化事务记录都是正式交付物。

## 5. 协议设计

### 5.1 顶层输入输出

`AgentInput` 至少包含：

- `schema_version`
- `user_message`
- `thread_id`
- `run_id`（调用方可给出，否则生成）
- `mode`
- `max_steps`
- `recursion_limit`
- `context`

`AgentOutput` 至少包含：

- `schema_version`
- `status`: `success | partial | error | interrupted | cycle_detected | limit_reached`
- `answer`
- `intent` 与置信度
- `citations: list[EvidenceItem]`
- `tool_results: list[ToolCallResponse]`
- `steps`、`cycle_triggers`、`checkpoint_id`
- `warnings`、`errors`、`metrics`

不得用非枚举的任意状态字符串。输出校验失败必须显式返回协议错误，不能把失败包装为成功。

### 5.2 中间协议

- `Decision`: `action = retrieve | call_tool | respond | stop`、理由、下一动作、工具参数。
- `EvidenceItem`: `source_id`、`page_path`、`chunk_id`、`score`、受限长度摘录和元数据。
- `ToolCallRequest/Response`: 工具名、结构化参数、超时、状态、结果/错误和耗时。
- `EvolutionProposal`: 依据 run/checkpoint、候选文件操作、理由、来源、未完成事项、所需验证。
- `EvolutionRecord`: 状态机、受影响文件、前后哈希、备份位置、验证报告和回滚结果。

所有协议可导出 JSON Schema，版本固定为 `1.0.0`。字段演进优先新增可选字段；破坏性变化必须提升主版本。

## 6. LangGraph 状态与拓扑

状态只保存：消息、意图、计划、证据、工具请求/响应、上下文摘要、artifact 引用、线程级循环历史、业务步数、终止原因、最终答案和指标。服务实例通过编译时闭包或运行时上下文注入，但不进入状态。

主图：

```text
START
  -> validate_input
  -> hydrate_context
  -> classify_and_plan
  -> retrieve_knowledge
  -> decide_action
  -> execute_tool
  -> validate_observation
  -> evaluate
       ├─ continue -> decide_action
       ├─ synthesize -> synthesize -> persist_outcome -> END
       └─ stop -> persist_outcome -> END
```

知识问答允许从 `retrieve_knowledge` 直接进入 `synthesize`。没有工具调用时不得制造空工具调用来推进循环。

循环指纹由线程状态中的规范化动作、工具名、参数摘要、观测摘要和目标构成。检测逻辑是无副作用函数，历史和计数写回 state。不同 `thread_id` 不共享计数。

终止采用三层约束：

1. 业务 `max_steps`：节点可预见地降级并输出部分结果。
2. 状态指纹重复阈值：先一次策略干预；仍重复则 `cycle_detected`。
3. LangGraph `recursion_limit`：作为运行时硬限制，捕获 `GraphRecursionError` 后读取最后 checkpoint 并返回 `limit_reached`。

## 7. 上下文与记忆

上下文按生命周期分离：

- **线程短期记忆：** checkpointer 中的消息和图状态。
- **跨线程长期记忆：** `memory/` 下经进化确认的 Markdown 记录；不与 checkpoint 混存。
- **结构化知识：** `knowledge/` 按主题组织，和时间线记忆分离。
- **大结果：** 写入 `data/langgraph/artifacts/`，状态只保存路径、哈希、媒体类型和摘要。

`ContextService` 生成 `essential + summary + recent`：系统约束和当前用户指令永不被摘要替换；历史摘要可重算；最近消息数量和估算 token 数同时受限。第一版使用仓库现有 token 估算，模型专用 tokenizer 作为后续可替换组件。

## 8. Checkpointer 与恢复语义

SQLite 是本地默认，内存后端只用于测试。每次 invoke/stream 必须传：

```python
{"configurable": {"thread_id": thread_id}, "recursion_limit": limit}
```

`recursion_limit` 是运行时顶层配置，不放入 `configurable`。

恢复分两类：

- **同线程新消息：** 向同一 `thread_id` 调用图，只追加新用户消息，由消息 reducer 保留历史。
- **故障/中断续跑：** 先 `graph.get_state(config)`；若 `snapshot.next` 非空，从该 checkpoint 继续，不重建“初始空状态”，也不重复已成功 super-step。

提供 `get_state`、`get_state_history`、列出线程和从指定 `checkpoint_id` 分支回放的门面。现有 `data/checkpoints.db` 不删除、不迁移；新实现写入 `data/langgraph/checkpoints.sqlite`，避免用新状态模式误读旧实验数据。

## 9. KnowledgeService：Wiki、RAG 与图谱

### 9.1 Markdown 是事实源

首次 `--init-knowledge` 将 6 条种子幂等生成 6 个 `knowledge/orbital-dynamics/<slug>.md` 页面。每页只表达一个主题，包含：标题、来源、稳定页面 ID、摘要、正文要点和相关页面。不得把 6 条种子描述为完整教材。

`knowledge/index.md` 按分类列出页面和一句摘要。`knowledge/log.md` 只追加 ingest/update/evolution/rollback 事件。重复初始化不重复建页、不重复索引。

### 9.2 派生检索

复用 `AerospaceRAG` 的向量、关键词和图谱能力，但使用独立 `data/langgraph/rag/`。索引输入来自 Wiki 页面，元数据至少包含 `page_id`、`page_path`、`chunk_id` 和内容哈希。第一版在 Wiki 发生提交或回滚后执行确定性的全量派生索引重建，以一致性优先；知识规模达到重访阈值后再改为按内容哈希增量更新。

图谱节点对应 Wiki 页面/航天概念，Markdown 交叉引用生成有向 `related_to` 边。原有内置航天图谱继续作为领域推导图，但每条回答引用必须回到可读页面或明确标为内置图谱知识。

### 9.3 可视化

复用现有知识云 HTML 生成器，补充 `page_path`，节点点击可打开对应 Markdown 页面。`--knowledge-graph <path>` 输出自包含 HTML 和结构化 JSON。验收仅证明文件可生成、包含正确节点/边/链接；不以主观美观作为通过条件。

## 10. MCPGateway

`MCPGateway` 统一提供 `list_tools()` 和 `call_tool(request)`。真实路径使用官方 MCP SDK 的 stdio client 连接 `python -m aerospace_agent.mcp.server`，完成 initialize、list_tools、call_tool 和关闭会话。同步 Agent 可用受控的异步桥接；禁止在已有事件循环中直接嵌套 `asyncio.run()`。

现有 MCP 入口必须同时修正两个协议问题：`stdio_server` 在实际运行函数内导入；stdio 模式的标准输出只能发送 MCP 协议帧，启动信息和诊断统一写入标准错误。工具结果使用官方 SDK 接受的内容类型，不能依赖未验证的裸字典返回。

`InProcessMCPGateway` 直接调用现有注册表，仅用于单元测试或依赖缺失时的显式降级。降级必须写入 `warnings`，不得声称标准 MCP 已测试。

工具选择先读取 MCP Tool 定义的 `inputSchema`，再产生参数。参数为空但 Schema 有必填项时，必须停止调用并返回 `invalid_arguments`，不能把异常吞掉后继续。

## 11. EvolutionService：自主进化事务

### 11.1 触发

支持：

- `--evolve <thread_id>` 手动复盘。
- `--evolve-due` 扫描满足策略的线程。
- REPL 常驻时按短周期检查：`enabled`、空闲时间超过阈值、对话轮数达到阈值或上下文接近容量。

一次性进程退出后无法在后台自动运行，因此不声称 `--task` 退出后仍会自动复盘；需要常驻 REPL 或外部调度调用 `--evolve-due`。

### 11.2 隔离复盘

复盘读取 checkpoint 快照和 run summary，在隔离上下文中输出 `EvolutionProposal`。允许候选类型：

- 写入/更新 `knowledge/` 页面和交叉引用。
- 写入 `memory/` 时间线或未完成事项。
- 在 `evolved_skills/` 新建或更新用户空间 Skill。
- 在 `workflows/evolved/` 沉淀可复用工具序列。

未完成事项默认只登记，不自动执行具有外部副作用的操作。模型提案不是事实；必须带来源并通过协议和验证门。

### 11.3 路径与保护

写入白名单只有：`knowledge/`、`memory/`、`evolved_skills/`、`workflows/evolved/` 和事务自身目录。禁止修改 `aerospace_agent/` 内置源码/技能、项目外路径、密钥、Git 历史或外部系统。解析后的绝对路径必须仍位于对应根目录，阻止 `..` 和链接逃逸。

### 11.4 事务状态机

```text
proposed -> staged -> backed_up -> validating
  -> committed
  -> validation_failed -> rolled_back
  -> commit_failed -> rolled_back
committed -> rollback_requested -> rolled_back | conflict
```

流程：

1. 把候选内容写入事务 `staging/`，不直接改正式文件。
2. 记录目标文件的存在性、内容、权限和 SHA256 到 `backup/` 与 `manifest.json`。
3. 验证 Schema、Markdown 链接、索引一致性、Skill manifest、工作流结构和受影响测试。
4. 写入事务日志后逐文件替换；任一失败立即按 manifest 恢复。
5. 提交成功后重建 Wiki 索引、RAG 和图谱，再写 `after` 哈希和验证报告。

多文件替换无法在普通文件系统上提供真正的全局原子性，因此使用写前日志和补偿恢复，不把它描述为数据库级原子提交。

### 11.5 手动回滚与冲突

`rollback(evolution_id)` 先比对当前文件哈希与该次提交的 `after` 哈希。只有未被后续修改的文件才自动还原；不一致返回 `conflict`，默认不覆盖。回滚后重建 Wiki/RAG/图谱并追加日志。强制覆盖不作为默认 CLI 选项。

## 12. 配置

`config/langgraph_agent.yaml` 至少包含：

```yaml
schema_version: "1.0.0"
llm:
  endpoint: "http://127.0.0.1:8000/v1"
  model: "qwythos"
runtime:
  max_steps: 15
  recursion_limit: 40
  cycle_max_repeats: 3
context:
  max_tokens: 8192
  recent_turns: 8
knowledge:
  workspace: "knowledge"
  data_dir: "data/langgraph/rag"
checkpoint:
  backend: "sqlite"
  path: "data/langgraph/checkpoints.sqlite"
evolution:
  enabled: true
  idle_minutes: 10
  min_turns: 6
  allowed_roots: ["knowledge", "memory", "evolved_skills", "workflows/evolved"]
mcp:
  transport: "stdio"
  command: "python"
  args: ["-m", "aerospace_agent.mcp.server"]
```

环境变量只覆盖 LLM 端点、模型和明确的运行参数。路径解析相对项目根目录，配置加载后验证所有目录边界。

## 13. 错误处理和可观测性

每个节点返回结构化错误，区分协议错误、检索为空、工具不可用、工具失败、循环、运行时递归、检查点错误、进化验证失败、回滚冲突和模型连接失败。

每次 run 记录：run/thread/checkpoint ID、节点步骤、工具耗时、RAG 命中和引用、循环干预、模型端点/模型名、总耗时和最终状态。日志不得包含密钥或无限制的大模型原始上下文。

失败降级：

- LLM 不可用：知识问答可返回带引用的检索结果；需要模型规划的任务返回明确 `partial`。
- MCP 不可用：不执行工具，返回 `tool_unavailable`；只有显式允许时才能使用进程内降级。
- RAG 空结果：明确无证据，不用模型常识伪装私域检索命中。
- Checkpointer 写入失败：不宣称对话已保存。

## 14. 测试与验收

### 14.1 单元测试

1. 所有 Pydantic 协议的有效/无效样例和 JSON Schema 导出。
2. 状态仅含可序列化值；消息 reducer 和上下文裁剪保持当前用户指令。
3. 循环指纹确定性、重复干预、业务上限和跨线程隔离。
4. Wiki 六页幂等生成、索引、日志、交叉引用和路径安全。
5. RAG 结果包含页面/块/来源，图谱边和 HTML 导出正确。
6. Evolution 成功提交、验证失败恢复、提交失败恢复、手动回滚、哈希冲突和白名单拒绝。
7. MCP 定义、参数校验和进程内网关行为。

### 14.2 LangGraph 集成测试

1. 无 LLM 的确定性知识查询完成全图并返回引用。
2. SQLite 同线程多轮追加消息；销毁并重建 Agent 后仍能恢复。
3. 注入一次性故障，确认 `snapshot.next` 非空；续跑不重复已完成步骤。
4. 获取状态历史，从指定 checkpoint 分支回放。
5. 构造重复状态，确认在限制内终止；两个线程的计数互不影响。
6. 触发 `GraphRecursionError`，确认输出 `limit_reached` 和最后 checkpoint，而非未捕获异常。

### 14.3 MCP 标准协议测试

用官方 MCP client 经 stdio 完成初始化、`list_tools` 和至少一个无外部引擎依赖的 `call_tool`。工具定义与处理器注册表一致；非法参数和不可用引擎返回结构化错误；服务 stdout 不含协议外启动文本。

### 14.4 本地 Qwen 真实验收

1. `/v1/models` 返回 `qwythos`。
2. 直接 chat completion 返回可解析响应。
3. Agent 完成英文种子覆盖的轨道动力学问答，输出包含 Wiki/RAG 引用。
4. 同一 thread 追问并确认 checkpoint 中保留前一轮上下文。
5. 记录模型、耗时、步骤和引用数。

模型返回非空只证明调用链工作。内容是否正确必须逐条比对种子证据；无法自动判断的语义质量标为未验证，不写成“通过”。

### 14.5 回归

先运行 `tests/langgraph_agent/`，再运行与 RAG、MCP、Skill、Memory 直接相关的既有测试，最后运行全套测试。环境中确实缺失的可选航天引擎可以按既有契约跳过，但必须列出跳过项和原因。所有测试写入独立临时目录，避免污染用户现有 `data/`。

## 15. 终端接口

必须支持：

```powershell
python start_langgraph_agent.py --init-knowledge
python start_langgraph_agent.py --knowledge-status
python start_langgraph_agent.py --knowledge-graph reports/langgraph_knowledge_graph.html
python start_langgraph_agent.py --task "What is two-body orbital dynamics?" --thread acceptance-qwen
python start_langgraph_agent.py --task "What assumptions did you mention?" --thread acceptance-qwen
python start_langgraph_agent.py --evolve acceptance-qwen
python start_langgraph_agent.py --evolve-due
python start_langgraph_agent.py --rollback <evolution_id>
python start_langgraph_agent.py --checkpoint-history acceptance-qwen
```

CLI 返回人类可读摘要，并可选择 JSON 输出以符合 `AgentOutput`/`EvolutionRecord` 协议。

## 16. 依赖与兼容

声明并测试兼容范围：Python 3.10–3.13、Pydantic 2、LangGraph 1.x、`langgraph-checkpoint-sqlite`、`langchain-core`、官方 `mcp` SDK。具体解析版本记录在终端验收报告中。

不使用 LangGraph beta `DeltaChannel`，避免将未稳定 API 放入基础持久层。当前实验性节点名和 checkpoint 模式不承诺向后兼容；旧数据库保留但不复用。公共导出尽量保留，已存在的 `LangGraphAerospaceAgent`、`AgentInput/Output`、`CycleDetector` 和 `EvolutionEngine` 名称继续可导入。

## 17. 反向审查与重访条件

最强反对意见是“已有 RAG、图谱、Memory 和 Skill，不应再建一套”。本设计不重写这些算法，而是用 `KnowledgeService` 统一调用并将数据隔离到新目录；新增的是 Wiki 事实源、一致性同步和事务边界。

以下情况触发架构复审：

- 多用户或并发写入需要真正的数据库事务。
- Agent 需要常驻系统服务而不是终端进程。
- 知识量增长到全量重建不可接受。
- 自主进化需要执行外部系统未完成事项。
- 旧 checkpoint 成为必须迁移的正式资产。

在这些条件出现前，不引入微服务、消息队列、远程向量数据库或无人值守外部副作用。

## 18. 参考依据

- LangGraph Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Graph API / recursion limit: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph backward compatibility: https://docs.langchain.com/oss/python/langgraph/backward-compatibility
- LangChain MCP: https://docs.langchain.com/oss/python/langchain/mcp
- CowAgent Personal Knowledge Base: https://docs.cowagent.ai/knowledge
- CowAgent Self-Evolution: https://docs.cowagent.ai/memory/self-evolution
