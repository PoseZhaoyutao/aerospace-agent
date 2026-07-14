# Aerospace Agent — 总体架构说明

> 基于 **LoopRecursive-CEO** 方法论设计的航天动力学 Agent 系统。
> Phase A（递归第一性原理）确保架构正确，Phase B（自主交付循环）确保落地收敛。

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户航天任务需求                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LoopEngine (八阶段元工作流)                      │
│  Plan → SelectEngine → RetrieveDemo → GenerateWorkflow              │
│    → Run → Validate → Fix → Save                                    │
│  + FirstPrinciplesAnalyzer (递归第一性原理分析)                      │
│  + LoopLedger (逐轮可追溯账本)                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AerospaceAgent (ReAct 编排器)                    │
│  think → act(MCP tool) → observe → ... → Final Answer               │
│  + CEO 上下文管理 (Essential/Compress/Offload 三层)                 │
│  + ModelRouter (本地小模型 ↔ 云端大模型 自动路由)                    │
└──────────────┬──────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────┬─────────────────────────────────────┐
│   RAG 知识系统 (Main Agent)   │   MCP Tools (12 个白名单工具)       │
│  ┌─────────────────────┐     │  check_engine_availability          │
│  │ RetrieverRouter     │     │  index_reference_demos              │
│  │  ├─ 文档检索(Hybrid)│     │  search_workflows                   │
│  │  ├─ 数据库检索      │     │  generate_astrodynamics_workflow    │
│  │  ├─ 代码检索        │     │  convert_time                       │
│  │  ├─ 记忆检索        │     │  transform_frame                    │
│  │  └─ 网络搜索        │     │  query_ephemeris_state              │
│  ├─────────────────────┤     │  convert_orbit_representation       │
│  │ EvidenceVerifier    │     │  propagate_orbit                    │
│  │  (声明↔证据支撑验证)│     │  compute_ground_access              │
│  ├─────────────────────┤     │  run_gmat_script                    │
│  │ Traceability        │     │  cross_validate_results             │
│  │  (答案溯源链)       │     │                                     │
│  └─────────────────────┘     └──────────────┬──────────────────────┘
└──────────────────────────────────────────────┤──────────────────────┘
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Canonical Astrodynamics Model (统一中间层)              │
│  Epoch · Frame · Body · OrbitState · AttitudeState · ForceModel     │
│  PropagatorConfig · GroundStation · SpacecraftConfig               │
│  WorkflowSpec · WorkflowResult · ValidationReport · LoopLedgerEntry │
│  规则：SI 单位 + epoch/frame 标签 + JSON 可序列化 + 无损往返        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    7 引擎 Adapter (可选插件)                         │
│  OrekitAdapter │ GMATAdapter │ SpiceyPyAdapter │ AstropyAdapter     │
│  PoliastroAdapter │ BasiliskAdapter │ STKAdapter                   │
│  规则：懒加载 · is_available() 闸门 · 不可用返回结构化结果不崩溃     │
│        所有输出转回 Canonical Model · LLM 不可直接调底层库          │
└─────────────────────────────────────────────────────────────────────┘
```

## 2. 五个第一性原理决策（Phase A 递归结论）

| 优先级 | 铰链决策 | 第一性原理 | 影响 |
|--------|----------|-----------|------|
| K1 | Canonical Astrodynamics Model | 物理状态必须 SI 单位显式标注 + epoch/frame 标签 + 无损往返 | 所有引擎经 adapter 转换到此统一模型 |
| K2 | Adapter 契约 | 可选插件、懒加载、is_available() 唯一闸门、不可用不崩溃 | 零引擎安装时 MCP server 仍可启动 |
| K3 | Loop 与 ReAct 集成 | Loop 是元工作流编排 ReAct 步骤，非替代 | Plan→Run→Validate→Fix 全程可追溯 |
| K4 | 本地小模型接口 | LLM 引擎无关，ModelRouter 按复杂度路由 | 简单任务→本地，复杂推理→云端 |
| K5 | RAG 路由化 | 多源路由 + 证据验证 + 溯源链 | 答案可追溯、可验证、可信任 |

## 3. Canonical Astrodynamics Model（统一中间层）

所有引擎的输入输出都必须经过此层，保证物理一致性。

### 3.1 Schema 清单

| Schema | 核心字段 | 单位/约束 |
|--------|----------|-----------|
| `Epoch` | value, scale(UTC/TAI/TT/TDB/ET), format(ISO/JD/MJD/UNIX) | 时间尺度必须显式 |
| `Frame` | name(GCRF/ICRF/EME2000/ITRF/TEME/...), center, realization | 每个状态必须绑定 |
| `Body` | name, mu(m³/s²), radius(m), gravity_model | SI 单位 |
| `OrbitState` | epoch, frame, representation, position_m[3], velocity_mps[3], elements | 笛卡尔↔开普勒无损互转 |
| `AttitudeState` | epoch, frame_from, frame_to, quaternion/euler/dcm | 标量在前四元数 |
| `ForceModel` | central_body, gravity, degree, order, drag, srp, third_body | 力学模型完整定义 |
| `PropagatorConfig` | engine, type, integrator, step_s, duration_s, tolerance | 传播精度可追溯 |
| `GroundStation` | name, lat_deg, lon_deg, alt_m, min_elevation_deg | WGS84 |
| `SpacecraftConfig` | name, mass_kg, drag_area_m2, srp_area_m2, cd, cr | 力学参数 |
| `WorkflowSpec` | id, goal, task_type, inputs, models, engine, steps, outputs, validation, failure_handling | 工作流完整规格 |
| `WorkflowResult` | workflow_id, status, outputs, state_history, validation, engine, units, frame, loop_ledger | 含溯源与验证 |
| `ValidationReport` | passed, checks, position_error_m, velocity_error_mps, confidence | 交叉验证报告 |

### 3.2 无损往返保证

```
canonical → engine-specific → 计算 → engine-specific → canonical
```
- `OrbitState.to_keplerian(mu)` / `to_cartesian(mu)`：二体解析互转，已实现
- 每个 Adapter 的能力方法返回 `OrbitState.to_dict()` 格式
- 数值精度通过 `cross_validate_results` 工具多引擎交叉验证

## 4. 7 引擎 Adapter 架构

### 4.1 Adapter 契约

每个 Adapter 必须实现：
```python
class BaseAdapter(ABC):
    def is_available(self) -> bool: ...      # 唯一闸门，绝不抛异常
    def version(self) -> str: ...             # 不可用返回 'unavailable'
    def capabilities(self) -> Set[str]: ...   # 能力集合
```

可选能力方法（默认返回 `unavailable_result`）：
```python
def propagate_orbit(self, initial_state, force_model, config) -> dict
def transform_frame(self, state, target_frame) -> dict
def query_ephemeris(self, target, observer, epoch, frame) -> dict
def convert_time(self, epoch, target_scale) -> dict
def compute_ground_access(self, orbit_state, station, ...) -> dict
def run_script(self, script_text, script_path, workspace) -> dict
```

### 4.2 引擎能力矩阵

| 引擎 | propagate | transform_frame | query_ephemeris | convert_time | ground_access | run_script | attitude | spherical_harmonics |
|------|-----------|-----------------|-----------------|--------------|---------------|------------|----------|---------------------|
| Orekit | ✓ | ✓ | — | ✓ | ✓ | — | — | ✓ |
| GMAT | ✓ | — | — | — | ✓ | ✓ | — | — |
| SpiceyPy | — | ✓ | ✓ | ✓ | — | — | — | — |
| Astropy | — | ✓ | ✓ | ✓ | — | — | — | — |
| Poliastro | ✓ | — | — | — | — | — | — | — |
| Basilisk | ✓ | — | — | — | — | — | ✓ | — |
| STK | ✓ | — | — | — | ✓ | — | ✓ | — |

### 4.3 安全规则

- **懒加载**：引擎库在方法内 import，模块顶层不 import
- **不崩溃**：`is_available()` / `version()` 全程 try/except
- **STK 许可**：COM 接口检测安装与授权，不可用返回结构化结果
- **工作区隔离**：GMAT/Basilisk/STK 脚本必须复制到 workspace 后执行
- **Kernel 白名单**：SPICE kernel 路径必须来自 kernel_registry 或用户显式授权
- **不复制商业代码**：对商业软件只建 metadata，不复制源码

## 5. Loop 引擎（八阶段自主交付循环）

### 5.1 八阶段流程

```
Plan (递归第一性原理分析)
  ↓ FirstPrinciplesAnalyzer: 关键词评分 → Top-K → 递归下钻 → v1 蓝图
SelectEngine (引擎能力匹配)
  ↓ check_engine_availability → 任务类型 → 优先级矩阵 → 选中引擎
RetrieveDemo (检索可复用工作流)
  ↓ search_workflows → 候选工作流列表
GenerateWorkflow (生成工作流规格)
  ↓ generate_astrodynamics_workflow → WorkflowSpec (YAML)
Run (执行工作流步骤)
  ↓ 逐步调用 MCP tools → WorkflowResult
Validate (验证结果)
  ↓ 输出存在性 + 错误检查 + 单位标注 + 交叉验证 → ValidationReport
Fix (最小修复)
  ↓ 分析失败检查项 → 切换引擎/重试/放宽阈值
Save (沉淀可复用)
  ↓ 成功工作流标记为 reusable
```

### 5.2 LoopLedger 逐轮记录

每轮 Loop 记录一条 `LoopLedgerEntry`：
- phase（八阶段之一）
- goal（当前目标）
- tools_used（使用的工具）
- inputs / outputs（输入输出）
- errors（错误信息）
- fix_action（修复动作）
- validation_result（验证结果）
- saved_as_reusable（是否保存为可复用）

### 5.3 与 ReAct 的关系

```
LoopEngine (元工作流)
  ├─ Plan: 递归分析 → 蓝图
  ├─ Run: 将 WorkflowSpec.steps 分发
  │    └─ AerospaceAgent.run() (ReAct 循环)
  │         think → act(MCP tool) → observe → ... → Final Answer
  ├─ Validate: 验证 ReAct 产出
  └─ Fix: 驱动 ReAct 重新执行失败步骤
```

## 6. RAG 知识系统（可路由、可验证、可追踪）

### 6.1 架构

```
用户查询
  ↓
RetrieverRouter (多源路由)
  ├─ 文档检索: HybridRetriever (向量 + BM25 + 知识图谱 三路混合)
  ├─ 数据库检索: 业务表/订单/日志
  ├─ 代码检索: 代码库/接口/README
  ├─ 记忆检索: 用户偏好/历史决策
  └─ 网络搜索: 最新信息
  ↓
合并去重 → 候选证据
  ↓
EvidenceVerifier (证据验证)
  ├─ 声明切分 → 逐条匹配证据
  ├─ 关键词重合度 (Jaccard)
  ├─ 数值一致性检查
  └─ 矛盾检测
  ↓
VerificationReport (支撑度评级)
  ↓
TraceabilityManager (溯源链构建)
  ↓
AnswerTrace (答案 + source_id 链 + 引用)
```

### 6.2 设计原则

- **不是"向量库 + 大模型"**，而是"可路由、可检索、可验证、可追踪的知识工具系统"
- **Hybrid Search 必须做**：向量（语义）+ BM25（精确术语）+ 知识图谱（概念联想）
- **元数据入库**：不只切 chunk，加 source_type / file_path / timestamp / tags
- **答案可溯源**：每个声明附 source_id，可回溯到原始文档

## 7. 本地小模型接口

### 7.1 ModelRouter 路由策略

| 任务复杂度 | 关键词 | 路由目标 |
|-----------|--------|---------|
| simple | convert/transform/query/check/转换/查询 | LocalLLM (本地小模型) |
| complex | design/analyze/plan/diagnose/设计/分析/规划 | OpenAICompatibleLLM (云端大模型) |
| unknown | — | 优先本地，回退云端 |

### 7.2 本地模型部署支持

LocalLLM 通过 OpenAI 兼容端点支持：
- **Ollama**: `http://localhost:11434/v1`（默认）
- **vLLM**: `http://localhost:8000/v1`
- **llama.cpp server**: `http://localhost:8080/v1`
- **LM Studio**: `http://localhost:1234/v1`

## 8. 安全模型

| 安全规则 | 实现 |
|---------|------|
| 不执行任意 shell 命令 | 工具白名单，LLM 不可直接调底层库 |
| 只在工作区读写 | `SandboxGuard.validate_path()` + `PathPolicy.is_allowed_write()` |
| GMAT/Basilisk/STK 先复制到工作区 | `SandboxGuard.prepare_workspace_copy()` |
| SPICE kernel 路径白名单 | `PathPolicy.validate_kernel_path()` + `KernelRegistry` |
| STK 检测授权 | `check_license("stk")` → COM 接口检测 |
| 不复制商业软件 Demo 代码 | `index_reference_demos` 只建 metadata |
| 所有结果含单位/坐标系/时间/引擎版本 | WorkflowResult 强制字段 |
| 所有失败返回结构化错误 | `error_handler` 装饰器，绝不静默失败 |

## 9. 项目目录结构

```
aerospace-agent/
├── aerospace_agent/              # 主 Agent 包（已有 + 增强）
│   ├── core/
│   │   ├── agent.py              # ReAct 编排器（已有）
│   │   ├── llm_interface.py      # LLM 接口（+LocalLLM +ModelRouter）
│   │   ├── context_manager.py    # CEO 上下文管理（已有）
│   │   └── memory.py             # 短期+长期记忆（已有）
│   ├── physics/                  # 物理引擎（已有）
│   ├── mcp_tools/                # 旧版 MCP 工具（已有）
│   ├── rag/
│   │   ├── retriever.py          # HybridRetriever 三路混合（已有）
│   │   ├── router.py             # RetrieverRouter 多源路由（新增）
│   │   ├── verifier.py           # EvidenceVerifier 证据验证（新增）
│   │   ├── trace.py              # Traceability 溯源链（新增）
│   │   └── ...                   # 向量库/关键词/知识图谱/文献管线（已有）
│   ├── workflows/                # 工作流注册表（已有）
│   ├── reporting/                # 报告生成（已有）
│   └── utils/git_manager.py      # Git 管理（已有）
├── astro_dynamics_mcp/           # 统一航天动力学 MCP Server（新增）
│   ├── pyproject.toml
│   ├── src/astro_dynamics_mcp/
│   │   ├── __init__.py
│   │   ├── server.py             # MCP Server 入口
│   │   ├── schemas/              # Canonical Astrodynamics Model（12 个 schema）
│   │   ├── adapters/             # 7 引擎适配器 + BaseAdapter
│   │   ├── tools/                # 12 个 MCP 工具
│   │   ├── loop/                 # Loop 引擎 + 递归第一性原理分析
│   │   ├── resources/            # 工作流目录 + Demo 索引 + Kernel 注册表
│   │   ├── safety/               # 许可检查 + 沙箱 + 路径策略
│   │   ├── prompts/              # MCP 提示模板
│   │   ├── workflows/            # 6 个 YAML 工作流模板
│   │   ├── examples/             # 请求示例 JSON
│   │   └── tests/                # 测试套件
│   └── README.md
├── data/                         # 持久化数据（已有）
├── docs/                         # 文档
│   ├── ARCHITECTURE.md           # 本文件
│   └── RUN_AND_INTEGRATE.md      # 运行与接入指南
└── README.md
```
