# Aerospace Agent — 航天导航控制 Agent 核心框架

基于 **ReAct（Reason+Act）循环** 的航天导航控制 Agent，集成可插拔 LLM 接口、
CEO 三层上下文管理、记忆系统、工具/工作流编排与命令行入口。仅依赖
`numpy` / `scipy` / `click`，无 API key 时自动回退到离线 MockLLM，开箱即用。

## 项目结构

```
aerospace_agent/
├── __init__.py              # 顶层包，导出版本号 0.1.0
├── cli.py                   # Click 命令行入口
├── core/
│   ├── __init__.py          # 导出 AerospaceAgent
│   ├── llm_interface.py     # 可插拔 LLM 接口（LLMInterface/OpenAICompatibleLLM/MockLLM）
│   ├── context_manager.py   # CEO 上下文管理（Compress/Essential/Offload）
│   ├── memory.py            # 短期+长期记忆（向量检索占位）
│   └── agent.py             # ReAct 主编排器 + create_default_agent
└── utils/
    ├── __init__.py
    └── git_manager.py       # Git 操作封装
```

## 安装

```bash
pip install -r requirements.txt
# 可选：开发安装
pip install -e .
```

## CLI 用法

```bash
# 运行地月转移轨道演示（无需 API key，自动使用 MockLLM）
python -m aerospace_agent.cli demo

# 执行一个任务
aerospace-agent run "设计地月转移轨道"

# 列出工作流 / 工具
aerospace-agent workflow list
aerospace-agent tool list

# 运行工作流（带参数）
aerospace-agent workflow run lunar_transfer --param altitude_km=300

# 知识库检索
aerospace-agent rag index ./docs
aerospace-agent rag query "地月转移 C3 能量"

# Git 包装
aerospace-agent git init
aerospace-agent git add .
aerospace-agent git commit "init"

# 交互式 REPL
aerospace-agent shell
```

### 环境变量（真实 LLM）

未设置时自动回退 MockLLM；设置后调用 OpenAI 兼容 API：

```bash
export AEROSPACE_LLM_API_KEY=sk-xxxx
export AEROSPACE_LLM_BASE_URL=https://api.openai.com/v1
export AEROSPACE_LLM_MODEL=gpt-4o-mini
```

## 架构图

```
                        ┌──────────────────────────────────────┐
                        │            AerospaceAgent            │
                        │         (ReAct 主编排器)              │
                        └────────────────┬─────────────────────┘
            ┌──────────────┬────────────┼────────────┬──────────────┐
            ▼              ▼            ▼            ▼              ▼
      ┌──────────┐  ┌────────────┐ ┌─────────┐ ┌─────────┐  ┌──────────┐
      │   LLM    │  │  Context   │ │ Memory  │ │  Tools  │  │ Workflows│
      │ Interface│  │  Manager   │ │ ST+LT   │ │ Registry│  │ Registry │
      └────┬─────┘  │  (CEO)     │ └────┬────┘ └────┬────┘  └────┬─────┘
           │        └─────┬──────┘      │           │            │
      OpenAI/Mock         │             │      orbit_calc     lunar_transfer
      ┌──────────┐   ┌────┴────┐   ┌────┴────┐   orbital_vel   orbit_design
      │Essential │   │Compress │   │Offload  │   calculator
      │(永不压缩)│   │(超阈值  │   │(外部文件│
      └──────────┘   │ 摘要)   │   │ +引用)  │
                     └─────────┘   └─────────┘

  ReAct 循环:  think ─▶ act(tool) ─▶ observe ─▶ ... ─▶ Final Answer  (≤ N 步)
```

## 关键设计

- **CEO 上下文管理**：Essential 层（任务规格/用户原始指令）永不压缩，原样保留；
  Compress 层超阈值摘要；Offload 层大块数据存外部文件，上下文只保留引用。
- **可插拔 LLM**：`create_llm()` 根据环境变量自动选择真实 API 或 MockLLM。
- **离线可用**：MockLLM 能识别“设计地月转移轨道”等任务并返回结构化响应，
  使 ReAct 循环可在无网络环境下完整演示。
- **简化向量检索**：长期记忆用 numpy 点积 + 随机投影做 embedding 占位，无需向量库。
