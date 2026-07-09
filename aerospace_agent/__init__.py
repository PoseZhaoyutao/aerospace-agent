"""aerospace_agent — 航天导航控制 Agent 核心框架。

基于 ReAct 循环的航天动力学智能 Agent，集成：

子包结构
--------
    core/        — Agent 编排器、上下文管理(CEO)、LLM 接口、三层记忆系统
    physics/     — 轨道力学物理计算 (常数 / 开普勒 / 二体 / Lambert /
                    拼凑圆锥 / 轨道机动 / 地月转移)
    mcp/         — 统一航天动力学 MCP Server（Canonical Model + 7 引擎适配器
                    + 12 MCP 工具 + Loop 引擎 + 安全模块）
    mcp_tools/   — MCP 工具的 Agent 适配层（桥接 MCP 工具到 ReAct Agent）
    rag/         — 可路由、可验证、可追踪的知识工具系统（RAG）
    skills/      — 技能模块（可复用能力单元，通过 SkillRegistry 统一管理）
    prompts/     — 系统提示词 + ReAct 模板 + 任务专属模板 + 上下文策略
    workflows/   — 航天任务工作流（轨道设计 / 发射窗口 / 轨迹分析）
    reporting/   — 报告生成（公式 / 图表 / 知识报告）
    utils/       — 辅助工具（Git 管理器等）

快速开始
--------
::

    from aerospace_agent.core import create_default_agent

    agent = create_default_agent()
    result = agent.run("设计一条 300km 停泊轨道的地月转移轨道")

架构总览
--------
1. **Agent (core/)**：ReAct 循环 + CEO 上下文管理 + 三层记忆
2. **MCP (mcp/)**：Canonical Astrodynamics Model + 7 引擎适配器 + Loop 引擎
3. **RAG (rag/)**：多源路由检索 + 证据验证 + 溯源链
4. **Memory (core/memory.py)**：短期 + 工作 + 长期记忆，MemoryManager 统一管理
5. **Skill (skills/)**：上下文 / 记忆 / RAG / Loop / 报告 五大技能
6. **Prompt (prompts/)**：系统提示词 + 7 类任务模板 + 上下文策略代码化
"""
from __future__ import annotations

__version__ = "0.7.0"

# ---- 延迟导入：避免循环依赖，按需加载子模块 ----

def __getattr__(name: str):
    """PEP 562 延迟导入，避免在 import aerospace_agent 时加载全部子包。"""
    # core 模块
    if name in ("AerospaceAgent", "create_default_agent"):
        from .core.agent import AerospaceAgent, create_default_agent
        return locals()[name]
    if name == "ContextManager":
        from .core.context_manager import ContextManager
        return ContextManager
    if name in ("LLMInterface", "MockLLM", "OpenAICompatibleLLM",
                "LocalLLM", "ModelRouter", "create_llm"):
        from .core import llm_interface
        return getattr(llm_interface, name)
    if name in ("ShortTermMemory", "WorkingMemory", "LongTermMemory",
                "MemoryManager"):
        from .core.memory import (
            ShortTermMemory, WorkingMemory, LongTermMemory, MemoryManager,
        )
        _map = {
            "ShortTermMemory": ShortTermMemory,
            "WorkingMemory": WorkingMemory,
            "LongTermMemory": LongTermMemory,
            "MemoryManager": MemoryManager,
        }
        return _map[name]

    # MCP 模块
    if name == "TOOL_REGISTRY":
        from .mcp.tools import TOOL_REGISTRY
        return TOOL_REGISTRY
    if name == "LoopEngine":
        from .mcp.loop.engine import LoopEngine
        return LoopEngine
    if name in ("run_minimal_orbit_experiment", "check_aerospace_invariants"):
        from . import experiment_runtime
        return getattr(experiment_runtime, name)

    # Skills 模块
    if name in ("SkillBase", "SkillRegistry", "ContextManagementSkill",
                "MemoryRecallSkill", "KnowledgeRetrievalSkill",
                "LoopOrchestrationSkill", "ReportGenerationSkill"):
        from .skills import (
            SkillBase, SkillRegistry, ContextManagementSkill,
            MemoryRecallSkill, KnowledgeRetrievalSkill,
            LoopOrchestrationSkill, ReportGenerationSkill,
        )
        _map = {
            "SkillBase": SkillBase,
            "SkillRegistry": SkillRegistry,
            "ContextManagementSkill": ContextManagementSkill,
            "MemoryRecallSkill": MemoryRecallSkill,
            "KnowledgeRetrievalSkill": KnowledgeRetrievalSkill,
            "LoopOrchestrationSkill": LoopOrchestrationSkill,
            "ReportGenerationSkill": ReportGenerationSkill,
        }
        return _map[name]

    # Prompts 模块
    if name in ("SYSTEM_PROMPT", "get_prompt", "list_task_types",
                "build_system_prompt", "build_react_messages"):
        from .prompts import (
            SYSTEM_PROMPT, get_prompt, list_task_types,
            build_system_prompt, build_react_messages,
        )
        _map = {
            "SYSTEM_PROMPT": SYSTEM_PROMPT,
            "get_prompt": get_prompt,
            "list_task_types": list_task_types,
            "build_system_prompt": build_system_prompt,
            "build_react_messages": build_react_messages,
        }
        return _map[name]

    # RAG 模块
    if name == "AerospaceRAG":
        from .rag.aerospace_rag import AerospaceRAG
        return AerospaceRAG

    raise AttributeError(f"module 'aerospace_agent' has no attribute {name!r}")


__all__ = [
    "__version__",
    # core
    "AerospaceAgent", "create_default_agent",
    "ContextManager",
    "LLMInterface", "MockLLM", "OpenAICompatibleLLM",
    "LocalLLM", "ModelRouter", "create_llm",
    "ShortTermMemory", "WorkingMemory", "LongTermMemory", "MemoryManager",
    # mcp
    "TOOL_REGISTRY", "LoopEngine",
    "run_minimal_orbit_experiment", "check_aerospace_invariants",
    # skills
    "SkillBase", "SkillRegistry",
    "ContextManagementSkill", "MemoryRecallSkill",
    "KnowledgeRetrievalSkill", "LoopOrchestrationSkill",
    "ReportGenerationSkill",
    # prompts
    "SYSTEM_PROMPT", "get_prompt", "list_task_types",
    "build_system_prompt", "build_react_messages",
    # rag
    "AerospaceRAG",
]
