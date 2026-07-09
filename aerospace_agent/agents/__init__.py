"""aerospace_agent.agents — 多Agent协作子包。

三Agent架构：
    TestAgent  — 测试覆盖 · 契约验证 · CI门禁
    FixAgent   — 自主修复 · 最小变更 · 验证闭环
    ArchAgent  — 依赖治理 · 配置统一 · 架构红线
    MultiAgentOrchestrator — 编排三Agent协作循环

设计原则（LoopRecursive-CEO Phase A）：
    K1 依赖图必须有向无环(DAG) — ArchAgent 守卫
    K2 上下文压缩必须控制LLM输入 — FixAgent 修复
    K3 工作流必须先解析变量再执行 — FixAgent 实现
    K4 关键路径必须有回归测试 — TestAgent 覆盖
    K5 复杂任务需要专业化Agent分工 — 编排器协调
"""
from .base import AgentBase, AgentResult, AgentRole
from .test_agent import TestAgent
from .fix_agent import FixAgent
from .arch_agent import ArchAgent
from .orchestrator import MultiAgentOrchestrator

__all__ = [
    "AgentBase",
    "AgentResult",
    "AgentRole",
    "TestAgent",
    "FixAgent",
    "ArchAgent",
    "MultiAgentOrchestrator",
]
