"""Loop 引擎 — LoopRecursive-CEO 的 Phase B 自主交付循环。

第一性原理（K3）：Loop 是编排 ReAct 步骤的元工作流，而非替代。
八阶段：Plan → Select Engine → Retrieve Demo → Generate Workflow → Run → Validate → Fix → Save
每轮记录 LoopLedgerEntry，确保可追溯、可回放、可复用。

与现有 AerospaceAgent 的集成方式：
    LoopEngine 不替代 ReAct 循环，而是在其之上编排：
    - Plan 阶段调用递归第一性原理分析（Phase A 递归下降）
    - Run 阶段将 WorkflowSpec 的 steps 分发给 ReAct agent 执行
    - Validate/Fix 阶段驱动 ReAct agent 做验证与修复
"""
from .engine import LoopEngine, LoopContext
from .recursion import FirstPrinciplesAnalyzer, DecisionNode

__all__ = ["LoopEngine", "LoopContext", "FirstPrinciplesAnalyzer", "DecisionNode"]
