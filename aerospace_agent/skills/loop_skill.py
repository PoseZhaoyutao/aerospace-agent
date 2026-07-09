"""Loop 编排技能 —— 触发 LoopEngine 执行复杂任务的自主交付循环。

LoopEngine 是 LoopRecursive-CEO 的 Phase B，八阶段循环:
Plan -> SelectEngine -> RetrieveDemo -> GenerateWorkflow -> Run -> Validate -> Fix -> Save
在 ReAct 循环之上编排，不替代而是增强 Agent 的多步推理能力。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .base import SkillBase


class LoopOrchestrationSkill(SkillBase):
    """Loop 编排技能。

    对复杂航天任务（如"设计地月转移轨道，精度<1km"）触发 LoopEngine，
    自动完成规划、引擎选择、工作流生成、执行、验证与修复的完整循环。
    """

    name: str = "loop_orchestration"
    description: str = "Loop 编排：触发 LoopEngine 执行复杂任务的八阶段自主交付循环"
    category: str = "mcp"

    def is_available(self) -> bool:
        """LoopEngine 依赖 mcp.loop 模块，惰性检测是否可导入。"""
        try:
            from aerospace_agent.mcp.loop import LoopEngine  # noqa: F401
            return True
        except Exception:
            return False

    def execute(self, agent, **kwargs) -> dict:
        """执行 Loop 编排。

        Args:
            agent: AerospaceAgent 实例
            goal: 任务目标描述（必填，如"设计地月转移轨道"）
            constraints: 约束条件列表（默认空）
            max_iterations: Loop 最大迭代轮次（默认 3）

        Returns:
            {"success", "result": {"status", "engine", "outputs", "ledger"}, "message"}
        """
        # 惰性导入 LoopEngine（重依赖）
        try:
            from aerospace_agent.mcp.loop import LoopEngine
        except Exception as exc:
            return self._error(f"mcp.loop 模块不可用: {exc}")

        goal: str = kwargs.get("goal", "")
        if not goal or not goal.strip():
            return self._error("缺少必填参数 goal（任务目标描述）")

        constraints: List[str] = kwargs.get("constraints", [])
        max_iterations: int = kwargs.get("max_iterations", 3)

        # 从 Agent 收集可用工具（原生 Tool + MCP BaseTool）
        tools: Dict[str, Any] = {}
        for tname, tool in getattr(agent, "tools", {}).items():
            tools[tname] = tool
        for tname, bt in getattr(agent, "mcp_tools", {}).items():
            # 包装 BaseTool.call(method, **kw) 为统一可调用接口
            tools[tname] = _make_mcp_wrapper(bt)

        llm = getattr(agent, "llm", None)

        # 创建并执行 Loop 引擎
        try:
            engine = LoopEngine(llm=llm, tools=tools,
                                max_iterations=max_iterations)
            wf_result = engine.execute(goal=goal, constraints=constraints)
        except Exception as exc:
            return self._error(f"Loop 引擎执行失败: {exc}")

        # 提取结构化结果
        result: Dict[str, Any] = {
            "status": getattr(wf_result, "status", "unknown"),
            "engine": getattr(wf_result, "engine", ""),
            "workflow_id": getattr(wf_result, "workflow_id", ""),
            "outputs": _safe_dict(getattr(wf_result, "outputs", {})),
            "errors": getattr(wf_result, "errors", []),
            "ledger_count": len(getattr(wf_result, "loop_ledger", [])),
        }
        validation = getattr(wf_result, "validation", None)
        if validation is not None:
            result["validation_passed"] = getattr(validation, "passed", False)
            result["validation_confidence"] = getattr(validation, "confidence", "")

        success = result["status"] == "success"
        return {
            "success": success,
            "result": result,
            "message": f"Loop 完成：状态={result['status']}，"
                       f"引擎={result['engine']}，账本={result['ledger_count']} 条",
        }

    @staticmethod
    def _error(message: str) -> dict:
        """返回标准化错误结果。"""
        return {"success": False, "result": None, "message": message}


def _make_mcp_wrapper(bt: Any):
    """将 MCP BaseTool 包装为 (method, **kw) 统一可调用函数。"""

    def wrapper(method: str = "", **kwargs):
        methods = getattr(bt, "list_methods", lambda: [])()
        if not method and methods:
            method = methods[0]
        return bt.call(method, **kwargs)

    return wrapper


def _safe_dict(obj: Any) -> dict:
    """安全地将对象转为可序列化字典。"""
    try:
        if isinstance(obj, dict):
            return obj
        return dict(obj)
    except Exception:
        return {}
