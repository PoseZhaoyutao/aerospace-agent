"""工具适配器 — 将现有工具桥接到新的 Tool 接口。

现有两种工具格式：
    1. agent.py 的 Tool (name, description, func) — 简单可调用包装
    2. research_tools/base.py 的 ResearchTool (dataclass with params)

本模块将它们适配为 core/tool.py 的 Tool 接口，
使 QueryEngine 可以统一调用所有工具。
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List, Optional

from .messages import AssistantMessage, ToolResultBlock
from .permissions import PermissionResult, allow_permission
from .tool import (
    Tool as ToolInterface,
    ToolInputSchema,
    ToolParam,
    ToolResult,
    ToolUseContext,
    ValidationResult,
)


# ======================================================================
# CallableWrapperTool — 包装简单可调用对象
# ======================================================================

class CallableWrapperTool(ToolInterface):
    """将简单可调用函数包装为 Tool 接口。

    用于 agent.py 中的 Tool(name, description, func) 格式。
    """

    def __init__(self, name: str, description: str, func: Callable):
        self._name = name
        self._description = description
        self._func = func
        self._input_schema = ToolInputSchema()  # 空 schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_schema(self) -> ToolInputSchema:
        return self._input_schema

    async def call(
        self,
        args: Dict[str, Any],
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: AssistantMessage,
        on_progress: Optional[Callable] = None,
    ) -> ToolResult:
        try:
            # 调用原始函数
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(**args)
            else:
                result = self._func(**args)
            return ToolResult(data=result)
        except Exception as e:
            return ToolResult(data=f"工具执行错误: {e}")

    async def description(
        self,
        input_data: Dict[str, Any],
        options: Dict[str, Any],
    ) -> str:
        return self._description

    def is_read_only(self, input_data: Dict[str, Any]) -> bool:
        """简单工具默认不确定，返回 False（保守策略）。"""
        return False

    def is_concurrency_safe(self, input_data: Dict[str, Any]) -> bool:
        """简单工具默认不并发安全。"""
        return False

    def user_facing_name(self, input_data: Optional[Dict[str, Any]] = None) -> str:
        return self._name


# ======================================================================
# ResearchToolAdapter — 包装 ResearchTool (dataclass)
# ======================================================================

class ResearchToolAdapter(ToolInterface):
    """将 research_tools/base.py 的 ResearchTool 适配为 Tool 接口。

    ResearchTool 格式:
        name: str
        description: str
        params: List[ResearchToolParam]
        func: Callable
    """

    def __init__(self, research_tool: Any):
        self._rt = research_tool
        # 构建 input schema
        params = []
        for p in getattr(research_tool, "params", []):
            params.append(ToolParam(
                name=p.name,
                type=getattr(p, "type", "string"),
                description=getattr(p, "description", ""),
                required=getattr(p, "required", False),
                default=getattr(p, "default", None),
                enum=getattr(p, "enum", None),
            ))
        self._input_schema = ToolInputSchema.from_params(params)
        # 只读工具判断
        self._is_read_only = self._detect_read_only()

    def _detect_read_only(self) -> bool:
        """根据工具名/描述判断是否只读。"""
        name = self._rt.name.lower()
        desc = getattr(self._rt, "description", "").lower()
        # 明确的写操作关键词
        write_keywords = [
            "save", "write", "create", "delete", "remove", "execute",
            "run", "install", "modify", "update", "set",
        ]
        for kw in write_keywords:
            if kw in name or kw in desc:
                return False
        return True

    @property
    def name(self) -> str:
        return self._rt.name

    @property
    def input_schema(self) -> ToolInputSchema:
        return self._input_schema

    async def call(
        self,
        args: Dict[str, Any],
        context: ToolUseContext,
        can_use_tool: Any,
        parent_message: AssistantMessage,
        on_progress: Optional[Callable] = None,
    ) -> ToolResult:
        try:
            # ResearchTool 的 func 接受 **kwargs
            func = getattr(self._rt, "func", None)
            if func is None:
                return ToolResult(data=f"工具 {self._rt.name} 没有可调用函数")

            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return ToolResult(data=result)
        except TypeError as e:
            # 参数不匹配 — 尝试传递整个 input 作为单个参数
            try:
                if func:
                    result = func(args)
                    return ToolResult(data=result)
            except Exception:
                pass
            return ToolResult(data=f"工具参数错误: {e}")
        except Exception as e:
            return ToolResult(data=f"工具执行错误: {type(e).__name__}: {e}")

    async def description(
        self,
        input_data: Dict[str, Any],
        options: Dict[str, Any],
    ) -> str:
        return getattr(self._rt, "description", self._rt.name)

    def is_read_only(self, input_data: Dict[str, Any]) -> bool:
        return self._is_read_only

    def is_concurrency_safe(self, input_data: Dict[str, Any]) -> bool:
        """只读工具可以并发执行。"""
        return self._is_read_only

    def user_facing_name(self, input_data: Optional[Dict[str, Any]] = None) -> str:
        return self._rt.name

    def to_openai_format(self) -> Dict:
        """转为 OpenAI 工具定义格式。"""
        return {
            "type": "function",
            "function": {
                "name": self._rt.name,
                "description": getattr(self._rt, "description", ""),
                "parameters": self._input_schema.to_dict(),
            },
        }


# ======================================================================
# 批量转换函数
# ======================================================================

def wrap_callable_tools(
    tools: List[Any],
) -> List[ToolInterface]:
    """将简单 Tool(name, description, func) 列表转为 ToolInterface 列表。"""
    result = []
    for t in tools:
        if isinstance(t, ToolInterface):
            result.append(t)
        elif hasattr(t, "name") and hasattr(t, "description") and hasattr(t, "func"):
            result.append(CallableWrapperTool(t.name, t.description, t.func))
        elif callable(t):
            result.append(CallableWrapperTool(
                getattr(t, "__name__", "anonymous"),
                getattr(t, "__doc__", ""),
                t,
            ))
    return result


def wrap_research_tools(
    registry: Any,
) -> List[ToolInterface]:
    """将 ResearchToolRegistry 中的所有工具转为 ToolInterface 列表。"""
    result = []
    if registry is None:
        return result
    for tool_name in registry.list_all():
        rt = registry.get(tool_name)
        if rt is not None:
            result.append(ResearchToolAdapter(rt))
    return result


def build_tools_def_for_query(
    tool_interfaces: List[ToolInterface],
    tool_names: Optional[set] = None,
) -> List[Dict]:
    """构建 OpenAI 格式工具定义列表。

    Args:
        tool_interfaces: ToolInterface 列表
        tool_names: 只包含这些名称的工具（None 表示全部）
    """
    tools_def = []
    for tool in tool_interfaces:
        if tool_names and tool.name not in tool_names:
            continue
        if isinstance(tool, ResearchToolAdapter):
            tools_def.append(tool.to_openai_format())
        else:
            tools_def.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool._description if hasattr(tool, "_description") else tool.name,
                    "parameters": tool.input_schema.to_dict(),
                },
            })
    return tools_def
