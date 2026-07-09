"""工具适配器 — 将现有 Tool 对象包装为 langchain_core BaseTool。

支持两种工具格式:
    - ``Tool`` (dataclass): name + description + func
    - ``BaseTool`` (MCP): BaseTool.call(method, **kwargs) 接口
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool as LCBaseTool
from pydantic import BaseModel, Field, create_model


class ToolAdapter(LCBaseTool):
    """将现有 Tool 包装为 langchain_core BaseTool。

    Args:
        name: 工具名称
        description: 工具描述
        func: 实际调用函数
        args_schema: 参数 schema（Pydantic model），None 表示接受任意 dict
    """

    name: str
    description: str
    func: Callable[..., Any] = Field(exclude=True)
    args_schema: Optional[Type[BaseModel]] = None

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行工具调用，返回字符串结果。"""
        try:
            result = self.func(**kwargs)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, default=str)
            return str(result)
        except TypeError as e:
            return (
                f"[TOOL ERROR] 工具 '{self.name}' 参数错误: {e}\n"
                f"提示: 请检查参数名和类型，或调用 list_tools 查看 '{self.name}' 的正确参数。"
            )
        except Exception as e:
            return (
                f"[TOOL ERROR] 工具 '{self.name}' 执行异常: {e}\n"
                f"提示: 如再次失败，请换用其他工具或调用 list_tools。"
            )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)


class _EmptyInput(BaseModel):
    """空参数 schema（工具无参数时使用），接受任意额外字段。"""
    class Config:
        extra = "allow"


def wrap_tools(
    native_tools: List[Any],
    mcp_tools: Optional[Dict[str, Any]] = None,
) -> List[ToolAdapter]:
    """将原生 Tool 和 MCP BaseTool 统一包装为 ToolAdapter 列表。

    Args:
        native_tools: 原生 Tool 对象列表（有 name/description/func 属性）
        mcp_tools: MCP 工具字典 {name: BaseTool}

    Returns:
        ToolAdapter 列表
    """
    wrapped = []

    # 原生 Tool
    for tool in native_tools:
        name = getattr(tool, "name", "unknown")
        desc = getattr(tool, "description", "")
        func = getattr(tool, "func", None) or getattr(tool, "__call__", None) or (lambda **kw: "工具不可用")
        wrapped.append(ToolAdapter(name=name, description=desc, func=func))

    # MCP BaseTool
    if mcp_tools:
        from .mcp_tool_adapter import MCPToolAdapter
        for name, bt in mcp_tools.items():
            desc = getattr(bt, "description", f"MCP 工具: {name}")
            wrapped.append(MCPToolAdapter(
                name=name,
                description=desc,
                mcp_tool=bt,
            ))

    return wrapped