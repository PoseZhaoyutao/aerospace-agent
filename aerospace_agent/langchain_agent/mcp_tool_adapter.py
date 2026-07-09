"""MCP 工具适配器 — 替代 wrap_tools 中的粗糙闭包。

MCPToolAdapter 直接持有 MCP BaseTool 引用，支持：
    - 自动探测可用方法（list_methods）
    - 智能方法选择（默认第一个方法）
    - 参数透传 + JSON 序列化返回
    - 清晰的错误信息
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool as LCBaseTool


class MCPToolAdapter(LCBaseTool):
    """将 MCP BaseTool 包装为 langchain_core BaseTool。

    Args:
        name: 工具名称
        description: 工具描述
        mcp_tool: MCP BaseTool 实例（有 call(method, **kwargs) 接口）
    """

    name: str
    description: str
    _mcp_tool: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, name: str, description: str, mcp_tool: Any, **kwargs: Any) -> None:
        super().__init__(name=name, description=description, **kwargs)
        self._mcp_tool = mcp_tool

    @property
    def _methods(self) -> List[str]:
        """获取 MCP 工具可用方法列表。"""
        list_methods = getattr(self._mcp_tool, "list_methods", None)
        if list_methods:
            try:
                return list_methods()
            except Exception:
                pass
        # 回退：尝试常用方法名
        for method in ("call", "execute", "run", "invoke"):
            if hasattr(self._mcp_tool, method):
                return [method]
        return ["call"]

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 MCP 工具调用。"""
        # 提取用户指定的 method，否则默认第一个可用方法
        method = kwargs.pop("method", None)
        if not method:
            method = self._methods[0]

        try:
            result = self._mcp_tool.call(method, **kwargs)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False, default=str)
            return str(result)
        except TypeError as e:
            # 参数错误
            return (
                f"[TOOL ERROR] MCP 工具 '{self.name}' 参数错误: {e}\n"
                f"可用方法: {', '.join(self._methods)}\n"
                f"提示: 请检查方法名和参数，或调用 list_tools 查看 '{self.name}' 的正确用法。"
            )
        except Exception as e:
            return (
                f"[TOOL ERROR] MCP 工具 '{self.name}' 执行异常: {e}\n"
                f"提示: 如再次失败，请换用其他工具。"
            )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        return self._run(*args, **kwargs)