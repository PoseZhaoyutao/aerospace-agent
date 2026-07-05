"""全局工具注册表。

创建 ``default_registry`` 并注册全部 6 个航天工具实例，提供
``get_available_tools()`` 与 ``get_tool(name)`` 便捷函数。
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import BaseTool, ToolRegistry
from .orekit_tool import OrekitTool
from .gmat_tool import GmatTool
from .spiceypy_tool import SpiceypyTool
from .astropy_tool import AstropyTool
from .basilisk_tool import BasiliskTool
from .stk_tool import StkTool

# 所有工具类（按规范顺序）
_ALL_TOOL_CLASSES = [
    OrekitTool,
    GmatTool,
    SpiceypyTool,
    AstropyTool,
    BasiliskTool,
    StkTool,
]


def _build_registry() -> ToolRegistry:
    """构建并填充默认注册表。"""
    reg = ToolRegistry()
    for cls in _ALL_TOOL_CLASSES:
        reg.register(cls())
    return reg


# 全局默认注册表（模块级单例）
default_registry: ToolRegistry = _build_registry()

# 兼容别名
tool_registry = default_registry


def get_available_tools() -> Dict[str, BaseTool]:
    """返回真实库可用的工具 {name: tool}。"""
    return default_registry.get_available_tools()


def get_tool(name: str) -> Optional[BaseTool]:
    """按名称获取工具实例，不存在返回 None。"""
    return default_registry.get_tool(name)


def list_tools() -> Dict[str, BaseTool]:
    """返回全部已注册工具 {name: tool}。"""
    return default_registry.list_all()


def get_status_summary() -> list:
    """返回所有工具的可用性状态摘要列表。"""
    return [
        {
            "name": t.name,
            "library": t.library_name,
            "available": t.is_available,
            "source": t.source,
            "methods": t.list_methods(),
        }
        for t in default_registry
    ]


if __name__ == "__main__":
    print("=== aerospace_agent.mcp_tools.registry 自测 ===")
    print(f"已注册工具数: {len(default_registry)}")
    print("\n工具状态总览:")
    for s in get_status_summary():
        status = "真实" if s["available"] else "回退"
        print(f"  {s['name']:12s} 依赖={s['library']:12s} "
              f"来源={status:4s} 方法={s['methods']}")

    print("\n可用工具 (真实模式):", list(get_available_tools().keys()))
    print("全部工具:", list(list_tools().keys()))

    # 按名称获取
    orekit = get_tool("orekit")
    print(f"\nget_tool('orekit') -> {orekit!r}")

    # 便捷调用
    astropy = get_tool("astropy")
    r = astropy.call("to_julian", date_str="2000-01-01 12:00:00")
    print(f"astropy.to_julian('2000-01-01 12:00:00') = {r['result']['jd']} "
          f"(source={r['source']})")
