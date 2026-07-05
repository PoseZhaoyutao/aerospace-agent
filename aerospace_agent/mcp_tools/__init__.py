"""aerospace_agent.mcp_tools —— MCP 风格航天工具接口包。

提供 6 个航天专业工具的统一接口，遵循"库可用则调用真实库，
不可用则优雅回退到内置物理引擎或返回明确'需安装'提示"的设计原则。

工具清单
--------
- OrekitTool      : 高精度轨道传播与坐标系转换（orekit）
- GmatTool        : 任务设计与脚本生成（GMAT 独立应用）
- SpiceypyTool    : 星历查询与月球状态（spiceypy）
- AstropyTool     : 时间/坐标/恒星时（astropy）
- BasiliskTool    : 仿真与 3D 轨迹可视化（Basilisk）
- StkTool         : STK COM 自动化（comtypes + STK）

统一调用入口
------------
每个工具通过 ``call(method, **kwargs)`` 调用，返回标准格式::

    {
        'success': bool,
        'source': 'real' | 'fallback' | 'unavailable',
        'result': Any,        # 成功时的结果
        'error': str | None,  # 失败时的错误信息
        'message': str,       # 人类可读说明
    }

快速使用
--------
::

    from aerospace_agent.mcp_tools import default_registry, get_tool

    orekit = get_tool("orekit")
    res = orekit.call("propagate",
                      initial_state=[6778e3, 0, 0, 0, 7660, 0],
                      times=[0, 1800, 3600])
    print(res["source"], res["result"]["states"])
"""

from __future__ import annotations

from .base import BaseTool, ToolRegistry, register_tool
from .orekit_tool import OrekitTool
from .gmat_tool import GmatTool
from .spiceypy_tool import SpiceypyTool
from .astropy_tool import AstropyTool
from .basilisk_tool import BasiliskTool
from .stk_tool import StkTool
from .registry import (
    default_registry,
    tool_registry,
    get_available_tools,
    get_tool,
    list_tools,
    get_status_summary,
)

__all__ = [
    # 基类
    "BaseTool",
    "ToolRegistry",
    "register_tool",
    # 工具类
    "OrekitTool",
    "GmatTool",
    "SpiceypyTool",
    "AstropyTool",
    "BasiliskTool",
    "StkTool",
    # 注册表与便捷函数
    "default_registry",
    "tool_registry",
    "get_available_tools",
    "get_tool",
    "list_tools",
    "get_status_summary",
]

__version__ = "0.1.0"
