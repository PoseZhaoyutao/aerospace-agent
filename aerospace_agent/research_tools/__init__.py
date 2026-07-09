"""research_tools 包——100+ 科研工具的统一入口。

导入此包即自动注册所有工具到全局 ResearchToolRegistry。

工具分类（10 个域，105 个原子操作）：
  - file_io:          15 个文件/IO 工具
  - data_processing:  15 个数据处理工具
  - math_compute:     15 个数学计算工具
  - visualization:    10 个可视化工具
  - text_document:    10 个文本/文档工具
  - web_network:      10 个网络/Web 工具
  - scientific_ref:   10 个科学引用工具
  - code_execution:   10 个代码执行工具
  - system_env:        5 个系统/环境工具
  - self_evolution:    5 个自进化工具

用法：
    from aerospace_agent.research_tools import get_registry
    registry = get_registry()
    print(registry.get_summary())        # 打印统计
    result = registry.call("save_file", path="a.txt", content="hello")
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, List, Optional

from .base import (
    ResearchTool,
    ResearchToolRegistry,
    ParamSpec,
    get_registry,
    register_tool,
)

_logger = logging.getLogger(__name__)

# 所有工具模块（按依赖顺序）
_TOOL_MODULES = [
    "file_io",
    "data_processing",
    "math_compute",
    "visualization",
    "text_document",
    "web_network",
    "scientific_ref",
    "code_execution",
    "system_env",
    "self_evolution",
]

_loaded = False


def _load_all() -> None:
    """懒加载所有工具模块——仅首次调用时执行。"""
    global _loaded
    if _loaded:
        return
    _loaded = True

    pkg_dir = os.path.dirname(__file__)
    for mod_name in _TOOL_MODULES:
        full_name = f"{__name__}.{mod_name}"
        try:
            importlib.import_module(full_name)
            _logger.debug("已加载工具模块: %s", full_name)
        except Exception as e:
            _logger.warning("工具模块 %s 加载失败: %s", full_name, e)


def _ensure_loaded() -> ResearchToolRegistry:
    """确保所有工具已加载，返回注册表。"""
    _load_all()
    return get_registry()


def get_all_schemas() -> List[str]:
    """获取所有工具的紧凑说明（供 LLM 系统提示）。"""
    return _ensure_loaded().get_schemas()


def get_all_json_schemas() -> List[Dict[str, Any]]:
    """获取所有工具的 JSON Schema（供 MCP 协议）。"""
    return _ensure_loaded().get_json_schemas()


def call_tool(name: str, **kwargs) -> Any:
    """按名称调用科研工具。"""
    return _ensure_loaded().call(name, **kwargs)


def list_tools(category: Optional[str] = None) -> List[str]:
    """列出工具名（可按分类过滤）。"""
    reg = _ensure_loaded()
    if category:
        return reg.list_by_category(category)
    return reg.list_all()


def get_tool_count() -> int:
    """获取已注册工具总数。"""
    return _ensure_loaded().count()


def get_categories() -> Dict[str, List[str]]:
    """获取分类→工具名列表映射。"""
    return _ensure_loaded().categories()


# 预加载（首次 import 时触发）
_load_all()

__all__ = [
    "ResearchTool",
    "ResearchToolRegistry",
    "ParamSpec",
    "get_registry",
    "register_tool",
    "get_all_schemas",
    "get_all_json_schemas",
    "call_tool",
    "list_tools",
    "get_tool_count",
    "get_categories",
]
