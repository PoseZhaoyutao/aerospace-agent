"""adapters —— 7 引擎适配器统一入口。

每个适配器都是可选插件（懒加载），未安装对应引擎时 is_available() 返回 False，
但不影响模块导入与 MCP server 启动。

工厂函数 get_all_adapters() 返回 {引擎名: 适配器实例} 字典，供 Loop 引擎按能力选型。
"""
from __future__ import annotations

from typing import Dict, List

from .base import BaseAdapter, AdapterError
from .orekit_adapter import OrekitAdapter
from .gmat_adapter import GMATAdapter
from .spiceypy_adapter import SpiceyPyAdapter
from .astropy_adapter import AstropyAdapter
from .poliastro_adapter import PoliastroAdapter
from .basilisk_adapter import BasiliskAdapter
from .stk_adapter import STKAdapter

__all__ = [
    "BaseAdapter", "AdapterError",
    "OrekitAdapter", "GMATAdapter", "SpiceyPyAdapter", "AstropyAdapter",
    "PoliastroAdapter", "BasiliskAdapter", "STKAdapter",
    "get_all_adapters", "get_adapter", "ADAPTER_REGISTRY",
]

#: 引擎名 → 适配器类的注册表（便于按名实例化）
ADAPTER_REGISTRY: Dict[str, type] = {
    "orekit": OrekitAdapter,
    "gmat": GMATAdapter,
    "spiceypy": SpiceyPyAdapter,
    "astropy": AstropyAdapter,
    "poliastro": PoliastroAdapter,
    "basilisk": BasiliskAdapter,
    "stk": STKAdapter,
}

#: 全部适配器类（保持顺序）
_ALL_ADAPTER_CLASSES: List[type] = [
    OrekitAdapter, GMATAdapter, SpiceyPyAdapter, AstropyAdapter,
    PoliastroAdapter, BasiliskAdapter, STKAdapter,
]


def get_all_adapters() -> Dict[str, BaseAdapter]:
    """实例化并返回全部 7 个适配器，以 {引擎名: 实例} 字典形式返回。

    每个适配器的 __init__ 都是轻量的（不触发引擎加载），可安全批量构造。
    引擎真正按需在能力方法内部懒加载。
    """
    result: Dict[str, BaseAdapter] = {}
    for cls in _ALL_ADAPTER_CLASSES:
        instance = cls()
        result[instance.engine_name] = instance
    return result


def get_adapter(name: str) -> BaseAdapter:
    """按引擎名获取单个适配器实例。

    K5-M3: 使用单例缓存，避免每次调用创建新实例
    （STK COM 连接、Orekit JVM 初始化、GMAT 缓存丢失）。

    Args:
        name: 引擎名（orekit/gmat/spiceypy/astropy/poliastro/basilisk/stk）
    Returns:
        适配器实例。未知名时返回 _UnavailableAdapter 占位（不抛异常）。
    """
    # K5-M3: 单例缓存
    if not hasattr(get_adapter, "_instances"):
        get_adapter._instances = {}
    cache_key = name.lower()
    if cache_key in get_adapter._instances:
        return get_adapter._instances[cache_key]

    cls = ADAPTER_REGISTRY.get(cache_key)
    if cls is None:
        from .base import BaseAdapter

        class _UnavailableAdapter(BaseAdapter):
            engine_name = cache_key

            def is_available(self) -> bool:
                return False

            def version(self) -> str:
                return "unavailable"

            def capabilities(self) -> set:
                return set()

        instance = _UnavailableAdapter()
    else:
        instance = cls()
    get_adapter._instances[cache_key] = instance
    return instance
