"""MCP 工具基类与注册表。

定义所有航天工具的统一抽象接口：

- ``BaseTool``：抽象基类，规定 ``call(method, **kwargs)`` 统一入口，
  返回 ``{success, source, result, message}`` 标准格式。
- ``ToolRegistry``：工具注册表，支持注册 / 查询 / 按可用性筛选。
- ``register_tool``：类装饰器，自动将工具类注册到指定注册表。

设计原则：库可用则调用真实库（source='real'），不可用则回退到内置
物理引擎或解析公式（source='fallback'），二者皆无则返回明确
"需安装"提示（source='unavailable'）。
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class BaseTool(ABC):
    """所有 MCP 风格航天工具的抽象基类。

    子类需设置类属性并实现 ``call`` 方法：

    - ``name``：工具唯一标识（如 'orekit'）
    - ``description``：工具用途描述
    - ``library_name``：所依赖的真实库导入名（如 'orekit'）
    - ``methods_schema``：支持的方法签名字典
    """

    # ---- 子类必须覆写的类属性 ----
    name: str = "base"
    description: str = "未定义"
    library_name: Optional[str] = None
    methods_schema: Dict[str, Dict[str, Any]] = {}

    # ---- 内部缓存 ----
    _availability_cache: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # 可用性检测
    # ------------------------------------------------------------------
    def _check_available(self) -> bool:
        """检测依赖库是否可导入，结果按 library_name 缓存。

        Returns
        -------
        bool
            True 表示真实库可用，False 表示需回退。
        """
        if self.library_name is None:
            return False
        cache_key = self.library_name
        if cache_key in BaseTool._availability_cache:
            return BaseTool._availability_cache[cache_key]
        try:
            importlib.import_module(self.library_name)
            available = True
        except ImportError:
            available = False
        BaseTool._availability_cache[cache_key] = available
        return available

    @property
    def is_available(self) -> bool:
        """真实库是否可用（只读属性，便于外部查询）。"""
        return self._check_available()

    @property
    def source(self) -> str:
        """当前工具的数据来源标识：'real' 或 'fallback'。"""
        return "real" if self.is_available else "fallback"

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------
    @abstractmethod
    def call(self, method: str, **kwargs) -> dict:
        """统一调用入口。

        Parameters
        ----------
        method : str
            要调用的方法名，应为 ``list_methods()`` 中的成员。
        **kwargs
            方法参数。

        Returns
        -------
        dict
            标准返回格式::

                {
                    'success': bool,
                    'source': 'real' | 'fallback' | 'unavailable',
                    'result': Any,      # 成功时的结果
                    'error': str,       # 失败时的错误信息（success=False 时）
                    'message': str,     # 人类可读说明
                }
        """

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        """返回工具元信息。"""
        return {
            "name": self.name,
            "description": self.description,
            "library_name": self.library_name,
            "available": self.is_available,
            "source": self.source,
            "methods": self.list_methods(),
            "methods_schema": self.methods_schema,
        }

    def list_methods(self) -> List[str]:
        """列出该工具支持的方法名。"""
        return list(self.methods_schema.keys())

    # ------------------------------------------------------------------
    # 辅助构造返回
    # ------------------------------------------------------------------
    @staticmethod
    def _ok(result: Any, source: str, message: str = "") -> dict:
        """构造成功返回。"""
        return {
            "success": True,
            "source": source,
            "result": result,
            "error": None,
            "message": message,
        }

    @staticmethod
    def _fail(error: str, source: str, message: str = "") -> dict:
        """构造失败返回。"""
        return {
            "success": False,
            "source": source,
            "result": None,
            "error": error,
            "message": message,
        }

    @staticmethod
    def _unavailable(method: str, library: str, install_hint: str = "") -> dict:
        """构造"不可用"返回（无回退路径时使用）。"""
        hint = f" 安装提示: {install_hint}" if install_hint else ""
        return {
            "success": False,
            "source": "unavailable",
            "result": None,
            "error": f"方法 '{method}' 需要安装 {library} 且无内置回退实现。{hint}",
            "message": f"该功能不可用，请安装 {library}。{hint}",
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"source={self.source!r} methods={self.list_methods()}>"
        )


class ToolRegistry:
    """工具注册表：管理多个 BaseTool 实例。

    提供按名称查询、按可用性筛选等便捷方法。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    # ---- 注册 / 注销 ----
    def register(self, tool: BaseTool) -> BaseTool:
        """注册一个工具实例。若同名已存在则覆盖。"""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"仅可注册 BaseTool 实例，收到 {type(tool)}")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> bool:
        """注销工具，返回是否成功移除。"""
        return self._tools.pop(name, None) is not None

    # ---- 查询 ----
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """按名称获取工具，不存在返回 None。"""
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> BaseTool:
        return self._tools[name]

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    # ---- 列表 / 筛选 ----
    def list_names(self) -> List[str]:
        """返回所有已注册工具名。"""
        return list(self._tools.keys())

    def list_all(self) -> Dict[str, BaseTool]:
        """返回 {name: tool} 全部工具。"""
        return dict(self._tools)

    def get_available_tools(self) -> Dict[str, BaseTool]:
        """返回真实库可用的工具 {name: tool}（source=='real'）。"""
        return {n: t for n, t in self._tools.items() if t.is_available}

    def get_fallback_tools(self) -> Dict[str, BaseTool]:
        """返回当前以回退模式运行的工具 {name: tool}。"""
        return {n: t for n, t in self._tools.items() if not t.is_available}

    def get_info_all(self) -> List[dict]:
        """返回所有工具的元信息列表。"""
        return [t.get_info() for t in self._tools.values()]

    def call(self, name: str, method: str, **kwargs) -> dict:
        """便捷调用：按工具名 + 方法名调用。"""
        tool = self.get_tool(name)
        if tool is None:
            return BaseTool._fail(
                error=f"未找到工具 '{name}'",
                source="unavailable",
                message=f"注册表中无 '{name}'，可用: {self.list_names()}",
            )
        return tool.call(method, **kwargs)


def register_tool(registry: ToolRegistry) -> Callable[[type], type]:
    """类装饰器工厂：将工具类实例化并注册到指定 registry。

    用法::

        @register_tool(my_registry)
        class MyTool(BaseTool):
            ...
    """

    def decorator(cls: type) -> type:
        instance = cls()
        registry.register(instance)
        return cls

    return decorator


if __name__ == "__main__":
    # ---- 自测 ----

    class DummyTool(BaseTool):
        name = "dummy"
        description = "测试用桩工具"
        library_name = "this_lib_does_not_exist_xyz"
        methods_schema = {
            "ping": {"params": {}, "returns": "str"},
        }

        def call(self, method: str, **kwargs) -> dict:
            if method == "ping":
                # 库不存在 -> 回退
                return self._ok("pong", "fallback", "回退响应")
            return self._fail(f"未知方法 {method}", self.source)

    reg = ToolRegistry()
    tool = reg.register(DummyTool())
    print("工具列表:", reg.list_names())
    print("工具信息:", tool.get_info())
    print("可用工具:", reg.get_available_tools())
    print("回退工具:", reg.get_fallback_tools())
    print("调用 ping:", reg.call("dummy", "ping"))
    print("调用未知:", reg.call("dummy", "nope"))

    @register_tool(reg)
    class AnotherTool(BaseTool):
        name = "another"
        description = "装饰器注册的工具"
        library_name = None
        methods_schema = {"echo": {}}

        def call(self, method: str, **kwargs) -> dict:
            return self._ok(kwargs.get("msg"), "fallback", "echo")

    print("装饰器注册后:", reg.list_names())
    print("echo:", reg.call("another", "echo", msg="hello"))
