"""ResearchTool 基类与注册器——100+ 科研工具的统一基础设施。

第一性原理：
  1. 每个工具是不可分解的原子操作（最小群论生成元）
  2. 统一接口：name + description + params + run() → 任意工具可互换
  3. 装饰器自动注册——新增工具只需 @register_tool，无需手动登记
  4. 懒加载——import 时不执行重依赖，首次调用才初始化
  5. 可组合——工具输出可直接作为另一工具输入（JSON 可序列化）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

_logger = logging.getLogger(__name__)


@dataclass
class ParamSpec:
    """工具参数规格。"""
    name: str
    type: str  # str | int | float | bool | list | dict | any
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ResearchTool:
    """科研工具基类——所有 100+ 工具的统一接口。

    最小群论原则：每个工具是一个不可分解的原子操作，
    通过组合可表达任意科研工作流。
    """
    name: str
    description: str
    category: str  # file_io | data_processing | math_compute | ...
    func: Callable[..., Any]
    params: List[ParamSpec] = field(default_factory=list)
    version: str = "1.0"
    author: str = "system"
    tags: List[str] = field(default_factory=list)

    def __call__(self, **kwargs) -> Any:
        """执行工具，返回 JSON 可序列化结果。"""
        # 填充默认值
        for p in self.params:
            if p.name not in kwargs and not p.required:
                kwargs[p.name] = p.default
        # 校验必填参数
        missing = [p.name for p in self.params if p.required and p.name not in kwargs]
        if missing:
            return {"status": "error", "reason": f"缺少必填参数: {missing}"}
        # 调用
        try:
            result = self.func(**kwargs)
            # 确保可序列化
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                return result
            return {"status": "success", "result": str(result)}
        except Exception as e:
            return {"status": "error", "reason": str(e), "tool": self.name}

    def to_schema(self) -> str:
        """返回紧凑的工具说明（供 LLM 系统提示）。"""
        params_str = ", ".join(
            f"{p.name}:{p.type}" + ("" if p.required else "?")
            for p in self.params
        )
        return f"- {self.name}({params_str}): {self.description}"

    def to_json_schema(self) -> Dict[str, Any]:
        """返回 JSON Schema（供 MCP 协议）。"""
        properties = {}
        required = []
        for p in self.params:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ResearchToolRegistry:
    """科研工具注册表——单例，管理所有工具的注册、查询、动态创建。"""

    _instance: Optional["ResearchToolRegistry"] = None

    def __new__(cls) -> "ResearchToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, ResearchTool] = {}
            cls._instance._categories: Dict[str, List[str]] = {}
            cls._instance._dynamic_dir: Optional[str] = None
        return cls._instance

    # ---- 注册 ----
    def register(self, tool: ResearchTool) -> None:
        """注册一个工具。"""
        self._tools[tool.name] = tool
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        if tool.name not in self._categories[tool.category]:
            self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> bool:
        """注销工具。"""
        tool = self._tools.pop(name, None)
        if tool and tool.category in self._categories:
            try:
                self._categories[tool.category].remove(name)
            except ValueError:
                pass
        return tool is not None

    # ---- 查询 ----
    def get(self, name: str) -> Optional[ResearchTool]:
        return self._tools.get(name)

    def call(self, name: str, **kwargs) -> Any:
        """按名称调用工具。"""
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "error", "reason": f"工具 '{name}' 不存在"}
        return tool(**kwargs)

    def list_all(self) -> List[str]:
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> List[str]:
        return self._categories.get(category, [])

    def categories(self) -> Dict[str, List[str]]:
        return dict(self._categories)

    def count(self) -> int:
        return len(self._tools)

    def get_schemas(self, category: Optional[str] = None) -> List[str]:
        """获取工具说明列表（紧凑格式，供 LLM）。"""
        if category:
            names = self.list_by_category(category)
        else:
            names = self.list_all()
        return [self._tools[n].to_schema() for n in names]

    def get_json_schemas(self, category: Optional[str] = None) -> List[Dict]:
        """获取 JSON Schema 列表（供 MCP）。"""
        if category:
            names = self.list_by_category(category)
        else:
            names = self.list_all()
        return [self._tools[n].to_json_schema() for n in names]

    def get_summary(self) -> str:
        """获取注册表摘要（分类统计）。"""
        lines = [f"ResearchToolRegistry: {self.count()} tools in {len(self._categories)} categories"]
        for cat, tools in sorted(self._categories.items()):
            lines.append(f"  {cat}: {len(tools)} tools")
        return "\n".join(lines)

    # ---- 动态工具创建（自进化）----
    def set_dynamic_dir(self, path: str) -> None:
        """设置动态工具保存目录。"""
        self._dynamic_dir = path

    @property
    def dynamic_dir(self) -> str:
        if self._dynamic_dir is None:
            import os
            self._dynamic_dir = os.path.join(os.getcwd(), "dynamic_tools")
        return self._dynamic_dir

    def create_dynamic_tool(
        self,
        name: str,
        description: str,
        category: str,
        code: str,
        params: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """动态创建新工具——自进化核心。

        流程：
          1. 生成 Python 文件到 dynamic_dir
          2. 动态 import
          3. 提取函数并注册到 registry
          4. 立即可用

        Args:
            name: 工具名（snake_case）
            description: 工具描述
            category: 工具分类
            code: Python 函数代码（def func(...): ...）
            params: 参数规格列表 [{name, type, description, required, default}]

        Returns:
            创建结果
        """
        import os
        import importlib.util
        import sys

        # 确保目录存在
        os.makedirs(self.dynamic_dir, exist_ok=True)

        # 安全检查：工具名必须合法
        if not name.replace("_", "").isalnum():
            return {"status": "error", "reason": "工具名必须为 snake_case 字母数字"}

        # 如果已存在，先注销旧版
        if name in self._tools:
            self.unregister(name)
            _logger.info("已注销旧版工具: %s", name)

        # 生成完整模块文件
        param_specs = params or []
        module_code = f'"""动态生成工具: {name}\n{description}\n"""\n\n'
        module_code += code
        module_code += "\n\n"
        module_code += f"_TOOL_META = {{\n"
        module_code += f"    'name': '{name}',\n"
        module_code += f"    'description': {repr(description)},\n"
        module_code += f"    'category': '{category}',\n"
        module_code += f"    'params': {repr(param_specs)},\n"
        module_code += f"}}\n"

        file_path = os.path.join(self.dynamic_dir, f"dyn_{name}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(module_code)

        # 动态 import
        try:
            spec = importlib.util.spec_from_file_location(f"dyn_{name}", file_path)
            if spec is None or spec.loader is None:
                return {"status": "error", "reason": "无法创建模块 spec"}
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"dyn_{name}"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            return {"status": "error", "reason": f"动态导入失败: {e}"}

        # 提取函数
        func = getattr(module, "func", None)
        if func is None:
            # 尝试用工具名作为函数名
            func = getattr(module, name, None)
        if func is None:
            return {"status": "error", "reason": "代码中未找到 func() 或同名函数"}

        # 构建参数规格
        p_specs = []
        for p in param_specs:
            p_specs.append(ParamSpec(
                name=p.get("name", ""),
                type=p.get("type", "any"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
            ))

        # 注册
        tool = ResearchTool(
            name=name,
            description=description,
            category=category,
            func=func,
            params=p_specs,
            version="dynamic",
            author="self_evolved",
        )
        self.register(tool)

        _logger.info("动态工具已创建并注册: %s (%s)", name, file_path)
        return {
            "status": "success",
            "tool": name,
            "file": file_path,
            "message": f"工具 '{name}' 已创建并注册，立即可用",
        }


# ---- 全局单例 ----
_registry = ResearchToolRegistry()


def get_registry() -> ResearchToolRegistry:
    """获取全局工具注册表单例。"""
    return _registry


def register_tool(
    name: str,
    description: str,
    category: str,
    params: Optional[List[Dict]] = None,
    tags: Optional[List[str]] = None,
):
    """装饰器：自动注册工具到全局注册表。

    用法：
        @register_tool("save_file", "保存内容到文件", "file_io",
                       params=[{"name":"path","type":"str","description":"文件路径"},
                               {"name":"content","type":"str","description":"文件内容"}])
        def save_file(path, content):
            with open(path, 'w') as f:
                f.write(content)
            return {"status":"success","path":path}
    """
    param_specs = []
    if params:
        for p in params:
            param_specs.append(ParamSpec(
                name=p.get("name", ""),
                type=p.get("type", "any"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
            ))

    def decorator(func: Callable) -> Callable:
        tool = ResearchTool(
            name=name,
            description=description,
            category=category,
            func=func,
            params=param_specs,
            tags=tags or [],
        )
        _registry.register(tool)
        return func

    return decorator
