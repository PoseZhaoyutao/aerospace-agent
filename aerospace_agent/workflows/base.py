"""工作流基类与注册表 (Workflow base classes and registry)。

本模块定义航天任务工作流的基础设施，供 Agent 编排器调用：

    * :class:`WorkflowResult` — 工作流执行结果的数据结构 (dataclass)。
    * :class:`BaseWorkflow`   — 工作流抽象基类，规定 ``execute`` 等接口。
    * :class:`WorkflowRegistry` — 工作流注册表，支持注册 / 查询 / 列出。
    * :func:`register_workflow` — 类装饰器，自动将工作流实例注册到注册表。

设计原则
--------
* 工作流是"步骤化"的：每一步执行都记录到 ``WorkflowResult.steps_log``，
  便于 Agent 追踪与报告。
* 物理计算真实调用 ``aerospace_agent.physics``，工具调用统一走
  ``aerospace_agent.mcp_tools`` 的 ``call(method, **kwargs)`` 接口。
* ``get_plan()`` 仅返回步骤计划文本，不执行，供 Agent 上下文使用。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 工作流执行结果
# ---------------------------------------------------------------------------
@dataclass
class WorkflowResult:
    """工作流执行结果。

    Attributes
    ----------
    success : bool
        工作流是否成功完成。
    result : Any
        工作流的主要输出 (字典 / 对象，由具体工作流定义)。
    steps_log : list[dict]
        步骤执行日志，每项为 dict，至少含 ``step`` / ``status`` / ``detail``。
    artifacts : list[str]
        产出物文件路径列表 (CSV / JSON / PNG 等)。
    summary : str
        人类可读的执行摘要。
    metadata : dict
        额外元信息 (执行耗时、工具来源、参数快照等)。
    """

    success: bool = False
    result: Any = None
    steps_log: List[dict] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    summary: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 工作流抽象基类
# ---------------------------------------------------------------------------
class BaseWorkflow(ABC):
    """航天任务工作流抽象基类。

    子类须设置以下类属性并实现 :meth:`execute`：

    - ``name``         : 工作流唯一标识 (如 'lunar_transfer')。
    - ``description``  : 工作流用途描述。
    - ``version``      : 版本号。
    - ``required_tools``: 所需 MCP 工具名列表 (如 ['spiceypy', 'basilisk'])。
    - ``steps``        : 步骤定义列表，每项为 dict，含 ``name`` 与 ``description``。
    """

    # ---- 子类应覆写的类属性 ----
    name: str = "base"
    description: str = "未定义工作流"
    version: str = "0.1.0"
    required_tools: List[str] = []
    steps: List[dict] = []

    # ------------------------------------------------------------------
    # 抽象接口
    # ------------------------------------------------------------------
    @abstractmethod
    def execute(self, **params) -> WorkflowResult:
        """执行工作流，返回 :class:`WorkflowResult`。

        子类实现应：
            1. 调用 :meth:`validate_params` 校验参数；
            2. 创建 :class:`WorkflowResult`；
            3. 按步骤执行，每步用 :meth:`_log_step` 记录日志；
            4. 填充 result / artifacts / summary / metadata；
            5. 返回结果。
        """

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> bool:
        """校验参数是否满足工作流要求。

        默认实现：检查 ``params`` 为字典且非 None。子类可覆写以增加
        具体校验逻辑 (类型、范围、互斥等)。

        Returns
        -------
        bool
            True 表示参数合法。
        """
        if params is None:
            return False
        if not isinstance(params, dict):
            return False
        return True

    # ------------------------------------------------------------------
    # 步骤计划 (不执行)
    # ------------------------------------------------------------------
    def get_plan(self) -> List[str]:
        """返回步骤计划文本列表，供 Agent 上下文使用 (不执行)。

        格式::

            ['1. <step_name>: <description>', '2. ...', ...]
        """
        plan: List[str] = []
        for idx, step in enumerate(self.steps, start=1):
            sname = step.get("name", f"step{idx}")
            sdesc = step.get("description", "")
            plan.append(f"{idx}. {sname}: {sdesc}")
        return plan

    # ------------------------------------------------------------------
    # 工具可用性检查
    # ------------------------------------------------------------------
    def check_tools(self) -> Dict[str, bool]:
        """检查所需工具的可用性。

        通过 ``aerospace_agent.mcp_tools.default_registry`` 查询每个工具
        是否存在且其依赖库可导入 (即 ``is_available``)。

        Returns
        -------
        dict
            ``{tool_name: bool}``，True 表示工具存在且真实库可用。
        """
        availability: Dict[str, bool] = {}
        if not self.required_tools:
            return availability
        try:
            # 延迟导入，避免循环依赖
            from aerospace_agent.mcp_tools import default_registry
            for tname in self.required_tools:
                tool = default_registry.get_tool(tname)
                availability[tname] = bool(tool is not None and tool.is_available)
        except Exception:
            # mcp_tools 不可用时，所有工具标记为不可用
            for tname in self.required_tools:
                availability[tname] = False
        return availability

    # ------------------------------------------------------------------
    # 步骤日志辅助方法 (供子类使用)
    # ------------------------------------------------------------------
    @staticmethod
    def _log_step(
        result: WorkflowResult,
        step: str,
        status: str = "success",
        detail: str = "",
        data: Any = None,
    ) -> dict:
        """向 ``result.steps_log`` 追加一条步骤记录。

        Parameters
        ----------
        result : WorkflowResult
            被记录的结果对象。
        step : str
            步骤名称。
        status : str
            步骤状态：'success' / 'failed' / 'skipped' / 'warning'。
        detail : str
            人类可读的步骤说明。
        data : Any, optional
            该步骤产生的数据 (可为 dict / 数值等)。

        Returns
        -------
        dict
            已追加的日志条目。
        """
        entry = {
            "step": step,
            "status": status,
            "detail": detail,
        }
        if data is not None:
            entry["data"] = data
        result.steps_log.append(entry)
        return entry

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        """返回工作流元信息。"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "required_tools": list(self.required_tools),
            "steps": list(self.steps),
            "tools_available": self.check_tools(),
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"version={self.version!r} steps={len(self.steps)}>"
        )


# ---------------------------------------------------------------------------
# 工作流注册表
# ---------------------------------------------------------------------------
class WorkflowRegistry:
    """工作流注册表：管理多个 :class:`BaseWorkflow` 实例。

    提供按名称查询、列出、获取元信息等便捷方法。
    """

    def __init__(self) -> None:
        self._workflows: Dict[str, BaseWorkflow] = {}

    # ---- 注册 / 注销 ----
    def register(self, workflow: BaseWorkflow) -> BaseWorkflow:
        """注册一个工作流实例。若同名已存在则覆盖。"""
        if not isinstance(workflow, BaseWorkflow):
            raise TypeError(
                f"仅可注册 BaseWorkflow 实例，收到 {type(workflow)}"
            )
        self._workflows[workflow.name] = workflow
        return workflow

    def unregister(self, name: str) -> bool:
        """注销工作流，返回是否成功移除。"""
        return self._workflows.pop(name, None) is not None

    # ---- 查询 ----
    def get_workflow(self, name: str) -> Optional[BaseWorkflow]:
        """按名称获取工作流，不存在返回 None。"""
        return self._workflows.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._workflows

    def __getitem__(self, name: str) -> BaseWorkflow:
        return self._workflows[name]

    def __iter__(self):
        return iter(self._workflows.values())

    def __len__(self) -> int:
        return len(self._workflows)

    # ---- 列表 ----
    def list_names(self) -> List[str]:
        """返回所有已注册工作流名。"""
        return list(self._workflows.keys())

    def list_all(self) -> Dict[str, BaseWorkflow]:
        """返回 {name: workflow} 全部工作流。"""
        return dict(self._workflows)

    def list_workflows(self) -> Dict[str, dict]:
        """返回 {name: info}，info 为各工作流的元信息。"""
        return {name: wf.get_info() for name, wf in self._workflows.items()}

    def get_info_all(self) -> List[dict]:
        """返回所有工作流的元信息列表。"""
        return [wf.get_info() for wf in self._workflows.values()]

    def execute(self, name: str, **params) -> WorkflowResult:
        """便捷调用：按工作流名执行。"""
        wf = self.get_workflow(name)
        if wf is None:
            res = WorkflowResult(success=False)
            res.summary = f"未找到工作流 '{name}'，可用: {self.list_names()}"
            res.metadata["error"] = f"workflow '{name}' not found"
            return res
        return wf.execute(**params)


# ---------------------------------------------------------------------------
# 模块级默认注册表与装饰器
# ---------------------------------------------------------------------------
# 全局默认工作流注册表 (模块级单例)。
workflow_registry: WorkflowRegistry = WorkflowRegistry()


def register_workflow(
    registry: Optional[WorkflowRegistry] = None,
) -> Callable[[type], type]:
    """类装饰器工厂：将工作流类实例化并注册到指定 registry。

    若 ``registry`` 为 None，则注册到模块级 :data:`workflow_registry`。

    用法::

        @register_workflow()
        class MyWorkflow(BaseWorkflow):
            name = "my_workflow"
            ...

    注意
    ----
    装饰器返回原类 (不替换类)，仅产生注册副作用，故被装饰类仍可正常
    被实例化、继承。注册发生在装饰器执行时 (类定义阶段)。
    """

    target_registry = registry if registry is not None else workflow_registry

    def decorator(cls: type) -> type:
        if not issubclass(cls, BaseWorkflow):
            raise TypeError(
                f"@register_workflow 仅可用于 BaseWorkflow 子类，收到 {cls}"
            )
        instance = cls()
        target_registry.register(instance)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.workflows.base 自测 ===")

    # --- WorkflowResult 默认值 ---
    r = WorkflowResult()
    print(f"默认 WorkflowResult: success={r.success}, steps_log={r.steps_log}, "
          f"artifacts={r.artifacts}")
    assert r.success is False
    assert r.steps_log == []
    assert r.artifacts == []

    # --- 定义一个测试工作流 ---
    reg = WorkflowRegistry()

    @register_workflow(reg)
    class _DemoWorkflow(BaseWorkflow):
        name = "demo"
        description = "演示工作流"
        version = "1.0.0"
        required_tools = ["spiceypy"]
        steps = [
            {"name": "s1", "description": "第一步"},
            {"name": "s2", "description": "第二步"},
        ]

        def execute(self, **params) -> WorkflowResult:
            res = WorkflowResult()
            self._log_step(res, "s1", "success", "第一步完成")
            self._log_step(res, "s2", "success", "第二步完成", data={"x": 1})
            res.success = True
            res.result = {"demo": True}
            res.summary = "演示工作流执行成功"
            return res

    print(f"\n注册后工作流列表: {reg.list_names()}")
    assert "demo" in reg
    assert len(reg) == 1

    wf = reg.get_workflow("demo")
    print(f"get_workflow('demo') -> {wf!r}")

    # get_plan (不执行)
    plan = wf.get_plan()
    print(f"get_plan(): {plan}")
    assert plan == ["1. s1: 第一步", "2. s2: 第二步"]

    # check_tools
    tools = wf.check_tools()
    print(f"check_tools(): {tools}")
    assert "spiceypy" in tools

    # validate_params
    assert wf.validate_params({"a": 1}) is True
    assert wf.validate_params(None) is False

    # execute
    result = reg.execute("demo")
    print(f"\nexecute('demo'): success={result.success}, "
          f"steps_log={len(result.steps_log)} 条, summary='{result.summary}'")
    assert result.success is True
    assert len(result.steps_log) == 2
    assert result.steps_log[1]["data"] == {"x": 1}

    # list_workflows
    infos = reg.list_workflows()
    print(f"\nlist_workflows() keys: {list(infos.keys())}")
    assert "demo" in infos
    assert infos["demo"]["version"] == "1.0.0"

    # 未找到工作流
    miss = reg.execute("not_exist")
    assert miss.success is False

    # 模块级默认 registry + 无参装饰器
    @register_workflow()
    class _GlobalWorkflow(BaseWorkflow):
        name = "global_demo"
        description = "全局注册演示"

        def execute(self, **params) -> WorkflowResult:
            res = WorkflowResult(success=True, summary="ok")
            return res

    print(f"\n模块级 workflow_registry: {workflow_registry.list_names()}")
    assert "global_demo" in workflow_registry

    print("\nbase 自测全部通过.")
