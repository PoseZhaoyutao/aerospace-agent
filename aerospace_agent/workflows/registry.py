"""工作流注册表 (Workflow registry)。

导入所有工作流模块 (触发 ``@register_workflow()`` 装饰器自动注册到
``base.workflow_registry``)，并对外提供：

    * ``default_workflow_registry`` — 已注册全部工作流实例的注册表
    * ``get_workflow(name)``       — 按名获取工作流实例
    * ``list_workflows()``         — 列出全部工作流元信息

注册的工作流：
    - orbit_design       (OrbitDesignWorkflow)
    - launch_window      (LaunchWindowWorkflow)
    - lunar_transfer     (TrajectoryAnalysisWorkflow)
    - basilisk_viz       (BasiliskVisualizationWorkflow)
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import BaseWorkflow, WorkflowRegistry, workflow_registry

# 导入各工作流模块, 触发 @register_workflow() 装饰器自动注册。
# 导入顺序: 基类 -> 各工作流 (依赖关系: trajectory_analysis 依赖 launch_window)
from .orbit_design import OrbitDesignWorkflow  # noqa: F401
from .launch_window import LaunchWindowWorkflow  # noqa: F401
from .trajectory_analysis import TrajectoryAnalysisWorkflow  # noqa: F401
from .basilisk_visualization import BasiliskVisualizationWorkflow  # noqa: F401


# 全局默认工作流注册表 (与 base.workflow_registry 同一实例, 已含全部工作流)
default_workflow_registry: WorkflowRegistry = workflow_registry


def get_workflow(name: str) -> Optional[BaseWorkflow]:
    """按名称获取工作流实例，不存在返回 None。

    Parameters
    ----------
    name : str
        工作流名 (如 'lunar_transfer', 'orbit_design')。
    """
    return default_workflow_registry.get_workflow(name)


def list_workflows() -> Dict[str, dict]:
    """返回 ``{name: info}``，info 为各工作流的元信息字典。

    每个 info 含: name, description, version, required_tools, steps,
    tools_available。
    """
    return default_workflow_registry.list_workflows()


def list_workflow_names() -> list:
    """返回所有已注册工作流名列表。"""
    return default_workflow_registry.list_names()


def execute_workflow(name: str, **params):
    """便捷调用：按工作流名执行并返回 WorkflowResult。"""
    return default_workflow_registry.execute(name, **params)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.workflows.registry 自测 ===")
    print(f"已注册工作流数: {len(default_workflow_registry)}")
    print(f"工作流名: {list_workflow_names()}")

    print("\n工作流状态总览:")
    for name, info in list_workflows().items():
        tools = info["tools_available"]
        tools_str = ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in tools.items()) or "无"
        print(f"  {name:18s} v{info['version']:5s} 步骤数={len(info['steps'])} "
              f"工具=[{tools_str}]")
        print(f"    描述: {info['description']}")

    # 按名获取
    wf = get_workflow("lunar_transfer")
    print(f"\nget_workflow('lunar_transfer') -> {wf!r}")
    assert wf is not None
    assert wf.name == "lunar_transfer"

    # 不存在的工作流
    miss = get_workflow("not_exist")
    assert miss is None

    # 确认全部 4 个工作流已注册
    expected = {"orbit_design", "launch_window", "lunar_transfer", "basilisk_viz"}
    assert set(list_workflow_names()) == expected, \
        f"工作流不匹配, 期望 {expected}, 实际 {set(list_workflow_names())}"

    print(f"\n>>> 校验通过: 全部 {len(expected)} 个工作流已注册")
