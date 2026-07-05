"""aerospace_agent.workflows —— 航天任务工作流包。

提供可被 Agent 编排器调用的步骤化工作流。每个工作流：
    * 继承 :class:`BaseWorkflow`，实现 ``execute(**params) -> WorkflowResult``；
    * 按 ``steps`` 定义逐步执行，记录到 ``WorkflowResult.steps_log``；
    * 物理计算真实调用 ``aerospace_agent.physics``；
    * 工具调用走 ``aerospace_agent.mcp_tools`` 的统一 ``call(method, **kwargs)`` 接口。

工作流清单
----------
- :class:`OrbitDesignWorkflow`            (name='orbit_design')   轨道设计
- :class:`LaunchWindowWorkflow`           (name='launch_window')  发射窗口分析
- :class:`TrajectoryAnalysisWorkflow`     (name='lunar_transfer') 地月转移轨迹分析 (核心)
- :class:`BasiliskVisualizationWorkflow`  (name='basilisk_viz')   Basilisk 可视化

快速使用
--------
::

    from aerospace_agent.workflows import (
        workflow_registry, default_workflow_registry, get_workflow, list_workflows,
    )

    # 列出所有工作流
    print(list_workflows())

    # 按名执行
    result = default_workflow_registry.execute("lunar_transfer", launch_date=None)
    print(result.summary, result.metadata["dv_total_km_s"])
"""

from __future__ import annotations

from .base import (
    BaseWorkflow,
    WorkflowResult,
    WorkflowRegistry,
    register_workflow,
    workflow_registry,
)
from .orbit_design import OrbitDesignWorkflow
from .launch_window import LaunchWindowWorkflow
from .trajectory_analysis import TrajectoryAnalysisWorkflow, build_formula_derivation
from .basilisk_visualization import BasiliskVisualizationWorkflow
from .registry import (
    default_workflow_registry,
    get_workflow,
    list_workflows,
    list_workflow_names,
    execute_workflow,
)

__all__ = [
    # 基类与数据结构
    "BaseWorkflow",
    "WorkflowResult",
    "WorkflowRegistry",
    "register_workflow",
    "workflow_registry",
    # 工作流类
    "OrbitDesignWorkflow",
    "LaunchWindowWorkflow",
    "TrajectoryAnalysisWorkflow",
    "BasiliskVisualizationWorkflow",
    # 公式推导
    "build_formula_derivation",
    # 注册表与便捷函数
    "default_workflow_registry",
    "get_workflow",
    "list_workflows",
    "list_workflow_names",
    "execute_workflow",
]

__version__ = "0.1.0"
