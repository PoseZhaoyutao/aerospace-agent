"""resources — 工作流目录、Demo 索引、Kernel 注册表。

导出：
    DemoIndex            示例索引与检索（Loop RetrieveDemo 阶段）
    WorkflowCatalog      工作流目录加载与检索（Loop GenerateWorkflow 阶段）
    KernelRegistry       SPICE 内核注册与路径验证（安全白名单）

    CATALOG_CATEGORIES   7 大工作流类别常量
    DEFAULT_KERNEL_SET   默认 SPICE 内核集
"""
from __future__ import annotations

from .demo_index import DemoIndex
from .workflow_catalog import WorkflowCatalog, CATALOG_CATEGORIES
from .kernel_registry import KernelRegistry, DEFAULT_KERNEL_SET

__all__ = [
    "DemoIndex",
    "WorkflowCatalog",
    "KernelRegistry",
    "CATALOG_CATEGORIES",
    "DEFAULT_KERNEL_SET",
]
