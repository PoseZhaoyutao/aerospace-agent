"""技能基类 —— 所有可复用能力单元的抽象基础。

SkillBase 定义了技能的标准接口：名称、描述、分类、执行方法。
每个具体技能继承 SkillBase 并实现 execute()，返回结构化结果字典。

分类体系:
    - context   : 上下文管理（CEO 三层压缩/卸载）
    - memory    : 记忆检索（短期/长期记忆召回）
    - rag       : 知识检索（多源路由 RAG）
    - mcp       : 工具编排（Loop 引擎等 MCP 能力）
    - analysis  : 分析计算（轨道力学分析等）
    - reporting : 报告生成（HTML 报告等）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class SkillBase(ABC):
    """技能抽象基类。

    所有可被 Agent 调用的能力单元均继承此类。子类必须实现 execute()，
    可选实现 is_available() 以声明运行时依赖的可用性。

    Attributes:
        name: 技能名称（唯一标识，用于注册与检索）
        description: 技能功能描述（供 Agent 决策时参考）
        category: 技能分类 ("context"/"memory"/"rag"/"mcp"/"analysis"/"reporting")
    """

    name: str = ""
    description: str = ""
    category: str = ""

    @abstractmethod
    def execute(self, agent, **kwargs) -> dict:
        """执行本技能。

        Args:
            agent: AerospaceAgent 实例，提供上下文/记忆/RAG/工具等能力
            **kwargs: 技能特定参数

        Returns:
            结构化结果字典::

                {
                    "success": bool,      # 执行是否成功
                    "result": Any,        # 执行结果（成功时）或 None
                    "message": str,       # 人类可读的状态/错误说明
                }
        """
        ...

    def is_available(self) -> bool:
        """检查技能依赖是否可用。

        默认返回 True（无外部依赖）。子类可覆盖此方法，
        在 execute() 之前供注册表或 Agent 预判技能可用性。
        """
        return True

    def info(self) -> dict:
        """返回技能元数据，供注册表列表展示与 Agent 决策参考。"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "available": self.is_available(),
        }

    def __repr__(self) -> str:
        """可读的技能表示。"""
        return f"<Skill {self.name} [{self.category}]>"
