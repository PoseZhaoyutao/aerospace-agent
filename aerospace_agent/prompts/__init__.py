"""航天动力学智能 Agent 提示词模块 (prompts)。

统一导出系统提示词、ReAct 循环模板、任务专属模板与 CEO 上下文管理策略，
并提供 ``get_prompt(task_type)`` 便捷获取函数。

子模块
------
- ``system_prompt``     : 系统提示词 (SYSTEM_PROMPT) 与构建函数
- ``react_template``    : ReAct 循环各阶段提示词模板
- ``task_templates``    : 七类任务专属提示词模板
- ``context_strategy``  : CEO 上下文管理策略（代码化）

快速使用
--------
::

    from aerospace_agent.prompts import get_prompt, SYSTEM_PROMPT

    # 按任务类型获取完整提示词配置
    cfg = get_prompt("lunar_transfer")
    system = cfg["system"]          # REACT_SYSTEM + 任务专属指令
    user_template = cfg["user_template"]  # 含 {task} 占位符
    tools_hint = cfg["tools_hint"]  # 推荐工具清单

    # 格式化用户提示
    user_msg = user_template.format(task="设计 300km 停泊轨道的地月转移轨道")
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .system_prompt import (
    SYSTEM_PROMPT,
    IDENTITY_LINE,
    build_system_prompt,
)
from .react_template import (
    REACT_SYSTEM,
    REACT_USER_TEMPLATE,
    REACT_OBSERVATION_TEMPLATE,
    FORMAT_GUIDE,
    build_react_messages,
)
from .task_templates import (
    ORBIT_DESIGN,
    LAUNCH_WINDOW,
    TRAJECTORY_ANALYSIS,
    GROUND_ACCESS,
    LUNAR_TRANSFER,
    CROSS_VALIDATION,
    LITERATURE_SEARCH,
    TASK_TEMPLATES,
    TASK_TYPES,
)
from .context_strategy import (
    CONTEXT_STRATEGY,
    ESSENTIAL_PRESERVE_RULES,
    COMPRESS_TRIGGER,
    OFFLOAD_TRIGGER,
    get_context_prompt,
    decide_action,
)

__all__ = [
    # 系统提示词
    "SYSTEM_PROMPT",
    "IDENTITY_LINE",
    "build_system_prompt",
    # ReAct 模板
    "REACT_SYSTEM",
    "REACT_USER_TEMPLATE",
    "REACT_OBSERVATION_TEMPLATE",
    "FORMAT_GUIDE",
    "build_react_messages",
    # 任务专属模板
    "ORBIT_DESIGN",
    "LAUNCH_WINDOW",
    "TRAJECTORY_ANALYSIS",
    "GROUND_ACCESS",
    "LUNAR_TRANSFER",
    "CROSS_VALIDATION",
    "LITERATURE_SEARCH",
    "TASK_TEMPLATES",
    "TASK_TYPES",
    # 上下文策略
    "CONTEXT_STRATEGY",
    "ESSENTIAL_PRESERVE_RULES",
    "COMPRESS_TRIGGER",
    "OFFLOAD_TRIGGER",
    "get_context_prompt",
    "decide_action",
    # 便捷函数
    "get_prompt",
    "list_task_types",
]

__version__ = "0.1.0"


def get_prompt(task_type: str) -> Dict[str, str]:
    """按任务类型获取完整提示词配置。

    返回的字典结构：
        {
            "task_type":   str,  # 任务类型键名
            "system":       str,  # REACT_SYSTEM + 任务专属系统指令
            "user_template": str,  # 用户任务输入模板（含 {task} 占位符）
            "tools_hint":    str,  # 推荐工具/技能清单
        }

    Args:
        task_type: 任务类型键名，支持以下值：
            - ``"orbit_design"``        : 轨道设计
            - ``"launch_window"``       : 发射窗口计算
            - ``"trajectory_analysis"`` : 轨迹分析
            - ``"ground_access"``       : 地面站可见性
            - ``"lunar_transfer"``      : 地月转移轨道
            - ``"cross_validation"``    : 交叉验证
            - ``"literature_search"``   : 文献检索
            - ``"general"``             : 通用任务（默认回退）

    Returns:
        提示词配置字典。未知类型回退到 ``"general"``。
    """
    task_key = task_type.strip().lower() if task_type else "general"
    template = TASK_TEMPLATES.get(task_key)

    if template is None:
        # 未知任务类型：回退到通用配置
        return {
            "task_type": "general",
            "system": REACT_SYSTEM,
            "user_template": (
                "【任务】\n{task}\n\n"
                "请按 ReAct 格式执行：先输出 Thought，再输出 Action 或 Final Answer。"
            ),
            "tools_hint": "使用可用工具列表中的任意工具，按需调用 RAG 与 Loop 引擎。",
        }

    # 组合 REACT_SYSTEM + 任务专属系统指令
    combined_system = REACT_SYSTEM + "\n\n" + template["system"]
    return {
        "task_type": task_key,
        "system": combined_system,
        "user_template": template["user_template"],
        "tools_hint": template["tools_hint"],
    }


def list_task_types() -> list:
    """返回所有支持的任务类型键名列表（含 ``"general"`` 回退项）。"""
    return TASK_TYPES + ["general"]
