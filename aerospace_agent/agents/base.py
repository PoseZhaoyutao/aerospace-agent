"""Agent 基类 — 三Agent共享的接口契约与工具方法。

第一性原理：
  1. 每个Agent有明确职责边界（role）和输入输出契约
  2. 共享观测性基础设施（StructuredLogger + MetricsCollector）
  3. 所有Agent操作返回结构化 AgentResult，便于编排器决策
  4. Agent间通过 LoopLedger 实现跨Agent可追溯
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.observability import get_logger, get_metrics


class AgentRole(Enum):
    """Agent 角色枚举。"""
    TEST = "test"
    FIX = "fix"
    ARCH = "arch"


@dataclass
class AgentResult:
    """Agent 操作结果——所有Agent的统一输出。"""
    agent: str                    # agent名称
    operation: str                # 操作名称
    success: bool                 # 是否成功
    data: Dict[str, Any] = field(default_factory=dict)  # 结构化结果数据
    issues: List[Dict] = field(default_factory=list)    # 发现的问题
    metrics: Dict[str, Any] = field(default_factory=dict)  # 操作指标
    duration_s: float = 0.0       # 耗时
    error: Optional[str] = None   # 错误信息

    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "operation": self.operation,
            "success": self.success,
            "data": self.data,
            "issues": self.issues,
            "metrics": self.metrics,
            "duration_s": round(self.duration_s, 3),
            "error": self.error,
        }


class AgentBase:
    """Agent 基类——共享日志、metrics、项目路径。"""

    role: AgentRole

    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.package_root = f"{project_root}/aerospace_agent"
        self.log = get_logger(f"agent.{self.role.value}")
        self.metrics = get_metrics()

    def _time_operation(self, operation: str):
        """上下文管理器——自动计时并记录metrics。"""
        return _TimedOp(self, operation)

    def _make_result(self, operation: str, success: bool,
                     data: Dict = None, issues: List[Dict] = None,
                     error: str = None, duration: float = 0.0) -> AgentResult:
        """构造标准化结果。"""
        result = AgentResult(
            agent=self.role.value,
            operation=operation,
            success=success,
            data=data or {},
            issues=issues or [],
            duration_s=duration,
            error=error,
        )
        self.metrics.inc(
            f"agent.{self.role.value}.operations",
            tags={"operation": operation,
                  "status": "success" if success else "error"})
        return result


class _TimedOp:
    """操作计时上下文管理器。"""

    def __init__(self, agent: AgentBase, operation: str):
        self._agent = agent
        self._operation = operation
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        self._agent.log.info(f"{self._operation}_start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self._start
        self._agent.metrics.record_timing(
            f"agent.{self._agent.role.value}.latency",
            duration,
            tags={"operation": self._operation})
        self._agent.log.info(
            f"{self._operation}_end",
            data={"duration_s": round(duration, 3),
                  "success": exc_type is None})
        return False
