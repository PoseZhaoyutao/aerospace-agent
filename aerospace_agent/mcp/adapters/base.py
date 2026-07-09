"""Adapter 基类 — 所有引擎适配器的统一契约。

第一性原理（K2）：
  1. 适配器是可选插件，懒加载（import 在方法内，不在模块顶层）
  2. is_available() 是唯一闸门——不可用时返回 False，不抛异常
  3. 所有输出必须转换回 Canonical Astrodynamics Model
  4. 不允许 LLM 直接调用底层库——只暴露白名单封装的高层方法
  5. 不可用的引擎绝不导致整个 MCP server 崩溃

每个 Adapter 必须实现：
    is_available() -> bool
    version() -> str
    capabilities() -> set[str]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set


class AdapterError(Exception):
    """适配器结构化错误。"""

    def __init__(self, engine: str, reason: str, recoverable: bool = True,
                 detail: str = ""):
        self.engine = engine
        self.reason = reason
        self.recoverable = recoverable
        self.detail = detail
        super().__init__(f"[{engine}] {reason}: {detail}" if detail
                         else f"[{engine}] {reason}")

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "reason": self.reason,
            "recoverable": self.recoverable,
            "detail": self.detail,
        }


class BaseAdapter(ABC):
    """引擎适配器基类。

    子类必须实现 is_available / version / capabilities。
    具体能力方法（propagate / transform_frame 等）按需实现，
    不可用时返回结构化 unavailable 结果而非抛异常。
    """

    #: 引擎名（子类覆盖）
    engine_name: str = "base"

    #: 该引擎支持的能力集合（子类覆盖）
    _capabilities: Set[str] = set()

    def __init__(self):
        self._available: Optional[bool] = None
        self._version: Optional[str] = None

    # ------------------------------------------------------------------
    # 必须实现的三个方法
    # ------------------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:
        """检测引擎是否安装可用。绝不抛异常——返回 False 即可。"""
        ...

    @abstractmethod
    def version(self) -> str:
        """返回引擎版本字符串。不可用时返回 'unavailable'。"""
        ...

    @abstractmethod
    def capabilities(self) -> Set[str]:
        """返回该引擎支持的能力集合。

        标准能力名：
            propagate_orbit, transform_frame, query_ephemeris,
            convert_orbit, compute_access, run_script,
            attitude_control, spherical_harmonics
        """
        ...

    # ------------------------------------------------------------------
    # 通用保护
    # ------------------------------------------------------------------
    def _require_available(self) -> None:
        """调用前检查可用性，不可用则抛 AdapterError。"""
        if self._available is None:
            self._available = self.is_available()
        if not self._available:
            raise AdapterError(
                self.engine_name, "engine_unavailable",
                recoverable=False,
                detail=f"{self.engine_name} 未安装或不可用",
            )

    def unavailable_result(self, operation: str) -> dict:
        """返回标准化的 unavailable 结构化结果（不抛异常）。"""
        return {
            "status": "unavailable",
            "engine": self.engine_name,
            "operation": operation,
            "reason": f"{self.engine_name} 未安装或不可用",
            "version": "unavailable",
        }

    def info(self) -> dict:
        """返回引擎信息摘要。"""
        return {
            "engine": self.engine_name,
            "available": self.is_available(),
            "version": self.version(),
            "capabilities": sorted(self.capabilities()),
        }

    # ------------------------------------------------------------------
    # 可选能力方法（子类按需覆盖；默认返回 unavailable）
    # ------------------------------------------------------------------
    def propagate_orbit(self, initial_state, force_model, config) -> dict:
        return self.unavailable_result("propagate_orbit")

    def transform_frame(self, state, target_frame: str) -> dict:
        return self.unavailable_result("transform_frame")

    def query_ephemeris(self, target: str, observer: str,
                        epoch, frame: str, **kwargs) -> dict:
        return self.unavailable_result("query_ephemeris")

    def convert_time(self, epoch, target_scale: str) -> dict:
        return self.unavailable_result("convert_time")

    def compute_ground_access(self, orbit_state, station, start_epoch,
                              stop_epoch, min_elevation_deg: float) -> dict:
        return self.unavailable_result("compute_ground_access")

    def run_script(self, script_text: str = "", script_path: str = "",
                   workspace: str = "") -> dict:
        return self.unavailable_result("run_script")
