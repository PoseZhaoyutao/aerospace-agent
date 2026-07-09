"""SpiceyPy 适配器 — 基于 NAIF SPICE 的星历查询与坐标系转换引擎。

第一性原理：
  1. SPICE 的全部能力依赖 kernel 文件（SPK/CK/PCK/LSK/FRAME），必须显式装载。
  2. 本适配器从 kernel 注册表获取所需 kernel 路径，懒加载 spiceypy。
  3. 时间、坐标系、星历三类操作的输出统一回写为 Canonical Model（SI 单位）。
  4. 未安装 spiceypy 或缺少 kernel 时返回结构化结果，绝不崩溃。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:
    from ..schemas import Epoch, OrbitState


class SpiceyPyAdapter(BaseAdapter):
    """SpiceyPy（SPICE）引擎适配器。

    能力：query_ephemeris / transform_frame / convert_time
    依赖：spiceypy（pip install spiceypy）+ kernel 文件集
    资源：kernel 注册表（通过 set_kernel_registry 或外部注入）
    """

    engine_name: str = "spiceypy"
    _capabilities: Set[str] = {
        "query_ephemeris", "transform_frame", "convert_time",
    }

    def __init__(self):
        super().__init__()
        self._kernel_registry: dict = {}
        self._loaded_kernels: set = set()

    # ------------------------------------------------------------------
    # 通用辅助
    # ------------------------------------------------------------------
    def _guard(self, operation: str):
        """可用性闸门：调用 _require_available()，不可用时返回 unavailable_result。"""
        try:
            self._require_available()
            return None
        except AdapterError:
            return self.unavailable_result(operation)

    def _error_result(self, operation: str, reason: str) -> dict:
        return {"status": "error", "engine": self.engine_name,
                "operation": operation, "reason": reason}

    def _todo_result(self, operation: str, message: str = "") -> dict:
        return {"status": "todo", "engine": self.engine_name,
                "operation": operation, "message": message}

    # ------------------------------------------------------------------
    # kernel 管理
    # ------------------------------------------------------------------
    def set_kernel_registry(self, registry: dict) -> None:
        """注入 kernel 注册表（name→path 映射）。"""
        self._kernel_registry = dict(registry)
        self._loaded_kernels.clear()

    def _ensure_kernels(self, names: list) -> None:
        """按需装载尚未加载的 kernel 文件。"""
        import spiceypy as spice
        for name in names:
            path = self._kernel_registry.get(name) or self._kernel_registry.get(name.upper())
            if path and name not in self._loaded_kernels:
                spice.furnsh(path)
                self._loaded_kernels.add(name)

    # ------------------------------------------------------------------
    # 契约实现
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 spiceypy 是否安装。绝不抛异常（kernel 缺失不视为不可用）。"""
        try:
            import spiceypy  # noqa: F401
            return True
        except Exception:
            return False

    def version(self) -> str:
        """返回 spiceypy.__version__，不可用时返回 'unavailable'。绝不抛异常。"""
        try:
            import spiceypy
            return getattr(spiceypy, "__version__", "unknown")
        except Exception:
            return "unavailable"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法
    # ------------------------------------------------------------------
    def query_ephemeris(self, target: str, observer: str,
                        epoch, frame: str, **kwargs) -> dict:
        """星历查询（spkezr），输出位置 km→m、速度 km/s→m/s 的 Canonical dict。"""
        unavail = self._guard("query_ephemeris")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            self._ensure_kernels(["spk", "lsk", "pck"])
            # epoch(ISO) → ephemeris time
            et = spice.str2et(epoch.value)
            # spkezr 返回 state[x,y,z,vx,vy,vz] (km, km/s) + 光行时 lt (s)
            state, lt = spice.spkezr(target, et, frame, "NONE", observer)
            # 单位换算: km→m, km/s→m/s
            position_m = [float(state[0]) * 1000.0,
                          float(state[1]) * 1000.0,
                          float(state[2]) * 1000.0]
            velocity_mps = [float(state[3]) * 1000.0,
                            float(state[4]) * 1000.0,
                            float(state[5]) * 1000.0]
            return {
                "status": "success",
                "target": target,
                "observer": observer,
                "position_m": position_m,
                "velocity_mps": velocity_mps,
                "epoch": epoch.value,
                "frame": frame,
                "light_time_s": float(lt),
            }
        except Exception as exc:
            return self._error_result("query_ephemeris", str(exc))

    def transform_frame(self, state, target_frame: str) -> dict:
        """坐标系转换（sxform/pxform），输出 Canonical OrbitState。"""
        unavail = self._guard("transform_frame")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            import numpy as np
            self._ensure_kernels(["fk", "ck", "pck"])

            # 帧名映射: Canonical → SPICE 内部帧名
            frame_map = {"GCRF": "J2000", "ITRF": "ITRF93", "ICRF": "J2000"}

            def _frame_str(name) -> str:
                if hasattr(name, "value"):
                    return name.value
                return str(name)

            source_frame_name = _frame_str(state.frame.name)
            target_frame_name = _frame_str(target_frame)
            source_frame = frame_map.get(source_frame_name, source_frame_name)
            target_frame_mapped = frame_map.get(target_frame_name, target_frame_name)

            et = spice.str2et(state.epoch.value)
            position = list(state.position_m)
            velocity = list(state.velocity_mps)

            # pxform: 3x3 位置旋转矩阵
            rotation = spice.pxform(source_frame, target_frame_mapped, et)
            new_pos = np.dot(rotation, position)

            # sxform: 6x6 状态旋转矩阵（含速度交叉项，对旋转系更准确）
            rotation6 = spice.sxform(source_frame, target_frame_mapped, et)
            state_vec = list(position) + list(velocity)
            new_state = np.dot(rotation6, state_vec)
            new_pos = new_state[:3]
            new_vel = new_state[3:]

            return {
                "status": "success",
                "position_m": [float(x) for x in new_pos],
                "velocity_mps": [float(x) for x in new_vel],
                "source_frame": source_frame_name,
                "target_frame": target_frame_name,
            }
        except Exception as exc:
            return self._error_result("transform_frame", str(exc))

    def convert_time(self, epoch, target_scale: str) -> dict:
        """时间转换（str2et/et2utc/unitim），输出 Canonical Epoch dict。"""
        unavail = self._guard("convert_time")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            self._ensure_kernels(["lsk", "pck"])
            # 先统一转为 ephemeris time
            et = spice.str2et(epoch.value)
            target = target_scale.upper()
            if target == "UTC":
                # ISOD: ISO 格式字符串
                result = spice.et2utc(et, "ISOD", 9)
            elif target in ("TDB", "ET"):
                # TDB/ET 即 ephemeris time 本身（秒，浮点）
                result = float(et)
            elif target == "TAI":
                result = spice.et2utc(et, "TAI", 9)
            else:
                return self._error_result(
                    "convert_time",
                    f"Unsupported target scale: {target_scale}")
            return {
                "status": "success",
                "epoch": {
                    "value": result,
                    "scale": target_scale,
                    "format": "ISO",
                },
            }
        except Exception as exc:
            return self._error_result("convert_time", str(exc))
