"""Astropy 适配器 — 基于 astropy 的时间转换与天文坐标系引擎。

第一性原理：
  1. astropy.time.Time 是跨尺度（UTC/TAI/TT/TDB）时间转换的事实标准，闰秒自动处理。
  2. astropy.coordinates 提供 ICRS/ITRS/GCRS 等帧变换，依赖 IERS A/B 表。
  3. 本适配器作为时间与帧转换的轻量后端，不承担高保真轨道传播。
  4. 懒加载 astropy，未安装时返回结构化结果，绝不崩溃。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Set

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:
    from ..schemas import Epoch, OrbitState


class AstropyAdapter(BaseAdapter):
    """Astropy 引擎适配器。

    能力：convert_time / transform_frame / query_ephemeris
    依赖：astropy（pip install astropy）+ IERS 表（首次自动下载）
    """

    engine_name: str = "astropy"
    _capabilities: Set[str] = {
        "convert_time", "transform_frame", "query_ephemeris",
    }

    def __init__(self):
        super().__init__()

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
    # 契约实现
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 astropy 是否安装。绝不抛异常。"""
        try:
            import astropy  # noqa: F401
            return True
        except Exception:
            return False

    def version(self) -> str:
        """返回 astropy.__version__，不可用时返回 'unavailable'。绝不抛异常。"""
        try:
            import astropy
            return getattr(astropy, "__version__", "unknown")
        except Exception:
            return "unavailable"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法
    # ------------------------------------------------------------------
    def convert_time(self, epoch, target_scale: str) -> dict:
        """时间尺度转换——astropy.time.Time 核心能力，输出 Canonical Epoch dict。"""
        unavail = self._guard("convert_time")
        if unavail is not None:
            return unavail
        try:
            t = self._build_astropy_time(epoch)
            converted = getattr(t, target_scale.lower())
            return {
                "status": "success", "engine": self.engine_name,
                "operation": "convert_time",
                "epoch": {
                    "value": self._time_to_value(converted, epoch),
                    "scale": target_scale,
                    "format": self._epoch_format(epoch),
                },
            }
        except Exception as exc:
            return self._error_result("convert_time", str(exc))

    def transform_frame(self, state, target_frame: str) -> dict:
        """坐标系转换 — astropy.coordinates 帧变换 (ICRS↔ITRS/GCRS)。"""
        unavail = self._guard("transform_frame")
        if unavail is not None:
            return unavail
        try:
            import numpy as np
            from astropy.coordinates import (
                ICRS, ITRS, GCRS, CartesianRepresentation,
            )
            from astropy import units as u
            from astropy.time import Time

            r = state.position_m
            v = state.velocity_mps
            if r is None or v is None:
                return self._error_result("transform_frame", "缺少 position_m 或 velocity_mps")

            # 构建 astropy 笛卡尔表示 (m → 无量纲,手动管理单位)
            pos = CartesianRepresentation(
                r[0] * u.m, r[1] * u.m, r[2] * u.m)
            vel = CartesianRepresentation(
                v[0] * u.m / u.s, v[1] * u.m / u.s, v[2] * u.m / u.s)

            # 解析源帧
            src_frame_name = (state.frame.name.value
                             if hasattr(state.frame, "name") and hasattr(state.frame.name, "value")
                             else str(getattr(state.frame, "name", "GCRF")))
            epoch = self._build_astropy_time(state.epoch)

            # 映射帧名到 astropy 类
            frame_map = {"ICRF": ICRS, "GCRF": GCRS, "GCRS": GCRS,
                         "ITRF": ITRS, "ITRS": ITRS}
            src_cls = frame_map.get(src_frame_name.upper(), GCRS)
            dst_cls = frame_map.get(target_frame.upper(), GCRS)

            # 创建源帧坐标 (带速度)
            src_coord = src_cls(pos.with_differentials(vel), obstime=epoch)
            # 变换到目标帧
            dst_coord = src_coord.transform_to(dst_cls(obstime=epoch))

            # 提取结果
            new_pos = [float(dst_coord.x / u.m),
                       float(dst_coord.y / u.m),
                       float(dst_coord.z / u.m)]
            diff = dst_coord.data.differentials.get("s")
            if diff is not None:
                new_vel = [float(diff.d_x / (u.m / u.s)),
                           float(diff.d_y / (u.m / u.s)),
                           float(diff.d_z / (u.m / u.s))]
            else:
                new_vel = v  # 速度未变换,保留原值

            return {
                "status": "success",
                "engine": self.engine_name,
                "operation": "transform_frame",
                "position_m": new_pos,
                "velocity_mps": new_vel,
                "source_frame": src_frame_name,
                "target_frame": target_frame,
            }
        except Exception as exc:
            return self._error_result("transform_frame", str(exc))

    def query_ephemeris(self, target: str, observer: str,
                        epoch, frame: str, **kwargs) -> dict:
        """天体星历查询 — astropy.coordinates.get_body_barycentric。"""
        unavail = self._guard("query_ephemeris")
        if unavail is not None:
            return unavail
        try:
            from astropy.coordinates import (
                solar_system_ephemeris, get_body_barycentric,
                get_body_barycentric_posvel,
            )
            from astropy import units as u
            from astropy.time import Time

            t = self._build_astropy_time(epoch)
            # 设置星历表 (默认 JPL ephemeris)
            with solar_system_ephemeris.set("jpl"):
                # 获取目标和观测者的日心坐标
                try:
                    target_posvel = get_body_barycentric_posvel(target, t)
                    obs_posvel = get_body_barycentric_posvel(observer, t)
                except Exception:
                    target_pos = get_body_barycentric(target, t)
                    obs_pos = get_body_barycentric(observer, t)
                    target_posvel = (target_pos, None)
                    obs_posvel = (obs_pos, None)

            # 相对位置 = 目标 - 观测者
            t_pos = target_posvel[0]
            o_pos = obs_posvel[0]
            rel_x = float((t_pos.x - o_pos.x) / u.m)
            rel_y = float((t_pos.y - o_pos.y) / u.m)
            rel_z = float((t_pos.z - o_pos.z) / u.m)

            rel_vel = None
            if target_posvel[1] is not None and obs_posvel[1] is not None:
                t_vel = target_posvel[1]
                o_vel = obs_posvel[1]
                rel_vel = [
                    float((t_vel.d_x - o_vel.d_x) / (u.m / u.s)),
                    float((t_vel.d_y - o_vel.d_y) / (u.m / u.s)),
                    float((t_vel.d_z - o_vel.d_z) / (u.m / u.s)),
                ]

            return {
                "status": "success",
                "engine": self.engine_name,
                "operation": "query_ephemeris",
                "target": target,
                "observer": observer,
                "position_m": [rel_x, rel_y, rel_z],
                "velocity_mps": rel_vel,
                "epoch": str(t.iso),
                "frame": frame or "ICRF",
            }
        except Exception as exc:
            return self._error_result("query_ephemeris", str(exc))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _epoch_format(epoch) -> str:
        """从 Canonical Epoch（对象或 dict）取 format 字符串。"""
        if isinstance(epoch, dict):
            return epoch.get("format", "ISO")
        return getattr(getattr(epoch, "format", None), "value", "ISO")

    @staticmethod
    def _build_astropy_time(epoch):
        """把 Canonical Epoch（对象或 dict）转为 astropy.time.Time。"""
        from astropy.time import Time
        if isinstance(epoch, dict):
            value, fmt, scale = epoch["value"], epoch.get("format", "ISO"), \
                epoch.get("scale", "UTC")
        else:
            value, fmt, scale = epoch.value, epoch.format.value, epoch.scale.value
        astropy_fmt = {"ISO": "isot", "JD": "jd", "MJD": "mjd",
                       "UNIX": "unix"}.get(fmt, "isot")
        return Time(value, format=astropy_fmt, scale=scale.lower(), precision=9)

    @staticmethod
    def _time_to_value(t, epoch) -> str:
        """把 astropy Time 按原始格式取值（ISO 用 isot 字符串）。"""
        fmt = AstropyAdapter._epoch_format(epoch)
        mapping = {"ISO": "isot", "JD": "jd", "MJD": "mjd", "UNIX": "unix"}
        return str(getattr(t, mapping.get(fmt, "isot")))
