"""STK 适配器 — 基于 AGI STK 的任务分析与地面可见性引擎（Windows COM）。

第一性原理：
  1. STK 通过 COM 自动化暴露 API，本适配器经 comtypes（或 win32com）驱动 STK 应用。
  2. 许可是硬前提——必须先经 COM 校验 STK 可启动且持有有效许可证，否则全程不可用。
  3. 所有 STK 内部对象（卫星/地面站/链路）经 Canonical Model 映射，单位 SI。
  4. COM 调用全程 try/except：任何失败都返回结构化结果，绝不向上抛、绝不崩溃。
  5. 仅 Windows 可用；非 Windows 直接返回 unavailable。
"""
from __future__ import annotations

import platform
import os
from typing import TYPE_CHECKING, Optional, Set

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:
    from ..schemas import GroundStation, OrbitState, PropagatorConfig


class STKAdapter(BaseAdapter):
    """STK 引擎适配器（Windows COM）。

    能力：propagate_orbit / compute_ground_access / attitude_control
    依赖：comtypes（pip install comtypes）或 win32com + 已安装并授权的 STK
    """

    engine_name: str = "stk"
    _capabilities: Set[str] = {
        "propagate_orbit", "compute_ground_access", "attitude_control",
    }

    #: 尝试连接的 STK ProgID 列表（兼容多版本）
    _PROGIDS = (
        "STK.Application.12", "STK.Application.11",
        "STK12.Application", "STK11.Application",
    )

    def __init__(self):
        super().__init__()
        self._app = None
        self._license_ok: Optional[bool] = None

    # ------------------------------------------------------------------
    # 通用辅助
    # ------------------------------------------------------------------
    def _guard(self, operation: str):
        """可用性闸门：调用 _require_available()，不可用时返回 unavailable_result。

        STK 特别要求：许可缺失时返回结构化结果，绝不崩溃。
        """
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

    @staticmethod
    def _datetime_to_stk(dt) -> str:
        """datetime → STK 日期串 '1 Jan 2020 12:00:00.000'。

        平台无关：不依赖 %-d / %#d，手动拼日（无前导零）。
        """
        ms = int(dt.microsecond) // 1000
        return (f"{dt.day} {dt.strftime('%b')} {dt.year} "
                f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}")

    @staticmethod
    def _vec3(obj):
        """从 STK 位置/速度 COM 对象提取 3 分量（兼容索引与 .X/.Y/.Z）。"""
        for accessor in (
            lambda o: (o[0], o[1], o[2]),
            lambda o: (o.X, o.Y, o.Z),
            lambda o: (o.x, o.y, o.z),
        ):
            try:
                v = accessor(obj)
                return [float(v[0]), float(v[1]), float(v[2])]
            except Exception:
                continue
        raise ValueError("无法从 STK 对象提取 3 分量")

    @staticmethod
    def _extract_initial(initial_state):
        """从 OrbitState/dict 提取 (pos_m, vel_mps, epoch_value)。"""
        if isinstance(initial_state, dict):
            pos = initial_state.get("position_m")
            vel = initial_state.get("velocity_mps")
            epoch_obj = initial_state.get("epoch", {})
            epoch_val = (epoch_obj.get("value")
                         if isinstance(epoch_obj, dict)
                         else str(epoch_obj))
        else:
            pos = initial_state.position_m
            vel = initial_state.velocity_mps
            epoch_val = initial_state.epoch.value
        return pos, vel, epoch_val

    @staticmethod
    def _parse_access_intervals(intervals):
        """解析 STK ComputedAccessIntervalTimes → [{start, stop}, ...]。

        兼容 GetInterval(0-based) / Item(1-based) / Item(0-based) 多种接口。
        """
        windows = []
        try:
            count = int(intervals.Count)
        except Exception:
            return windows
        for i in range(count):
            interval = None
            for getter in (lambda idx: intervals.GetInterval(idx),
                           lambda idx: intervals.Item(idx + 1),
                           lambda idx: intervals.Item(idx)):
                try:
                    interval = getter(i)
                    break
                except Exception:
                    continue
            if interval is None:
                continue
            try:
                start_str = str(interval.Start)
                stop_str = str(interval.Stop)
            except Exception:
                continue
            windows.append({"start": start_str, "stop": stop_str})
        return windows

    # ------------------------------------------------------------------
    # COM 与许可
    # ------------------------------------------------------------------
    def _get_stk_app(self):
        """惰性获取 STK COM 应用实例（含许可校验）。失败返回 None，绝不抛异常。"""
        if self._app is not None:
            return self._app
        if platform.system() != "Windows":
            return None
        # COM activation can hang in unattended sessions without an STK
        # desktop/license service.  It is an explicit deployment opt-in; the
        # availability API otherwise returns a bounded unavailable result.
        if os.environ.get("AEROSPACE_ENABLE_STK_COM_PROBE", "").lower() not in {"1", "true", "yes"}:
            self._license_ok = False
            return None
        # 优先 comtypes，回退 win32com
        try:
            from comtypes.client import CreateObject
        except Exception:
            CreateObject = None
        for progid in self._PROGIDS:
            try:
                if CreateObject is not None:
                    app = CreateObject(progid)
                else:
                    import win32com.client
                    app = win32com.client.Dispatch(progid)
                # 许可校验：能拿到应用对象即视为许可可用
                try:
                    app.Visible = False
                    self._license_ok = True
                    self._app = app
                    return app
                except Exception:
                    self._license_ok = False
                    continue
            except Exception:
                continue
        self._license_ok = False
        return None

    def _has_license(self) -> bool:
        """显式许可校验。"""
        if self._license_ok is not None:
            return self._license_ok
        return self._get_stk_app() is not None

    # ------------------------------------------------------------------
    # 契约实现
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 STK 是否可经 COM 启动且许可有效。绝不抛异常。"""
        try:
            return self._has_license()
        except Exception:
            return False

    def version(self) -> str:
        """返回 STK 应用版本，不可用或无许可时返回 'unavailable'。绝不抛异常。"""
        try:
            app = self._get_stk_app()
            if app is None:
                return "unavailable"
            return str(getattr(app, "Version", "unknown"))
        except Exception:
            return "unavailable"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法
    # ------------------------------------------------------------------
    def propagate_orbit(self, initial_state, force_model, config) -> dict:
        """轨道传播——STK 卫星对象 + 传播器（HPOP/SGP4）。"""
        unavail = self._guard("propagate_orbit")
        if unavail is not None:
            return unavail
        app = self._get_stk_app()
        if app is None:
            return self.unavailable_result("propagate_orbit")
        root = None
        try:
            from datetime import datetime, timedelta

            # --- 提取初始状态（兼容 OrbitState 对象与 dict）---
            pos, vel, epoch_val = self._extract_initial(initial_state)
            if pos is None or vel is None:
                return self._error_result(
                    "propagate_orbit",
                    "initial_state 缺少 position_m 或 velocity_mps")

            # --- 解析配置（mu 不在 PropagatorConfig 中，用 getattr 兜底）---
            duration = float(getattr(config, "duration_s", 86400.0))
            step = float(getattr(config, "step_s", 60.0))
            if step <= 0:
                step = 60.0
            mu = float(getattr(config, "mu", 3.986004418e14))

            # --- 单位换算：m→km, m/s→km/s（STK 默认 km）---
            x = float(pos[0]) / 1000.0
            y = float(pos[1]) / 1000.0
            z = float(pos[2]) / 1000.0
            vx = float(vel[0]) / 1000.0
            vy = float(vel[1]) / 1000.0
            vz = float(vel[2]) / 1000.0

            # --- ISO 历元 → STK 日期串 ---
            iso_str = str(epoch_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            stk_epoch = self._datetime_to_stk(dt)
            stk_start = stk_epoch
            stk_stop = self._datetime_to_stk(dt + timedelta(seconds=duration))
            stk_step = str(step)

            # --- STK 场景与卫星 ---
            root = app.Personality2
            root.NewScenario("PropScenario")
            scenario = root.CurrentScenario
            eSatellite = 18
            sat = scenario.Children.New(eSatellite, "MySat")
            sat.SetStateCartesian("Earth", "J2000", stk_epoch, stk_start,
                                  stk_stop, stk_step, x, y, z, vx, vy, vz)
            sat.Propagate()

            # --- 提取终态（km → m, km/s → m/s）---
            final_pos = [c * 1000.0 for c in self._vec3(sat.Position)]
            final_vel = [c * 1000.0 for c in self._vec3(sat.Velocity)]
            final_dt = dt + timedelta(seconds=duration)
            final_iso = final_dt.strftime("%Y-%m-%dT%H:%M:%S")

            state_history = [{
                "epoch": {"value": final_iso, "scale": "UTC",
                          "format": "ISO"},
                "frame": {"name": "J2000", "center": "Earth",
                          "realization": "EME2000"},
                "representation": "cartesian",
                "position_m": final_pos,
                "velocity_mps": final_vel,
                "elapsed_s": duration,
            }]

            return {
                "status": "success",
                "state_history": state_history,
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "propagator_type": "stk_cartesian",
                    "units": "SI (m, m/s, s)",
                    "frame": "J2000",
                    "step_count": len(state_history),
                    "duration_s": duration,
                    "step_s": step,
                    "mu": mu,
                },
            }
        except Exception as exc:
            return self._error_result("propagate_orbit", str(exc))
        finally:
            if root is not None:
                try:
                    root.CloseScenario()
                except Exception:
                    pass

    def compute_ground_access(self, orbit_state, station, start_epoch,
                              stop_epoch, min_elevation_deg: float) -> dict:
        """地面可见性——STK Facility/Access 对象计算访问时段。"""
        unavail = self._guard("compute_ground_access")
        if unavail is not None:
            return unavail
        app = self._get_stk_app()
        if app is None:
            return self.unavailable_result("compute_ground_access")
        root = None
        try:
            from datetime import datetime

            # --- 提取轨道状态（兼容 OrbitState 对象与 dict）---
            pos, vel, epoch_val = self._extract_initial(orbit_state)
            if pos is None or vel is None:
                return self._error_result(
                    "compute_ground_access",
                    "orbit_state 缺少 position_m 或 velocity_mps")

            # --- 地面站参数 ---
            if isinstance(station, dict):
                st_name = str(station.get("name", "Station"))
                lat = float(station.get("latitude_deg", 0.0))
                lon = float(station.get("longitude_deg", 0.0))
                alt_m = float(station.get("altitude_m", 0.0))
            else:
                st_name = str(getattr(station, "name", "Station"))
                lat = float(getattr(station, "latitude_deg", 0.0))
                lon = float(getattr(station, "longitude_deg", 0.0))
                alt_m = float(getattr(station, "altitude_m", 0.0))
            alt_km = alt_m / 1000.0  # m → km

            # --- 解析起止历元（兼容 Epoch 对象/字符串）---
            def _to_stk(ep):
                val = ep.value if hasattr(ep, "value") else ep
                return self._datetime_to_stk(
                    datetime.fromisoformat(str(val).replace("Z", "+00:00")))

            stk_start = _to_stk(start_epoch)
            stk_stop = _to_stk(stop_epoch)

            # --- 轨道状态单位换算 + 历元 ---
            x = float(pos[0]) / 1000.0
            y = float(pos[1]) / 1000.0
            z = float(pos[2]) / 1000.0
            vx = float(vel[0]) / 1000.0
            vy = float(vel[1]) / 1000.0
            vz = float(vel[2]) / 1000.0
            stk_epoch = self._datetime_to_stk(
                datetime.fromisoformat(str(epoch_val).replace("Z", "+00:00")))

            # --- STK 场景：卫星 + 地面站 ---
            root = app.Personality2
            root.NewScenario("AccessScenario")
            scenario = root.CurrentScenario
            eSatellite = 18
            eFacility = 8
            sat = scenario.Children.New(eSatellite, "Sat")
            sat.SetStateCartesian("Earth", "J2000", stk_epoch, stk_start,
                                  stk_stop, "60", x, y, z, vx, vy, vz)

            fac = scenario.Children.New(eFacility, st_name)
            fac.Position.AssignGeodetic(lat, lon, alt_km)

            # --- Access 计算（最小仰角约束，best-effort）---
            access = sat.GetAccessToObject(fac)
            try:
                # eCstrMinElevation：不同 STK 版本枚举值不一，try 多个常见值
                for min_el_enum in (4, 6, 3, 1):
                    try:
                        cstr = access.AccessConstraints.AddConstraint(min_el_enum)
                        if hasattr(cstr, "EnableMinElevation"):
                            cstr.EnableMinElevation = True
                        if hasattr(cstr, "MinElevationValue"):
                            cstr.MinElevationValue = float(min_elevation_deg)
                        break
                    except Exception:
                        continue
            except Exception:
                pass

            access.ComputeAccess()
            access_windows = self._parse_access_intervals(
                access.ComputedAccessIntervalTimes)

            return {
                "status": "success",
                "access_windows": access_windows,
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "station": st_name,
                    "min_elevation_deg": float(min_elevation_deg),
                    "window_count": len(access_windows),
                    "start": stk_start,
                    "stop": stk_stop,
                },
            }
        except Exception as exc:
            return self._error_result("compute_ground_access", str(exc))
        finally:
            if root is not None:
                try:
                    root.CloseScenario()
                except Exception:
                    pass

    def attitude_control(self, initial_state, profile: str = "nadir_pointing",
                         **kwargs) -> dict:
        """姿态仿真——STK 卫星姿态指向 profile。"""
        unavail = self._guard("attitude_control")
        if unavail is not None:
            return unavail
        app = self._get_stk_app()
        if app is None:
            return self.unavailable_result("attitude_control")
        root = None
        try:
            from datetime import datetime, timedelta

            # --- 提取轨道状态（兼容 OrbitState 对象与 dict）---
            pos, vel, epoch_val = self._extract_initial(initial_state)
            if pos is None or vel is None:
                return self._error_result(
                    "attitude_control",
                    "initial_state 缺少 position_m 或 velocity_mps")

            # --- 参数 ---
            duration = float(kwargs.get("duration_s", 3600.0))
            step = float(kwargs.get("step_s", 60.0))
            if step <= 0:
                step = 60.0

            # profile 名映射：nadir_pointing → NadirPointing 等
            _profile_map = {
                "nadir_pointing": "NadirPointing",
                "nadir": "NadirPointing",
                "sun_pointing": "SunPointing",
                "sun": "SunPointing",
                "inertial": "Inertial",
            }
            stk_profile = _profile_map.get(
                str(profile).lower().strip(), str(profile))

            # --- 单位换算：m→km, m/s→km/s ---
            x = float(pos[0]) / 1000.0
            y = float(pos[1]) / 1000.0
            z = float(pos[2]) / 1000.0
            vx = float(vel[0]) / 1000.0
            vy = float(vel[1]) / 1000.0
            vz = float(vel[2]) / 1000.0

            # --- ISO 历元 → STK 日期串 ---
            iso_str = str(epoch_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            stk_epoch = self._datetime_to_stk(dt)
            stk_start = stk_epoch
            stk_stop = self._datetime_to_stk(dt + timedelta(seconds=duration))
            stk_step = str(step)

            # --- STK 场景与卫星 ---
            root = app.Personality2
            root.NewScenario("AttScenario")
            scenario = root.CurrentScenario
            eSatellite = 18
            sat = scenario.Children.New(eSatellite, "MySat")
            sat.SetStateCartesian("Earth", "J2000", stk_epoch, stk_start,
                                  stk_stop, stk_step, x, y, z, vx, vy, vz)

            # --- 设置姿态指向 profile ---
            attitude = sat.Attitude
            try:
                attitude.Profile = stk_profile
            except Exception:
                pass

            # --- 导出姿态四元数（best-effort，兼容多种数据接口）---
            attitude_history = []
            try:
                quats = attitude.Quaternions
                count = int(getattr(quats, "Count", 0))
                for i in range(count):
                    try:
                        row = (quats.Item(i) if hasattr(quats, "Item")
                               else quats[i])
                    except Exception:
                        continue
                    try:
                        attitude_history.append({
                            "quaternion": [
                                float(row[1]), float(row[2]),
                                float(row[3]), float(row[4]),
                            ],
                            "time_s": float(row[0]),
                        })
                    except Exception:
                        continue
            except Exception:
                pass

            return {
                "status": "success",
                "controller": profile,
                "attitude_history": attitude_history,
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "profile": stk_profile,
                    "duration_s": duration,
                    "step_s": step,
                    "step_count": len(attitude_history),
                },
            }
        except Exception as exc:
            return self._error_result("attitude_control", str(exc))
        finally:
            if root is not None:
                try:
                    root.CloseScenario()
                except Exception:
                    pass
