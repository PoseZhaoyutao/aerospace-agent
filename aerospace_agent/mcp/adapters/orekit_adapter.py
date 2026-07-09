"""Orekit 适配器 — 基于 Orekit 的高保真轨道动力学引擎。

第一性原理：
  1. Orekit 提供 IERS 级坐标系转换、球谐引力与数值积分，是本框架的"金标准"参考引擎。
  2. 依赖 orekitdata 物理常数包——必须显式定位，否则精度无意义。
  3. 所有计算在 Orekit 内部完成，输出再转回 Canonical OrbitState（SI 单位）。
  4. 懒加载：orekit 仅在方法内部 import，模块顶层不依赖，保证未安装时仍可导入。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:  # 仅用于类型标注，运行时不导入（避免未安装时报错）
    from ..schemas import ForceModel, OrbitState, PropagatorConfig


class OrekitAdapter(BaseAdapter):
    """Orekit 引擎适配器。

    能力：propagate_orbit / transform_frame / convert_time / spherical_harmonics
    依赖：orekit（pip install orekit）+ orekitdata 物理常数包
    资源：OREKIT_DATA 环境变量 或 ~/.orekitdata 目录
    """

    engine_name: str = "orekit"
    _capabilities: Set[str] = {
        "propagate_orbit", "transform_frame", "convert_time", "spherical_harmonics",
    }

    def __init__(self):
        super().__init__()
        self._orekit_initialized: bool = False

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
    # 资源定位与初始化
    # ------------------------------------------------------------------
    def _find_orekit_data(self) -> Optional[str]:
        """定位 orekitdata 路径：优先 OREKIT_DATA 环境变量，其次 ~/.orekitdata。"""
        env_path = os.environ.get("OREKIT_DATA")
        if env_path and Path(env_path).exists():
            return env_path
        home_data = Path.home() / ".orekitdata"
        if home_data.exists():
            return str(home_data)
        return None

    def _ensure_orekit_vm(self) -> None:
        """初始化 Orekit JVM 并加载物理常数（幂等）。失败则抛异常。"""
        if self._orekit_initialized:
            return
        import orekit
        from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data
        data_path = self._find_orekit_data()
        if data_path is None:
            target = Path.home() / ".orekitdata"
            download_orekit_data(str(target))  # 首次使用自动下载常数包
            data_path = str(target)
        orekit.initVM()
        setup_orekit_curdir(data_path)
        self._orekit_initialized = True

    # ------------------------------------------------------------------
    # 契约实现
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 orekit 是否安装且 orekitdata 可定位。绝不抛异常。"""
        try:
            import orekit  # noqa: F401
        except Exception:
            return False
        return self._find_orekit_data() is not None

    def version(self) -> str:
        """返回 orekit.__version__，不可用时返回 'unavailable'。绝不抛异常。"""
        try:
            import orekit
            return getattr(orekit, "__version__", "unknown")
        except Exception:
            return "unavailable"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法
    # ------------------------------------------------------------------
    def propagate_orbit(self, initial_state, force_model, config) -> dict:
        """数值轨道传播。

        TODO: 将 Canonical OrbitState 转 Orekit SpacecraftState，按 ForceModel
              组装 NumericalPropagator（球谐/阻力/SRP/三体），积分 config.duration_s
              后按 output_step_s 采样输出，再转回 Canonical OrbitState 列表。
        """
        unavail = self._guard("propagate_orbit")
        if unavail is not None:
            return unavail
        try:
            self._ensure_orekit_vm()
        except Exception:
            return self._error_result("propagate_orbit",
                                      "Orekit JVM 初始化或 orekitdata 加载失败")
        try:
            from datetime import datetime, timedelta
            from org.orekit.orbits import CartesianOrbit
            from org.orekit.frames import FramesFactory
            from org.orekit.time import AbsoluteDate, TimeScalesFactory
            from org.orekit.utils import PVCoordinates, IERSConventions
            from org.hipparchus.geometry.euclidean.threed import Vector3D

            # --- 提取初始状态（兼容 OrbitState 对象与 dict）---
            if isinstance(initial_state, dict):
                pos = initial_state.get("position_m")
                vel = initial_state.get("velocity_mps")
                epoch_obj = initial_state.get("epoch", {})
                epoch_val = (epoch_obj.get("value")
                             if isinstance(epoch_obj, dict)
                             else str(epoch_obj))
                frame_obj = initial_state.get("frame", {})
                src_frame_name = (frame_obj.get("name", "GCRF")
                                  if isinstance(frame_obj, dict)
                                  else str(frame_obj))
            else:
                pos = initial_state.position_m
                vel = initial_state.velocity_mps
                epoch_val = initial_state.epoch.value
                src_frame_name = (initial_state.frame.name.value
                                  if hasattr(initial_state.frame, "name")
                                  else str(initial_state.frame))

            if pos is None or vel is None:
                return self._error_result(
                    "propagate_orbit",
                    "initial_state 缺少 position_m 或 velocity_mps")

            # --- 解析力学模型（兼容 dict 与 ForceModel 对象）---
            if isinstance(force_model, dict):
                gravity = force_model.get("gravity", "point_mass")
                sh_degree = force_model.get("degree", 0)
                sh_order = force_model.get("order", sh_degree)
            else:
                gravity = getattr(force_model, "gravity", "point_mass")
                sh_degree = getattr(force_model, "degree", 0)
                sh_order = getattr(force_model, "order", sh_degree)

            # --- 解析配置 ---
            duration = float(getattr(config, "duration_s", 86400.0))
            _output_step_raw = getattr(config, "output_step_s", None)
            output_step = (float(_output_step_raw)
                           if _output_step_raw else duration)
            mu = float(getattr(config, "mu", 3.986004418e14))

            # --- ISO 历元 → AbsoluteDate ---
            iso_str = str(epoch_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            utc = TimeScalesFactory.getUTC()
            abs_date = AbsoluteDate(dt.year, dt.month, dt.day, dt.hour,
                                    dt.minute,
                                    dt.second + dt.microsecond * 1e-6, utc)

            # --- 源帧名 → Orekit Frame ---
            _fkey = src_frame_name.upper().strip()
            if _fkey in ("ITRF", "BODYFIXED"):
                inertial_frame = FramesFactory.getITRF(
                    IERSConventions.IERS_2010, False)
            elif _fkey in ("EME2000", "J2000"):
                inertial_frame = FramesFactory.getEME2000()
            elif _fkey == "TEME":
                inertial_frame = FramesFactory.getTEME()
            else:  # GCRF / ICRF / 默认
                inertial_frame = FramesFactory.getGCRF()

            # --- PVCoordinates + CartesianOrbit ---
            position = Vector3D(float(pos[0]), float(pos[1]), float(pos[2]))
            velocity = Vector3D(float(vel[0]), float(vel[1]), float(vel[2]))
            pv = PVCoordinates(position, velocity)
            orbit = CartesianOrbit(pv, inertial_frame, abs_date, mu)

            # --- 组装传播器 ---
            use_sh = (gravity == "spherical_harmonics" and sh_degree >= 2)
            if use_sh:
                from org.orekit.propagation.numerical import (
                    NumericalPropagator)
                from org.orekit.propagation import SpacecraftState
                from org.orekit.forces.gravity import (
                    HolmesFeatherstoneAttractionModel)
                from org.orekit.forces.gravity.potential import (
                    GravityFieldFactory)
                from org.hipparchus.ode.nonstiff import (
                    DormandPrince853Integrator)

                provider = GravityFieldFactory.getNormalizedProvider(
                    int(sh_degree), int(sh_order))
                itrf = FramesFactory.getITRF(
                    IERSConventions.IERS_2010, False)
                gravity_model = HolmesFeatherstoneAttractionModel(
                    itrf, provider)
                integrator = DormandPrince853Integrator(
                    0.001, 600.0, 1e-10, 1e-10)
                propagator = NumericalPropagator(integrator)
                propagator.addForceModel(gravity_model)
                propagator.setInitialState(SpacecraftState(orbit))
                propagator_type = "numerical_spherical_harmonics"
            else:
                from org.orekit.propagation.analytical import (
                    KeplerianPropagator)
                propagator = KeplerianPropagator(orbit)
                propagator_type = "keplerian_two_body"

            # --- 按 output_step_s 采样 ---
            n_steps = max(1, int(duration / output_step) + 1)
            state_history = []
            for i in range(n_steps):
                t = min(i * output_step, duration)
                target_date = abs_date.shiftedBy(t)
                sstate = propagator.propagate(target_date)
                pv_out = sstate.getPVCoordinates(inertial_frame)
                p_out = pv_out.getPosition()
                v_out = pv_out.getVelocity()
                sample_dt = dt + timedelta(seconds=t)
                sample_iso = sample_dt.strftime("%Y-%m-%dT%H:%M:%S")
                entry = {
                    "epoch": {"value": sample_iso, "scale": "UTC",
                              "format": "ISO"},
                    "frame": {"name": src_frame_name, "center": "Earth",
                              "realization": "IERS2010"},
                    "representation": "cartesian",
                    "position_m": [p_out.getX(), p_out.getY(),
                                   p_out.getZ()],
                    "velocity_mps": [v_out.getX(), v_out.getY(),
                                     v_out.getZ()],
                    "elapsed_s": t,
                }
                state_history.append(entry)
                if t >= duration:
                    break

            return {
                "status": "success",
                "state_history": state_history,
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "propagator_type": propagator_type,
                    "units": "SI (m, m/s, s)",
                    "frame": src_frame_name,
                    "step_count": len(state_history),
                    "duration_s": duration,
                    "output_step_s": output_step,
                    "mu": mu,
                    "force_model": gravity,
                },
            }
        except Exception as exc:
            return self._error_result("propagate_orbit",
                                      f"传播失败: {exc}")

    def transform_frame(self, state, target_frame: str) -> dict:
        """坐标系转换（依赖 Orekit IERS 帧链，GCRF↔ITRF 等）。"""
        unavail = self._guard("transform_frame")
        if unavail is not None:
            return unavail
        try:
            self._ensure_orekit_vm()
        except Exception:
            return self._error_result("transform_frame", "Orekit JVM 初始化失败")
        try:
            from datetime import datetime
            from org.orekit.frames import FramesFactory
            from org.orekit.time import AbsoluteDate, TimeScalesFactory
            from org.orekit.utils import PVCoordinates, IERSConventions
            from org.hipparchus.geometry.euclidean.threed import Vector3D

            # --- 提取状态（兼容 OrbitState 对象与 dict）---
            if isinstance(state, dict):
                pos = state.get("position_m")
                vel = state.get("velocity_mps")
                epoch_obj = state.get("epoch", {})
                epoch_val = (epoch_obj.get("value")
                             if isinstance(epoch_obj, dict)
                             else str(epoch_obj))
                frame_obj = state.get("frame", {})
                src_frame_name = (frame_obj.get("name", "GCRF")
                                  if isinstance(frame_obj, dict)
                                  else str(frame_obj))
            else:
                pos = state.position_m
                vel = state.velocity_mps
                epoch_val = state.epoch.value
                src_frame_name = (state.frame.name.value
                                  if hasattr(state.frame, "name")
                                  else str(state.frame))

            if pos is None or vel is None:
                return self._error_result(
                    "transform_frame",
                    "state 缺少 position_m 或 velocity_mps")

            # --- 帧名 → Orekit Frame 映射 ---
            def _map_frame(name):
                key = name.upper().strip()
                if key in ("ITRF", "BODYFIXED"):
                    return FramesFactory.getITRF(
                        IERSConventions.IERS_2010, False)
                if key in ("EME2000", "J2000"):
                    return FramesFactory.getEME2000()
                if key == "TEME":
                    return FramesFactory.getTEME()
                # GCRF / ICRF / 默认
                return FramesFactory.getGCRF()

            source_frame = _map_frame(src_frame_name)
            target_frame_name = str(target_frame).upper().strip()
            target_frame_obj = _map_frame(target_frame_name)

            # --- ISO 历元 → AbsoluteDate ---
            iso_str = str(epoch_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            utc = TimeScalesFactory.getUTC()
            abs_date = AbsoluteDate(dt.year, dt.month, dt.day, dt.hour,
                                    dt.minute,
                                    dt.second + dt.microsecond * 1e-6, utc)

            # --- PVCoordinates → 帧变换 ---
            position = Vector3D(float(pos[0]), float(pos[1]), float(pos[2]))
            velocity = Vector3D(float(vel[0]), float(vel[1]), float(vel[2]))
            pv = PVCoordinates(position, velocity)

            transform = source_frame.getTransformTo(target_frame_obj,
                                                    abs_date)
            pv_out = transform.transformPVCoordinates(pv)
            p_out = pv_out.getPosition()
            v_out = pv_out.getVelocity()

            return {
                "status": "success",
                "position_m": [p_out.getX(), p_out.getY(), p_out.getZ()],
                "velocity_mps": [v_out.getX(), v_out.getY(),
                                 v_out.getZ()],
                "source_frame": src_frame_name,
                "target_frame": target_frame_name,
                "epoch": str(epoch_val),
                "engine": self.engine_name,
                "units": "SI (m, m/s)",
            }
        except Exception as exc:
            return self._error_result("transform_frame",
                                      f"坐标系转换失败: {exc}")

    def convert_time(self, epoch, target_scale: str) -> dict:
        """时间尺度转换（UTC/TAI/TT/TDB），输出 Canonical Epoch dict。"""
        unavail = self._guard("convert_time")
        if unavail is not None:
            return unavail
        try:
            self._ensure_orekit_vm()
        except Exception:
            return self._error_result("convert_time", "Orekit JVM 初始化失败")
        try:
            from datetime import datetime
            from org.orekit.time import AbsoluteDate, TimeScalesFactory

            # --- 提取历元（兼容 Epoch 对象、dict、字符串）---
            if isinstance(epoch, dict):
                epoch_val = epoch.get("value")
                src_scale = epoch.get("scale", "UTC")
            elif hasattr(epoch, "value"):
                epoch_val = epoch.value
                src_scale = (epoch.scale.value
                             if hasattr(epoch.scale, "value")
                             else str(epoch.scale))
            else:
                epoch_val = str(epoch)
                src_scale = "UTC"

            target = str(target_scale).upper().strip()

            # --- 时间尺度名 → Orekit TimeScale ---
            def _map_scale(name):
                key = name.upper().strip()
                if key == "UTC":
                    return TimeScalesFactory.getUTC()
                if key == "TAI":
                    return TimeScalesFactory.getTAI()
                if key == "TT":
                    return TimeScalesFactory.getTT()
                if key == "TDB":
                    return TimeScalesFactory.getTDB()
                raise ValueError(f"不支持的时间尺度: {name}")

            source_ts = _map_scale(src_scale)
            target_ts = _map_scale(target)

            # --- ISO → AbsoluteDate（以源尺度构建）---
            iso_str = str(epoch_val).replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_str)
            abs_date = AbsoluteDate(dt.year, dt.month, dt.day, dt.hour,
                                    dt.minute,
                                    dt.second + dt.microsecond * 1e-6,
                                    source_ts)

            # --- 转换到目标尺度并输出 ISO 字符串 ---
            output_str = abs_date.toString(target_ts)

            return {
                "status": "success",
                "epoch": {
                    "value": output_str,
                    "scale": target,
                    "format": "ISO",
                },
                "engine": self.engine_name,
            }
        except Exception as exc:
            return self._error_result("convert_time",
                                      f"时间转换失败: {exc}")

    def spherical_harmonics(self, body_name: str = "Earth",
                            degree: int = 70, order: int = 70) -> dict:
        """球谐引力模型查询（Orekit 专有扩展能力）。"""
        unavail = self._guard("spherical_harmonics")
        if unavail is not None:
            return unavail
        try:
            self._ensure_orekit_vm()
        except Exception:
            return self._error_result("spherical_harmonics", "Orekit JVM 初始化失败")
        try:
            from org.orekit.forces.gravity.potential import GravityFieldFactory

            provider = GravityFieldFactory.getNormalizedProvider(
                int(degree), int(order))
            provider_mu = provider.getMu()
            max_degree = provider.getMaxDegree()
            max_order = provider.getMaxOrder()

            return {
                "status": "success",
                "body": body_name,
                "degree": int(degree),
                "order": int(order),
                "engine": self.engine_name,
                "mu": provider_mu,
                "max_degree": max_degree,
                "max_order": max_order,
            }
        except Exception as exc:
            return self._error_result("spherical_harmonics",
                                      f"球谐引力模型加载失败: {exc}")
