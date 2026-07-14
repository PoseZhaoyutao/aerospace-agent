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
        "load_kernels", "list_loaded_kernels",
        "compute_observation_geometry", "compute_occultation",
        "two_line_elements_to_state",
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

    # ------------------------------------------------------------------
    # kernel 加载与管理
    # ------------------------------------------------------------------
    def load_kernels(self, paths: list) -> dict:
        """加载指定路径的 SPICE kernel 文件。"""
        unavail = self._guard("load_kernels")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            loaded = []
            failed = []
            for path in paths:
                try:
                    spice.furnsh(path)
                    self._loaded_kernels.add(path)
                    loaded.append(path)
                except Exception as exc:
                    failed.append({"path": path, "error": str(exc)})
            return {
                "status": "success",
                "loaded": loaded,
                "failed": failed,
                "total_loaded": len(self._loaded_kernels),
            }
        except Exception as exc:
            return self._error_result("load_kernels", str(exc))

    def list_loaded_kernels(self) -> dict:
        """列出当前已加载的全部 SPICE kernel 文件。"""
        unavail = self._guard("list_loaded_kernels")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            count = spice.ktotal("ALL")
            kernels = []
            for i in range(count):
                try:
                    data = spice.kdata(i, "ALL")
                    kernels.append({
                        "file": data[0],
                        "type": data[1],
                        "source": data[2],
                    })
                except Exception:
                    kernels.append({"file": "<unknown>", "type": "unknown", "source": "unknown"})
            return {
                "status": "success",
                "count": count,
                "kernels": kernels,
                "tracked_loaded": list(self._loaded_kernels),
            }
        except Exception as exc:
            return self._error_result("list_loaded_kernels", str(exc))

    # ------------------------------------------------------------------
    # 观测几何与掩星
    # ------------------------------------------------------------------
    def compute_observation_geometry(self, target: str, observer: str,
                                     epoch, frame: str, **kwargs) -> dict:
        """计算观测几何：光照角、距离、亚观测点/亚日点。

        基于 SPICE ilumin/subpnt/subslr，输出角度（deg）、距离（m）。
        注：ilumin 需要目标天体定义椭球模型（PCK），否则返回部分结果。
        """
        unavail = self._guard("compute_observation_geometry")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            import numpy as np
            self._ensure_kernels(["spk", "lsk", "pck"])

            et = spice.str2et(epoch.value)
            # 目标天体固联坐标系
            body_fixed = "IAU_" + target.upper()

            # 目标相对观测者的状态
            state, lt = spice.spkezr(target, et, frame, "NONE", observer)
            pos = state[:3]

            # 距离 (km->m)
            distance = float(spice.vnorm(pos)) * 1000.0

            # 太阳相对目标的距离
            sun_state, _ = spice.spkezr("SUN", et, frame, "NONE", target)
            sun_distance = float(spice.vnorm(sun_state[:3])) * 1000.0

            # 亚观测者点 (sub-observer point)
            subobs_lat, subobs_lon = None, None
            try:
                spoint, _, _ = spice.subpnt(
                    "NEAR POINT/ELLIPSOID", target, et, body_fixed, "NONE", observer
                )
                _, lon_obs, lat_obs = spice.recrad(spoint)
                subobs_lat = float(lat_obs) * spice.dpr()
                subobs_lon = float(lon_obs) * spice.dpr()
            except Exception:
                pass

            # 亚日点 (sub-solar point)
            subsol_lat, subsol_lon = None, None
            try:
                spoint_sun, _, _ = spice.subslr(
                    "NEAR POINT/ELLIPSOID", target, et, body_fixed, "NONE", observer
                )
                _, lon_sun, lat_sun = spice.recrad(spoint_sun)
                subsol_lat = float(lat_sun) * spice.dpr()
                subsol_lon = float(lon_sun) * spice.dpr()
            except Exception:
                pass

            # 光照角 (ilumin: 需要亚观测者点作为表面点)
            phase, solar, emissn, visible = None, None, None, None
            try:
                spoint, _, _ = spice.subpnt(
                    "NEAR POINT/ELLIPSOID", target, et, body_fixed, "NONE", observer
                )
                _, srfvec, ph, sl, em, vis = spice.ilumin(
                    "ELLIPSOID", target, "SUN", et, body_fixed, "NONE", observer, spoint
                )
                phase = float(ph) * spice.dpr()
                solar = float(sl) * spice.dpr()
                emissn = float(em) * spice.dpr()
                visible = bool(vis)
            except Exception:
                pass

            return {
                "status": "success",
                "target": target,
                "observer": observer,
                "epoch": epoch.value,
                "frame": frame,
                "phase_angle_deg": phase,
                "solar_incidence_angle_deg": solar,
                "emission_angle_deg": emissn,
                "visible": visible,
                "target_observer_distance_m": distance,
                "target_sun_distance_m": sun_distance,
                "light_time_s": float(lt),
                "sub_observer_lat_deg": subobs_lat,
                "sub_observer_lon_deg": subobs_lon,
                "sub_solar_lat_deg": subsol_lat,
                "sub_solar_lon_deg": subsol_lon,
            }
        except Exception as exc:
            return self._error_result("compute_observation_geometry", str(exc))

    def compute_occultation(self, target: str, occulting_body: str,
                             observer: str, epoch, frame: str, **kwargs) -> dict:
        """计算掩星事件——判断目标天体是否被掩星体遮挡。

        基于角距离比较：目标与掩星体视圆面是否重叠。
        返回 occultation_code: 0=无掩星, 1=部分掩星, 2=全掩星。
        """
        unavail = self._guard("compute_occultation")
        if unavail is not None:
            return unavail
        try:
            import spiceypy as spice
            import numpy as np
            self._ensure_kernels(["spk", "lsk", "pck"])

            et = spice.str2et(epoch.value)

            # 目标与掩星体相对观测者的状态
            state_tgt, _ = spice.spkezr(target, et, frame, "NONE", observer)
            state_occ, _ = spice.spkezr(occulting_body, et, frame, "NONE", observer)

            pos_tgt = np.array(state_tgt[:3])
            pos_occ = np.array(state_occ[:3])

            norm_tgt = float(spice.vnorm(pos_tgt))
            norm_occ = float(spice.vnorm(pos_occ))

            # 角距离
            cos_sep = float(np.dot(pos_tgt, pos_occ)) / (norm_tgt * norm_occ)
            cos_sep = max(-1.0, min(1.0, cos_sep))
            angular_sep_rad = float(np.arccos(cos_sep))

            # 天体半径 (km) → 视半径 (rad)
            try:
                _, radii_tgt = spice.bodvrd(target, "RADII", 3)
                rad_tgt = float(radii_tgt[0])  # 取赤道半径
            except Exception:
                rad_tgt = 0.0
            try:
                _, radii_occ = spice.bodvrd(occulting_body, "RADII", 3)
                rad_occ = float(radii_occ[0])
            except Exception:
                rad_occ = 0.0

            ang_rad_tgt = float(np.arctan2(rad_tgt, norm_tgt)) if norm_tgt > 0 else 0.0
            ang_rad_occ = float(np.arctan2(rad_occ, norm_occ)) if norm_occ > 0 else 0.0

            # 掩星判定
            if angular_sep_rad > (ang_rad_tgt + ang_rad_occ):
                code = 0  # 无掩星
            elif angular_sep_rad <= abs(ang_rad_tgt - ang_rad_occ):
                if ang_rad_tgt < ang_rad_occ:
                    code = 2  # 全掩星
                else:
                    code = 0  # 目标比掩星体大，无全掩
            else:
                code = 1  # 部分掩星

            return {
                "status": "success",
                "target": target,
                "occulting_body": occulting_body,
                "observer": observer,
                "epoch": epoch.value,
                "frame": frame,
                "occultation_code": code,
                "occultation_type": {0: "none", 1: "partial", 2: "full"}.get(code, "unknown"),
                "angular_separation_deg": angular_sep_rad * spice.dpr(),
                "target_angular_radius_deg": ang_rad_tgt * spice.dpr(),
                "occulting_body_angular_radius_deg": ang_rad_occ * spice.dpr(),
            }
        except Exception as exc:
            return self._error_result("compute_occultation", str(exc))

    # ------------------------------------------------------------------
    # TLE 转换
    # ------------------------------------------------------------------
    def two_line_elements_to_state(self, line1: str, line2: str,
                                    epoch, frame: str, **kwargs) -> dict:
        """TLE（两行轨道根数）→ 轨道状态转换。

        解析 TLE 格式字符串，提取开普勒轨道根数，解 Kepler 方程得到
        笛卡尔位置/速度。使用二体模型（不含 SGP4 长期项）。
        注意：此简化结果精度有限，高精度需求请使用 sgp4 库。
        """
        unavail = self._guard("two_line_elements_to_state")
        if unavail is not None:
            return unavail
        try:
            import math

            self._ensure_kernels(["lsk", "pck"])

            # ---- 解析 TLE Line 2 ----
            try:
                inc_deg = float(line2[8:16].strip())
                raan_deg = float(line2[17:25].strip())
                ecc_str = line2[26:33].strip()
                if "." not in ecc_str:
                    eccentricity = float("0." + ecc_str)
                else:
                    eccentricity = float(ecc_str)
                argp_deg = float(line2[34:42].strip())
                mean_anom_deg = float(line2[43:51].strip())
                mean_motion_rpd = float(line2[52:63].strip())
            except (ValueError, IndexError) as exc:
                return self._error_result(
                    "two_line_elements_to_state",
                    f"TLE 解析失败: {exc}。请检查 TLE 格式是否正确。"
                )

            # ---- 基本参数 ----
            GM = 3.986004418e14  # Earth GM (m^3/s^2)
            n = mean_motion_rpd * 2.0 * math.pi / 86400.0  # rad/s
            semi_major_axis = (GM / (n * n)) ** (1.0 / 3.0)

            inc_rad = math.radians(inc_deg)
            raan_rad = math.radians(raan_deg)
            argp_rad = math.radians(argp_deg)
            mean_anom_rad = math.radians(mean_anom_deg)

            # ---- 解 Kepler 方程 (Newton-Raphson) ----
            E = mean_anom_rad
            for _ in range(20):
                dE = (mean_anom_rad - (E - eccentricity * math.sin(E))) / (
                    1.0 - eccentricity * math.cos(E)
                )
                E += dE
                if abs(dE) < 1e-14:
                    break

            # ---- 真近点角 ----
            cos_f = (math.cos(E) - eccentricity) / (1.0 - eccentricity * math.cos(E))
            sin_f = (math.sqrt(1.0 - eccentricity * eccentricity) * math.sin(E)) / (
                1.0 - eccentricity * math.cos(E)
            )
            true_anom_rad = math.atan2(sin_f, cos_f)

            # ---- 轨道面上的位置和速度 ----
            r = semi_major_axis * (1.0 - eccentricity * math.cos(E))
            x_orb = r * math.cos(true_anom_rad)
            y_orb = r * math.sin(true_anom_rad)

            h = math.sqrt(GM * semi_major_axis * (1.0 - eccentricity * eccentricity))
            vx_orb = -(GM / h) * math.sin(true_anom_rad)
            vy_orb = (GM / h) * (eccentricity + math.cos(true_anom_rad))

            # ---- 旋转到 ECI (J2000) ----
            cos_raan = math.cos(raan_rad)
            sin_raan = math.sin(raan_rad)
            cos_inc = math.cos(inc_rad)
            sin_inc = math.sin(inc_rad)
            cos_argp = math.cos(argp_rad)
            sin_argp = math.sin(argp_rad)

            # 旋转矩阵分量
            r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_inc
            r12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc
            r13 = sin_raan * sin_inc
            r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_inc
            r22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc
            r23 = -cos_raan * sin_inc
            r31 = sin_argp * sin_inc
            r32 = cos_argp * sin_inc
            r33 = cos_inc

            px = r11 * x_orb + r12 * y_orb
            py = r21 * x_orb + r22 * y_orb
            pz = r31 * x_orb + r32 * y_orb
            vx = r11 * vx_orb + r12 * vy_orb
            vy = r21 * vx_orb + r22 * vy_orb
            vz = r31 * vx_orb + r32 * vy_orb

            return {
                "status": "success",
                "position_m": [px, py, pz],
                "velocity_mps": [vx, vy, vz],
                "frame": frame,
                "units": "SI (m, m/s)",
                "orbital_elements": {
                    "inclination_deg": inc_deg,
                    "raan_deg": raan_deg,
                    "eccentricity": eccentricity,
                    "argument_of_perigee_deg": argp_deg,
                    "mean_anomaly_deg": mean_anom_deg,
                    "true_anomaly_deg": math.degrees(true_anom_rad),
                    "mean_motion_rev_per_day": mean_motion_rpd,
                    "semi_major_axis_m": semi_major_axis,
                },
                "note": (
                    "假设: 二体开普勒运动。TLE 的 SGP4 长期项（大气阻力、J2 等）"
                    "未包含。相邻历元精度约 km 级。高精度传播请使用 sgp4 库。"
                ),
            }
        except Exception as exc:
            return self._error_result("two_line_elements_to_state", str(exc))
