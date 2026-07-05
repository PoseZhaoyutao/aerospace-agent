"""Orekit 接口工具 —— 高精度轨道传播与坐标系转换。

依赖库：orekit (Python 包装的 Orekit Java 库，需 Java 运行时)。

单位约定：SI（米 m、米/秒 m/s、m^3/s^2），与 aerospace_agent.physics 一致。

真实模式（orekit 可用）：
    - 需下载 orekit-data.zip 并在 ``initialize`` 时加载（含 EOP、章动等数据）。
    - 使用 NumericalPropagator + DormandPrince54 数值积分器传播轨道。
    - 通过 FramesFactory / Transform 完成 EME2000/MOD/TOD/ITRF 坐标系转换。

回退模式（orekit 不可用）：
    - ``propagate`` 调用 ``aerospace_agent.physics.two_body.propagate_orbit``
      （import 置于方法内部以避免 import 循环）。
    - ``convert_frame`` 使用 numpy 实现近似坐标系旋转（岁差/章动/GMST）。
    - ``initialize`` 在回退模式下记录配置即可。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseTool


# 地球标准引力常数 (m^3/s^2)，与 aerospace_agent.physics.constants 一致
MU_EARTH = 3.986004418e14
# 地球赤道半径 (m)
R_EARTH = 6378137.0
# J2000 历元对应的儒略日
JD_J2000 = 2451545.0
# 约化儒略日 J2000
MJD_J2000 = 51544.5


def _jd_to_t(jd: float) -> float:
    """儒略日 -> 自 J2000 起的世纪数 T（用于岁差/章动公式）。"""
    return (jd - (MJD_J2000 + 2400000.5)) / 36525.0


def _gmst_rad(jd: float) -> float:
    """格林尼治平恒星时（弧度），IAU 1982 公式（Vallado）。

    GMST(秒) = 67310.54841 + (876600*3600 + 8640184.812866)*T
               + 0.093104*T^2 - 6.2e-6*T^3
    其中 T = (JD - 2451545.0)/36525。J2000 历元 GMST = 18.6974 h。
    """
    t = _jd_to_t(jd)
    theta_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    theta_rad = math.radians(theta_sec / 240.0) % (2.0 * math.pi)
    if theta_rad < 0:
        theta_rad += 2.0 * math.pi
    return theta_rad


def _precession_matrix(t: float) -> np.ndarray:
    """岁差矩阵 P（IAU 1976 三次旋转），ECI(J2000) -> MOD。"""
    zeta = math.radians((2306.2181 * t + 0.30188 * t * t + 0.017998 * t ** 3) / 3600.0)
    theta = math.radians((2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) / 3600.0)
    z = math.radians((2306.2181 * t + 1.09468 * t * t + 0.018203 * t ** 3) / 3600.0)

    def _rz(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def _ry(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    return _rz(-z) @ _ry(theta) @ _rz(-zeta)


def _nutation_matrix(t: float) -> np.ndarray:
    """章动矩阵 N（IAU 1980 主项近似），MOD -> TOD。"""
    omega = math.radians(125.0445 - 1934.136 * t)
    delta_psi = math.radians(-17.20 * math.sin(omega) / 3600.0)
    delta_eps = math.radians(9.20 * math.cos(omega) / 3600.0)
    eps0 = math.radians(23.439291 - 0.0130042 * t)
    eps = eps0 + delta_eps

    def _rx(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def _rz(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    return _rx(-eps) @ _rz(-delta_psi) @ _rx(eps0)


def _itrf_rotation(jd: float) -> np.ndarray:
    """EME2000(ECI) -> ITRF(ECEF) 的近似旋转矩阵（仅 z 轴 GMST）。

    注：完整转换还需极移与章动，此处采用工程近似。
    """
    theta = _gmst_rad(jd)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


class OrekitTool(BaseTool):
    """Orekit 高精度轨道传播工具。"""

    name = "orekit"
    description = "高精度轨道数值传播与坐标系转换（EME2000/MOD/TOD/ITRF），SI 单位"
    library_name = "orekit"

    methods_schema = {
        "propagate": {
            "params": {"initial_state": "list(6) [m,m/s]", "times": "list[s]",
                       "mu": "float[m^3/s^2]"},
            "returns": "ndarray(N,6)",
            "description": "用数值积分器传播轨道状态",
        },
        "convert_frame": {
            "params": {"state": "list(6)", "from_frame": "str",
                       "to_frame": "str", "epoch": "float(jd)"},
            "returns": "list(6)",
            "description": "坐标系转换：EME2000, MOD, TOD, ITRF",
        },
        "initialize": {
            "params": {"propagator_type": "str"},
            "returns": "dict",
            "description": "初始化传播器（真实模式需加载 orekit-data）",
        },
    }

    def __init__(self) -> None:
        self._initialized = False
        self._propagator_type = "keplerian"
        self._orekit_vm = False

    # ------------------------------------------------------------------
    # 真实模式实现
    # ------------------------------------------------------------------
    def _init_real(self, propagator_type: str = "keplerian") -> dict:
        """真实模式初始化 Orekit JVM 与数据。"""
        import orekit  # noqa: F401
        from orekit.pyhelpers import setup_orekit_curdir

        orekit.initVM()
        self._orekit_vm = True
        try:
            setup_orekit_curdir()
            data_loaded = True
            message = "orekit-data 已加载"
        except Exception:
            data_loaded = False
            message = (
                "未找到 orekit-data。请下载 orekit-data.zip 并解压到工作目录，"
                "否则 EOP/章动数据缺失。可调用 "
                "from orekit.pyhelpers import download_orekit_data; "
                "download_orekit_data() 自动下载。"
            )
        self._initialized = True
        self._propagator_type = propagator_type
        return {"vm_started": True, "data_loaded": data_loaded,
                "propagator_type": propagator_type, "message": message}

    def _propagate_real(
        self, initial_state: Sequence[float], times: Sequence[float], mu: float
    ) -> np.ndarray:
        """真实模式：Orekit NumericalPropagator 数值传播。"""
        import orekit
        from orekit import JArray_double
        from org.orekit.orbits import CartesianOrbit
        from org.orekit.frames import FramesFactory
        from org.orekit.propagation.numerical import NumericalPropagator
        from org.hipparchus.ode.nonstiff import DormandPrince54Integrator
        from org.orekit.time import AbsoluteDate

        if not self._orekit_vm:
            orekit.initVM()

        frame = FramesFactory.getEME2000()
        epoch = AbsoluteDate.J2000_EPOCH
        pos = JArray_double([float(x) for x in initial_state[:3]])
        vel = JArray_double([float(x) for x in initial_state[3:6]])
        orbit = CartesianOrbit(pos, vel, frame, epoch, float(mu))

        integrator = DormandPrince54Integrator(1e-3, 600.0, 1e-8, 1e-8)
        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(orbit.getType())
        propagator.setInitialState(orbit)

        results = []
        for dt in times:
            date = epoch.shiftedBy(float(dt))
            state = propagator.propagate(date)
            pv = state.getPVCoordinates(frame)
            p = pv.getPosition()
            v = pv.getVelocity()
            results.append([p.getX(), p.getY(), p.getZ(),
                            v.getX(), v.getY(), v.getZ()])
        return np.array(results)

    def _convert_frame_real(
        self, state: Sequence[float], from_frame: str, to_frame: str, epoch: float
    ) -> List[float]:
        """真实模式：Orekit Transform 坐标系转换。"""
        from org.orekit.frames import FramesFactory
        from org.orekit.time import AbsoluteDate
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.utils import PVCoordinates

        def _get_frame(name: str):
            mapping = {
                "EME2000": FramesFactory.getEME2000,
                "MOD": lambda: FramesFactory.getMOD(False),
                "TOD": lambda: FramesFactory.getTOD(False),
                "ITRF": FramesFactory.getITRF,
            }
            key = name.upper().replace("J2000", "EME2000")
            if key not in mapping:
                raise ValueError(f"未知坐标系: {name}")
            return mapping[key]()

        f_from = _get_frame(from_frame)
        f_to = _get_frame(to_frame)
        date = AbsoluteDate.J2000_EPOCH.shiftedBy((epoch - JD_J2000) * 86400.0)

        pos = Vector3D(float(state[0]), float(state[1]), float(state[2]))
        vel = Vector3D(float(state[3]), float(state[4]), float(state[5]))
        transform = f_from.getTransformTo(f_to, date)
        pv_out = transform.transformPVCoordinates(PVCoordinates(pos, vel))
        p = pv_out.getPosition()
        v = pv_out.getVelocity()
        return [p.getX(), p.getY(), p.getZ(), v.getX(), v.getY(), v.getZ()]

    # ------------------------------------------------------------------
    # 回退模式实现
    # ------------------------------------------------------------------
    def _propagate_fallback(
        self, initial_state: Sequence[float], times: Sequence[float], mu: float
    ) -> np.ndarray:
        """回退模式：调用内置 two_body 传播器。

        import 置于方法内部，并用 try/except 防止 import 循环。
        使用 propagate_orbit 沿时间数组传播（普适变量法，SI 单位）。
        """
        try:
            from aerospace_agent.physics import two_body
        except ImportError as e:
            raise ImportError(
                "回退传播需要 aerospace_agent.physics.two_body，但导入失败: " + str(e)
            ) from e
        r0 = np.asarray(initial_state[:3], dtype=float)
        v0 = np.asarray(initial_state[3:6], dtype=float)
        return two_body.propagate_orbit(r0, v0, mu, list(times))

    def _convert_frame_fallback(
        self, state: Sequence[float], from_frame: str, to_frame: str, epoch: float
    ) -> List[float]:
        """回退模式：基于 numpy 的近似坐标系转换。

        以 EME2000(ECI) 为中枢，MOD/TOD/ITRF 之间通过 ECI 中转。
        旋转矩阵与单位无关，仅作用于方向，故对任意单位的状态向量均适用。
        """
        pos = np.array(state[:3], dtype=float)
        vel = np.array(state[3:6], dtype=float)
        jd = float(epoch)
        t = _jd_to_t(jd)

        norm = {"EME2000": "ECI", "J2000": "ECI",
                "MOD": "MOD", "TOD": "TOD", "ITRF": "ITRF"}
        f = norm.get(from_frame.upper(), from_frame.upper())
        to = norm.get(to_frame.upper(), to_frame.upper())
        if f == to:
            return list(state)

        def _to_eci(p, v, frame):
            if frame == "ECI":
                return p, v
            if frame == "MOD":
                P = _precession_matrix(t)
                return P.T @ p, P.T @ v
            if frame == "TOD":
                M = _nutation_matrix(t) @ _precession_matrix(t)
                return M.T @ p, M.T @ v
            if frame == "ITRF":
                R = _itrf_rotation(jd)
                return R.T @ p, R.T @ v
            raise ValueError(f"未知源坐标系: {frame}")

        def _from_eci(p, v, frame):
            if frame == "ECI":
                return p, v
            if frame == "MOD":
                P = _precession_matrix(t)
                return P @ p, P @ v
            if frame == "TOD":
                M = _nutation_matrix(t) @ _precession_matrix(t)
                return M @ p, M @ v
            if frame == "ITRF":
                R = _itrf_rotation(jd)
                return R @ p, R @ v
            raise ValueError(f"未知目标坐标系: {frame}")

        p_eci, v_eci = _to_eci(pos, vel, f)
        p_out, v_out = _from_eci(p_eci, v_eci, to)
        return list(p_out) + list(v_out)

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs) -> dict:
        if method == "initialize":
            return self._call_initialize(**kwargs)
        if method == "propagate":
            return self._call_propagate(**kwargs)
        if method == "convert_frame":
            return self._call_convert_frame(**kwargs)
        return self._fail(f"未知方法: {method}", self.source,
                          f"可用方法: {self.list_methods()}")

    def _call_initialize(self, propagator_type: str = "keplerian") -> dict:
        if self.is_available:
            try:
                info = self._init_real(propagator_type)
                return self._ok(info, "real",
                                "Orekit 真实模式初始化完成。" + info["message"])
            except Exception as e:
                self._initialized = True
                self._propagator_type = propagator_type
                return self._ok(
                    {"propagator_type": propagator_type, "data_loaded": False},
                    "fallback",
                    f"真实模式初始化失败({e})，已回退到内置二体传播器。",
                )
        self._initialized = True
        self._propagator_type = propagator_type
        return self._ok(
            {"propagator_type": propagator_type, "engine": "two_body"},
            "fallback",
            "orekit 不可用，回退到 aerospace_agent.physics.two_body 二体传播。",
        )

    def _call_propagate(
        self, initial_state: Sequence[float], times: Sequence[float],
        mu: float = MU_EARTH,
    ) -> dict:
        if not self._initialized:
            self._call_initialize()
        times = list(times)
        if self.is_available:
            try:
                res = self._propagate_real(initial_state, times, mu)
                return self._ok(
                    {"states": res.tolist(), "n": len(times), "mu": mu},
                    "real", "Orekit NumericalPropagator 传播完成。",
                )
            except Exception as e:
                try:
                    res = self._propagate_fallback(initial_state, times, mu)
                    return self._ok(
                        {"states": res.tolist(), "n": len(times), "mu": mu},
                        "fallback", f"真实模式失败({e})，已回退到二体传播。",
                    )
                except Exception as e2:
                    return self._fail(str(e2), "fallback", "回退传播也失败")
        try:
            res = self._propagate_fallback(initial_state, times, mu)
            return self._ok(
                {"states": res.tolist(), "n": len(times), "mu": mu},
                "fallback",
                "回退模式：aerospace_agent.physics.two_body 二体传播完成。",
            )
        except Exception as e:
            return self._fail(str(e), "fallback", "回退传播失败")

    def _call_convert_frame(
        self, state: Sequence[float], from_frame: str, to_frame: str, epoch: float
    ) -> dict:
        if self.is_available:
            try:
                res = self._convert_frame_real(state, from_frame, to_frame, epoch)
                return self._ok(
                    {"state": res, "from": from_frame, "to": to_frame, "epoch": epoch},
                    "real", "Orekit Transform 坐标系转换完成。",
                )
            except Exception as e:
                try:
                    res = self._convert_frame_fallback(state, from_frame, to_frame, epoch)
                    return self._ok(
                        {"state": res, "from": from_frame, "to": to_frame, "epoch": epoch},
                        "fallback", f"真实模式失败({e})，已回退到近似坐标系转换。",
                    )
                except Exception as e2:
                    return self._fail(str(e2), "fallback", "回退转换也失败")
        try:
            res = self._convert_frame_fallback(state, from_frame, to_frame, epoch)
            return self._ok(
                {"state": res, "from": from_frame, "to": to_frame, "epoch": epoch},
                "fallback",
                "回退模式：numpy 近似坐标系转换（岁差/章动/GMST）。",
            )
        except Exception as e:
            return self._fail(str(e), "fallback", "回退转换失败")


if __name__ == "__main__":
    tool = OrekitTool()
    print("工具信息:", {k: v for k, v in tool.get_info().items()
          if k != "methods_schema"})

    print("\n--- initialize ---")
    print(tool.call("initialize", propagator_type="keplerian"))

    # 传播（回退模式，SI 单位：m, m/s）
    print("\n--- propagate ---")
    state0 = [6778e3, 0.0, 0.0, 0.0, 7660.0, 0.0]  # ISS 近似圆轨道
    times = [0.0, 1800.0, 3600.0, 5400.0]
    r = tool.call("propagate", initial_state=state0, times=times, mu=MU_EARTH)
    print("source:", r["source"], "n:", r["result"]["n"])
    print("初始 (km):", [round(x / 1e3, 4) for x in r["result"]["states"][0]])
    print("末端位置半径 (km):", round(np.linalg.norm(r["result"]["states"][-1][:3]) / 1e3, 4))

    # 坐标系转换（回退模式）
    print("\n--- convert_frame EME2000 -> ITRF ---")
    r2 = tool.call("convert_frame", state=state0, from_frame="EME2000",
                   to_frame="ITRF", epoch=JD_J2000)
    print("source:", r2["source"])
    print("ITRF 状态 (km):", [round(x / 1e3, 4) for x in r2["result"]["state"]])
    # 往返一致性
    r3 = tool.call("convert_frame", state=r2["result"]["state"],
                   from_frame="ITRF", to_frame="EME2000", epoch=JD_J2000)
    print("往返 EME2000 (km):", [round(x / 1e3, 4) for x in r3["result"]["state"]])
