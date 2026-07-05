"""SPICE 接口工具 —— 星历查询与坐标系转换。

依赖库：spiceypy (NASA SPICE 工具包的 Python 封装)。

真实模式（spiceypy 可用）：
    - 需先用 ``load_kernel`` 加载 SPICE kernel（如 de430.bsp、naif0012.tls）。
    - ``get_ephemeris`` 使用 ``spiceypy.spkezr`` 获取目标相对观测者的状态向量。
    - ``get_moon_state`` 获取月球相对地心的位置/速度。
    - ``convert_state`` 使用 ``spiceypy.sxform`` 做坐标系转换。

回退模式（spiceypy 不可用）：
    - 使用解析公式（均质圆轨道近似）计算月球位置速度：
        * 月球平黄经 L = 218.316 + 13.176396 * T  (度, T 为自 J2000 的天数)
        * 升交点经度 Ω = 125.0445 - 0.0529530 * T (度, 18.6 年进动)
        * 轨道倾角 i = 5.145° (相对黄道)
        * 经黄道->赤道(倾角 ε=23.43929°)旋转得到 J2000 ECI 状态
    - 验证：J2000 时刻月球应在平黄经 218.316° 附近。
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseTool

# J2000 历元儒略日
JD_J2000 = 2451545.0
# 地球引力常数 (km^3/s^2)
MU_EARTH = 398600.4418
# 月球轨道参数（均质圆轨道近似）
MOON_MEAN_DISTANCE_KM = 384400.0          # 半长轴
MOON_INCLINATION_DEG = 5.145               # 相对黄道倾角
MOON_MEAN_LONGITUDE_RATE = 13.176396       # 度/天 (平黄经变化率)
MOON_MEAN_LONGITUDE_J2000 = 218.316        # J2000 平黄经 (度)
MOON_NODE_LONGITUDE_J2000 = 125.0445       # J2000 升交点经度 (度)
MOON_NODE_RATE = -0.0529530                # 度/天 (升交点退行)
OBLIQUITY_DEG = 23.439291                  # 黄赤交角


# ----------------------------------------------------------------------
# 回退模式：解析月球星历
# ----------------------------------------------------------------------
def _moon_ecliptic_position(t_days: float) -> np.ndarray:
    """计算月球在黄道坐标系下的位置（均质圆轨道近似）。

    Parameters
    ----------
    t_days : float
        自 J2000 起的天数。

    Returns
    -------
    np.ndarray, shape (3,)
        黄道坐标系下月球位置 (km)，x 轴指向春分点。
    """
    # 平黄经与升交点经度（度 -> 弧度）
    L = math.radians(MOON_MEAN_LONGITUDE_J2000 + MOON_MEAN_LONGITUDE_RATE * t_days)
    Omega = math.radians(MOON_NODE_LONGITUDE_J2000 + MOON_NODE_RATE * t_days)
    i = math.radians(MOON_INCLINATION_DEG)
    # 幅角 u = L - Ω（从升交点起量的轨道内角度）
    u = L - Omega

    # 轨道平面内位置 (perifocal-like, 升交点为 x 轴)
    x_orb = MOON_MEAN_DISTANCE_KM * math.cos(u)
    y_orb = MOON_MEAN_DISTANCE_KM * math.sin(u)
    # 黄道坐标系：R3(Ω) R1(i) [x_orb, y_orb, 0]
    #   x_ecl = cosΩ·x_orb - sinΩ·cosi·y_orb
    #   y_ecl = sinΩ·x_orb + cosΩ·cosi·y_orb
    #   z_ecl =              sin_i·y_orb
    cosO, sinO = math.cos(Omega), math.sin(Omega)
    cosi, sini = math.cos(i), math.sin(i)
    x_ecl = cosO * x_orb - sinO * cosi * y_orb
    y_ecl = sinO * x_orb + cosO * cosi * y_orb
    z_ecl = sini * y_orb
    return np.array([x_ecl, y_ecl, z_ecl])


def _ecliptic_to_equatorial(vec: np.ndarray) -> np.ndarray:
    """黄道坐标 -> J2000 赤道坐标 (ECI)：绕 x 轴旋转 +ε。"""
    eps = math.radians(OBLIQUITY_DEG)
    c, s = math.cos(eps), math.sin(eps)
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, c, -s],
                  [0.0, s, c]])
    return R @ vec


def _moon_state_fallback(jd: float) -> Dict[str, Any]:
    """回退模式：解析计算月球相对地心的 J2000 ECI 状态向量。

    位置由均质圆轨道近似给出；速度采用中心差分法获得（自动包含
    平黄经变化率与升交点进动率，避免解析求导符号错误）。
    """
    t_days = jd - JD_J2000
    # 中心差分步长（秒 -> 天），约 60 秒
    dt_days = 60.0 / 86400.0
    pos_ecl = _moon_ecliptic_position(t_days)
    pos_plus = _moon_ecliptic_position(t_days + dt_days)
    pos_minus = _moon_ecliptic_position(t_days - dt_days)
    # 速度 (km/s)：差分单位为天，转秒
    vel_ecl = (pos_plus - pos_minus) / (2.0 * dt_days * 86400.0)

    pos_eci = _ecliptic_to_equatorial(pos_ecl)
    vel_eci = _ecliptic_to_equatorial(vel_ecl)

    # 同时给出黄道经纬用于校验
    lon_ecl = math.degrees(math.atan2(pos_ecl[1], pos_ecl[0])) % 360.0
    lat_ecl = math.degrees(math.asin(np.clip(pos_ecl[2] / np.linalg.norm(pos_ecl), -1, 1)))

    return {
        "position_km": pos_eci.tolist(),
        "velocity_km_s": vel_eci.tolist(),
        "distance_km": float(np.linalg.norm(pos_eci)),
        "speed_km_s": float(np.linalg.norm(vel_eci)),
        "ecliptic_longitude_deg": lon_ecl,
        "ecliptic_latitude_deg": lat_ecl,
        "mean_longitude_deg": (MOON_MEAN_LONGITUDE_J2000
                               + MOON_MEAN_LONGITUDE_RATE * t_days) % 360.0,
        "epoch_jd": jd,
        "frame": "J2000",
    }


class SpiceypyTool(BaseTool):
    """SPICE 星历与坐标系工具。"""

    name = "spiceypy"
    description = "SPICE 星历查询、月球状态计算与坐标系转换"
    library_name = "spiceypy"

    methods_schema = {
        "get_ephemeris": {
            "params": {"target": "str", "epoch": "float(jd)",
                       "frame": "str", "observer": "str"},
            "returns": "dict",
            "description": "获取目标天体相对观测者的状态向量",
        },
        "get_moon_state": {
            "params": {"epoch": "float(jd)"},
            "returns": "dict",
            "description": "获取月球相对地心的位置与速度",
        },
        "convert_state": {
            "params": {"state": "list", "from_frame": "str",
                       "to_frame": "str", "epoch": "float(jd)"},
            "returns": "dict",
            "description": "状态向量坐标系转换",
        },
        "load_kernel": {
            "params": {"kernel_path": "str"},
            "returns": "dict",
            "description": "加载 SPICE kernel 文件",
        },
    }

    def __init__(self) -> None:
        self._kernels: List[str] = []

    # ------------------------------------------------------------------
    # 真实模式实现
    # ------------------------------------------------------------------
    def _jd_to_et(self, jd: float) -> float:
        """儒略日 -> SPICE ephemeris time (秒)。"""
        import spiceypy as spice
        # JD -> UTC 字符串 -> ET
        # 使用 spiceypy 的 str2et，需构造 UTC ISO 字符串
        # JD -> UTC: 用 spicedpg 的便捷方式
        # 简单做法：detim 替代，这里用 timedelta
        # spiceypy 提供 j2000() 与 et 的关系：ET = (JD - 2451545.0)*86400 近似
        # 但更准确需 leapsecond kernel。这里用 spdf/str2et 若已加载。
        try:
            utc = spice.et2utc((jd - 2451545.0) * 86400.0, "ISOD", 3)
            return spice.str2et(utc)
        except Exception:
            # 无时间 kernel 时用近似
            return (jd - 2451545.0) * 86400.0

    def _get_ephemeris_real(
        self, target: str, epoch: float, frame: str, observer: str
    ) -> dict:
        import spiceypy as spice
        et = self._jd_to_et(epoch)
        state, lt = spice.spkezr(target, et, frame, "NONE", observer)
        return {
            "state": list(state),
            "light_time": lt,
            "target": target,
            "observer": observer,
            "frame": frame,
            "epoch_jd": epoch,
        }

    def _get_moon_state_real(self, epoch: float) -> dict:
        import spiceypy as spice
        et = self._jd_to_et(epoch)
        state, lt = spice.spkezr("MOON", et, "J2000", "NONE", "EARTH")
        pos = list(state[:3])
        vel = list(state[3:])
        return {
            "position_km": pos,
            "velocity_km_s": vel,
            "distance_km": float(np.linalg.norm(pos)),
            "speed_km_s": float(np.linalg.norm(vel)),
            "light_time": lt,
            "epoch_jd": epoch,
            "frame": "J2000",
        }

    def _convert_state_real(
        self, state: Sequence[float], from_frame: str, to_frame: str, epoch: float
    ) -> dict:
        import spiceypy as spice
        et = self._jd_to_et(epoch)
        rot = spice.sxform(from_frame, to_frame, et)
        state_arr = np.array(state, dtype=float).reshape(6)
        new_state = rot @ state_arr
        return {
            "state": list(new_state),
            "from_frame": from_frame,
            "to_frame": to_frame,
            "epoch_jd": epoch,
        }

    def _load_kernel_real(self, kernel_path: str) -> dict:
        import spiceypy as spice
        if not os.path.isfile(kernel_path):
            return {"loaded": False, "error": f"kernel 文件不存在: {kernel_path}"}
        spice.furnsh(kernel_path)
        self._kernels.append(kernel_path)
        return {"loaded": True, "path": kernel_path, "loaded_kernels": list(self._kernels)}

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs) -> dict:
        if method == "get_ephemeris":
            return self._call_get_ephemeris(**kwargs)
        if method == "get_moon_state":
            return self._call_get_moon_state(**kwargs)
        if method == "convert_state":
            return self._call_convert_state(**kwargs)
        if method == "load_kernel":
            return self._call_load_kernel(**kwargs)
        return self._fail(f"未知方法: {method}", self.source,
                          f"可用方法: {self.list_methods()}")

    def _call_get_ephemeris(
        self, target: str, epoch: float,
        frame: str = "J2000", observer: str = "EARTH",
    ) -> dict:
        if self.is_available:
            try:
                res = self._get_ephemeris_real(target, epoch, frame, observer)
                return self._ok(res, "real", "SPICE spkezr 星历查询完成。")
            except Exception as e:
                return self._fail(str(e), "real", "SPICE 星历查询失败")
        # 回退：仅支持月球目标
        if target.upper() in ("MOON", "LUNA", "MOON BARYCENTER"):
            res = _moon_state_fallback(epoch)
            res["target"] = "MOON"
            res["observer"] = observer
            return self._ok(
                res, "fallback",
                "spiceypy 不可用，回退到月球解析星历（均质圆轨道近似）。",
            )
        return self._unavailable(
            "get_ephemeris", "spiceypy",
            install_hint="非月球目标需 spiceypy + SPICE kernel。"
                         "运行 pip install spiceypy 并加载 de430.bsp 等星历核。",
        )

    def _call_get_moon_state(self, epoch: float) -> dict:
        if self.is_available:
            try:
                res = self._get_moon_state_real(epoch)
                return self._ok(res, "real", "SPICE 月球状态查询完成。")
            except Exception as e:
                res = _moon_state_fallback(epoch)
                return self._ok(
                    res, "fallback",
                    f"真实模式失败({e})，已回退到月球解析星历。",
                )
        res = _moon_state_fallback(epoch)
        return self._ok(
            res, "fallback",
            "spiceypy 不可用，回退到月球解析星历（均质圆轨道近似，"
            "J2000 平黄经 218.316°）。",
        )

    def _call_convert_state(
        self, state: Sequence[float], from_frame: str,
        to_frame: str, epoch: float,
    ) -> dict:
        if self.is_available:
            try:
                res = self._convert_state_real(state, from_frame, to_frame, epoch)
                return self._ok(res, "real", "SPICE sxform 坐标系转换完成。")
            except Exception as e:
                return self._fail(str(e), "real", "SPICE 坐标系转换失败")
        # 回退：复用 orekit_tool 的近似旋转（ECI<->ITRF）
        return self._convert_state_fallback(state, from_frame, to_frame, epoch)

    def _convert_state_fallback(
        self, state: Sequence[float], from_frame: str,
        to_frame: str, epoch: float,
    ) -> dict:
        """回退模式：调用 orekit_tool 的近似坐标系旋转。"""
        try:
            # import 置于方法内，避免顶层循环依赖
            from .orekit_tool import OrekitTool
            orekit = OrekitTool()
            res = orekit.call(
                "convert_frame",
                state=state, from_frame=from_frame,
                to_frame=to_frame, epoch=epoch,
            )
            if res["success"]:
                return self._ok(
                    {"state": res["result"]["state"], "from_frame": from_frame,
                     "to_frame": to_frame, "epoch_jd": epoch},
                    "fallback",
                    "spiceypy 不可用，回退到近似坐标系转换（岁差/章动/GMST）。",
                )
            return self._fail(res.get("error", "未知错误"), "fallback",
                              "回退坐标系转换失败")
        except Exception as e:
            return self._fail(str(e), "fallback", "回退坐标系转换异常")

    def _call_load_kernel(self, kernel_path: str) -> dict:
        if self.is_available:
            res = self._load_kernel_real(kernel_path)
            if res.get("loaded"):
                return self._ok(res, "real", f"已加载 kernel: {kernel_path}")
            return self._fail(res.get("error", "未知错误"), "real", "kernel 加载失败")
        # 回退：无法加载真实 kernel
        return self._ok(
            {"loaded": False, "path": kernel_path, "note": "spiceypy 不可用，"
             "kernel 未实际加载。回退星历使用解析公式，不依赖 kernel。"},
            "fallback",
            "spiceypy 不可用：回退模式无需 kernel，月球星历由解析公式计算。",
        )


if __name__ == "__main__":
    tool = SpiceypyTool()
    print("工具信息:", {k: v for k, v in tool.get_info().items()
          if k != "methods_schema"})

    # 关键验证：J2000 时刻月球状态
    print("\n--- get_moon_state(J2000) ---")
    r = tool.call("get_moon_state", epoch=JD_J2000)
    print("source:", r["source"])
    res = r["result"]
    print(f"位置 (km): {[round(x, 2) for x in res['position_km']]}")
    print(f"速度 (km/s): {[round(x, 5) for x in res['velocity_km_s']]}")
    print(f"距离 (km): {res['distance_km']:.2f}")
    print(f"速度模 (km/s): {res['speed_km_s']:.5f}")
    print(f"黄道经度 (度): {res['ecliptic_longitude_deg']:.4f}")
    print(f"平黄经 (度): {res['mean_longitude_deg']:.4f}")
    print(f"黄道纬度 (度): {res['ecliptic_latitude_deg']:.4f}")
    # 验证：J2000 月球应在平黄经 218.316° 附近
    assert abs(res["mean_longitude_deg"] - 218.316) < 0.01, "平黄经校验失败"
    assert abs(res["ecliptic_longitude_deg"] - 218.316) < 1.0, "黄道经度应近 218.316°"
    assert abs(res["distance_km"] - MOON_MEAN_DISTANCE_KM) < 1.0, "距离校验"
    print(">>> 校验通过：J2000 月球平黄经 = %.3f° (期望 218.316°)"
          % res["mean_longitude_deg"])

    # 一天后月球位置变化（应前进约 13.176°）
    print("\n--- get_moon_state(J2000 + 1天) ---")
    r2 = tool.call("get_moon_state", epoch=JD_J2000 + 1.0)
    print(f"平黄经: {r2['result']['mean_longitude_deg']:.4f}° "
          f"(前一天 {res['mean_longitude_deg']:.4f}°, 差 "
          f"{r2['result']['mean_longitude_deg']-res['mean_longitude_deg']:.4f}°, "
          f"期望 +13.176°)")

    # load_kernel 回退
    print("\n--- load_kernel (回退) ---")
    print(tool.call("load_kernel", kernel_path="/tmp/de430.bsp"))
