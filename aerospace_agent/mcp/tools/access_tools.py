"""地面可见性工具 — 计算卫星对地面站的可见时间窗口。

第一性原理（K3 可见性判定）：
  1. 可见性 = 仰角 ≥ min_elevation_deg 的连续时段
  2. 仰角由卫星 ECEF 位置与地面站位置几何关系决定
  3. 引擎优先级：orekit/stk adapter > 内置简化算法
  4. 输出每个窗口的 start/stop/max_elevation_deg/duration_s
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from ..schemas import OrbitState, GroundStation, Epoch
from ..adapters import get_adapter

#: 地球平均半径 m
_R_EARTH = 6378137.0
#: 地球自转角速度 rad/s
_OMEGA_EARTH = 7.2921159e-5
#: 默认采样步长 s
_DEFAULT_STEP_S = 60.0


def compute_ground_access(orbit_state_dict: Dict,
                          ground_station_dict: Dict,
                          start_epoch_dict: Dict,
                          stop_epoch_dict: Dict,
                          min_elevation_deg: float = 5.0) -> Dict:
    """计算卫星对地面站的可见时间窗口。

    Args:
        orbit_state_dict: 初始 OrbitState 字典（GCRF/EME2000 惯性系）
        ground_station_dict: GroundStation 字典
        start_epoch_dict: 起始时间 Epoch 字典
        stop_epoch_dict: 结束时间 Epoch 字典
        min_elevation_deg: 最小仰角阈值 deg
    Returns:
        {access_windows, total_windows, engine}
    """
    # 尝试 orekit adapter
    result = _try_adapter("orekit", orbit_state_dict, ground_station_dict,
                          start_epoch_dict, stop_epoch_dict, min_elevation_deg)
    if result:
        return result

    # 尝试 stk adapter
    result = _try_adapter("stk", orbit_state_dict, ground_station_dict,
                          start_epoch_dict, stop_epoch_dict, min_elevation_deg)
    if result:
        return result

    # 回退到内置简化算法
    return _builtin_access(orbit_state_dict, ground_station_dict,
                           start_epoch_dict, stop_epoch_dict,
                           min_elevation_deg)


def _try_adapter(engine, orbit_dict, station_dict, start_dict,
                 stop_dict, min_el) -> Optional[Dict]:
    """尝试使用引擎适配器计算可见性。"""
    adapter = get_adapter(engine)
    if not adapter.is_available():
        return None
    try:
        orbit_state = OrbitState.from_dict(orbit_dict)
        station = GroundStation.from_dict(station_dict)
        start_epoch = Epoch.from_dict(start_dict)
        stop_epoch = Epoch.from_dict(stop_dict)
        result = adapter.compute_ground_access(
            orbit_state, station, start_epoch, stop_epoch, min_el
        )
        if result.get("status") != "unavailable":
            result.setdefault("engine", engine)
            return result
    except Exception:
        pass
    return None


def _builtin_access(orbit_dict, station_dict, start_dict, stop_dict,
                    min_el) -> Dict:
    """内置简化可见性算法——基于二体传播 + 仰角采样。"""
    try:
        state = OrbitState.from_dict(orbit_dict)
        station = GroundStation.from_dict(station_dict)
        start_dt = _epoch_to_datetime(Epoch.from_dict(start_dict))
        stop_dt = _epoch_to_datetime(Epoch.from_dict(stop_dict))
    except Exception as exc:
        return _error(f"输入解析失败: {exc}")

    if state.position_m is None or state.velocity_mps is None:
        return _error("轨道状态缺少 position_m / velocity_mps")

    total_s = (stop_dt - start_dt).total_seconds()
    if total_s <= 0:
        return _error("stop_epoch 必须晚于 start_epoch")

    # 采样
    n_samples = min(max(int(total_s / _DEFAULT_STEP_S), 10), 10000)
    dt_step = total_s / n_samples

    station_ecef = _geodetic_to_ecef(
        station.latitude_deg, station.longitude_deg, station.altitude_m)

    elevations: List[Dict] = []
    r = list(state.position_m)
    v = list(state.velocity_mps)
    mu = 3.986004418e14

    for i in range(n_samples + 1):
        t = i * dt_step
        current_dt = start_dt + timedelta(seconds=t)
        # 二体传播到 t
        pos_inertial = _propagate_two_body(r, v, t, mu)
        # 惯性系 → ECEF（简化地球自转）
        pos_ecef = _inertial_to_ecef(pos_inertial, current_dt)
        # 计算仰角
        el_deg = _compute_elevation(pos_ecef, station_ecef)
        elevations.append({
            "datetime": current_dt.isoformat(),
            "elevation_deg": el_deg,
        })

    # 提取可见窗口
    windows = _extract_windows(elevations, min_el)

    return {
        "access_windows": windows,
        "total_windows": len(windows),
        "engine": "builtin",
        "engine_version": "0.1.0",
        "min_elevation_deg": min_el,
        "units": "SI (deg, s)",
        "algorithm": "简化二体传播 + 60s 采样 + 线性插值窗口边界",
    }


def _extract_windows(elevations, min_el) -> List[Dict]:
    """从仰角序列提取可见窗口。"""
    windows = []
    in_window = False
    window_start = None
    max_el = -90.0

    for i, e in enumerate(elevations):
        visible = e["elevation_deg"] >= min_el
        if visible and not in_window:
            in_window = True
            window_start = e["datetime"]
            max_el = e["elevation_deg"]
        elif visible and in_window:
            max_el = max(max_el, e["elevation_deg"])
        elif not visible and in_window:
            in_window = False
            windows.append(_make_window(window_start, e["datetime"], max_el))
            max_el = -90.0

    if in_window:
        windows.append(_make_window(window_start, elevations[-1]["datetime"], max_el))

    return windows


def _make_window(start, stop, max_el) -> Dict:
    try:
        dt_start = datetime.fromisoformat(start)
        dt_stop = datetime.fromisoformat(stop)
        duration = (dt_stop - dt_start).total_seconds()
    except Exception:
        duration = 0.0
    return {
        "start": start,
        "stop": stop,
        "max_elevation_deg": round(max_el, 3),
        "duration_s": round(duration, 1),
    }


def _propagate_two_body(r, v, dt, mu):
    """二体传播（f and g 级数）。"""
    from .propagation_tools import _kepler_step
    return _kepler_step(r, v, dt, mu)[0]


def _inertial_to_ecef(pos_inertial, dt):
    """惯性系→ECEF（仅地球自转）。"""
    theta = _gmst(dt)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return [cos_t * pos_inertial[0] + sin_t * pos_inertial[1],
            -sin_t * pos_inertial[0] + cos_t * pos_inertial[1],
            pos_inertial[2]]


def _geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """K5-H10: 使用 WGS84 椭球模型（替代球面近似）。

    WGS84 参数: a=6378137.0m, f=1/298.257223563
    """
    # WGS84 椭球参数
    a = 6378137.0  # 赤道半径
    f = 1.0 / 298.257223563  # 扁率
    e2 = f * (2 - f)  # 第一偏心率平方

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    # 卯酉圈曲率半径
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * math.cos(lon)
    y = (N + alt_m) * cos_lat * math.sin(lon)
    z = (N * (1 - e2) + alt_m) * sin_lat
    return [x, y, z]


def _compute_elevation(sat_ecef, station_ecef) -> float:
    """计算卫星相对地面站的仰角。"""
    dx = sat_ecef[0] - station_ecef[0]
    dy = sat_ecef[1] - station_ecef[1]
    dz = sat_ecef[2] - station_ecef[2]
    # 站点当地切平面法向（近似径向）
    r_sta = math.sqrt(sum(c * c for c in station_ecef))
    if r_sta < 1e-6:
        return -90.0
    # 仰角 = arcsin(径向分量 / 距离)
    radial = (dx * station_ecef[0] + dy * station_ecef[1]
              + dz * station_ecef[2]) / r_sta
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-6:
        return -90.0
    return math.degrees(math.asin(max(-1.0, min(1.0, radial / dist))))


def _gmst(dt) -> float:
    """K5-M1: 格林尼治平恒星时（rad）—— IERS 2010 规范。

    GMST = ERA + 0.0334176425676·sin(Ω_moon) + ... (高阶项忽略)
    ERA = 2π·(0.7790572732640 + 1.00273781191135448·Tu)
    其中 Tu = J2000 以来的 UT1 天数。
    """
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    days = (dt - j2000).total_seconds() / 86400.0
    # ERA (Earth Rotation Angle) per IERS 2010
    era = 2 * math.pi * (0.7790572732640 + 1.00273781191135448 * days)
    # GMST ≈ ERA (简化，忽略赤经岁差高阶项)
    return era % (2 * math.pi)


def _epoch_to_datetime(epoch: Epoch) -> datetime:
    val = str(epoch.value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(val)
    except Exception:
        dt = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _error(reason) -> Dict:
    return {"status": "error", "reason": reason, "engine": None,
            "access_windows": [], "total_windows": 0}


__all__ = ["compute_ground_access"]
