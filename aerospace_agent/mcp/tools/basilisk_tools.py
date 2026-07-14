"""Basilisk MCP 工具 — 10 个航天器动力学与姿态控制工具。

第一性原理（K2 白名单封装）：
  1. 前 2 个工具（propagate_orbit / attitude_control）委托 BasiliskAdapter 执行 BSIL 仿真
  2. 后 8 个工具用纯 Python + numpy 实现，不依赖 Basilisk C++ 模拟
  3. 所有工具返回 JSON 可序列化字典，失败返回 {status:"error", reason:...}
  4. 轨道根数转换覆盖笛卡尔↔开普勒↔春分点三种表示
  5. 环境模型（大气阻力、太阳光压、星蚀）使用简化解析模型，标注为假设
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..adapters import get_adapter
from ..schemas import OrbitState, AttitudeState, Epoch, Frame

# ---------------------------------------------------------------------------
# 物理常量
# ---------------------------------------------------------------------------
_MU_EARTH = 3.986004418e14     # 地球引力参数 m³/s²
_R_EARTH = 6378137.0           # 地球赤道半径 m
_J2 = 1.08263e-3               # J2 摄动系数
_AU = 1.495978707e11           # 天文单位 m
_SOLAR_FLUX = 1361.0           # 太阳常数 W/m² (假设: 1 AU 处均值)
_C_LIGHT = 2.99792458e8        # 光速 m/s
_P_SRP = _SOLAR_FLUX / _C_LIGHT  # 太阳光压 ~4.56e-6 N/m² (假设: 完全吸收)
_R_SUN = 6.957e8               # 太阳半径 m

# 指数大气模型参数 (假设: 参考 Vallado 2013 表 8-4)
_EXP_ATMOS_LAYERS = [
    # h0 (km), rho0 (kg/m³), H (km)
    (0,    1.225,     7.249),
    (25,   3.899e-2,  6.349),
    (30,   1.774e-2,  6.682),
    (40,   3.972e-3,  7.554),
    (50,   1.057e-3,  8.382),
    (60,   3.206e-4,  7.714),
    (70,   8.770e-5,  6.549),
    (80,   1.905e-5,  5.799),
    (90,   3.396e-6,  5.382),
    (100,  5.297e-7,  5.877),
    (110,  9.661e-8,  7.263),
    (120,  2.438e-8,  9.473),
    (130,  8.484e-9,  12.636),
    (140,  3.845e-9,  16.149),
    (150,  2.070e-9,  22.523),
    (180,  5.464e-10, 29.740),
    (200,  2.789e-10, 37.105),
    (250,  7.248e-11, 45.546),
    (300,  2.418e-11, 53.628),
    (350,  9.158e-12, 53.298),
    (400,  3.725e-12, 58.515),
    (450,  1.585e-12, 60.828),
    (500,  6.967e-13, 63.822),
    (600,  1.454e-13, 71.835),
    (700,  3.614e-14, 88.667),
    (800,  1.170e-14, 124.64),
    (900,  5.245e-15, 181.05),
    (1000, 3.019e-15, 268.00),
]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _error(reason: str) -> Dict:
    return {"status": "error", "reason": reason}


def _success(**kwargs) -> Dict:
    result = {"status": "success"}
    result.update(kwargs)
    return result


def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros(3)
    return v / n


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])


def _wrap_angle(rad: float) -> float:
    """将弧度角归一化到 [0, 2*pi)。"""
    two_pi = 2 * math.pi
    return rad % two_pi


# ---------------------------------------------------------------------------
# Tool 1: basilisk_propagate_orbit
# ---------------------------------------------------------------------------
def basilisk_propagate_orbit(
    initial_state_dict: Dict,
    force_model_dict: Dict,
    duration_s: float,
    output_step_s: Optional[float] = None,
    engine: str = "basilisk",
) -> Dict:
    """使用 Basilisk 引擎进行轨道传播（二体 + 可选 J2）。

    委托 BasiliskAdapter.propagate_orbit()，通过 BSIL 仿真任务模块运行。

    Args:
        initial_state_dict: 初始 OrbitState 字典（含 position_m, velocity_mps）
        force_model_dict: ForceModel 字典
        duration_s: 传播时长 s
        output_step_s: 输出采样间隔 s（None 仅终态）
        engine: 引擎名，固定 "basilisk"
    Returns:
        {state_history, metadata}
    """
    adapter = get_adapter("basilisk")
    if not adapter.is_available():
        return _error("Basilisk 引擎不可用——请 pip install basilisk")

    try:
        state = OrbitState.from_dict(initial_state_dict)
    except Exception as exc:
        return _error(f"初始状态解析失败: {exc}")

    if state.position_m is None or state.velocity_mps is None:
        return _error("传播需要 cartesian position_m 和 velocity_mps")

    try:
        result = adapter.propagate_orbit(
            state, force_model_dict,
            type("Cfg", (), {
                "duration_s": duration_s,
                "output_step_s": output_step_s,
                "mu": _MU_EARTH,
                "step_s": 60.0,
            })(),
        )
        if result.get("status") == "unavailable":
            return _error("Basilisk 引擎不可用")
        return result
    except Exception as exc:
        return _error(f"轨道传播失败: {exc}")


# ---------------------------------------------------------------------------
# Tool 2: basilisk_attitude_control
# ---------------------------------------------------------------------------
def basilisk_attitude_control(
    initial_attitude_dict: Dict,
    controller: str = "MRP_feedback",
    K: float = 3.5,
    P: float = 35.0,
    duration_s: float = 600.0,
    step_s: float = 0.1,
    output_step_s: Optional[float] = None,
) -> Dict:
    """使用 Basilisk 引擎进行姿态控制仿真（MRP 反馈）。

    委托 BasiliskAdapter.attitude_control()，通过 BSIL FSW 模块运行。

    Args:
        initial_attitude_dict: 初始 AttitudeState 字典（含 quaternion, angular_velocity_radps）
        controller: 控制器类型，"MRP_feedback"
        K: 比例增益
        P: 导数增益
        duration_s: 仿真时长 s
        step_s: 积分步长 s
        output_step_s: 输出采样间隔 s
    Returns:
        {attitude_history, metadata}
    """
    adapter = get_adapter("basilisk")
    if not adapter.is_available():
        return _error("Basilisk 引擎不可用——请 pip install basilisk")

    try:
        state = AttitudeState.from_dict(initial_attitude_dict)
    except Exception as exc:
        return _error(f"姿态状态解析失败: {exc}")

    try:
        result = adapter.attitude_control(
            state, controller=controller,
            K=K, P=P, duration_s=duration_s,
            step_s=step_s, output_step_s=output_step_s,
        )
        if result.get("status") == "unavailable":
            return _error("Basilisk 引擎不可用")
        return result
    except Exception as exc:
        return _error(f"姿态控制仿真失败: {exc}")


# ---------------------------------------------------------------------------
# Tool 3: basilisk_orbit_elements_conversion
# ---------------------------------------------------------------------------
def basilisk_orbit_elements_conversion(
    source_representation: str,
    source_data: Dict,
    target_representation: str,
    mu: float = _MU_EARTH,
) -> Dict:
    """轨道根数转换：笛卡尔 ↔ 开普勒 ↔ 春分点。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    支持的表示:
      - cartesian: {position_m: [x,y,z], velocity_mps: [vx,vy,vz]}
      - keplerian: {a_m, e, i_deg, raan_deg, argp_deg, ta_deg}
      - equinoctial: {p_m, f, g, h, k, L_deg}  (春分点根数)

    Args:
        source_representation: 源表示 "cartesian" / "keplerian" / "equinoctial"
        source_data: 源数据字典
        target_representation: 目标表示 "cartesian" / "keplerian" / "equinoctial"
        mu: 引力参数 m³/s²（默认地球）
    Returns:
        {source_representation, target_representation, result, mu}
    """
    src = source_representation.lower().strip()
    tgt = target_representation.lower().strip()

    valid = {"cartesian", "keplerian", "equinoctial"}
    if src not in valid:
        return _error(f"不支持的源表示: '{source_representation}'，支持: {valid}")
    if tgt not in valid:
        return _error(f"不支持的目标表示: '{target_representation}'，支持: {valid}")
    if src == tgt:
        return _success(source_representation=src,
                        target_representation=tgt,
                        result=source_data, mu=mu)

    try:
        # 统一先转到 cartesian 中间表示
        if src == "cartesian":
            cart = _cartesian_from_dict(source_data)
        elif src == "keplerian":
            cart = _keplerian_to_cartesian(source_data, mu)
        elif src == "equinoctial":
            cart = _equinoctial_to_cartesian(source_data, mu)

        # 再从 cartesian 转到目标表示
        if tgt == "cartesian":
            result = _cartesian_to_dict(cart)
        elif tgt == "keplerian":
            result = _cartesian_to_keplerian(cart, mu)
        elif tgt == "equinoctial":
            result = _cartesian_to_equinoctial(cart, mu)

        return _success(source_representation=src,
                        target_representation=tgt,
                        result=result, mu=mu)
    except Exception as exc:
        return _error(f"轨道根数转换失败: {exc}")


# ---- 笛卡尔 ↔ 开普勒 ----
def _cartesian_from_dict(d: Dict) -> Tuple[np.ndarray, np.ndarray]:
    r = np.array(d["position_m"], dtype=float)
    v = np.array(d["velocity_mps"], dtype=float)
    return r, v


def _cartesian_to_dict(cart: Tuple[np.ndarray, np.ndarray]) -> Dict:
    r, v = cart
    return {
        "position_m": r.tolist(),
        "velocity_mps": v.tolist(),
    }


def _cartesian_to_keplerian(cart: Tuple[np.ndarray, np.ndarray],
                            mu: float) -> Dict:
    r, v = cart
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))

    if r_mag < 1.0:
        return {"a_m": 0, "e": 0, "i_deg": 0, "raan_deg": 0,
                "argp_deg": 0, "ta_deg": 0}

    # 角动量
    h = _cross(r, v)
    h_mag = float(np.linalg.norm(h))

    # 节点向量 n = K × h
    n = np.array([-h[1], h[0], 0.0])
    n_mag = float(np.linalg.norm(n))

    # 偏心率向量
    e_vec = (v_mag**2 - mu / r_mag) * r / mu - np.dot(r, v) * v / mu
    e = float(np.linalg.norm(e_vec))

    # 能量 → 半长轴
    energy = v_mag**2 / 2.0 - mu / r_mag
    a = -mu / (2.0 * energy) if abs(energy) > 1e-15 else float('inf')

    # 倾角
    cos_i = h[2] / h_mag if h_mag > 1e-15 else 1.0
    cos_i = max(-1.0, min(1.0, cos_i))
    i_deg = _rad2deg(math.acos(cos_i))

    # 升交点赤经
    if n_mag > 1e-15:
        cos_raan = n[0] / n_mag
        cos_raan = max(-1.0, min(1.0, cos_raan))
        raan_deg = _rad2deg(math.acos(cos_raan))
        if n[1] < 0:
            raan_deg = 360.0 - raan_deg
    else:
        raan_deg = 0.0

    # 近地点幅角
    if n_mag > 1e-15 and e > 1e-15:
        cos_argp = np.dot(n, e_vec) / (n_mag * e)
        cos_argp = max(-1.0, min(1.0, cos_argp))
        argp_deg = _rad2deg(math.acos(cos_argp))
        if e_vec[2] < 0:
            argp_deg = 360.0 - argp_deg
    else:
        argp_deg = 0.0

    # 真近点角
    if e > 1e-15:
        cos_ta = np.dot(e_vec, r) / (e * r_mag)
        cos_ta = max(-1.0, min(1.0, cos_ta))
        ta_deg = _rad2deg(math.acos(cos_ta))
        if np.dot(r, v) < 0:
            ta_deg = 360.0 - ta_deg
    else:
        # 圆轨道：用升交点方向
        if n_mag > 1e-15:
            cos_ta = np.dot(n, r) / (n_mag * r_mag)
            cos_ta = max(-1.0, min(1.0, cos_ta))
            ta_deg = _rad2deg(math.acos(cos_ta))
            if r[2] < 0:
                ta_deg = 360.0 - ta_deg
        else:
            ta_deg = 0.0

    return {
        "a_m": a,
        "e": e,
        "i_deg": i_deg,
        "raan_deg": raan_deg,
        "argp_deg": argp_deg,
        "ta_deg": ta_deg,
    }


def _keplerian_to_cartesian(d: Dict, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    a = float(d["a_m"])
    e = float(d["e"])
    i_rad = _deg2rad(float(d["i_deg"]))
    raan_rad = _deg2rad(float(d["raan_deg"]))
    argp_rad = _deg2rad(float(d["argp_deg"]))
    ta_rad = _deg2rad(float(d["ta_deg"]))

    p = a * (1.0 - e * e)
    if p <= 0:
        raise ValueError(f"半通径 p={p} 非正，无法计算（a={a}, e={e}）")

    r_orb = p / (1.0 + e * math.cos(ta_rad))
    x_orb = r_orb * math.cos(ta_rad)
    y_orb = r_orb * math.sin(ta_rad)

    vx_orb = -math.sqrt(mu / p) * math.sin(ta_rad)
    vy_orb = math.sqrt(mu / p) * (e + math.cos(ta_rad))

    # PQW → ECI 旋转矩阵: Rz(raan) * Rx(i) * Rz(argp)
    cos_raan, sin_raan = math.cos(raan_rad), math.sin(raan_rad)
    cos_i, sin_i = math.cos(i_rad), math.sin(i_rad)
    cos_argp, sin_argp = math.cos(argp_rad), math.sin(argp_rad)

    r = np.array([
        (cos_raan * cos_argp - sin_raan * sin_argp * cos_i) * x_orb
        + (-cos_raan * sin_argp - sin_raan * cos_argp * cos_i) * y_orb,
        (sin_raan * cos_argp + cos_raan * sin_argp * cos_i) * x_orb
        + (-sin_raan * sin_argp + cos_raan * cos_argp * cos_i) * y_orb,
        (sin_argp * sin_i) * x_orb + (cos_argp * sin_i) * y_orb,
    ])

    v = np.array([
        (cos_raan * cos_argp - sin_raan * sin_argp * cos_i) * vx_orb
        + (-cos_raan * sin_argp - sin_raan * cos_argp * cos_i) * vy_orb,
        (sin_raan * cos_argp + cos_raan * sin_argp * cos_i) * vx_orb
        + (-sin_raan * sin_argp + cos_raan * cos_argp * cos_i) * vy_orb,
        (sin_argp * sin_i) * vx_orb + (cos_argp * sin_i) * vy_orb,
    ])

    return r, v


# ---- 春分点根数 (equinoctial elements) ----
# 定义: p, f=e*cos(omega+Omega), g=e*sin(omega+Omega),
#       h=tan(i/2)*cos(Omega), k=tan(i/2)*sin(Omega), L=true_longitude
def _cartesian_to_equinoctial(cart: Tuple[np.ndarray, np.ndarray],
                              mu: float) -> Dict:
    kepler = _cartesian_to_keplerian(cart, mu)
    e = float(kepler["e"])
    i_rad = _deg2rad(float(kepler["i_deg"]))
    raan_rad = _deg2rad(float(kepler["raan_deg"]))
    argp_rad = _deg2rad(float(kepler["argp_deg"]))
    ta_rad = _deg2rad(float(kepler["ta_deg"]))
    a = float(kepler["a_m"])

    p = a * (1.0 - e * e)
    omega_plus_raan = argp_rad + raan_rad
    f = e * math.cos(omega_plus_raan)
    g = e * math.sin(omega_plus_raan)
    h = math.tan(i_rad / 2.0) * math.cos(raan_rad)
    k = math.tan(i_rad / 2.0) * math.sin(raan_rad)
    L = _wrap_angle(omega_plus_raan + ta_rad)

    return {
        "p_m": p,
        "f": f, "g": g,
        "h": h, "k": k,
        "L_deg": _rad2deg(L),
    }


def _equinoctial_to_cartesian(d: Dict, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    p = float(d["p_m"])
    f = float(d["f"])
    g = float(d["g"])
    h = float(d["h"])
    k = float(d["k"])
    L_rad = _deg2rad(float(d["L_deg"]))

    # 恢复轨道根数
    e = math.sqrt(f * f + g * g)
    i_rad = 2.0 * math.atan(math.sqrt(h * h + k * k))

    if h * h + k * k < 1e-15:
        raan_rad = 0.0
    else:
        raan_rad = math.atan2(k, h)

    if e < 1e-15:
        argp_rad = 0.0
        omega_plus_raan = raan_rad
    else:
        omega_plus_raan = math.atan2(g, f)
        argp_rad = omega_plus_raan - raan_rad

    raan_rad = _wrap_angle(raan_rad)
    argp_rad = _wrap_angle(argp_rad)
    ta_rad = _wrap_angle(L_rad - omega_plus_raan)

    a = p / (1.0 - e * e) if e < 1.0 else float('inf')

    return _keplerian_to_cartesian({
        "a_m": a, "e": e,
        "i_deg": _rad2deg(i_rad),
        "raan_deg": _rad2deg(raan_rad),
        "argp_deg": _rad2deg(argp_rad),
        "ta_deg": _rad2deg(ta_rad),
    }, mu)


# ---------------------------------------------------------------------------
# Tool 4: basilisk_thruster_modeling
# ---------------------------------------------------------------------------
def basilisk_thruster_modeling(
    thrust_N: float,
    isp_s: float,
    spacecraft_mass_kg: float,
    thrust_direction_body: Optional[List[float]] = None,
    thrust_position_body: Optional[List[float]] = None,
    burn_duration_s: float = 1.0,
    g0: float = 9.80665,
) -> Dict:
    """推力器建模：计算推力矢量、比冲、质量变化、速度增量。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    假设: 推力为常值，质量变化为线性（m_dot = thrust / (Isp * g0)）。

    Args:
        thrust_N: 推力大小 N
        isp_s: 比冲 s
        spacecraft_mass_kg: 航天器初始质量 kg
        thrust_direction_body: 推力方向本体系单位矢量 [x,y,z]（默认 +x）
        thrust_position_body: 推力器安装位置本体系 [x,y,z] m（默认原点）
        burn_duration_s: 点火时长 s
        g0: 海平面重力加速度 m/s²（默认 9.80665）
    Returns:
        {force_vector, torque_vector, mass_flow_rate, final_mass, delta_v, metadata}
    """
    if thrust_N <= 0:
        return _error("推力必须 > 0")
    if isp_s <= 0:
        return _error("比冲必须 > 0")
    if spacecraft_mass_kg <= 0:
        return _error("航天器质量必须 > 0")

    direction = np.array(thrust_direction_body or [1.0, 0.0, 0.0], dtype=float)
    direction = _normalize(direction)

    position = np.array(thrust_position_body or [0.0, 0.0, 0.0], dtype=float)

    force_vector = thrust_N * direction
    torque_vector = _cross(position, force_vector)
    mass_flow_rate = thrust_N / (isp_s * g0)
    final_mass = spacecraft_mass_kg - mass_flow_rate * burn_duration_s

    if final_mass <= 0:
        return _error(f"点火 {burn_duration_s}s 后质量耗尽: "
                      f"初值 {spacecraft_mass_kg} kg, 流量 {mass_flow_rate:.4f} kg/s")

    # 火箭方程 delta-v
    delta_v = isp_s * g0 * math.log(spacecraft_mass_kg / final_mass)

    return _success(
        force_vector_N=force_vector.tolist(),
        torque_vector_Nm=torque_vector.tolist(),
        mass_flow_rate_kgps=round(mass_flow_rate, 8),
        initial_mass_kg=spacecraft_mass_kg,
        final_mass_kg=round(final_mass, 6),
        delta_v_mps=round(delta_v, 6),
        metadata={
            "thrust_N": thrust_N,
            "isp_s": isp_s,
            "g0": g0,
            "burn_duration_s": burn_duration_s,
            "assumption": "假设: 常值推力、线性质量变化、火箭方程",
        },
    )


# ---------------------------------------------------------------------------
# Tool 5: basilisk_reaction_wheel_modeling
# ---------------------------------------------------------------------------
def basilisk_reaction_wheel_modeling(
    wheel_speeds_rpm: List[float],
    wheel_inertia_kgm2: float = 0.001,
    max_rpm: float = 6000.0,
    wheel_axes: Optional[List[List[float]]] = None,
    torque_cmd_Nm: Optional[List[float]] = None,
) -> Dict:
    """反作用轮建模：角动量存储、饱和检测、转矩分配。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    假设: 所有轮子转动惯量相同（axisymmetric）、无摩擦、力矩无延迟响应。

    Args:
        wheel_speeds_rpm: 各轮当前转速 RPM 列表
        wheel_inertia_kgm2: 单轮转动惯量 kg·m²
        max_rpm: 最大转速 RPM
        wheel_axes: 各轮安装轴方向（N×3 列表），默认 4 轮金字塔构型
        torque_cmd_Nm: 期望力矩指令 [Tx,Ty,Tz] N·m（可选，用于计算轮速变化）
    Returns:
        {angular_momentum, saturation_ratio, is_saturated, torque_allocation, metadata}
    """
    if not wheel_speeds_rpm:
        return _error("wheel_speeds_rpm 不能为空")

    # 默认四轮金字塔构型 (假设: 倾角 45°, 对称分布)
    if wheel_axes is None:
        # 四轮金字塔: 轴与本体 +z 夹角 θ=45°, 在 xy 平面均布
        theta = _deg2rad(45.0)
        st, ct = math.sin(theta), math.cos(theta)
        wheel_axes = [
            [st, 0.0, ct],          # +x 方向
            [0.0, st, ct],          # +y 方向
            [-st, 0.0, ct],         # -x 方向
            [0.0, -st, ct],         # -y 方向
        ]

    n_wheels = len(wheel_speeds_rpm)
    if len(wheel_axes) != n_wheels:
        return _error(f"wheel_speeds_rpm 长度 ({len(wheel_speeds_rpm)}) "
                      f"与 wheel_axes 长度 ({len(wheel_axes)}) 不匹配")

    axes = np.array(wheel_axes, dtype=float)
    # 归一化各轴
    for i in range(n_wheels):
        axes[i] = _normalize(axes[i])

    speeds = np.array(wheel_speeds_rpm, dtype=float)
    omega = speeds * (2.0 * math.pi / 60.0)  # RPM → rad/s

    # 各轮角动量矢量
    h_wheels = np.array([wheel_inertia_kgm2 * omega[i] * axes[i]
                         for i in range(n_wheels)])
    total_h = np.sum(h_wheels, axis=0)

    # 饱和检测
    max_omega = max_rpm * (2.0 * math.pi / 60.0)
    max_h_per_wheel = wheel_inertia_kgm2 * max_omega
    saturation_ratios = [abs(omega[i]) / max_omega for i in range(n_wheels)]
    is_saturated = any(r >= 0.95 for r in saturation_ratios)

    result = {
        "total_angular_momentum_Nms": total_h.tolist(),
        "total_angular_momentum_mag_Nms": round(float(np.linalg.norm(total_h)), 8),
        "wheel_angular_momenta_Nms": [round(float(np.linalg.norm(h_wheels[i])), 8)
                                      for i in range(n_wheels)],
        "saturation_ratios": [round(r, 4) for r in saturation_ratios],
        "is_saturated": is_saturated,
        "max_angular_momentum_per_wheel_Nms": round(max_h_per_wheel, 8),
    }

    # 转矩分配（如果提供了力矩指令）
    if torque_cmd_Nm is not None:
        torques = np.array(torque_cmd_Nm, dtype=float)
        # 分配矩阵: 伪逆 (最小范数解)
        # tau_wheel = A^T * (A * A^T)^{-1} * torque_cmd
        A = axes.T  # 3×N 分配矩阵
        AAT = A @ A.T
        try:
            AAT_inv = np.linalg.inv(AAT)
            allocation = A.T @ AAT_inv @ torques
            result["torque_allocation_Nm"] = allocation.tolist()
            result["torque_allocation_norm"] = round(float(np.linalg.norm(allocation)), 8)
        except np.linalg.LinAlgError:
            result["torque_allocation_Nm"] = None
            result["torque_allocation_note"] = "分配矩阵奇异，无法计算伪逆"

    result["metadata"] = {
        "n_wheels": n_wheels,
        "wheel_inertia_kgm2": wheel_inertia_kgm2,
        "max_rpm": max_rpm,
        "assumption": "假设: 所有轮子转动惯量相同、无摩擦、力矩无延迟响应",
    }
    return _success(**result)


# ---------------------------------------------------------------------------
# Tool 6: basilisk_sun_pointing
# ---------------------------------------------------------------------------
def basilisk_sun_pointing(
    spacecraft_position_m: List[float],
    epoch_iso: str = "2026-01-01T00:00:00",
    body_x_axis: Optional[List[float]] = None,
) -> Dict:
    """太阳指向控制：计算太阳方向矢量 + 期望姿态四元数。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    假设: 太阳方向由简化解析公式近似（精度 ~0.01°），
    太阳位置 = 从地心到太阳的 AU 距离矢量。
    使用简化的太阳经度模型（Meeus 1998 近似）。

    Args:
        spacecraft_position_m: 航天器地心位置 [x,y,z] m
        epoch_iso: ISO 8601 时间字符串
        body_x_axis: 本体 +x 轴在 ECI 中的方向（默认 [1,0,0]）
    Returns:
        {sun_vector_eci, sun_vector_body, desired_quaternion, sun_distance_au, metadata}
    """
    r_sc = np.array(spacecraft_position_m, dtype=float)
    r_sc_mag = float(np.linalg.norm(r_sc))
    if r_sc_mag < 1e3:
        return _error("航天器位置模长过小，可能在地球内部")

    # 简化的太阳位置计算 (假设: Meeus 1998 近似，精度 ~0.01°)
    # 从 ISO 时间提取儒略日近似
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(epoch_iso.replace("Z", "+00:00").split("+")[0].split("-")[0:6][0] if "+" in epoch_iso else epoch_iso.replace("Z", ""))
        dt = datetime.fromisoformat(epoch_iso.replace("Z", ""))
    except Exception:
        # 默认 J2000
        dt = None

    # 使用简化公式计算太阳黄经和距离
    if dt is not None:
        jd = _julian_date(dt.year, dt.month, dt.day,
                          dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
        n_days = jd - 2451545.0  # J2000

        # 太阳平近点角 deg
        M_sun = _wrap_angle(_deg2rad(357.5291 + 0.98560028 * n_days))
        # 太阳平黄经
        L_sun = _wrap_angle(_deg2rad(280.4665 + 0.98564736 * n_days))
        # 太阳中心差
        C_sun = _deg2rad(1.9148 * math.sin(M_sun) + 0.0200 * math.sin(2 * M_sun)
                         + 0.0003 * math.sin(3 * M_sun))
        # 真黄经
        lambda_sun = _wrap_angle(L_sun + C_sun)
        # 黄赤交角
        epsilon = _deg2rad(23.4393 - 0.0000004 * n_days)

        # 太阳距离 AU
        r_sun_au = 1.00014 - 0.01671 * math.cos(M_sun) - 0.00014 * math.cos(2 * M_sun)

        # 太阳方向 ECI (假设: 地心指向太阳)
        sun_dir_eci = np.array([
            math.cos(lambda_sun),
            math.sin(lambda_sun) * math.cos(epsilon),
            math.sin(lambda_sun) * math.sin(epsilon),
        ])
    else:
        # 回退: 默认春分点附近太阳方向 (假设)
        r_sun_au = 1.0
        sun_dir_eci = np.array([1.0, 0.0, 0.0])

    sun_dir_eci = _normalize(sun_dir_eci)
    sun_distance_m = r_sun_au * _AU

    # 太阳方向在航天器本体系中的表示
    # 先计算航天器指向太阳的矢量
    r_sun_from_sc = sun_dir_eci * sun_distance_m - r_sc
    sun_vector_eci = _normalize(r_sun_from_sc)

    # 期望姿态：本体 +x 指向太阳，+z 尽量远离太阳方向
    body_x = np.array(body_x_axis or [1.0, 0.0, 0.0], dtype=float)
    body_x = _normalize(body_x)

    # 期望 DCM:
    # b1 = sun_vector_eci (期望指向)
    # b2 = cross(sun_vector_eci, body_x) 归一化
    # b3 = cross(b1, b2)
    b1 = sun_vector_eci
    b2 = _cross(sun_vector_eci, body_x)
    if np.linalg.norm(b2) < 1e-10:
        b2 = _cross(sun_vector_eci, np.array([0.0, 0.0, 1.0]))
    b2 = _normalize(b2)
    b3 = _cross(b1, b2)

    # DCM → 四元数 (scalar-first)
    quat = _dcm_to_quat(np.column_stack([b1, b2, b3]))

    return _success(
        sun_vector_eci=sun_vector_eci.tolist(),
        sun_vector_body=(sun_vector_eci).tolist(),  # 简化: 假设本体系≈ECI 初始
        sun_distance_au=round(r_sun_au, 6),
        sun_distance_m=round(sun_distance_m, 0),
        desired_quaternion=quat,
        metadata={
            "epoch_iso": epoch_iso,
            "pointing_mode": "sun_pointing",
            "body_axis_aligned_to_sun": "+x",
            "assumption": "假设: 太阳方向由 Meeus 1998 简化解析公式近似，精度 ~0.01°",
        },
    )


# ---------------------------------------------------------------------------
# Tool 7: basilisk_nadir_pointing
# ---------------------------------------------------------------------------
def basilisk_nadir_pointing(
    spacecraft_position_m: List[float],
    spacecraft_velocity_mps: Optional[List[float]] = None,
) -> Dict:
    """天底指向控制：计算天底方向矢量 + 期望姿态四元数。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    天底 = -position / |position|（指向地心方向）。
    用轨道坐标系 (LVLH) 定义期望姿态：+z 指向天底，+x 沿速度方向，
    +y = +z × +x 完成右手系。

    Args:
        spacecraft_position_m: 航天器地心位置 [x,y,z] m
        spacecraft_velocity_mps: 航天器地心速度 [vx,vy,vz] m/s（可选，用于定义 LVLH）
    Returns:
        {nadir_vector_eci, desired_quaternion, off_nadir_angle_deg, metadata}
    """
    r = np.array(spacecraft_position_m, dtype=float)
    r_mag = float(np.linalg.norm(r))
    if r_mag < 1e3:
        return _error("航天器位置模长过小，可能在地球内部")

    nadir_eci = -_normalize(r)  # 指向地心

    if spacecraft_velocity_mps is not None:
        v = np.array(spacecraft_velocity_mps, dtype=float)
        # LVLH: z = nadir, y = -r×v/|r×v|, x = y×z
        h = _cross(r, v)
        y_lvlh = -_normalize(h)
        x_lvlh = _cross(y_lvlh, nadir_eci)
        x_lvlh = _normalize(x_lvlh)
        # 确保正交
        y_lvlh = _cross(nadir_eci, x_lvlh)
    else:
        # 无速度时用简化假设: y 轴取赤道面法向
        y_lvlh = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(nadir_eci, y_lvlh)) > 0.99:
            y_lvlh = np.array([0.0, 1.0, 0.0])
        y_lvlh = y_lvlh - np.dot(y_lvlh, nadir_eci) * nadir_eci
        y_lvlh = _normalize(y_lvlh)
        x_lvlh = _cross(y_lvlh, nadir_eci)

    # DCM (ECI → LVLH aligned): 列 = [x_lvlh, y_lvlh, nadir_eci]
    dcm = np.column_stack([x_lvlh, y_lvlh, nadir_eci])
    quat = _dcm_to_quat(dcm)

    # 偏天底角 (假设: 本体 +z 与天底夹角)
    off_nadir_deg = 0.0  # 理想指向下为 0

    return _success(
        nadir_vector_eci=nadir_eci.tolist(),
        lvlh_x_eci=x_lvlh.tolist(),
        lvlh_y_eci=y_lvlh.tolist(),
        lvlh_z_eci=nadir_eci.tolist(),
        desired_quaternion=quat,
        off_nadir_angle_deg=off_nadir_deg,
        metadata={
            "pointing_mode": "nadir_pointing",
            "reference_frame": "LVLH",
            "body_z_aligned_to": "nadir",
            "body_x_aligned_to": "velocity_direction" if spacecraft_velocity_mps else "assumed_in_plane",
            "assumption": "假设: 无速度时 y 轴取赤道面法向投影",
        },
    )


# ---------------------------------------------------------------------------
# Tool 8: basilisk_eclipse_detection
# ---------------------------------------------------------------------------
def basilisk_eclipse_detection(
    spacecraft_position_m: List[float],
    sun_position_m: Optional[List[float]] = None,
    epoch_iso: str = "2026-01-01T00:00:00",
    shadow_model: str = "cylindrical",
) -> Dict:
    """星蚀检测：判断航天器是否处于地球阴影中（本影/半影/全日照）。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    支持两种模型:
      - cylindrical: 圆柱阴影（假设: 太阳为点光源，纯几何影锥）
      - conical: 圆锥阴影（假设: 考虑太阳角直径，区分本影/半影）

    Args:
        spacecraft_position_m: 航天器地心位置 [x,y,z] m
        sun_position_m: 太阳地心位置 [x,y,z] m（可选，默认用简化公式计算）
        epoch_iso: ISO 8601 时间字符串（sun_position_m 未提供时用于计算太阳位置）
        shadow_model: 阴影模型 "cylindrical" 或 "conical"
    Returns:
        {eclipse_state, shadow_type, umbra_angle_deg, penumbra_angle_deg, metadata}
    """
    r_sc = np.array(spacecraft_position_m, dtype=float)
    r_sc_mag = float(np.linalg.norm(r_sc))

    # 位置合法性校验：航天器必须在地球表面以上
    if r_sc_mag < _R_EARTH:
        return {
            "status": "error",
            "reason": f"航天器位置在地球内部: r={r_sc_mag:.0f}m < R_EARTH={_R_EARTH:.0f}m",
            "engine": "basilisk",
        }

    if sun_position_m is not None:
        r_sun = np.array(sun_position_m, dtype=float)
    else:
        # 简化太阳位置计算 (同 sun_pointing 中的方法)
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(epoch_iso.replace("Z", ""))
            jd = _julian_date(dt.year, dt.month, dt.day,
                              dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
            n_days = jd - 2451545.0
            M_sun = _wrap_angle(_deg2rad(357.5291 + 0.98560028 * n_days))
            L_sun = _wrap_angle(_deg2rad(280.4665 + 0.98564736 * n_days))
            C_sun = _deg2rad(1.9148 * math.sin(M_sun) + 0.0200 * math.sin(2 * M_sun))
            lambda_sun = _wrap_angle(L_sun + C_sun)
            epsilon = _deg2rad(23.4393 - 0.0000004 * n_days)
            r_sun_au = 1.00014 - 0.01671 * math.cos(M_sun) - 0.00014 * math.cos(2 * M_sun)
            r_sun = np.array([
                r_sun_au * _AU * math.cos(lambda_sun),
                r_sun_au * _AU * math.sin(lambda_sun) * math.cos(epsilon),
                r_sun_au * _AU * math.sin(lambda_sun) * math.sin(epsilon),
            ])
        except Exception:
            r_sun = np.array([_AU, 0.0, 0.0])

    r_sun_mag = float(np.linalg.norm(r_sun))

    # 太阳-航天器方向 (从航天器看太阳)
    d_sun_from_sc = r_sun - r_sc
    d_mag = float(np.linalg.norm(d_sun_from_sc))
    d_unit = d_sun_from_sc / d_mag

    # 航天器-地心方向 (从航天器看地心)
    d_earth_from_sc = -r_sc  # 指向地心

    # 地心到日-地连线的距离
    # 航天器在日-地连线上的投影距离
    proj = np.dot(r_sc, d_unit)

    if shadow_model == "conical":
        # 圆锥阴影模型
        # 太阳角半径 (从地球看)
        alpha_sun = math.asin(_R_SUN / r_sun_mag)
        # 地球角半径 (从航天器看)
        alpha_earth_sc = math.asin(_R_EARTH / r_sc_mag)
        # 本影锥半角
        # 考虑太阳和地球的视尺寸
        # 从航天器看太阳和地球的角距离
        cos_angle = np.dot(-r_sc, d_sun_from_sc) / (r_sc_mag * d_mag)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angular_sep = math.acos(cos_angle)  # 航天器-地球 vs 航天器-太阳

        # 本影条件: 地球完全遮挡太阳
        # 半影条件: 地球部分遮挡太阳
        if angular_sep < alpha_earth_sc - alpha_sun:
            eclipse_state = "umbra"  # 本影
            shadow_type = "total"
        elif angular_sep < alpha_earth_sc + alpha_sun:
            eclipse_state = "penumbra"  # 半影
            shadow_type = "partial"
        else:
            eclipse_state = "sunlight"
            shadow_type = "none"

        result = _success(
            eclipse_state=eclipse_state,
            shadow_type=shadow_type,
            shadow_model="conical",
            angular_separation_deg=round(_rad2deg(angular_sep), 6),
            sun_angular_radius_deg=round(_rad2deg(alpha_sun), 6),
            earth_angular_radius_deg=round(_rad2deg(alpha_earth_sc), 6),
            umbra_angle_deg=round(_rad2deg(alpha_earth_sc - alpha_sun), 6),
            penumbra_angle_deg=round(_rad2deg(alpha_earth_sc + alpha_sun), 6),
            metadata={
                "epoch_iso": epoch_iso,
                "assumption": "假设: 圆锥阴影模型，考虑太阳角直径 (~0.267°)",
            },
        )
    else:
        # 圆柱阴影模型 (默认)
        # 判断: 航天器是否在天体圆柱阴影内
        # 阴影轴 = 日地连线方向，航天器在阴影轴上的投影距地心距离
        # 阴影圆柱半径 = R_EARTH
        r_perp = math.sqrt(abs(r_sc_mag**2 - proj**2))  # 到阴影轴的垂直距离

        if proj < 0 and r_perp < _R_EARTH:
            eclipse_state = "umbra"
            shadow_type = "total"
        elif proj < 0 and r_perp < _R_EARTH * 1.05:
            eclipse_state = "penumbra"
            shadow_type = "partial"
        else:
            eclipse_state = "sunlight"
            shadow_type = "none"

        result = _success(
            eclipse_state=eclipse_state,
            shadow_type=shadow_type,
            shadow_model="cylindrical",
            distance_to_shadow_axis_m=round(r_perp, 0),
            projection_along_axis_m=round(proj, 0),
            shadow_radius_m=round(_R_EARTH, 0),
            metadata={
                "epoch_iso": epoch_iso,
                "assumption": "假设: 圆柱阴影模型，太阳为点光源，阴影半径 = R_EARTH",
            },
        )
    return result


# ---------------------------------------------------------------------------
# Tool 9: basilisk_atmospheric_drag
# ---------------------------------------------------------------------------
def basilisk_atmospheric_drag(
    spacecraft_position_m: List[float],
    spacecraft_velocity_mps: Optional[List[float]] = None,
    drag_coefficient: float = 2.2,
    area_m2: float = 1.0,
    spacecraft_mass_kg: float = 100.0,
    atmosphere_model: str = "exponential",
    F10_7: float = 150.0,
    Ap: float = 15.0,
) -> Dict:
    """大气阻力模型：计算大气密度与阻力加速度。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    支持两种模型:
      - exponential: 指数大气（分层参考密度，假设: Vallado 2013 表 8-4）
      - nrlmsise_simplified: 简化版 NRLMSISE-00（F10.7 + Ap 修正，假设: 仅含主要项）

    注意: 简化版 NRLMSISE-00 仅提供数量级估计，精度远低于完整版。

    Args:
        spacecraft_position_m: 航天器地心位置 [x,y,z] m
        spacecraft_velocity_mps: 航天器速度 [vx,vy,vz] m/s（可选，用于计算阻力方向）
        drag_coefficient: 阻力系数 Cd（默认 2.2）
        area_m2: 迎风面积 m²
        spacecraft_mass_kg: 航天器质量 kg
        atmosphere_model: 大气模型 "exponential" 或 "nrlmsise_simplified"
        F10_7: 10.7 cm 太阳射电通量 sfu（nrlmsise 模式使用）
        Ap: 地磁指数（nrlmsise 模式使用）
    Returns:
        {density_kgpm3, drag_acceleration_mps2, drag_direction, altitude_km, metadata}
    """
    r = np.array(spacecraft_position_m, dtype=float)
    r_mag = float(np.linalg.norm(r))
    altitude_km = (r_mag - _R_EARTH) / 1000.0

    if altitude_km < 0:
        return _error("航天器在地球表面以下，无法计算大气阻力")

    if atmosphere_model == "nrlmsise_simplified":
        # 简化 NRLMSISE-00 (假设: 仅含主要项，精度 ~30-50%)
        density = _nrlmsise_simplified(altitude_km, F10_7, Ap)
        model_note = "假设: 简化版 NRLMSISE-00，仅含主要项，精度 ~30-50%"
    else:
        density = _exponential_atmosphere(altitude_km)
        model_note = "假设: 指数大气模型 (Vallado 2013 表 8-4)，无太阳/地磁活动修正"

    if spacecraft_velocity_mps is not None:
        v = np.array(spacecraft_velocity_mps, dtype=float)
        v_mag = float(np.linalg.norm(v))
        if v_mag > 1e-10:
            # 假设: 大气与地球共转，转速 omega_E × r
            omega_e = 7.2921159e-5  # rad/s
            v_atm = np.array([-omega_e * r[1], omega_e * r[0], 0.0])
            v_rel = v - v_atm
            v_rel_mag = float(np.linalg.norm(v_rel))
            drag_dir = -_normalize(v_rel) if v_rel_mag > 1e-10 else np.zeros(3)
            drag_accel_mag = 0.5 * density * drag_coefficient * area_m2 * v_rel_mag**2 / spacecraft_mass_kg
            drag_accel = drag_accel_mag * drag_dir
        else:
            drag_accel = np.zeros(3)
            drag_dir = np.zeros(3)
            v_rel_mag = 0.0
    else:
        # 假设: 圆轨道速度
        v_mag = math.sqrt(_MU_EARTH / r_mag)
        v_rel_mag = v_mag
        drag_dir = np.array([0.0, 0.0, 0.0])  # 无速度方向无法确定
        drag_accel_mag = 0.5 * density * drag_coefficient * area_m2 * v_rel_mag**2 / spacecraft_mass_kg
        drag_accel = np.array([drag_accel_mag, 0.0, 0.0])  # 方向未知

    return _success(
        density_kgpm3=density,
        altitude_km=round(altitude_km, 3),
        drag_acceleration_mps2=drag_accel.tolist(),
        drag_acceleration_mag_mps2=round(float(np.linalg.norm(drag_accel)), 12),
        drag_direction=drag_dir.tolist(),
        ballistic_coefficient_m2kg=round(area_m2 / spacecraft_mass_kg, 6),
        metadata={
            "atmosphere_model": atmosphere_model,
            "drag_coefficient": drag_coefficient,
            "area_m2": area_m2,
            "spacecraft_mass_kg": spacecraft_mass_kg,
            "relative_velocity_mps": round(v_rel_mag, 3) if spacecraft_velocity_mps else None,
            "assumption": model_note,
        },
    )


def _exponential_atmosphere(altitude_km: float) -> float:
    """指数大气密度模型 (假设: Vallado 2013 表 8-4 分层参考密度)。"""
    if altitude_km > 1000:
        return 1e-17  # 极高空外推

    # 找到所在层
    for i in range(len(_EXP_ATMOS_LAYERS) - 1):
        h0, rho0, H = _EXP_ATMOS_LAYERS[i]
        h1, _, _ = _EXP_ATMOS_LAYERS[i + 1]
        if h0 <= altitude_km < h1:
            return rho0 * math.exp(-(altitude_km - h0) / H)

    # 最后一层外推
    h0, rho0, H = _EXP_ATMOS_LAYERS[-1]
    return rho0 * math.exp(-(altitude_km - h0) / H)


def _nrlmsise_simplified(altitude_km: float, F10_7: float, Ap: float) -> float:
    """简化版 NRLMSISE-00 密度估计 (假设: 仅含主要项，精度 ~30-50%)。

    参考: Picone et al. 2002, "NRLMSISE-00 empirical model of the atmosphere"
    简化: 仅用指数拟合 + F10.7/Ap 修正因子，忽略周日变化、纬度、季节。
    """
    # 基础指数密度
    rho_base = _exponential_atmosphere(altitude_km)

    # F10.7 太阳活动修正 (假设: 线性修正，参考均值 150 sfu)
    f10_factor = 1.0 + 0.5 * (F10_7 - 150.0) / 150.0

    # Ap 地磁活动修正 (假设: 仅在 >200 km 显著)
    if altitude_km > 200:
        ap_factor = 1.0 + 0.02 * (Ap - 15.0) / 15.0
    else:
        ap_factor = 1.0

    return rho_base * max(0.1, f10_factor) * max(0.5, ap_factor)


# ---------------------------------------------------------------------------
# Tool 10: basilisk_solar_radiation_pressure
# ---------------------------------------------------------------------------
def basilisk_solar_radiation_pressure(
    spacecraft_position_m: List[float],
    spacecraft_mass_kg: float = 100.0,
    area_m2: float = 1.0,
    reflectivity_coefficient: float = 1.2,
    sun_position_m: Optional[List[float]] = None,
    epoch_iso: str = "2026-01-01T00:00:00",
    shadow_model: str = "cylindrical",
) -> Dict:
    """太阳光压模型：计算 SRP 加速度（球模型 + 圆柱阴影）。

    纯 numpy 实现，不依赖 Basilisk 仿真。

    模型: 球形航天器假设，cannonball 模型。
    a_srp = -Cr * (A/m) * P_srp * (1 AU / r_sun)^2 * d_sun

    其中:
      Cr = 1 + reflectivity_coefficient (假设: 镜面反射系数 0.2 → Cr=1.2)
      P_srp = 4.56e-6 N/m² (1 AU 处太阳光压)
      d_sun = 太阳方向单位矢量

    Args:
        spacecraft_position_m: 航天器地心位置 [x,y,z] m
        spacecraft_mass_kg: 航天器质量 kg
        area_m2: 迎光面积 m²
        reflectivity_coefficient: 反射系数（0=完全吸收, 1=完全镜面反射, 默认 0.2）
        sun_position_m: 太阳地心位置 [x,y,z] m（可选）
        epoch_iso: ISO 8601 时间字符串
        shadow_model: 阴影模型 "cylindrical" 或 "none"
    Returns:
        {srp_acceleration_mps2, srp_force_N, eclipse_factor, sun_distance_au, metadata}
    """
    r_sc = np.array(spacecraft_position_m, dtype=float)
    if spacecraft_mass_kg <= 0:
        return _error("航天器质量必须 > 0")

    if sun_position_m is not None:
        r_sun = np.array(sun_position_m, dtype=float)
    else:
        # 简化太阳位置
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(epoch_iso.replace("Z", ""))
            jd = _julian_date(dt.year, dt.month, dt.day,
                              dt.hour + dt.minute / 60.0 + dt.second / 3600.0)
            n_days = jd - 2451545.0
            M_sun = _wrap_angle(_deg2rad(357.5291 + 0.98560028 * n_days))
            L_sun = _wrap_angle(_deg2rad(280.4665 + 0.98564736 * n_days))
            C_sun = _deg2rad(1.9148 * math.sin(M_sun) + 0.0200 * math.sin(2 * M_sun))
            lambda_sun = _wrap_angle(L_sun + C_sun)
            epsilon = _deg2rad(23.4393 - 0.0000004 * n_days)
            r_sun_au = 1.00014 - 0.01671 * math.cos(M_sun) - 0.00014 * math.cos(2 * M_sun)
            r_sun = np.array([
                r_sun_au * _AU * math.cos(lambda_sun),
                r_sun_au * _AU * math.sin(lambda_sun) * math.cos(epsilon),
                r_sun_au * _AU * math.sin(lambda_sun) * math.sin(epsilon),
            ])
        except Exception:
            r_sun = np.array([_AU, 0.0, 0.0])

    # 太阳-航天器矢量
    d_sun_sc = r_sun - r_sc
    d_mag = float(np.linalg.norm(d_sun_sc))
    d_unit = d_sun_sc / d_mag
    r_sun_au_val = d_mag / _AU

    # 星蚀因子
    eclipse_factor = 1.0
    if shadow_model == "cylindrical":
        r_sc_mag = float(np.linalg.norm(r_sc))
        d_unit_earth_to_sun = _normalize(r_sun)
        proj = np.dot(r_sc, d_unit_earth_to_sun)
        r_perp = math.sqrt(abs(r_sc_mag**2 - proj**2))
        if proj < 0 and r_perp < _R_EARTH:
            eclipse_factor = 0.0  # 本影，完全遮挡
        elif proj < 0 and r_perp < _R_EARTH * 1.1:
            # 半影过渡 (假设: 线性插值)
            frac = (r_perp - _R_EARTH) / (_R_EARTH * 0.1)
            eclipse_factor = max(0.0, min(1.0, frac))

    # cannonball 模型
    Cr = 1.0 + reflectivity_coefficient
    # 距离修正: (1 AU)^2 / r^2
    distance_factor = (1.0 / r_sun_au_val)**2
    srp_mag = -Cr * (area_m2 / spacecraft_mass_kg) * _P_SRP * distance_factor * eclipse_factor
    srp_accel = srp_mag * d_unit
    srp_force = srp_accel * spacecraft_mass_kg

    return _success(
        srp_acceleration_mps2=srp_accel.tolist(),
        srp_acceleration_mag_mps2=round(abs(srp_mag), 12),
        srp_force_N=srp_force.tolist(),
        srp_force_mag_N=round(abs(srp_mag) * spacecraft_mass_kg, 12),
        eclipse_factor=round(eclipse_factor, 4),
        sun_distance_au=round(r_sun_au_val, 6),
        sun_direction_eci=d_unit.tolist(),
        reflectivity_cr=round(Cr, 4),
        metadata={
            "model": "cannonball",
            "shadow_model": shadow_model,
            "reflectivity_coefficient": reflectivity_coefficient,
            "Cr": round(Cr, 4),
            "P_srp_Npm2": _P_SRP,
            "assumption": "假设: 球形航天器 cannonball 模型，圆柱阴影，"
                          "太阳常数 1361 W/m² 为 1 AU 处均值",
        },
    )


# ---------------------------------------------------------------------------
# 辅助: 儒略日计算 + DCM → 四元数
# ---------------------------------------------------------------------------
def _julian_date(year: int, month: int, day: int, day_frac: float = 0.0) -> float:
    """计算儒略日 (假设: 简化公式，适用 1901-2099 年)。"""
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd + day_frac


def _dcm_to_quat(dcm: np.ndarray) -> List[float]:
    """DCM (3x3) → 四元数 scalar-first [q0, q1, q2, q3]。

    假设: DCM 为正交矩阵（可能含微小数值误差）。
    """
    trace = float(dcm[0, 0] + dcm[1, 1] + dcm[2, 2])

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q0 = 0.25 * s
        q1 = (dcm[2, 1] - dcm[1, 2]) / s
        q2 = (dcm[0, 2] - dcm[2, 0]) / s
        q3 = (dcm[1, 0] - dcm[0, 1]) / s
    elif dcm[0, 0] > dcm[1, 1] and dcm[0, 0] > dcm[2, 2]:
        s = math.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]) * 2.0
        q0 = (dcm[2, 1] - dcm[1, 2]) / s
        q1 = 0.25 * s
        q2 = (dcm[0, 1] + dcm[1, 0]) / s
        q3 = (dcm[0, 2] + dcm[2, 0]) / s
    elif dcm[1, 1] > dcm[2, 2]:
        s = math.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2]) * 2.0
        q0 = (dcm[0, 2] - dcm[2, 0]) / s
        q1 = (dcm[0, 1] + dcm[1, 0]) / s
        q2 = 0.25 * s
        q3 = (dcm[1, 2] + dcm[2, 1]) / s
    else:
        s = math.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1]) * 2.0
        q0 = (dcm[1, 0] - dcm[0, 1]) / s
        q1 = (dcm[0, 2] + dcm[2, 0]) / s
        q2 = (dcm[1, 2] + dcm[2, 1]) / s
        q3 = 0.25 * s

    # 归一化
    q_norm = math.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    if q_norm > 1e-15:
        q0, q1, q2, q3 = q0 / q_norm, q1 / q_norm, q2 / q_norm, q3 / q_norm
    else:
        q0, q1, q2, q3 = 1.0, 0.0, 0.0, 0.0

    return [round(float(q0), 12), round(float(q1), 12),
            round(float(q2), 12), round(float(q3), 12)]


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
__all__ = [
    "basilisk_propagate_orbit",
    "basilisk_attitude_control",
    "basilisk_orbit_elements_conversion",
    "basilisk_thruster_modeling",
    "basilisk_reaction_wheel_modeling",
    "basilisk_sun_pointing",
    "basilisk_nadir_pointing",
    "basilisk_eclipse_detection",
    "basilisk_atmospheric_drag",
    "basilisk_solar_radiation_pressure",
]