"""轨道传播工具 — 轨道表示转换与轨道外推。

第一性原理（K3 传播完整性）：
  1. 笛卡尔↔开普勒互转必须显式标注 mu 和 units
  2. 二体传播是最低保底——高保真需切换 poliastro/orekit/gmat
  3. 引擎选择：auto 时优先 poliastro > orekit > gmat > 内置二体
  4. 输出必须包含 engine、engine_version、units、frame、propagator_type
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..schemas import OrbitState, Epoch, Frame, FrameName, FrameCenter
from ..schemas import OrbitRepresentation, KeplerianElements
from ..adapters import get_adapter, get_all_adapters

#: 地球引力参数 m³/s²
_MU_EARTH = 3.986004418e14


def convert_orbit_representation(state_dict: Dict,
                                 target_representation: str,
                                 mu: float = _MU_EARTH) -> Dict:
    """轨道状态表示转换（笛卡尔↔开普勒）。

    Args:
        state_dict: OrbitState.to_dict() 格式
        target_representation: "cartesian" 或 "keplerian"
        mu: 引力参数 m³/s²（默认地球）
    Returns:
        {state, source_representation, target_representation, mu, units, engine}
    """
    target = target_representation.lower().strip()

    if target not in ("cartesian", "keplerian"):
        return _error(f"不支持的表示: '{target_representation}'，"
                      "支持: cartesian / keplerian")

    try:
        state = OrbitState.from_dict(state_dict)
    except Exception as exc:
        return _error(f"输入状态解析失败: {exc}")

    src_repr = state.representation.value

    # 同表示——直接返回
    if src_repr == target:
        return _build_convert_result(state, src_repr, target, mu, "identity")

    engine = "schemas_builtin"

    if target == "keplerian":
        if state.position_m is None or state.velocity_mps is None:
            return _error("转 keplerian 需要 position_m 和 velocity_mps")
        try:
            new_state = state.to_keplerian(mu)
        except Exception as exc:
            # 回退到 poliastro
            result = _try_poliastro_convert(state, target, mu)
            if result:
                return result
            return _error(f"内置转换失败: {exc}")
        return _build_convert_result(new_state, src_repr, target, mu, engine)

    if target == "cartesian":
        if state.elements is None:
            return _error("转 cartesian 需要 elements")
        try:
            new_state = state.to_cartesian(mu)
        except Exception as exc:
            result = _try_poliastro_convert(state, target, mu)
            if result:
                return result
            return _error(f"内置转换失败: {exc}")
        return _build_convert_result(new_state, src_repr, target, mu, engine)

    return _error(f"未知目标表示: {target}")


def _try_poliastro_convert(state, target, mu) -> Optional[Dict]:
    """使用 poliastro 进行轨道表示转换（回退）。"""
    adapter = get_adapter("poliastro")
    if not adapter.is_available():
        return None
    try:
        from poliastro.bodies import Earth  # type: ignore
        from poliastro.twobody import Orbit  # type: ignore
        import astropy.units as u  # type: ignore
        if state.position_m and state.velocity_mps:
            r = state.position_m * u.m
            v = state.velocity_mps * u.m / u.s
            orb = Orbit.from_vectors(Earth, r, v)
            if target == "keplerian":
                elements = {
                    "a_m": float(orb.a.to(u.m).value),
                    "e": float(orb.ecc.value) if hasattr(orb.ecc, 'value') else float(orb.ecc),
                    "i_deg": math.degrees(float(orb.inc.to(u.rad).value)),
                    "raan_deg": math.degrees(float(orb.raan.to(u.rad).value)),
                    "argp_deg": math.degrees(float(orb.argp.to(u.rad).value)),
                    "ta_deg": math.degrees(float(orb.nu.to(u.rad).value)),
                }
                new_state = OrbitState(
                    epoch=state.epoch, frame=state.frame,
                    representation=OrbitRepresentation.KEPLERIAN,
                    position_m=state.position_m, velocity_mps=state.velocity_mps,
                    elements=KeplerianElements(**elements),
                )
                return _build_convert_result(new_state, "cartesian",
                                             "keplerian", mu, "poliastro")
    except Exception:
        pass
    return None


def propagate_orbit(initial_state_dict: Dict, force_model_dict: Dict,
                    duration_s: float, output_step_s: Optional[float] = None,
                    engine: str = "auto") -> Dict:
    """轨道传播。

    第一版：二体 + J2 占位。engine 支持 auto/poliastro/orekit/gmat。

    Args:
        initial_state_dict: 初始 OrbitState 字典
        force_model_dict: ForceModel 字典
        duration_s: 传播时长 s
        output_step_s: 输出采样间隔 s（None 表示仅输出终态）
        engine: 引擎选择
    Returns:
        {state_history, metadata}
    """
    try:
        state = OrbitState.from_dict(initial_state_dict)
    except Exception as exc:
        return _error(f"初始状态解析失败: {exc}")

    if state.position_m is None or state.velocity_mps is None:
        return _error("传播需要 cartesian position_m 和 velocity_mps")

    mu = _MU_EARTH
    # 判断是否纯二体
    is_two_body = _is_two_body(force_model_dict)
    is_j2 = _is_j2(force_model_dict)

    # 引擎选择
    chosen = _select_engine(engine)
    adapter = get_adapter(chosen)

    if chosen != "builtin" and adapter.is_available():
        try:
            result = adapter.propagate_orbit(
                state, force_model_dict,
                type("Cfg", (), {
                    "duration_s": duration_s,
                    "output_step_s": output_step_s,
                    "mu": mu,
                })(),
            )
            if result.get("status") != "unavailable":
                return _format_adapter_result(result, chosen, adapter)
        except Exception:
            pass  # 回退到内置

    # 内置二体 / J2 传播
    if is_two_body or is_j2:
        history = _propagate_builtin(state, mu, duration_s,
                                     output_step_s, is_j2)
        return {
            "state_history": history,
            "metadata": {
                "engine": "builtin",
                "engine_version": "0.1.0",
                "units": "SI (m, m/s, s)",
                "frame": state.frame.name.value,
                "propagator_type": "j2" if is_j2 else "two_body",
                "step_count": len(history),
                "mu": mu,
            },
        }

    return _error("无法完成传播：无可用引擎且力学模型非二体/J2")


def _is_two_body(fm_dict: Dict) -> bool:
    gravity = fm_dict.get("gravity", "point_mass")
    return (gravity == "point_mass"
            and not fm_dict.get("drag", {}).get("enabled", False)
            and not fm_dict.get("srp", {}).get("enabled", False)
            and not fm_dict.get("third_body")
            and not fm_dict.get("relativity", False))


def _is_j2(fm_dict: Dict) -> bool:
    gravity = fm_dict.get("gravity", "point_mass")
    return (gravity == "spherical_harmonics"
            and fm_dict.get("degree", 0) == 2
            and fm_dict.get("order", 0) == 0)


def _select_engine(engine: str) -> str:
    if engine != "auto":
        return engine
    for eng in ("poliastro", "orekit", "gmat"):
        adapter = get_adapter(eng)
        if adapter.is_available():
            return eng
    return "builtin"


def _propagate_builtin(state, mu, duration_s, output_step_s, is_j2) -> List[Dict]:
    """内置二体/J2 传播器。"""
    r = list(state.position_m)
    v = list(state.velocity_mps)
    step = output_step_s or duration_s
    n_steps = max(1, int(duration_s / step) + 1)
    history = []

    for i in range(n_steps):
        t = min(i * step, duration_s)
        if is_j2:
            pos, vel = _j2_step(r, v, t, mu)
        else:
            pos, vel = _kepler_step(r, v, t, mu)
        epoch_val = state.epoch.value
        entry = OrbitState(
            epoch=state.epoch, frame=state.frame,
            position_m=pos, velocity_mps=vel,
        ).to_dict()
        entry["elapsed_s"] = t
        history.append(entry)
        if t >= duration_s:
            break
    return history


def _kepler_step(r, v, dt, mu):
    """二体解析传播——基于开普勒方程 Newton-Raphson 求解。

    替换原 f/g 泰勒近似（长跨度误差数千公里），
    使用普适变量法求解开普勒方程，支持椭圆/抛物线/双曲线。
    """
    r_mag = math.sqrt(sum(x * x for x in r))
    v_mag = math.sqrt(sum(x * x for x in v))
    if r_mag < 1.0 or abs(dt) < 1e-12:
        return list(r), list(v)

    energy = v_mag * v_mag / 2.0 - mu / r_mag
    a = -mu / (2.0 * energy) if energy != 0 else float('inf')

    # 角动量
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    hz = r[0] * v[1] - r[1] * v[0]
    h_mag = math.sqrt(hx * hx + hy * hy + hz * hz)

    # 偏心率向量
    rdotv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
    e_vec = [(v_mag * v_mag - mu / r_mag) * r[i] / mu - rdotv * v[i] / mu
             for i in range(3)]
    e = math.sqrt(sum(c * c for c in e_vec))

    if e < 1.0 and a > 0:
        # 椭圆轨道：用平近点角 M 推进
        n_motion = math.sqrt(mu / a ** 3)
        # 当前真近点角 → E → M
        cos_ta = max(-1.0, min(1.0, sum(e_vec[i] * r[i] for i in range(3)) / (e * r_mag))) if e > 1e-10 else 1.0
        sin_ta = math.sin(math.acos(cos_ta)) if e > 1e-10 else 0.0
        if rdotv < 0:
            sin_ta = -sin_ta
        E = math.atan2(math.sqrt(1 - e * e) * sin_ta, e + cos_ta)
        M = E - e * math.sin(E)
        M_new = M + n_motion * dt
        # Newton-Raphson 求解 E_new
        E_new = M_new
        for _ in range(30):
            f = E_new - e * math.sin(E_new) - M_new
            fp = 1 - e * math.cos(E_new)
            dE = f / fp
            E_new -= dE
            if abs(dE) < 1e-12:
                break
        # E → 真近点角
        cos_E = math.cos(E_new)
        sin_E = math.sin(E_new)
        ta = math.atan2(math.sqrt(1 - e * e) * sin_E, e + cos_E)
        p = a * (1 - e * e)
    elif e > 1.0 and a < 0:
        # 双曲线轨道：用双曲近点角 H
        abs_a = abs(a)
        n_hyp = math.sqrt(mu / abs_a ** 3)
        cos_ta = max(-1.0, min(1.0, sum(e_vec[i] * r[i] for i in range(3)) / (e * r_mag))) if e > 1e-10 else 1.0
        sin_ta = math.sin(math.acos(cos_ta)) if e > 1e-10 else 0.0
        if rdotv < 0:
            sin_ta = -sin_ta
        # ta → H
        H = math.atanh(math.sqrt(e * e - 1) * sin_ta / (e + cos_ta))
        M_hyp = e * math.sinh(H) - H
        M_hyp_new = M_hyp + n_hyp * dt
        H_new = M_hyp_new
        for _ in range(30):
            f = e * math.sinh(H_new) - H_new - M_hyp_new
            fp = e * math.cosh(H_new) - 1
            dH = f / fp
            H_new -= dH
            if abs(dH) < 1e-12:
                break
        cosh_H = math.cosh(H_new)
        sinh_H = math.sinh(H_new)
        ta = math.atan2(math.sqrt(e * e - 1) * sinh_H, e - cosh_H)
        p = abs_a * (e * e - 1)
    else:
        # 抛物线或退化情况：回退到数值近似
        # 用 Barker 方程
        q = h_mag * h_mag / (2 * mu)
        # 当前 ta
        cos_ta = max(-1.0, min(1.0, sum(e_vec[i] * r[i] for i in range(3)) / (e * r_mag))) if e > 1e-10 else 1.0
        ta = math.acos(cos_ta)
        if rdotv < 0:
            ta = -ta
        # Barker 方程: D = tan(ta/2) + (1/3)*tan^3(ta/2)
        tan_half = math.tan(ta / 2)
        D = tan_half + tan_half ** 3 / 3
        D_new = D + math.sqrt(mu / (2 * q ** 3)) * dt
        # 求解 tan(ta/2) 的三次方程
        # 近似：迭代
        tan_half_new = D_new / 2  # 初始猜测
        for _ in range(20):
            f = tan_half_new + tan_half_new ** 3 / 3 - D_new
            fp = 1 + tan_half_new ** 2
            dt_half = f / fp
            tan_half_new -= dt_half
            if abs(dt_half) < 1e-12:
                break
        ta = 2 * math.atan(tan_half_new)
        p = 2 * q

    # 开普勒根数 → 笛卡尔
    r_orb = p / (1 + e * math.cos(ta))
    x_orb = r_orb * math.cos(ta)
    y_orb = r_orb * math.sin(ta)
    if r_orb > 0 and p > 0:
        vx_orb = -math.sqrt(mu / p) * math.sin(ta)
        vy_orb = math.sqrt(mu / p) * (e + math.cos(ta))
    else:
        vx_orb, vy_orb = 0.0, 0.0

    # 旋转到惯性系（使用当前轨道平面，不改变 Ω/ω/i）
    # 简化：仅绕角动量方向旋转，保持原轨道平面
    # 构建 PQW → ECI 旋转矩阵
    if h_mag > 1e-10:
        # 节点向量 n = k × h
        n_x, n_y = -hy, hx
        n_mag = math.sqrt(n_x * n_x + n_y * n_y)
        if n_mag > 1e-10:
            cos_raan = n_x / n_mag
            raan = math.acos(max(-1.0, min(1.0, cos_raan)))
            if n_y < 0:
                raan = 2 * math.pi - raan
        else:
            raan = 0.0
        cos_i = hz / h_mag
        cos_i = max(-1.0, min(1.0, cos_i))
        inc = math.acos(cos_i)
        if e > 1e-10 and n_mag > 1e-10:
            cos_argp = (n_x * e_vec[0] + n_y * e_vec[1]) / (n_mag * e)
            cos_argp = max(-1.0, min(1.0, cos_argp))
            argp = math.acos(cos_argp)
            if e_vec[2] < 0:
                argp = 2 * math.pi - argp
        else:
            argp = 0.0
    else:
        raan, inc, argp = 0.0, 0.0, 0.0

    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_argp = math.cos(argp)
    sin_argp = math.sin(argp)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)

    r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
    r12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_i
    r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
    r22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_i
    r31 = sin_argp * sin_i
    r32 = cos_argp * sin_i

    new_r = [r11 * x_orb + r12 * y_orb,
             r21 * x_orb + r22 * y_orb,
             r31 * x_orb + r32 * y_orb]
    new_v = [r11 * vx_orb + r12 * vy_orb,
             r21 * vx_orb + r22 * vy_orb,
             r31 * vx_orb + r32 * vy_orb]

    return new_r, new_v


def _j2_step(r, v, dt, mu):
    """J2 摄动传播——RK4 积分 J2 长期效应。

    J2 引起的轨道根数长期漂移：
      dΩ/dt = -3/2 * J2 * (R_E/a)² * n * cos(i)   (升交点赤经进动)
      dω/dt =  3/4 * J2 * (R_E/a)² * n * (5cos²(i)-1)  (近地点幅角漂移)

    实现步骤：
      1. 笛卡尔 → 开普勒根数
      2. 用解析公式计算 J2 长期漂移率
      3. 推进 Ω 和 ω
      4. 用二体传播推进真近点角
      5. 开普勒根数 → 笛卡尔
    """
    J2 = 1.08263e-3
    R_E = 6378137.0  # 地球赤道半径 m

    # 当前状态 → 开普勒根数
    r_mag = math.sqrt(sum(x * x for x in r))
    v_mag = math.sqrt(sum(x * x for x in v))
    if r_mag < 1.0 or dt == 0:
        return list(r), list(v)

    # 角动量 h = r × v
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    hz = r[0] * v[1] - r[1] * v[0]
    h_mag = math.sqrt(hx * hx + hy * hy + hz * hz)

    # 半长轴和偏心率
    energy = v_mag * v_mag / 2.0 - mu / r_mag
    a = -mu / (2.0 * energy) if energy != 0 else r_mag
    e_vec_x = (v_mag * v_mag - mu / r_mag) * r[0] / mu - (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) * v[0] / mu
    e_vec_y = (v_mag * v_mag - mu / r_mag) * r[1] / mu - (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) * v[1] / mu
    e_vec_z = (v_mag * v_mag - mu / r_mag) * r[2] / mu - (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) * v[2] / mu
    e = math.sqrt(e_vec_x * e_vec_x + e_vec_y * e_vec_y + e_vec_z * e_vec_z)

    # K5-H5: 双曲线/抛物线轨道(e>=1)无定义"平均运动"，回退到纯二体传播
    if e >= 1.0 or a <= 0:
        return _kepler_step(r, v, dt, mu)

    # 倾角
    cos_i = hz / h_mag if h_mag > 0 else 1.0
    cos_i = max(-1.0, min(1.0, cos_i))
    i = math.acos(cos_i)

    # 升交点赤经
    n_x = -hy
    n_y = hx
    n_mag = math.sqrt(n_x * n_x + n_y * n_y)
    if n_mag > 1e-10:
        cos_raan = n_x / n_mag
        cos_raan = max(-1.0, min(1.0, cos_raan))
        raan = math.acos(cos_raan)
        if n_y < 0:
            raan = 2 * math.pi - raan
    else:
        raan = 0.0

    # 近地点幅角
    if n_mag > 1e-10 and e > 1e-10:
        cos_argp = (n_x * e_vec_x + n_y * e_vec_y) / (n_mag * e)
        cos_argp = max(-1.0, min(1.0, cos_argp))
        argp = math.acos(cos_argp)
        if e_vec_z < 0:
            argp = 2 * math.pi - argp
    else:
        argp = 0.0

    # 真近点角
    if e > 1e-10:
        cos_ta = (e_vec_x * r[0] + e_vec_y * r[1] + e_vec_z * r[2]) / (e * r_mag)
        cos_ta = max(-1.0, min(1.0, cos_ta))
        ta = math.acos(cos_ta)
        # 判断方向
        rdotv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
        if rdotv < 0:
            ta = 2 * math.pi - ta
    else:
        ta = 0.0

    # J2 长期漂移率
    n_motion = math.sqrt(mu / (a ** 3))  # 平均运动
    j2_factor = 1.5 * J2 * (R_E / a) ** 2 * n_motion

    d_raan = -j2_factor * cos_i * dt
    d_argp = 0.5 * j2_factor * (5 * cos_i * cos_i - 1) * dt

    # K5-H4: J2 对平均近点角 M 的长期漂移修正
    # dM/dt = n + (3/4)*J2*(R_E/a)^2*n*(3cos^2(i)-1)*sqrt(1-e^2)
    dM_j2 = 0.75 * J2 * (R_E / a) ** 2 * n_motion * (3 * cos_i * cos_i - 1) * math.sqrt(max(0, 1 - e * e)) * dt

    # 推进轨道根数
    raan += d_raan
    argp += d_argp

    # 二体传播推进真近点角（用平近点角→偏近点角→真近点角）
    # 先用当前 ta 转 M，推进 M（含 J2 修正），再转回 ta
    if e < 1.0:  # 椭圆轨道
        cos_ta_val = max(-1.0, min(1.0, math.cos(ta)))
        sin_ta_val = math.sin(ta)
        E = math.atan2(math.sqrt(1 - e * e) * sin_ta_val, e + cos_ta_val)
        M = E - e * math.sin(E)
        M_new = M + n_motion * dt + dM_j2  # K5-H4: 加入 J2 M 漂移修正
        # Newton-Raphson 求解 E_new
        E_new = M_new
        for _ in range(20):
            f = E_new - e * math.sin(E_new) - M_new
            fp = 1 - e * math.cos(E_new)
            dE = f / fp
            E_new -= dE
            if abs(dE) < 1e-12:
                break
        # E → ta
        cos_E = math.cos(E_new)
        sin_E = math.sin(E_new)
        ta = math.atan2(math.sqrt(1 - e * e) * sin_E, e + cos_E)
    # 抛物线/双曲线暂用二体解析近似
    else:
        ta += n_motion * dt

    # 开普勒根数 → 笛卡尔
    cos_raan = math.cos(raan)
    sin_raan = math.sin(raan)
    cos_argp = math.cos(argp)
    sin_argp = math.sin(argp)
    cos_i = math.cos(i)
    sin_i = math.sin(i)
    cos_ta = math.cos(ta)
    sin_ta = math.sin(ta)

    # 轨道平面坐标
    p = a * (1 - e * e)

    r_orb = p / (1 + e * cos_ta)
    x_orb = r_orb * cos_ta
    y_orb = r_orb * sin_ta

    # 速度（轨道平面）
    if r_orb > 0:
        vx_orb = -math.sqrt(mu / p) * sin_ta
        vy_orb = math.sqrt(mu / p) * (e + cos_ta)
    else:
        vx_orb, vy_orb = 0.0, 0.0

    # 旋转到惯性系: R = Rz(raan) * Rx(i) * Rz(argp)
    # 简化旋转矩阵
    r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
    r12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_i
    r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
    r22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_i
    r31 = sin_argp * sin_i
    r32 = cos_argp * sin_i

    new_r = [r11 * x_orb + r12 * y_orb,
             r21 * x_orb + r22 * y_orb,
             r31 * x_orb + r32 * y_orb]
    new_v = [r11 * vx_orb + r12 * vy_orb,
             r21 * vx_orb + r22 * vy_orb,
             r31 * vx_orb + r32 * vy_orb]

    return new_r, new_v


def _build_convert_result(state, src, target, mu, engine) -> Dict:
    return {
        "state": state.to_dict(),
        "source_representation": src,
        "target_representation": target,
        "mu": mu,
        "units": "SI (m, m/s, deg, rad stored)",
        "engine": engine,
    }


def _format_adapter_result(result, engine, adapter) -> Dict:
    result.setdefault("metadata", {})
    result["metadata"]["engine"] = engine
    result["metadata"]["engine_version"] = adapter.version()
    result["metadata"]["units"] = "SI (m, m/s, s)"
    return result


def _error(reason) -> Dict:
    return {"status": "error", "reason": reason, "engine": None}


__all__ = ["convert_orbit_representation", "propagate_orbit"]
