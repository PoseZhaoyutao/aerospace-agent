"""轨道机动 (Orbital maneuvers) 辅助函数。

本模块实现常见的脉冲 (impulsive) 轨道机动:
    * Hohmann 转移 (共面圆轨道间最省能量的双脉冲转移)
    * 双椭圆转移 (Bi-elliptic)
    * 改变倾角 (plane change)
    * 相位机动 (phasing orbit)

所有公式基于 Vis-viva 方程与冲量假设, 单位 SI。
"""

from __future__ import annotations

import math


def _vis_viva(mu, r, a):
    """Vis-viva: v = sqrt(mu (2/r - 1/a))."""
    return math.sqrt(mu * (2.0 / r - 1.0 / a))


def hohmann_transfer_delta_v(r1, r2, mu):
    """共面圆轨道 Hohmann 转移的 delta-V 与飞行时间。

    推导
    -----
    两圆轨道半径 r1 < r2, 转移椭圆半长轴:
        a_t = (r1 + r2) / 2
    转移椭圆偏心率: e_t = (r2 - r1)/(r2 + r1).

    第 1 次脉冲 (在 r1 内圆切向加速到转移椭圆近地点):
        dv1 = sqrt(mu(2/r1 - 1/a_t)) - sqrt(mu/r1)
    第 2 次脉冲 (在 r2 转移椭圆远地点加速到外圆):
        dv2 = sqrt(mu/r2) - sqrt(mu(2/r2 - 1/a_t))
    飞行时间 = 转移椭圆半周期:
        t = pi * sqrt(a_t^3 / mu)

    返回 (dv1, dv2, dv_total, t_transfer, a_transfer)
    """
    r1 = float(r1); r2 = float(r2); mu = float(mu)
    if r1 <= 0 or r2 <= 0:
        raise ValueError("r1, r2 必须 > 0")
    a_t = 0.5 * (r1 + r2)
    v1_circ = math.sqrt(mu / r1)
    v2_circ = math.sqrt(mu / r2)
    v1_peri = _vis_viva(mu, r1, a_t)   # 转移椭圆在 r1 的速度
    v2_apo = _vis_viva(mu, r2, a_t)    # 转移椭圆在 r2 的速度
    dv1 = v1_peri - v1_circ
    dv2 = v2_circ - v2_apo
    t_transfer = math.pi * math.sqrt(a_t ** 3 / mu)
    e_t = abs(r2 - r1) / (r2 + r1)
    return {
        "dv1": dv1,
        "dv2": dv2,
        "dv_total": dv1 + dv2,
        "t_transfer": t_transfer,
        "a_transfer": a_t,
        "e_transfer": e_t,
    }


def bielliptic_transfer(r1, r2, rb, mu):
    """双椭圆转移 (Bi-elliptic transfer)。

    通过远地点 rb (rb > r2 > r1) 的两段椭圆完成转移, 三次脉冲。

    推导
    -----
    椭圆 1: r1 -> rb,  a1 = (r1 + rb)/2
    椭圆 2: rb -> r2,  a2 = (r2 + rb)/2
        dv1 = sqrt(mu(2/r1 - 1/a1)) - sqrt(mu/r1)              (内圆 -> 椭圆1)
        dv2 = sqrt(mu(2/rb - 1/a2)) - sqrt(mu(2/rb - 1/a1))   (椭圆1 -> 椭圆2, 在 rb)
        dv3 = sqrt(mu/r2) - sqrt(mu(2/r2 - 1/a2))             (椭圆2 -> 外圆, 在 r2)
    飞行时间 = 两段半周期之和。

    返回 dict 含 dv1, dv2, dv3, dv_total, t_transfer, a1, a2.
    """
    r1 = float(r1); r2 = float(r2); rb = float(rb); mu = float(mu)
    if not (r1 > 0 and r2 > 0 and rb > max(r1, r2)):
        raise ValueError("要求 rb > r2 > r1 > 0")
    a1 = 0.5 * (r1 + rb)
    a2 = 0.5 * (r2 + rb)
    v1_circ = math.sqrt(mu / r1)
    v2_circ = math.sqrt(mu / r2)
    v1_peri = _vis_viva(mu, r1, a1)
    v_rb_1 = _vis_viva(mu, rb, a1)
    v_rb_2 = _vis_viva(mu, rb, a2)
    v2_apo = _vis_viva(mu, r2, a2)
    dv1 = v1_peri - v1_circ
    dv2 = v_rb_2 - v_rb_1   # 在 rb 处继续加速 (同向)
    dv3 = v2_circ - v2_apo  # 在 r2 处减速进入圆轨道
    t1 = math.pi * math.sqrt(a1 ** 3 / mu)
    t2 = math.pi * math.sqrt(a2 ** 3 / mu)
    return {
        "dv1": dv1, "dv2": dv2, "dv3": dv3,
        "dv_total": dv1 + dv2 + dv3,
        "t_transfer": t1 + t2,
        "a1": a1, "a2": a2,
    }


def plane_change_delta_v(v, delta_i):
    """纯倾角改变 (速度大小不变) 的 delta-V。

    推导
    -----
    速度向量 v, 改变倾角 delta_i (速度方向转过 delta_i, 模不变).
    由余弦定理, 速度增量:
        dv = 2 v sin(delta_i / 2)

    在圆轨道上做纯倾角改变最贵 (因 v 最大); 常在大半径/远地点进行。
    """
    v = float(v); delta_i = float(delta_i)
    return 2.0 * v * math.sin(delta_i / 2.0)


def phasing_orbit(r_circ, r_phase, mu, target_lead_angle=0.0):
    """相位机动 (phasing orbit)。

    在半径 r_circ 的圆轨道上, 航天器进入半长轴不同的相位轨道 (近/远地点
    r_phase), 飞行整数圈后回到原相位点与目标交会。

    推导
    -----
    相位轨道半长轴: a_ph = (r_circ + r_phase)/2
    相位轨道周期:   T_ph = 2 pi sqrt(a_ph^3 / mu)
    圆轨道周期:     T_c  = 2 pi sqrt(r_circ^3 / mu)
    若飞行 k 圈后追上 (目标领先 target_lead_angle 弧度), 需满足:
        k * T_ph = T_c * (1 - target_lead_angle/(2 pi))   (追击)
    进入/离开相位轨道的 delta-V (各一次, 切向):
        dv_enter = |sqrt(mu(2/r_circ - 1/a_ph)) - sqrt(mu/r_circ)|
        dv_exit  = 同 enter (在相位轨道近/远地点回到圆轨道)

    参数
    -----
    r_circ           : 圆轨道半径 [m]
    r_phase          : 相位轨道另一端半径 (近或远地点) [m]
    mu               : 引力参数
    target_lead_angle: 目标领先角 [rad] (默认 0, 即完全追上)

    返回 dict 含 a_ph, T_ph, T_c, dv_enter, dv_exit, dv_total.
    """
    r_circ = float(r_circ); r_phase = float(r_phase); mu = float(mu)
    a_ph = 0.5 * (r_circ + r_phase)
    T_ph = 2.0 * math.pi * math.sqrt(a_ph ** 3 / mu)
    T_c = 2.0 * math.pi * math.sqrt(r_circ ** 3 / mu)
    v_circ = math.sqrt(mu / r_circ)
    v_ph_at_circ = _vis_viva(mu, r_circ, a_ph)
    dv_enter = abs(v_ph_at_circ - v_circ)
    dv_exit = dv_enter  # 对称机动
    return {
        "a_ph": a_ph,
        "T_ph": T_ph,
        "T_c": T_c,
        "dv_enter": dv_enter,
        "dv_exit": dv_exit,
        "dv_total": dv_enter + dv_exit,
        "period_ratio": T_ph / T_c,
    }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.physics.orbital_maneuvers 自测 ===")
    from .constants import MU_EARTH, R_EARTH

    # Hohmann: LEO 300km -> GEO
    r1 = R_EARTH + 300e3
    r2 = 42164e3  # GEO 半径
    h = hohmann_transfer_delta_v(r1, r2, MU_EARTH)
    print(f"Hohmann LEO->GEO: dv1={h['dv1']:.2f} m/s, dv2={h['dv2']:.2f} m/s, "
          f"dv_total={h['dv_total']:.2f} m/s, t={h['t_transfer']/3600:.2f} h")
    # 精确值 ~3893 m/s (LEO 300km -> GEO); 经典文献常四舍五入为 "~3.9 km/s"
    assert 3850 < h["dv_total"] < 3950, f"Hohmann dv_total 应 ~3.89 km/s, 得 {h['dv_total']:.1f}"
    assert abs(h["t_transfer"] / 3600 - 5.28) < 0.1

    # 双椭圆: rb = 100000 km
    be = bielliptic_transfer(r1, r2, 100000e3, MU_EARTH)
    print(f"双椭圆 (rb=100000km): dv_total={be['dv_total']:.2f} m/s, "
          f"t={be['t_transfer']/3600:.2f} h")

    # 倾角改变: v=7.8 km/s, 28.5°
    dv_pc = plane_change_delta_v(7800.0, math.radians(28.5))
    print(f"倾角改变 28.5° @ v=7.8km/s: dv={dv_pc:.2f} m/s (期望 ~3825)")
    assert abs(dv_pc - 3825) < 20

    # 相位轨道
    ph = phasing_orbit(r1, r1 - 100e3, MU_EARTH)
    print(f"相位轨道 (r_phase=r1-100km): T_ph/T_c={ph['period_ratio']:.5f}, "
          f"dv_total={ph['dv_total']:.2f} m/s")
    assert ph["T_ph"] < ph["T_c"]  # 更低轨道, 周期更短

    print("orbital_maneuvers 自测全部通过.")
