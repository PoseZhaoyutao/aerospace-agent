"""物理常数（SI 单位）。

本模块集中存放航天轨道力学计算所需的物理常数，所有数值均以 SI 单位
（米、千克、秒、弧度）给出，并附来源注释。便于整个 aerospace_agent 包
统一引用，避免不同模块出现数值不一致。

约定
-----
* ``mu``  = 引力参数 G*M，单位 m^3 s^-2
* ``R``   = 平均/赤道半径，单位 m
* ``a``   = 轨道半长轴，单位 m
* 角度内部一律使用弧度；提供 ``DEG2RAD`` 用于度->弧度转换。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 角度 / 时间换算因子
# ---------------------------------------------------------------------------
# 1 度 = pi/180 弧度。
DEG2RAD: float = math.pi / 180.0
# 1 天 = 86400 秒（平太阳日）。
DAY2SEC: float = 86400.0
SEC2DAY: float = 1.0 / DAY2SEC

# ---------------------------------------------------------------------------
# 地球 (Earth)
# ---------------------------------------------------------------------------
# 地球引力参数 mu_earth = G * M_earth。
# 来源: GRACE 卫星解算的地球质心引力常数 (IERS Technical Note 36, 2010)，
#       与 CODATA 2014 的 G 配合后给出的标准值。
MU_EARTH: float = 3.986004418e14  # m^3 s^-2

# 地球赤道半径 (WGS-84 参考椭球 semi-major axis)。
# 来源: NIMA WGS-84, 1997。
R_EARTH: float = 6378.137e3  # m

# 地球二阶带谐系数 J2（扁率项），用于长期摄动 (J2 摄动) 计算。
# 来源: EGM96 重力场模型。
J2_EARTH: float = 1.08263e-3  # dimensionless

# 地球自转角速度（相对惯性系，恒星日）。
# 来源: IERS Conventions 2010。
OMEGA_EARTH: float = 7.2921159e-5  # rad s^-1

# ---------------------------------------------------------------------------
# 月球 (Moon)
# ---------------------------------------------------------------------------
# 月球引力参数 mu_moon = G * M_moon。
# 来源: JPL DE430 历表 / Lunar Prospector LP150Q 模型。
MU_MOON: float = 4.9048695e12  # m^3 s^-2

# 月球平均半径 (IAU 2009)。
R_MOON: float = 1737.4e3  # m

# 月球绕地球公转轨道半长轴（地月平均距离）。
# 来源: IAU 2009 平均距离 384400 km。
A_MOON: float = 384400e3  # m

# 月球轨道偏心率。
E_MOON: float = 0.0549  # dimensionless

# 月球轨道相对黄道面的倾角。
# 来源: IAU 2009 (5.145°)。
I_MOON_DEG: float = 5.145  # deg
I_MOON: float = I_MOON_DEG * DEG2RAD  # rad

# 月球恒星公转周期（恒星月）。
# 来源: 27.32166 天 (恒星月)；此处取常用 27.3216 天。
T_MOON: float = 27.3216 * DAY2SEC  # s

# 月球平均轨道角速度（圆轨道近似）。
OMEGA_MOON: float = 2.0 * math.pi / T_MOON  # rad s^-1

# ---------------------------------------------------------------------------
# 太阳 (Sun)
# ---------------------------------------------------------------------------
# 日心引力常数 mu_sun = G * M_sun。
# 来源: JPL DE430 历表给出的日心引力常数 (Pitjeva & Standish 2009)。
MU_SUN: float = 1.32712440018e20  # m^3 s^-2

# 天文单位。
AU: float = 1.495978707e11  # m

# ---------------------------------------------------------------------------
# 月球引力作用球 (Sphere of Influence, SOI)
# ---------------------------------------------------------------------------
# 拉普拉斯 SOI 半径公式（三体问题中次天体引力主导区域）:
#
#   r_SOI = a * ( m_secondary / m_primary )^(2/5)
#
# 推导要点: 在限制性三体问题中，比较次天体对航天器的引力加速度与
# 主天体潮汐 (差分) 加速度，令两者量级相等即得 SOI 边界；幂指数 2/5
# 来自加速度比 ~ r^3 与 ~ R^2 的平衡。对地月系统:
#
#   r_SOI_moon = a_moon * (mu_moon / mu_earth)^(2/5)
#
# 代入数值: (4.9048695e12 / 3.986004418e14)^0.4 ≈ 0.1722
#          r_SOI ≈ 384400 km * 0.1722 ≈ 66180 km ≈ 6.62e7 m
R_SOI_MOON: float = A_MOON * (MU_MOON / MU_EARTH) ** (2.0 / 5.0)  # m

# ---------------------------------------------------------------------------
# 便于集中访问的数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Body:
    """天体常数集合（只读）。"""
    name: str
    mu: float          # m^3 s^-2
    radius: float      # m
    extra: dict


EARTH = _Body("Earth", MU_EARTH, R_EARTH, {"J2": J2_EARTH, "omega": OMEGA_EARTH})
MOON = _Body(
    "Moon",
    MU_MOON,
    R_MOON,
    {
        "a": A_MOON,
        "e": E_MOON,
        "i": I_MOON,
        "T": T_MOON,
        "omega": OMEGA_MOON,
        "r_soi": R_SOI_MOON,
    },
)
SUN = _Body("Sun", MU_SUN, AU, {})


class U:
    """方便属性式访问的常数容器。

    用法::

        from aerospace_agent.physics.constants import U
        print(U.mu_earth, U.R_earth, U.R_soi_moon)

    所有属性均为 SI 单位的浮点数。
    """

    # 角度/时间
    deg2rad = DEG2RAD
    day2sec = DAY2SEC
    sec2day = SEC2DAY

    # 地球
    mu_earth = MU_EARTH
    MU_EARTH = MU_EARTH
    R_earth = R_EARTH
    R_EARTH = R_EARTH
    J2 = J2_EARTH
    J2_EARTH = J2_EARTH
    omega_earth = OMEGA_EARTH
    OMEGA_EARTH = OMEGA_EARTH

    # 月球
    mu_moon = MU_MOON
    MU_MOON = MU_MOON
    R_moon = R_MOON
    R_MOON = R_MOON
    a_moon = A_MOON
    A_MOON = A_MOON
    e_moon = E_MOON
    E_MOON = E_MOON
    i_moon = I_MOON
    I_MOON = I_MOON
    i_moon_deg = I_MOON_DEG
    T_moon = T_MOON
    T_MOON = T_MOON
    omega_moon = OMEGA_MOON
    OMEGA_MOON = OMEGA_MOON
    R_soi_moon = R_SOI_MOON
    R_SOI_MOON = R_SOI_MOON

    # 太阳
    mu_sun = MU_SUN
    MU_SUN = MU_SUN
    AU = AU


# 字典形式（便于批量遍历或序列化）
CONSTANTS: dict = {
    "mu_earth": MU_EARTH,
    "R_earth": R_EARTH,
    "J2": J2_EARTH,
    "omega_earth": OMEGA_EARTH,
    "mu_moon": MU_MOON,
    "R_moon": R_MOON,
    "a_moon": A_MOON,
    "e_moon": E_MOON,
    "i_moon": I_MOON,
    "T_moon": T_MOON,
    "omega_moon": OMEGA_MOON,
    "R_soi_moon": R_SOI_MOON,
    "mu_sun": MU_SUN,
    "AU": AU,
    "deg2rad": DEG2RAD,
    "day2sec": DAY2SEC,
}


if __name__ == "__main__":
    # 自测: 打印关键常数并核对月球 SOI 半径
    print("=== aerospace_agent.physics.constants ===")
    print(f"mu_earth      = {MU_EARTH:.6e} m^3/s^2")
    print(f"R_earth       = {R_EARTH:.3f} m")
    print(f"J2            = {J2_EARTH:.6e}")
    print(f"omega_earth   = {OMEGA_EARTH:.7e} rad/s")
    print(f"mu_moon       = {MU_MOON:.7e} m^3/s^2")
    print(f"R_moon        = {R_MOON:.1f} m")
    print(f"a_moon        = {A_MOON:.3e} m")
    print(f"e_moon        = {E_MOON}")
    print(f"i_moon        = {I_MOON_DEG} deg")
    print(f"T_moon        = {T_MOON:.1f} s = {T_MOON/DAY2SEC:.4f} day")
    print(f"mu_sun        = {MU_SUN:.6e} m^3/s^2")
    print(f"R_soi_moon    = {R_SOI_MOON:.3e} m = {R_SOI_MOON/1e3:.1f} km")
    assert abs(R_SOI_MOON - 66.2e6) < 1e6, "月球 SOI 半径应在 ~66200 km 附近"
    print("SOI 校验通过 (~66200 km).")
