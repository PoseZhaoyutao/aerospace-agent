"""aerospace_agent.physics — 航天轨道力学物理计算模块。

子模块:
    constants        物理常数 (SI)
    kepler           开普勒轨道力学 (根数 / 状态向量 / 开普勒方程)
    two_body         二体问题传播 (普适变量法)
    lambert          Lambert 问题求解器
    patched_conic    拼凑圆锥近似 (地月转移核心)
    orbital_maneuvers 轨道机动 (Hohmann / 双椭圆 / 改变倾角 / 相位)
    moon_transfer    地月转移轨道设计
"""

from __future__ import annotations

# 物理常数
from .constants import (
    U,
    MU_EARTH,
    R_EARTH,
    J2_EARTH,
    OMEGA_EARTH,
    MU_MOON,
    R_MOON,
    A_MOON,
    E_MOON,
    I_MOON,
    T_MOON,
    OMEGA_MOON,
    MU_SUN,
    R_SOI_MOON,
    DEG2RAD,
    DAY2SEC,
    CONSTANTS,
    EARTH,
    MOON,
    SUN,
)

# 开普勒力学
from .kepler import (
    KeplerOrbit,
    elements_to_state,
    state_to_elements,
    kepler_solve,
    true_anomaly,
    eccentric_anomaly,
    perifocal_to_eci_matrix,
    R1,
    R2,
    R3,
)

# 二体传播
from .two_body import (
    propagate_two_body,
    propagate_orbit,
    stumpff,
)

# Lambert
from .lambert import (
    solve_lambert,
    lambert_universal,
)

# 拼凑圆锥
from .patched_conic import (
    PatchedConic,
)

# 轨道机动
from .orbital_maneuvers import (
    hohmann_transfer_delta_v,
    bielliptic_transfer,
    plane_change_delta_v,
    phasing_orbit,
)

# 地月转移
from .moon_transfer import (
    MoonTransfer,
)

__all__ = [
    # constants
    "U", "MU_EARTH", "R_EARTH", "J2_EARTH", "OMEGA_EARTH",
    "MU_MOON", "R_MOON", "A_MOON", "E_MOON", "I_MOON", "T_MOON", "OMEGA_MOON",
    "MU_SUN", "R_SOI_MOON", "DEG2RAD", "DAY2SEC", "CONSTANTS",
    "EARTH", "MOON", "SUN",
    # kepler
    "KeplerOrbit", "elements_to_state", "state_to_elements",
    "kepler_solve", "true_anomaly", "eccentric_anomaly",
    "perifocal_to_eci_matrix", "R1", "R2", "R3",
    # two_body
    "propagate_two_body", "propagate_orbit", "stumpff",
    # lambert
    "solve_lambert", "lambert_universal",
    # patched_conic
    "PatchedConic",
    # maneuvers
    "hohmann_transfer_delta_v", "bielliptic_transfer",
    "plane_change_delta_v", "phasing_orbit",
    # moon_transfer
    "MoonTransfer",
]
