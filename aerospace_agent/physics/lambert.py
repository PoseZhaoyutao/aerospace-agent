"""Lambert 问题求解器 (universal variable formulation)。

Lambert 问题: 给定两个位置矢量 r1, r2 与飞行时间 dt, 在中心引力 mu 下
求连接两点的开普勒轨道 (即两端的速度 v1, v2)。

本模块采用 Bate-Mueller-White《Fundamentals of Astrodynamics》第7章 /
Curtis《Orbital Mechanics》Algorithm 5.2 的 *普适变量法*，统一处理
椭圆、抛物线、双曲线，并支持多圈 (revolutions)。

核心推导
----------
记 r1 = |r1|, r2 = |r2|,  转移角 dtheta (按方向取 0~2pi)。

几何参数 A 的推导 (用真近点角增量 dnu = dtheta, 因为 Lambert 问题中两位置
矢量的夹角即为转移轨道上的真近点角增量):

由椭圆/双曲线几何关系, 在 perifocal 系中
    r1 = p/(1 + e cos nu1) ,   r2 = p/(1 + e cos nu2),   nu2 - nu1 = dtheta
其中 p 为半通径。由三角恒等式可证 (详见 Curtis 推导):

    A = sqrt(r1 * r2) * sqrt(1 + cos(dtheta))
      = sqrt(2 * r1 * r2) * cos(dtheta / 2)

  - dtheta in (0, pi):  cos(dtheta/2) > 0  => A > 0  (短路径 / 顺行直达)
  - dtheta = pi:        cos(dtheta/2) = 0  => A = 0  (Hohmann 180° 奇异)
  - dtheta in (pi,2pi): cos(dtheta/2) < 0  => A < 0  (长路径 / 绕远)

  *注*: 一些文献写 A = sin(dtheta) sqrt(r1 r2) / (1 - cos(dtheta)),
        利用恒等式 sin(dtheta)/(1-cos(dtheta)) = cot(dtheta/2) = cos(dtheta/2)/sin(dtheta/2),
        等价于 sqrt(r1 r2)(1+cos dtheta)/sin dtheta, 经化简即为上面的形式
        (但 sin(dtheta) 形式在 dtheta=pi 处分母奇异, 需特殊处理; 本实现用 cos 形式更鲁棒,
        唯一奇异点为 dtheta=pi)。直接用 sin(dtheta)*sqrt(r1 r2) 是 *错误* 的,
        仅在圆轨道 90° 转移时数值巧合成立。

引入 z = alpha * chi^2 (alpha 为轨道比能量相关量, chi 普适变量),
Stumpff 函数 C(z), S(z)。定义:

    y(z) = r1 + r2 + A * (z S(z) - 1) / sqrt(C(z))     # 要求 C(z) > 0, y > 0

普适形式的 Lambert 时间方程:

    sqrt(mu) dt = (y(z)/C(z))^(3/2) S(z) + A sqrt(y(z))

即

    F(z) = (y/C)^(3/2) S + A sqrt(y) - sqrt(mu) dt = 0

解出 z 后, 令 chi = sqrt(y / C), 用 Lagrange 系数 f, g:

    f    = 1 - y / r1
    g    = A sqrt(y / mu)
    gdot = 1 - y / r2

    v1 = (r2 - f r1) / g
    v2 = (gdot r2 - r1) / g

z 的取值范围 (椭圆):
    0 圈:   z in (-inf, 4 pi^2)            (含双曲线 z<0)
    N 圈:   z in (4 pi^2 N^2, 4 pi^2 (N+1)^2)
其中 z = (Delta E)^2 (Delta E 为偏近点角增量), 故多圈区间按 (2 pi N)^2 划分。
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional

import numpy as np
from scipy.optimize import brentq

from .constants import MU_EARTH
from .two_body import stumpff


_FOUR_PI2 = 4.0 * math.pi * math.pi


class LambertResult(NamedTuple):
    """Lambert 求解完整结果。"""
    v1: np.ndarray            # (3,) 起点速度 [m/s]
    v2: np.ndarray            # (3,) 终点速度 [m/s]
    dv_total: float           # |v1| + |v2|, 参考总速度量 [m/s]
    z: float                  # 求得的普适变量参数 z = alpha chi^2
    y: float                  # y(z)
    transfer_angle: float     # 转移角 [rad]
    revolutions: int          # 圈数


def _transfer_angle(r1, r2, r1mag, r2mag, direction):
    """计算转移角 dtheta (按方向取 0~2pi)。"""
    cos_dt = float(np.dot(r1, r2)) / (r1mag * r2mag)
    cos_dt = max(-1.0, min(1.0, cos_dt))
    dtheta = math.acos(cos_dt)  # [0, pi]
    cz = float(np.cross(r1, r2)[2])
    if direction == "prograde":
        if cz < 0.0:
            dtheta = 2.0 * math.pi - dtheta
    elif direction == "retrograde":
        if cz > 0.0:
            dtheta = 2.0 * math.pi - dtheta
    else:
        raise ValueError("direction 必须为 'prograde' 或 'retrograde'")
    return dtheta


def _y_and_F(z, A, r1mag, r2mag, mu, dt):
    """计算 y(z) 与 F(z); 不可行或溢出返回 (None, None)。

    F(z) = (y/C)^(3/2) S + A sqrt(y) - sqrt(mu) dt
    要求 C(z) > 0 且 y(z) > 0。对大负 z (双曲线远场) cosh 会溢出,
    用 sq > 700 阈值判溢出。
    """
    if z > 0.0:
        sq = math.sqrt(z)
        C = (1.0 - math.cos(sq)) / z
        S = (sq - math.sin(sq)) / (sq * sq * sq)
    elif z < 0.0:
        sq = math.sqrt(-z)
        if sq > 700.0:
            return None, None  # cosh/sinh 溢出
        C = (math.cosh(sq) - 1.0) / (-z)
        S = (math.sinh(sq) - sq) / (sq * sq * sq)
    else:
        C = 0.5
        S = 1.0 / 6.0
    if C <= 0.0:
        return None, None
    sqrtC = math.sqrt(C)
    y = r1mag + r2mag + A * (z * S - 1.0) / sqrtC
    if y <= 0.0:
        return None, None
    F = (y / C) ** 1.5 * S + A * math.sqrt(y) - math.sqrt(mu) * dt
    return y, F


def _sample_z_grid(z_lo, z_hi, revolutions):
    """构造 z 采样网格。

    0 圈区间 (z_lo, 4 pi^2): 含负半轴 (双曲线) 与 (0, 4 pi^2) (椭圆)。
    负半轴用 *几何* 采样 (z 从 -1e-3 到 z_lo, 因 F 在大负 z 处变化平缓),
    正半轴用 *密集线性* 采样 (0 到 4 pi^2, 因椭圆根分布密集且 F 在 4 pi^2 处->+inf)。
    多圈区间 (4 pi^2 N^2, 4 pi^2 (N+1)^2) 用线性采样。
    """
    if revolutions == 0:
        # 负半轴: 几何采样 z = -10^k, k 从 -3 到 log10(-z_lo)
        neg_pts = []
        if z_lo < -1e-3:
            k_hi = math.log10(-z_lo)
            ks = np.linspace(-3.0, k_hi, 80)
            neg_pts = list(-10.0 ** ks)
            neg_pts.reverse()  # 从小 (大负) 到大 (接近 0)
        # 正半轴: 密集线性 0 -> 4 pi^2
        pos_pts = list(np.linspace(1e-6, _FOUR_PI2 - 1e-6, 500))
        # 加 z=0
        return neg_pts + [0.0] + pos_pts
    else:
        return list(np.linspace(z_lo, z_hi, 800))


def _find_roots_in_range(z_lo, z_hi, A, r1mag, r2mag, mu, dt, revolutions=0):
    """在 [z_lo, z_hi] 内扫描 F(z) 的符号变化, 用 brentq 精化。

    返回 *物理有效* 的根: 满足 |F(z)| / (sqrt(mu) dt) < 1e-6
    (即时间残差远小于总飞行时间)。这可滤除大负 z 处 cosh/sinh 溢出导致的
    假 sign-change (catastrophic cancellation 产生的伪根)。
    """
    zs = _sample_z_grid(z_lo, z_hi, revolutions)
    Fs = []
    for z in zs:
        _, F = _y_and_F(z, A, r1mag, r2mag, mu, dt)
        Fs.append(F)

    sqrt_mu_dt = math.sqrt(mu) * dt
    # 物理根的时间残差应 << sqrt(mu)*dt; 阈值取 1e-6 倍 (相对).
    # 伪根的 |F| 通常 ~ 1e30 以上 (cosh 溢出量级), 远超此阈值。
    f_tol = 1e-6 * sqrt_mu_dt

    roots: List[float] = []
    for i in range(len(zs) - 1):
        a, b = zs[i], zs[i + 1]
        fa, fb = Fs[i], Fs[i + 1]
        if fa is None or fb is None:
            continue
        if not (math.isfinite(fa) and math.isfinite(fb)):
            continue
        if fa == 0.0:
            roots.append(a)
            continue
        if fa * fb < 0.0:
            def _f(z, A=A, r1mag=r1mag, r2mag=r2mag, mu=mu, dt=dt):
                _, F = _y_and_F(z, A, r1mag, r2mag, mu, dt)
                return F if F is not None else float("nan")
            try:
                root = brentq(_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=200)
            except ValueError:
                continue
            # 验证根的物理有效性: |F(root)| 必须足够小
            _, F_root = _y_and_F(root, A, r1mag, r2mag, mu, dt)
            if F_root is None or not math.isfinite(F_root):
                continue
            if abs(F_root) > f_tol:
                continue  # 伪根 (cosh 溢出导致的假 sign-change)
            roots.append(root)
    return roots


def lambert_universal(r1, r2, dt, mu, direction="prograde", revolutions=0,
                      tol=1e-10, multi_rev_solution="short") -> LambertResult:
    """普适变量法 Lambert 求解 (返回完整结果)。

    参数
    -----
    r1, r2     : (3,) 起止位置 [m]
    dt         : 飞行时间 [s], 必须 > 0
    mu         : 引力参数 [m^3/s^2]
    direction  : 'prograde' (顺行, 默认) 或 'retrograde' (逆行)
    revolutions: 完整圈数 N (>=0)。N=0 为 0~180° 或 0~360° 直达;
                 N>=1 为多圈解 (可能有 0/1/2 个解)
    multi_rev_solution : 'short' 或 'long', 多圈时选择短/长解
    tol        : 容差
    返回
    -----
    LambertResult(v1, v2, dv_total, z, y, transfer_angle, revolutions)
    """
    if dt <= 0:
        raise ValueError("dt 必须 > 0")
    r1 = np.asarray(r1, dtype=float).ravel()
    r2 = np.asarray(r2, dtype=float).ravel()
    r1mag = float(np.linalg.norm(r1))
    r2mag = float(np.linalg.norm(r2))
    if r1mag == 0.0 or r2mag == 0.0:
        raise ValueError("r1, r2 不能为零向量")

    dtheta = _transfer_angle(r1, r2, r1mag, r2mag, direction)

    # 用 A = sqrt(2 r1 r2) cos(dtheta/2)。奇异点为 dtheta = pi (cos(pi/2)=0),
    # 即 Hohmann 180° 直线转移; 此时 g = A sqrt(y/mu) -> 0, v1,v2 发散。
    # 微扰 dtheta 偏离 pi 一小量 (1e-8 rad), 误差可忽略。
    if abs(dtheta - math.pi) < 1e-8:
        dtheta = math.pi - 1e-8

    # A = sqrt(r1 r2 (1 + cos dtheta)) = sqrt(2 r1 r2) cos(dtheta/2)
    # 短路径 (dtheta<pi): A>0; 长路径 (dtheta>pi): A<0 (自动处理方向)。
    A = math.sqrt(2.0 * r1mag * r2mag) * math.cos(dtheta / 2.0)

    # z 的有效区间
    if revolutions == 0:
        # z in (-inf, 4 pi^2): 含双曲线 (z<0)、抛物线 (z=0)、椭圆 (0<z<4pi^2)
        # 负半轴下界取 -1000: 对应 sqrt(-z) ~ 31.6, cosh(31.6) ~ 4e13 (可控),
        # 覆盖 v_inf 高达 ~300 km/s 的极端双曲线 (alpha = -v_inf^2/mu ~ -2.5e-10,
        # chi ~ sqrt(mu)*dt/r ~ 6e4, z = alpha*chi^2 ~ -90)。更负的 z 会让
        # cosh/sinh 溢出导致 F(z) 假 sign-change (catastrophic cancellation),
        # 由 _find_roots_in_range 的 |F| 过滤兜底。
        z_lo = -1000.0
        z_hi = _FOUR_PI2 - 1e-6
    else:
        N = int(revolutions)
        z_lo = _FOUR_PI2 * N * N + 1e-6
        z_hi = _FOUR_PI2 * (N + 1) * (N + 1) - 1e-6

    roots = _find_roots_in_range(z_lo, z_hi, A, r1mag, r2mag, mu, dt,
                                 revolutions=revolutions)

    if not roots:
        raise RuntimeError(
            f"Lambert 无解: 在 z 区间内未找到根 (dt={dt}s 可能对该圈数不可行)"
        )

    # 多圈时可能有两个解, 按需选择
    if len(roots) == 1:
        z = roots[0]
    else:
        # 两个根: 较小的 z 对应较短的飞行时间 (短解), 较大 z 对应长解
        roots.sort()
        z = roots[0] if multi_rev_solution == "short" else roots[-1]

    y, _ = _y_and_F(z, A, r1mag, r2mag, mu, dt)
    C, S = stumpff(z)

    # Lagrange 系数 (推导见模块 docstring)
    f = 1.0 - y / r1mag
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2mag

    if abs(g) < 1e-14:
        raise RuntimeError("Lambert 奇异: g ≈ 0 (转移角接近 0 或 pi)")

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    dv_total = float(np.linalg.norm(v1)) + float(np.linalg.norm(v2))

    return LambertResult(
        v1=v1, v2=v2, dv_total=dv_total, z=z, y=y,
        transfer_angle=dtheta, revolutions=revolutions,
    )


def solve_lambert(r1, r2, dt, mu, direction="prograde", revolutions=0,
                  tol=1e-10):
    """Lambert 问题求解 (便捷接口, 返回 (v1, v2))。

    用法::

        v1, v2 = solve_lambert(r1, r2, dt, mu)

    如需完整结果 (含 z, dv_total 等), 请用 :func:`lambert_universal`。
    """
    res = lambert_universal(r1, r2, dt, mu, direction, revolutions, tol)
    return res.v1, res.v2


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.physics.lambert 自测 ===")
    mu = MU_EARTH

    # 黄金标准校验: 用两体传播得到 (r0,v0) -> (r1,v_at_r1), 再用 Lambert 反求。
    # solve_lambert(r0, r1, dt) 返回 (v_at_r0, v_at_r1) -- 即起点和终点的速度。
    # 故 v_at_r0 应与 v0 一致, v_at_r1 应与传播得到的 v_at_r1 一致。
    from .two_body import propagate_two_body

    # 用 a=20000km, e=0.4 的椭圆, 周期 T ~ 7.82h; dt in [1,3,5,7]h 均 < T (0 圈)
    a = 20000e3
    e = 0.4
    T_orb = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    print(f"椭圆 a={a/1e3}km e={e}, 周期 T={T_orb/3600:.4f} h")
    r_per = a * (1 - e)
    v_per = math.sqrt(mu * (2.0 / r_per - 1.0 / a))
    r0 = np.array([r_per, 0.0, 0.0])
    v0 = np.array([0.0, v_per, 0.0])
    # 取若干 dt (均 < T, 0 圈), 验证 Lambert 反求
    for dt_h in [1.0, 3.0, 5.0, 7.0]:
        dt = dt_h * 3600.0
        r1, v_at_r1_true = propagate_two_body(r0, v0, mu, dt)
        v_at_r0_lam, v_at_r1_lam = solve_lambert(r0, r1, dt, mu, direction="prograde")
        # 起点速度应与 v0 一致, 终点速度应与 v_at_r1_true 一致
        err_v0 = np.linalg.norm(v_at_r0_lam - v0)
        err_v1 = np.linalg.norm(v_at_r1_lam - v_at_r1_true)
        # 校验: 用 v_at_r0_lam 重新传播 dt 应到 r1
        r1_check, v_at_r1_check = propagate_two_body(r0, v_at_r0_lam, mu, dt)
        err_r = np.linalg.norm(r1_check - r1)
        print(f"  dt={dt_h:.0f}h: |v0 误差|={err_v0:.3e}, |v1 误差|={err_v1:.3e} m/s, "
              f"|r1 反传误差|={err_r:.3e} m")
        assert err_v0 < 1e-3, f"v0 误差过大: {err_v0}"
        assert err_v1 < 1e-3, f"v1 误差过大: {err_v1}"
        assert err_r < 1e-3

    # 多圈 Lambert 校验: dt > T (1 圈), 用 revolutions=1 求解
    # 注: 避开 dt = 1.5T (会使卫星停在 180° 对侧, dtheta=pi 奇异); 用 1.3T
    dt_multi = 1.3 * T_orb  # 1.3 圈
    r1m, v_at_r1m_true = propagate_two_body(r0, v0, mu, dt_multi)
    v_at_r0m, v_at_r1m = solve_lambert(r0, r1m, dt_multi, mu,
                                       direction="prograde", revolutions=1)
    err_v0m = np.linalg.norm(v_at_r0m - v0)
    err_v1m = np.linalg.norm(v_at_r1m - v_at_r1m_true)
    r1m_check, _ = propagate_two_body(r0, v_at_r0m, mu, dt_multi)
    err_rm = np.linalg.norm(r1m_check - r1m)
    print(f"  多圈 dt=1.5T: |v0 误差|={err_v0m:.3e}, |v1 误差|={err_v1m:.3e} m/s, "
          f"|r1 反传误差|={err_rm:.3e} m")
    assert err_v0m < 1e-3, f"多圈 v0 误差过大: {err_v0m}"
    assert err_v1m < 1e-3, f"多圈 v1 误差过大: {err_v1m}"
    assert err_rm < 1e-3

    # 双曲线 Lambert: 高能量短时间转移
    r_a = np.array([8000e3, 0.0, 0.0])
    r_b = np.array([0.0, 8000e3, 0.0])
    dt = 1800.0  # 30 分钟, 较短 -> 高能量
    v1, v2 = solve_lambert(r_a, r_b, dt, mu, direction="prograde")
    # 反传校验
    r_b_check, v_b_check = propagate_two_body(r_a, v1, mu, dt)
    print(f"  双曲线短转移: |r_b 反传误差|={np.linalg.norm(r_b_check-r_b):.3e} m, "
          f"|v_b 误差|={np.linalg.norm(v_b_check-v2):.3e} m/s")
    assert np.linalg.norm(r_b_check - r_b) < 1e-3
    assert np.linalg.norm(v_b_check - v2) < 1e-6

    # retrograde 校验
    v1_ret, v2_ret = solve_lambert(r_a, r_b, dt, mu, direction="retrograde")
    r_b_ret, _ = propagate_two_body(r_a, v1_ret, mu, dt)
    print(f"  retrograde: |r_b 反传误差|={np.linalg.norm(r_b_ret-r_b):.3e} m")
    assert np.linalg.norm(r_b_ret - r_b) < 1e-3

    print("lambert 自测全部通过.")
