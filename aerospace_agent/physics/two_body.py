"""二体问题传播 (Two-body propagation)。

本模块用 *普适变量法* (universal variable formulation, Bate-Mueller-White
第4章 / Vallado 第2章) 实现二体状态的精确传播，统一处理椭圆、抛物线、
双曲线轨道，且支持 dt 为负 (向后传播)。

普适变量法核心
----------------
给定 t0 时刻状态 (r0, v0), 求 t0+dt 时刻状态 (r, v)。令

    alpha = 2/|r0| - |v0|^2/mu        # 轨道比能量的 -2 倍
                                       # alpha>0: 椭圆; =0: 抛物线; <0: 双曲线

    vr0   = (r0 . v0) / |r0|           # 初始径向速度

引入普适变量 chi (普适偏近点角), 以及 Stumpff 函数 C(z), S(z),
z = alpha * chi^2。普适开普勒方程为:

    sqrt(mu) dt = (r0 vr0 / sqrt(mu)) chi^2 C(z)
                  + (1 - alpha r0) chi^3 S(z)
                  + r0 chi

用牛顿法解出 chi 后, 由 f, g 级数得到新状态:

    f      = 1 - chi^2 / r0 * C(z)
    g      = dt - chi^3 / sqrt(mu) * S(z)
    r_vec  = f r0 + g v0
    r      = |r_vec|
    fdot   = sqrt(mu)/r0 * (z S(z) - 1) / r * chi
    gdot   = 1 - chi^2 / r * C(z)
    v_vec  = fdot r0 + gdot v0

满足 f*gdot - fdot*g = 1 (相空间保面积)。
"""

from __future__ import annotations

import math

import numpy as np

from .constants import MU_EARTH


def stumpff(z: float):
    """Stumpff 函数 C(z), S(z)。

    定义 (级数):
        C(z) = sum_{k>=0} (-z)^k / (2k+2)! = (1 - cos sqrt(z))/z   (z>0)
                                                = (cosh sqrt(-z) - 1)/(-z)  (z<0)
                                                = 1/2                       (z=0)
        S(z) = sum_{k>=0} (-z)^k / (2k+3)! = (sqrt(z) - sin sqrt(z))/sqrt(z)^3  (z>0)
                                                = (sinh sqrt(-z) - sqrt(-z))/sqrt(-z)^3 (z<0)
                                                = 1/6                                  (z=0)

    这些函数在全实轴解析，连接椭圆 (z>0) / 抛物线 (z=0) / 双曲线 (z<0)。
    """
    z = float(z)
    if z > 0.0:
        sq = math.sqrt(z)
        C = (1.0 - math.cos(sq)) / z
        S = (sq - math.sin(sq)) / (sq ** 3)
    elif z < 0.0:
        sq = math.sqrt(-z)
        C = (math.cosh(sq) - 1.0) / (-z)
        S = (math.sinh(sq) - sq) / (sq ** 3)
    else:
        C = 1.0 / 2.0
        S = 1.0 / 6.0
    return C, S


def _stumpff_arr(z):
    """支持标量与 array 输入的 Stumpff 函数。"""
    z = np.asarray(z, dtype=float)
    C = np.empty_like(z)
    S = np.empty_like(z)
    pos = z > 0.0
    neg = z < 0.0
    zero = ~(pos | neg)

    if np.any(pos):
        zp = np.sqrt(z[pos])
        C[pos] = (1.0 - np.cos(zp)) / z[pos]
        S[pos] = (zp - np.sin(zp)) / zp ** 3
    if np.any(neg):
        zn = np.sqrt(-z[neg])
        C[neg] = (np.cosh(zn) - 1.0) / (-z[neg])
        S[neg] = (np.sinh(zn) - zn) / zn ** 3
    C[zero] = 0.5
    S[zero] = 1.0 / 6.0
    return C, S


def _universal_F(chi, alpha, r0mag, vr0, mu, dt):
    """普适开普勒方程 F(chi) 及其导数 F'(chi)。

    F(chi)  = (r0 vr0/sqrt(mu)) chi^2 C(z) + (1 - alpha r0) chi^3 S(z)
              + r0 chi - sqrt(mu) dt,    z = alpha chi^2
    F'(chi) = (r0 vr0/sqrt(mu)) chi (1 - z S) + (1 - alpha r0) chi^2 C + r0

    用到 Stumpff 恒等式 d/dchi[chi^2 C] = chi(1 - z S), d/dchi[chi^3 S] = chi^2 C.
    对双曲线大 |z| 用 sqrt(-z) > 700 阈值避免 cosh/sinh 溢出, 返回 inf.
    """
    sqrt_mu = math.sqrt(mu)
    z = alpha * chi * chi
    if z > 0.0:
        sq = math.sqrt(z)
        C = (1.0 - math.cos(sq)) / z
        S = (sq - math.sin(sq)) / (sq * sq * sq)
    elif z < 0.0:
        sq = math.sqrt(-z)
        if sq > 700.0:
            return math.inf, math.inf  # 溢出, 视为过大
        C = (math.cosh(sq) - 1.0) / (-z)
        S = (math.sinh(sq) - sq) / (sq * sq * sq)
    else:
        C = 0.5
        S = 1.0 / 6.0

    A = r0mag * vr0 / sqrt_mu
    B = 1.0 - alpha * r0mag
    chi2 = chi * chi
    chi3 = chi2 * chi
    F = A * chi2 * C + B * chi3 * S + r0mag * chi - sqrt_mu * dt
    Fp = A * chi * (1.0 - z * S) + B * chi2 * C + r0mag
    return F, Fp


def propagate_two_body(r0, v0, mu, dt, tol=1e-12):
    """普适变量法二体传播 (Bate-Mueller-White)。

    用 brentq 在 chi 上求根 (主根), 对椭圆/抛物线/双曲线统一鲁棒处理,
    支持 dt 为负 (向后传播)。

    参数
    -----
    r0, v0 : (3,) 初始位置 [m] / 速度 [m/s]
    mu     : 引力参数 [m^3/s^2]
    dt     : 传播时间 [s]，可为负 (向后传播)
    tol    : 求根容差
    返回
    -----
    r, v : (3,) 传播后位置 / 速度
    """
    from scipy.optimize import brentq

    r0 = np.asarray(r0, dtype=float).ravel()
    v0 = np.asarray(v0, dtype=float).ravel()
    r0mag = float(np.linalg.norm(r0))
    v0mag = float(np.linalg.norm(v0))
    sqrt_mu = math.sqrt(mu)

    if dt == 0.0:
        return r0.copy(), v0.copy()

    alpha = 2.0 / r0mag - v0mag * v0mag / mu
    vr0 = float(np.dot(r0, v0)) / r0mag

    # F(chi) 单调 (F' = r > 0); 主根 chi 与 dt 同号.
    # F(0) = -sqrt(mu) dt. 对 dt>0 需 chi_hi 使 F>0; dt<0 对称.
    def Ffun(chi):
        F, _ = _universal_F(chi, alpha, r0mag, vr0, mu, dt)
        return F

    # 初始 bracket 上界 (按抛物线近似 chi ~ sqrt(mu) dt / r0 量级)
    chi_hi = sqrt_mu * abs(dt) / r0mag
    if chi_hi < 1.0:
        chi_hi = 1.0

    # 1) 若上界溢出 (双曲线大 dt), 折半直到有限
    while not math.isfinite(Ffun(chi_hi)):
        chi_hi *= 0.5
        if chi_hi < 1e-6:
            break

    # 2) 调整上界使 F(chi_hi) 与 F(0) 异号
    F0 = -sqrt_mu * dt  # F(0)
    if dt > 0:
        # 需要 F(chi_hi) > 0
        Fhi = Ffun(chi_hi)
        # 若仍 <0, 翻倍; 若溢出, 折半回退
        it = 0
        while Fhi < 0 and it < 200:
            chi_hi *= 2.0
            Fhi = Ffun(chi_hi)
            it += 1
            if not math.isfinite(Fhi):
                # 翻倍溢出: 退回并在 [prev, chi_hi] 内必存在有限正 F
                while not math.isfinite(Ffun(chi_hi)):
                    chi_hi *= 0.5
                Fhi = Ffun(chi_hi)
                break
        chi_root = brentq(Ffun, 0.0, chi_hi, xtol=tol, rtol=tol, maxiter=200)
    else:
        chi_lo = -chi_hi
        Flo = Ffun(chi_lo)
        it = 0
        while Flo > 0 and it < 200:
            chi_lo *= 2.0
            Flo = Ffun(chi_lo)
            it += 1
            if not math.isfinite(Flo):
                while not math.isfinite(Ffun(chi_lo)):
                    chi_lo *= 0.5
                Flo = Ffun(chi_lo)
                break
        chi_root = brentq(Ffun, chi_lo, 0.0, xtol=tol, rtol=tol, maxiter=200)

    chi = chi_root

    # f, g 表达式
    z = alpha * chi * chi
    C, S = stumpff(z)
    f = 1.0 - chi * chi / r0mag * C
    g = dt - chi ** 3 / sqrt_mu * S
    r_vec = f * r0 + g * v0
    r = float(np.linalg.norm(r_vec))

    # Lagrange 系数导数 (守恒律 f*gdot - fdot*g = 1)
    fdot = sqrt_mu / r0mag / r * (z * S - 1.0) * chi
    gdot = 1.0 - chi * chi / r * C

    v_vec = fdot * r0 + gdot * v0
    return r_vec, v_vec


def propagate_orbit(r0, v0, mu, t_array, method="universal"):
    """沿时间数组传播二体轨道。

    参数
    -----
    r0, v0   : (3,) 初始状态 (对应 t_array 的第一个时刻或 t=0)
    mu       : 引力参数
    t_array  : (N,) 时间序列 [s]，相对初始时刻
    method   : 'universal' (默认, 普适变量) 或 'ivp' (scipy solve_ivp)

    返回
    -----
    states : (N, 6) ndarray, 每行 [rx,ry,rz,vx,vy,vz]
    """
    t_array = np.asarray(t_array, dtype=float).ravel()
    N = t_array.size
    out = np.zeros((N, 6), dtype=float)

    if method == "universal":
        for k, t in enumerate(t_array):
            r, v = propagate_two_body(r0, v0, mu, t)
            out[k, :3] = r
            out[k, 3:] = v
    elif method == "ivp":
        # 用 scipy.integrate.solve_ivp 做二体积分 (备选方案)
        from scipy.integrate import solve_ivp

        def _dyn(t, y):
            r = y[:3]
            rr = np.linalg.norm(r)
            ax = -mu * r / rr ** 3
            return np.concatenate([y[3:], ax])

        sol = solve_ivp(
            _dyn,
            (t_array[0], t_array[-1]),
            np.concatenate([r0, v0]),
            t_eval=t_array,
            rtol=1e-10,
            atol=1e-12,
            method="DOP853",
        )
        out = sol.y.T
    else:
        raise ValueError(f"未知 method: {method}")
    return out


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.physics.two_body 自测 ===")
    mu = MU_EARTH

    # 椭圆轨道: a=8000km, e=0.3
    a = 8000e3
    e = 0.3
    r_p = a * (1.0 - e)  # 近地点距离
    v_p = math.sqrt(mu * (2.0 / r_p - 1.0 / a))  # 近地点速度 (vis-viva)
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, v_p, 0.0])

    T = 2.0 * math.pi * math.sqrt(a ** 3 / mu)
    print(f"椭圆 a={a/1e3}km e={e}, 周期 T={T/3600:.4f} h")

    # 整周期后应回到原位
    r1, v1 = propagate_two_body(r0, v0, mu, T)
    print(f"传播 T 后 |r-r0|={np.linalg.norm(r1-r0):.3e} m, |v-v0|={np.linalg.norm(v1-v0):.3e} m/s")
    assert np.allclose(r1, r0, atol=1e-3)
    assert np.allclose(v1, v0, atol=1e-6)

    # 半周期后应在远地点
    r2, v2 = propagate_two_body(r0, v0, mu, T / 2.0)
    r_apo_expected = a * (1.0 + e)
    print(f"传播 T/2 后 |r|={np.linalg.norm(r2)/1e3:.3f} km (期望远地点 {r_apo_expected/1e3:.3f} km)")
    assert abs(np.linalg.norm(r2) - r_apo_expected) < 1e-3

    # 向后传播 (dt<0): 传播 +T/3 再 -T/3 应回到原位
    r3, v3 = propagate_two_body(r0, v0, mu, T / 3.0)
    r4, v4 = propagate_two_body(r3, v3, mu, -T / 3.0)
    print(f"前 T/3 再回 -T/3: |r-r0|={np.linalg.norm(r4-r0):.3e} m")
    assert np.allclose(r4, r0, atol=1e-3)
    assert np.allclose(v4, v0, atol=1e-6)

    # 双曲线轨道校验 (alpha<0): 近地点 r=7000km, v_inf=3 km/s
    # 双曲线比能量 eps = v_inf^2 / 2, 故任意时刻应满足 v^2 - 2 mu/r = v_inf^2
    rp = 7000e3
    v_inf = 3000.0
    v_per = math.sqrt(v_inf ** 2 + 2.0 * mu / rp)
    rh0 = np.array([rp, 0.0, 0.0])
    vh0 = np.array([0.0, v_per, 0.0])
    # 传播到远距离: 检验能量守恒 v^2 - 2 mu/r == v_inf^2 (精确)
    # 双曲线比能量 eps = v_inf^2 / 2, 故任意时刻 v^2 - 2 mu/r = v_inf^2 (守恒)
    rh_far, vh_far = propagate_two_body(rh0, vh0, mu, 5 * 86400.0)
    r_far = float(np.linalg.norm(rh_far))
    v_far = float(np.linalg.norm(vh_far))
    energy_err = abs((v_far ** 2 - 2.0 * mu / r_far) - v_inf ** 2)
    print(f"双曲线远场 r={r_far/1e9:.3f}e9 m, |v|={v_far:.2f} m/s, "
          f"能量误差={energy_err:.3e} (应精确守恒 v_inf^2={v_inf**2:.0f})")
    assert energy_err < 1e-3, "双曲线能量应精确守恒"
    # 传播到极远距离 (60 天): 仍检验能量守恒 (而非 |v|->v_inf,
    # 因 60 天时 r 仍有限 ~1.6e10 m, 2 mu/r ~ 5e4, |v| 比 v_inf 大约 8 m/s)
    rh_asy, vh_asy = propagate_two_body(rh0, vh0, mu, 60 * 86400.0)
    r_asy = float(np.linalg.norm(rh_asy))
    v_asy = float(np.linalg.norm(vh_asy))
    energy_err2 = abs((v_asy ** 2 - 2.0 * mu / r_asy) - v_inf ** 2)
    print(f"双曲线 60 天 r={r_asy/1e9:.3f}e9 m, |v|={v_asy:.3f} m/s "
          f"(理论 sqrt(v_inf^2+2mu/r)={math.sqrt(v_inf**2+2*mu/r_asy):.3f}), "
          f"能量误差={energy_err2:.3e}")
    assert energy_err2 < 1e-3, "双曲线能量应精确守恒"

    # 守恒律 f*gdot - fdot*g = 1 (通过再反推隐式校验)

    # 轨迹数组
    ts = np.linspace(0, T, 50)
    states = propagate_orbit(r0, v0, mu, ts)
    print(f"轨迹数组 shape={states.shape}, 首末点距离差={np.linalg.norm(states[-1,:3]-r0):.3e} m")
    assert states.shape == (50, 6)
    print("two_body 自测全部通过.")
