"""开普勒轨道力学 (Kepler orbital mechanics)。

本模块实现经典轨道根数 (classical orbital elements) 与状态向量
(位置 r、速度 v) 之间的相互转换，以及开普勒方程求解。

经典轨道根数
-------------
    a     半长轴 (semi-major axis)            [m]
    e     偏心率 (eccentricity)                [-]
    i     倾角 (inclination)                   [rad]
    raan  升交点赤经 (RAAN, capital Omega)     [rad]
    argp  近地点幅角 (arg of perigee, omega)   [rad]
    nu    真近点角 (true anomaly, nu)          [rad]

所有角度内部以弧度表示，所有长度以米表示（SI）。

坐标系约定
-----------
* ECI (Earth-Centered Inertial) 惯性系：原点在地心，x 轴指向春分点，
  z 轴指向北极，y 轴成右手系。
* PQW (Perifocal) 近焦点坐标系：P 轴指向近地点，Q 轴在轨道面内沿运动
  方向超前 90°，W 轴沿角动量方向 (P x Q)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH


# ---------------------------------------------------------------------------
# 旋转矩阵
# ---------------------------------------------------------------------------
def R1(theta: float) -> np.ndarray:
    """绕 x 轴旋转 theta 的主动旋转矩阵 (3x3)。

    R_x(theta) = [[1,       0,        0],
                  [0, cos t, -sin t],
                  [0, sin t,  cos t]]
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ], dtype=float)


def R2(theta: float) -> np.ndarray:
    """绕 y 轴旋转 theta 的主动旋转矩阵。"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def R3(theta: float) -> np.ndarray:
    """绕 z 轴旋转 theta 的主动旋转矩阵。

    R_z(theta) = [[cos t, -sin t, 0],
                  [sin t,  cos t, 0],
                  [    0,      0, 1]]

    注: 这里使用“数学标准”主动旋转约定 (上三角为 -sin)，它与
    Vallado 教材的 R3 (下三角为 -sin) 在转角上差一个符号。两者最终
    得到的 PQW->ECI 合成矩阵完全等价 (见 perifocal_to_eci 注释)。
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def perifocal_to_eci_matrix(raan: float, i: float, argp: float) -> np.ndarray:
    """PQW -> ECI 的旋转矩阵。

    推导
    -----
    PQW 坐标系相对 ECI 的姿态由三个欧拉角确定 (3-1-3 序列):

        1. 先绕 ECI 的 z 轴 (K) 转 RAAN -> x 轴指向升交点 (node line);
        2. 再绕新的 x 轴 (node line) 转 i  -> 得到轨道平面;
        3. 再绕新的 z 轴 (轨道面法向 W) 转 argp -> x 轴指向近地点 (P).

    故把 PQW 中向量变到 ECI，应做反向旋转的合成 (主动旋转观点):

        r_ECI = R_z(RAAN) * R_x(i) * R_z(argp) * r_PQW

    即::

        M = R3(raan) @ R1(i) @ R3(argp)

    验证 (标量情形): 取 raan=i=argp=0 得单位阵；取 i=0, argp=0 得
    R3(raan)，即平面内绕 z 旋转，符合赤道圆轨道的几何。
    """
    return R3(raan) @ R1(i) @ R3(argp)


# ---------------------------------------------------------------------------
# 状态向量 <-> 轨道根数
# ---------------------------------------------------------------------------
def elements_to_state(a, e, i, raan, argp, nu, mu=MU_EARTH):
    """由经典轨道根数计算 ECI 状态向量 (r, v)。

    推导
    -----
    1) 轨道方程 (圆锥曲线, 焦点在地心):

           p
       r = ---------   ,   p = a (1 - e^2)  (半通径)
           1 + e cos nu

       位置在 PQW 系中:
           r_PQW = ( r cos nu,  r sin nu,  0 )

    2) 速度由角动量 h = sqrt(mu * p) 守恒导出:
           v_r   = (mu/h) e sin nu                       (径向分量)
           v_t   = (mu/h) (1 + e cos nu)                 (横向分量)
       在 PQW 系:
           v_PQW = ( -sqrt(mu/p) sin nu,  sqrt(mu/p)(e + cos nu),  0 )

       推导: dr/dt = (dr/dnu)(dnu/dt), dnu/dt = h/r^2;
             dr/dnu = p e sin nu / (1+e cos nu)^2 = (r^2/h)(mu/h) e sin nu ...
             整理即得上式。

    3) PQW -> ECI:
           r_ECI = M @ r_PQW,   v_ECI = M @ v_PQW
       其中 M = R3(raan) R1(i) R3(argp) (见 perifocal_to_eci_matrix).

    返回
    -----
    r : (3,) ndarray  位置 [m]
    v : (3,) ndarray  速度 [m/s]
    """
    a = float(a); e = float(e)
    p = a * (1.0 - e * e)  # 半通径 [m]

    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)

    # PQW 系位置 / 速度
    r_mag = p / (1.0 + e * cos_nu)
    r_pqw = np.array([r_mag * cos_nu, r_mag * sin_nu, 0.0])

    sqrt_mu_p = math.sqrt(mu / p)
    v_pqw = np.array([
        -sqrt_mu_p * sin_nu,
        sqrt_mu_p * (e + cos_nu),
        0.0,
    ])

    M = perifocal_to_eci_matrix(raan, i, argp)
    r = M @ r_pqw
    v = M @ v_pqw
    return r, v


def state_to_elements(r, v, mu=MU_EARTH, eps=1e-12):
    """由 ECI 状态向量 (r, v) 计算经典轨道根数。

    采用角动量向量法与偏心率向量法 (Bate-Mueller-White / Curtis)。

    推导
    -----
    设 r = |r|, v = |v|。

    1) 比能量 (specific energy):
           eps = v^2/2 - mu/r
       半长轴 (轨道为椭圆时 eps<0):
           a = - mu / (2 eps) = 1 / (2/r - v^2/mu)

    2) 比角动量向量:
           h = r x v,   h = |h|
       倾角:
           cos i = h_z / h
       升交点向量 (node vector, 指向升交点):
           n = K x h = (-h_y, h_x, 0),   n = |n|

    3) 偏心率向量 (Laplace-Runge-Lenz):
           e_vec = (v x h)/mu - r/r
       其模即偏心率 e = |e_vec|，方向指向近地点。

    4) RAAN (capital Omega):
           cos RAAN = n_x / n
       若 n_y < 0 则 RAAN 在 (pi, 2pi)，取 2pi - arccos。
       (赤道轨道 n=0 时 RAAN 未定义，置 0。)

    5) 近地点幅角 (argp, omega):
           cos argp = (n . e_vec) / (n e)
       若 e_vec_z < 0 则取 2pi - arccos。
       (圆轨道 e=0 时 argp 未定义，置 0。)

    6) 真近点角 (nu):
           cos nu = (e_vec . r) / (e r)
       若 r.v >= 0 (远离近地点) 取正，否则取 2pi - arccos。

    返回: dict 含 a, e, i, raan, argp, nu (以及 h_mag, energy)
    """
    r = np.asarray(r, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    r_mag = float(np.linalg.norm(r))
    v_mag = float(np.linalg.norm(v))

    # 角动量
    h = np.cross(r, v)
    h_mag = float(np.linalg.norm(h))

    # 节点向量 n = K x h, K=(0,0,1) -> n = (-h_y, h_x, 0)
    n = np.array([-h[1], h[0], 0.0])
    n_mag = float(np.linalg.norm(n))

    # 偏心率向量
    e_vec = np.cross(v, h) / mu - r / r_mag
    e = float(np.linalg.norm(e_vec))

    # 能量与半长轴
    energy = 0.5 * v_mag * v_mag - mu / r_mag
    if abs(2.0 / r_mag - v_mag * v_mag / mu) > eps:
        a = 1.0 / (2.0 / r_mag - v_mag * v_mag / mu)
    else:
        a = math.inf  # 抛物线

    # 倾角
    i = math.acos(np.clip(h[2] / h_mag, -1.0, 1.0)) if h_mag > eps else 0.0

    # RAAN
    if n_mag > eps:
        raan = math.acos(np.clip(n[0] / n_mag, -1.0, 1.0))
        if n[1] < 0.0:
            raan = 2.0 * math.pi - raan
    else:
        raan = 0.0  # 赤道轨道，未定义

    # argp
    if n_mag > eps and e > eps:
        argp = math.acos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1.0, 1.0))
        if e_vec[2] < 0.0:
            argp = 2.0 * math.pi - argp
    else:
        argp = 0.0  # 圆/赤道，未定义

    # nu
    if e > eps:
        nu = math.acos(np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0))
        if np.dot(r, v) < 0.0:
            nu = 2.0 * math.pi - nu
    else:
        nu = 0.0  # 圆轨道，未定义

    return {
        "a": a,
        "e": e,
        "i": i,
        "raan": raan,
        "argp": argp,
        "nu": nu,
        "h_mag": h_mag,
        "energy": energy,
    }


# ---------------------------------------------------------------------------
# 开普勒方程求解 (Kepler's equation)
# ---------------------------------------------------------------------------
def kepler_solve(M, e, tol=1e-12, max_iter=100):
    """牛顿迭代解开普勒方程  M = E - e sin E   (椭圆, 0 <= e < 1)。

    推导
    -----
    平近点角 M = n (t - t_p) 与偏近点角 E 的关系:
        M = E - e sin E
    其中 n = sqrt(mu/a^3) 为平均运动。该方程对 E 无显式解，用牛顿迭代:

        f(E) = E - e sin E - M
        f'(E) = 1 - e cos E
        E_{k+1} = E_k - f(E_k)/f'(E_k)

    初值取 E0 = M + e sin M (对 e 不太大时收敛快, 近似三阶)。
    支持广播: M 可为标量或 array；返回同形状。
    """
    M = np.asarray(M, dtype=float)
    e = float(e)
    if not (0.0 <= e < 1.0):
        raise ValueError(f"kepler_solve 仅支持椭圆 (0<=e<1), 得到 e={e}")

    E = M + e * np.sin(M)            # 初值
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def true_anomaly(E, e):
    """由偏近点角 E 求真近点角 nu (椭圆)。

    推导
    -----
    由几何关系 (a, b 为半长/半短轴):
        cos nu = (cos E - e) / (1 - e cos E)
        sin nu = (sqrt(1-e^2) sin E) / (1 - e cos E)
    等价的半角公式:
        tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    用 atan2 同时确定象限:
        nu = 2 * atan2( sqrt(1+e) sin(E/2),  sqrt(1-e) cos(E/2) )
    支持 array 输入。
    """
    E = np.asarray(E, dtype=float)
    e = float(e)
    return 2.0 * np.arctan2(
        math.sqrt(1.0 + e) * np.sin(E / 2.0),
        math.sqrt(1.0 - e) * np.cos(E / 2.0),
    )


def eccentric_anomaly(nu, e):
    """由真近点角 nu 求偏近点角 E (椭圆)。"""
    nu = np.asarray(nu, dtype=float)
    e = float(e)
    return 2.0 * np.arctan2(
        math.sqrt(1.0 - e) * np.sin(nu / 2.0),
        math.sqrt(1.0 + e) * np.cos(nu / 2.0),
    )


# ---------------------------------------------------------------------------
# KeplerOrbit 类
# ---------------------------------------------------------------------------
@dataclass
class KeplerOrbit:
    """由经典轨道根数定义的开普勒轨道。

    参数
    -----
    a, e, i, raan, argp, nu0 : 经典根数 (SI, 弧度)
        nu0 为历元 t0 时刻的真近点角。
    mu : 引力参数 (默认地球)
    t0 : 历元 (默认 0)；仅作记录用，传播以相对时间 t 计算。

    方法
    -----
    state_vector(t) : 返回 t 时刻 (相对 t0) 的 (r, v)
    period()        : 轨道周期 (椭圆)
    """

    a: float
    e: float
    i: float
    raan: float
    argp: float
    nu0: float
    mu: float = MU_EARTH
    t0: float = 0.0

    def __post_init__(self):
        if not (0.0 <= self.e < 1.0):
            raise ValueError("KeplerOrbit 仅支持椭圆 (0<=e<1)")
        # 历元时刻的偏近点角 / 平近点角，用于时间传播
        self._E0 = float(eccentric_anomaly(self.nu0, self.e))
        self._M0 = self._E0 - self.e * math.sin(self._E0)
        self._n = math.sqrt(self.mu / self.a ** 3)  # 平均运动 [rad/s]

    def period(self) -> float:
        """轨道周期 T = 2 pi sqrt(a^3 / mu) [s]。"""
        return 2.0 * math.pi * math.sqrt(self.a ** 3 / self.mu)

    def mean_motion(self) -> float:
        """平均运动 n = sqrt(mu / a^3) [rad/s]。"""
        return self._n

    def true_anomaly_at(self, t: float) -> float:
        """t 时刻 (相对 t0) 的真近点角。"""
        M = self._M0 + self._n * t          # 平近点角线性增长
        E = kepler_solve(M, self.e)
        nu = true_anomaly(E, self.e)
        return float(np.atleast_1d(nu)[0])

    def state_vector(self, t: float):
        """返回 t 时刻 (相对 t0) 的 ECI 状态 (r, v)。

        推导
        -----
        平近点角随时间线性增长: M(t) = M0 + n*t;
        由 M 解开普勒方程得 E; 由 E 得 nu;
        最后用 elements_to_state 把 (a,e,i,raan,argp,nu) 转为 (r,v)。
        """
        nu = self.true_anomaly_at(t)
        return elements_to_state(
            self.a, self.e, self.i, self.raan, self.argp, nu, self.mu
        )


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.physics.kepler 自测 ===")

    mu = MU_EARTH
    # 圆轨道: a=7000 km, e=0, i=30°
    a = 7000e3
    e = 0.0
    i = math.radians(30.0)
    raan = math.radians(45.0)
    argp = 0.0
    nu0 = 0.0

    r, v = elements_to_state(a, e, i, raan, argp, nu0, mu)
    print(f"圆轨道 r = {np.round(r/1e3,3)} km,  |r|={np.linalg.norm(r)/1e3:.3f} km")
    print(f"        v = {np.round(v,3)} m/s, |v|={np.linalg.norm(v):.3f} m/s")
    assert abs(np.linalg.norm(r) - a) < 1e-6
    assert abs(np.linalg.norm(v) - math.sqrt(mu / a)) < 1e-6

    # 往返校验: state -> elements -> state
    els = state_to_elements(r, v, mu)
    print("反演根数:", {k: round(math.degrees(els[k]), 6) if k in ("i","raan","argp","nu") else round(els[k],6)
                        for k in ("a","e","i","raan","argp","nu")})
    assert abs(els["a"] - a) < 1e-3
    assert abs(els["e"] - e) < 1e-9
    assert abs(els["i"] - i) < 1e-9
    assert abs(els["raan"] - raan) < 1e-9

    # 椭圆轨道传播校验: 一个周期后回到原位
    a2 = 8000e3; e2 = 0.3; i2 = math.radians(20); raan2 = math.radians(10)
    argp2 = math.radians(30); nu0_2 = math.radians(15)
    orb = KeplerOrbit(a2, e2, i2, raan2, argp2, nu0_2, mu=mu)
    T = orb.period()
    r0, v0 = orb.state_vector(0.0)
    rT, vT = orb.state_vector(T)   # 整周期后应回到原位
    print(f"椭圆 a={a2/1e3}km e={e2} 周期 T={T/3600:.3f} h")
    print(f"  |r(0)|={np.linalg.norm(r0)/1e3:.3f} km, |r(T)|={np.linalg.norm(rT)/1e3:.3f} km")
    assert np.allclose(r0, rT, atol=1e-3), "整周期应回到原位"
    assert np.allclose(v0, vT, atol=1e-6)

    # 开普勒方程校验
    M = math.radians(60.0); e_t = 0.5
    E = float(kepler_solve(M, e_t))
    print(f"开普勒方程: M={math.degrees(M):.1f}°, e={e_t} -> E={math.degrees(E):.6f}°")
    assert abs((E - e_t * math.sin(E)) - M) < 1e-10
    nu_t = float(true_anomaly(E, e_t))
    print(f"  -> nu = {math.degrees(nu_t):.6f}°")
    print("kepler 自测全部通过.")
