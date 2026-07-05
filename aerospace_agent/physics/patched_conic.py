"""拼凑圆锥近似法 (Patched Conic Approximation)。

地月转移的核心简化方法: 把三体问题拆成若干段二体问题, 在引力作用球
(Sphere of Influence, SOI) 边界做速度匹配。对地月转移:

    1. 地心段 (geocentric):  航天器在地球引力下沿双曲线 (或大椭圆) 逃逸,
       在地球 SOI 边界具有双曲剩余速度 v_inf (相对地球).
    2. 月心段 (selenocentric): 进入月球 SOI 后, 以 v_inf_moon (相对月球)
       沿双曲线接近月球, 在近月点制动捕获.

关键公式
----------
* Vis-viva 方程 (能量守恒):

      v^2 = mu ( 2/r - 1/a )

  其中 a 为半长轴 (双曲线 a<0). 由此:
      - 任意点速度: v(r) = sqrt( mu (2/r - 1/a) )
      - 双曲剩余速度: v_inf^2 = v^2 - 2 mu/r  =  -mu/a   (a<0, 故 v_inf^2>0)

* 比能量 (specific orbital energy):

      eps = v^2/2 - mu/r = -mu/(2a)
      v_inf^2 = 2 eps

* 双曲线近地点速度 (periselene): 设近月点半径 r_p, 则

      v_p^2 = v_inf^2 + 2 mu_moon / r_p     (由 vis-viva, a = -mu_moon/v_inf^2)

* 圆轨道速度: v_circ = sqrt(mu/r).

* 月球 SOI 半径 (Laplace): r_SOI = a_moon (mu_moon/mu_earth)^(2/5).

* 捕获 delta-V (单次脉冲, 双曲线 -> 圆轨道):

      dv_capture = v_p - v_circ
"""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    MU_EARTH,
    MU_MOON,
    R_SOI_MOON,
)


class PatchedConic:
    """拼凑圆锥近似法工具集 (地月转移)。

    参数
    -----
    mu_earth, mu_moon : 引力参数 [m^3/s^2]
    r_soi             : 月球引力作用球半径 [m]
    """

    def __init__(self, mu_earth=MU_EARTH, mu_moon=MU_MOON, r_soi=R_SOI_MOON):
        self.mu_earth = mu_earth
        self.mu_moon = mu_moon
        self.r_soi = r_soi

    # ------------------------------------------------------------------
    # 地心段: 双曲逃逸
    # ------------------------------------------------------------------
    def earth_escape(self, r0, v0, mu_earth=None):
        """从地球引力场逃逸的双曲剩余速度与 C3。

        推导
        -----
        比能量 eps = v0^2/2 - mu_earth/r0;  对双曲线 eps > 0。
        双曲剩余速度 (相对地球, 在无穷远):
            v_inf^2 = 2 eps = v0^2 - 2 mu_earth / r0
        特征能量 C3 = v_inf^2 (发射能量指标, 单位 m^2/s^2)。
        v_inf 方向近似为 v0 方向 (远场渐近)。

        返回 dict: v_inf(标量), C3, v_inf_vec(向量), energy, is_hyperbolic
        """
        if mu_earth is None:
            mu_earth = self.mu_earth
        r0 = np.asarray(r0, dtype=float).ravel()
        v0 = np.asarray(v0, dtype=float).ravel()
        r0mag = float(np.linalg.norm(r0))
        v0mag = float(np.linalg.norm(v0))
        energy = 0.5 * v0mag * v0mag - mu_earth / r0mag  # 比能量
        v_inf_sq = 2.0 * energy  # = v0^2 - 2 mu/r
        is_hyp = v_inf_sq > 0.0
        v_inf = math.sqrt(max(v_inf_sq, 0.0))
        v_inf_vec = (v0 / v0mag) * v_inf if v0mag > 0 else np.zeros(3)
        return {
            "v_inf": v_inf,            # 双曲剩余速度标量 [m/s]
            "v_inf_vec": v_inf_vec,    # 双曲剩余速度向量 (近似沿 v0)
            "C3": v_inf_sq,            # 特征能量 [m^2/s^2]
            "energy": energy,          # 比能量 [J/kg]
            "is_hyperbolic": is_hyp,
            "r0": r0mag,
            "v0": v0mag,
        }

    # ------------------------------------------------------------------
    # 月心段: 双曲接近 + 捕获
    # ------------------------------------------------------------------
    def moon_arrival(self, v_inf_moon, r_perilune, mu_moon=None):
        """以 v_inf_moon 进入月球 SOI, 在近月点 r_perilune 处的捕获机动。

        推导
        -----
        双曲线 vis-viva: v^2 = v_inf^2 + 2 mu_moon / r  (a = -mu_moon/v_inf^2).
        近月点 (r = r_perilune):
            v_perilune = sqrt( v_inf_moon^2 + 2 mu_moon / r_perilune )
        制动到近月圆轨道:
            v_circ = sqrt( mu_moon / r_perilune )
            dv_capture = v_perilune - v_circ

        返回 dict: v_perilune, v_circ, dv_capture, v_inf, ...
        """
        if mu_moon is None:
            mu_moon = self.mu_moon
        v_inf = float(v_inf_moon)
        r_p = float(r_perilune)
        v_perilune = math.sqrt(v_inf * v_inf + 2.0 * mu_moon / r_p)
        v_circ = math.sqrt(mu_moon / r_p)
        dv_capture = v_perilune - v_circ
        # 双曲线半长轴 (负), 偏心率
        a_hyp = -mu_moon / (v_inf * v_inf) if v_inf > 0 else -math.inf
        # 近月点 r_p = a(1-e) -> e = 1 - r_p/a  (a<0)
        e_hyp = 1.0 - r_p / a_hyp if a_hyp != 0 and math.isfinite(a_hyp) else math.inf
        return {
            "v_inf": v_inf,
            "v_perilune": v_perilune,
            "v_circ": v_circ,
            "dv_capture": dv_capture,
            "a_hyp": a_hyp,
            "e_hyp": e_hyp,
            "r_perilune": r_p,
        }

    # ------------------------------------------------------------------
    # SOI 判定
    # ------------------------------------------------------------------
    def sphere_of_influence_check(self, r_sc, r_moon, r_soi=None):
        """判断航天器是否进入月球引力作用球。

        |r_sc - r_moon| < r_soi  => True
        """
        if r_soi is None:
            r_soi = self.r_soi
        r_sc = np.asarray(r_sc, dtype=float).ravel()
        r_moon = np.asarray(r_moon, dtype=float).ravel()
        dist = float(np.linalg.norm(r_sc - r_moon))
        return dist < r_soi, dist

    # ------------------------------------------------------------------
    # 边界速度匹配 (地心 -> 月心)
    # ------------------------------------------------------------------
    def soi_velocity_match(self, v_geo_at_soi, v_moon):
        """SOI 边界地心速度 -> 月心速度的转换。

        推导
        -----
        在 SOI 边界, 航天器相对地心的速度 v_geo (地心段远场近似为 v_inf_earth)
        与月球相对地心速度 v_moon 之差, 即为航天器相对月球的速度:

            v_inf_moon = v_geo - v_moon     (向量)

        其模 |v_inf_moon| 决定月心双曲线的能量。
        """
        v_geo = np.asarray(v_geo_at_soi, dtype=float).ravel()
        v_moon = np.asarray(v_moon, dtype=float).ravel()
        v_inf_moon_vec = v_geo - v_moon
        return v_inf_moon_vec, float(np.linalg.norm(v_inf_moon_vec))


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.physics.patched_conic 自测 ===")
    pc = PatchedConic()

    # 地球逃逸: LEO 200km 圆轨道 + 速度增量到双曲
    r_leo = 6378.137e3 + 200e3
    v_circ_leo = math.sqrt(MU_EARTH / r_leo)
    # 给一个 v_inf = 3 km/s 的逃逸
    v_inf_target = 3000.0
    v_esc_leo = math.sqrt(v_inf_target ** 2 + 2 * MU_EARTH / r_leo)
    r0 = np.array([r_leo, 0.0, 0.0])
    v0 = np.array([0.0, v_esc_leo, 0.0])
    esc = pc.earth_escape(r0, v0)
    print(f"LEO 逃逸: v_inf={esc['v_inf']:.2f} m/s (目标 {v_inf_target}), "
          f"C3={esc['C3']:.3e} m^2/s^2, hyperbolic={esc['is_hyperbolic']}")
    assert abs(esc["v_inf"] - v_inf_target) < 1e-6
    assert abs(esc["C3"] - v_inf_target ** 2) < 1e-3

    # 月球到达: v_inf_moon = 830 m/s, 近月点 100km
    r_lmo = 1737.4e3 + 100e3
    arr = pc.moon_arrival(830.0, r_lmo)
    print(f"月球到达: v_perilune={arr['v_perilune']:.2f} m/s, "
          f"v_circ={arr['v_circ']:.2f} m/s, dv_capture={arr['dv_capture']:.2f} m/s")
    assert arr["v_perilune"] > arr["v_circ"]
    assert arr["e_hyp"] > 1.0

    # SOI 判定
    r_moon_pos = np.array([384400e3, 0.0, 0.0])
    inside, dist = pc.sphere_of_influence_check(
        np.array([384400e3 - 50000e3, 0.0, 0.0]), r_moon_pos
    )
    print(f"SOI 判定: 距月球 {dist/1e3:.0f} km, 进入 SOI={inside}")
    assert inside is True
    inside2, dist2 = pc.sphere_of_influence_check(
        np.array([384400e3 - 100000e3, 0.0, 0.0]), r_moon_pos
    )
    assert inside2 is False

    # 边界速度匹配
    v_geo = np.array([0.0, 200.0, 0.0])
    v_moon = np.array([0.0, 1022.0, 0.0])
    vinf_vec, vinf_mag = pc.soi_velocity_match(v_geo, v_moon)
    print(f"SOI 速度匹配: v_inf_moon = {vinf_mag:.2f} m/s (期望 {1022-200:.0f})")
    assert abs(vinf_mag - (1022 - 200)) < 1e-6

    print("patched_conic 自测全部通过.")
