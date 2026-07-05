"""地月转移轨道设计 (Earth-Moon transfer design)。

本模块是整个 physics 包的演示核心, 实现:

    * :meth:`MoonTransfer.hohmann_transfer`   — Hohmann 近似地月转移 (含月球运动)
    * :meth:`MoonTransfer.launch_window`      — 发射窗口 (相位角) 分析, scipy 寻优
    * :meth:`MoonTransfer.design_trajectory`  — 完整拼凑圆锥 (patched conic) 轨迹
    * :meth:`MoonTransfer.porkchop_plot_data` — porkchop 图 (C3/飞行时间) 网格

简化假设 (便于演示, 在方法注释中说明):
    * 月球绕地做圆周运动 (半径 a_moon, 角速度 omega_moon), 位于 xy 平面;
      倾角默认忽略 (可在 _moon_state 中开启).
    * 月球在 start_date 时刻位于 +x 轴 (作为相位参考历元).
    * 停泊轨道 (LEO) 近地点方向固定在惯性 +x; 发射窗口即月球相位匹配时刻.
    * 各段均用二体 / 拼凑圆锥近似, 不做完整三体数值积分.

单位: 全部 SI (m, s, m/s, rad).
"""

from __future__ import annotations

import datetime
import math
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import (
    A_MOON,
    DAY2SEC,
    DEG2RAD,
    MU_EARTH,
    MU_MOON,
    OMEGA_MOON,
    R_EARTH,
    R_MOON,
    R_SOI_MOON,
    T_MOON,
)
from .patched_conic import PatchedConic


def _to_date(d) -> datetime.datetime:
    """把 date/datetime/str 转为 datetime.datetime。"""
    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, datetime.date):
        return datetime.datetime(d.year, d.month, d.day)
    if isinstance(d, str):
        return datetime.datetime.fromisoformat(d)
    raise TypeError(f"不支持日期类型: {type(d)}")


def _wrap_pi(angle: float) -> float:
    """把角度归一化到 (-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MoonTransfer:
    """地月转移轨道设计器。"""

    def __init__(
        self,
        mu_earth=MU_EARTH,
        mu_moon=MU_MOON,
        R_earth=R_EARTH,
        R_moon=R_MOON,
        a_moon=A_MOON,
        omega_moon=OMEGA_MOON,
        T_moon=T_MOON,
        r_soi=R_SOI_MOON,
    ):
        self.mu_earth = mu_earth
        self.mu_moon = mu_moon
        self.R_earth = R_earth
        self.R_moon = R_moon
        self.a_moon = a_moon
        self.omega_moon = omega_moon
        self.T_moon = T_moon
        self.r_soi = r_soi
        self.pc = PatchedConic(mu_earth, mu_moon, r_soi)

    # ------------------------------------------------------------------
    # 月球星历 (简化圆轨道)
    # ------------------------------------------------------------------
    def _moon_state(self, t_sec, include_inclination=False):
        """t_sec 时刻月球的地心位置与速度 (圆轨道近似)。

        约定: t_sec=0 时月球在 +x 轴, 沿 +y 方向运动 (顺行)。
        """
        theta = self.omega_moon * t_sec
        r = self.a_moon
        pos = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
        v = self.omega_moon * self.a_moon
        vel = np.array([-v * math.sin(theta), v * math.cos(theta), 0.0])
        if include_inclination:
            # 绕 x 轴抬起倾角 i_moon (近似, 忽略节点进动)
            from .constants import I_MOON
            c, s = math.cos(I_MOON), math.sin(I_MOON)
            pos = np.array([pos[0], pos[1] * c, pos[1] * s])
            vel = np.array([vel[0], vel[1] * c, vel[1] * s])
        return pos, vel

    # ------------------------------------------------------------------
    # 1. Hohmann 转移
    # ------------------------------------------------------------------
    def hohmann_transfer(self, altitude_leo=200e3, altitude_lmo=100e3):
        """Hohmann 近似地月转移 (考虑月球运动).

        推导
        -----
        1) 转移椭圆半长轴 (地心, 近地点 LEO, 远地点月球轨道):
               a_tr = (r_leo + r_moon) / 2
           其中 r_leo = R_earth + altitude_leo,  r_moon = a_moon (月球轨道半径).

        2) Vis-viva 给出各点速度:
               v(r, a) = sqrt( mu_earth (2/r - 1/a) )
           - LEO 圆轨道速度:        v_leo    = sqrt(mu_earth / r_leo)
           - 转移椭圆近地点速度:    v_tr_per = v(r_leo, a_tr)
           - 转移椭圆远地点速度:    v_tr_apo = v(r_moon, a_tr)
           - 月球轨道速度 (圆):     v_moon   = sqrt(mu_earth / r_moon)

        3) LEO 离轨脉冲 (切向加速):
               dv1 = v_tr_per - v_leo

        4) 月球相对速度 (远地点处, Hohmann 近似下 v_tr_apo 与 v_moon 共线反向):
               v_inf_moon = v_moon - v_tr_apo
           月球特征能量 C3_moon = v_inf_moon^2.

        5) 近月点制动 (LOI): 转移到 r_lmo 圆轨道.
           月心双曲线近月点速度:
               v_perilune = sqrt( v_inf_moon^2 + 2 mu_moon / r_lmo )
           近月圆轨道速度:
               v_circ_lmo = sqrt( mu_moon / r_lmo )
               dv2 = v_perilune - v_circ_lmo

        6) 飞行时间 = 转移椭圆半周期:
               t_flight = pi sqrt(a_tr^3 / mu_earth)   (约 5 天)

        返回 dict 含全部中间量与 dv1, dv2, dv_total, t_flight, C3 等.
        """
        r_leo = self.R_earth + altitude_leo
        r_lmo = self.R_moon + altitude_lmo
        r_moon = self.a_moon

        def vvisc(r, a):
            return math.sqrt(self.mu_earth * (2.0 / r - 1.0 / a))

        a_tr = 0.5 * (r_leo + r_moon)
        v_leo = math.sqrt(self.mu_earth / r_leo)
        v_tr_per = vvisc(r_leo, a_tr)
        v_tr_apo = vvisc(r_moon, a_tr)
        v_moon = math.sqrt(self.mu_earth / r_moon)

        dv1 = v_tr_per - v_leo
        # Hohmann 近似: v_tr_apo 与 v_moon 反向共线, 月球追上航天器
        v_inf_moon = v_moon - v_tr_apo
        C3_moon = v_inf_moon * v_inf_moon
        # 地心转移轨道比能量 (负, 束缚)
        energy_geo = -self.mu_earth / (2.0 * a_tr)

        # 近月点制动
        v_perilune = math.sqrt(v_inf_moon ** 2 + 2.0 * self.mu_moon / r_lmo)
        v_circ_lmo = math.sqrt(self.mu_moon / r_lmo)
        dv2 = v_perilune - v_circ_lmo

        dv_total = dv1 + dv2
        # 飞行时间 = 半周期
        T_tr = 2.0 * math.pi * math.sqrt(a_tr ** 3 / self.mu_earth)
        t_flight = T_tr / 2.0

        e_tr = (r_moon - r_leo) / (r_moon + r_leo)

        return {
            "r_leo": r_leo,
            "r_lmo": r_lmo,
            "r_moon": r_moon,
            "a_transfer": a_tr,
            "e_transfer": e_tr,
            "v_leo": v_leo,
            "v_tr_perigee": v_tr_per,
            "v_tr_apogee": v_tr_apo,
            "v_moon": v_moon,
            "v_inf_moon": v_inf_moon,
            "C3_moon": C3_moon,
            "energy_geo": energy_geo,
            "v_perilune": v_perilune,
            "v_circ_lmo": v_circ_lmo,
            "dv1": dv1,             # LEO 离轨
            "dv2": dv2,             # LOI 制动
            "dv_total": dv_total,
            "t_flight": t_flight,
            "T_transfer": T_tr,
            "altitude_leo": altitude_leo,
            "altitude_lmo": altitude_lmo,
        }

    # ------------------------------------------------------------------
    # 2. 发射窗口
    # ------------------------------------------------------------------
    def required_phase_angle(self, t_flight):
        """所需月球相位角 [rad]。

        推导
        -----
        设停泊轨道近地点 (出发方向) 固定在惯性 +x; 转移椭圆远地点在 -x.
        航天器经 t_flight 到达 -x (远地点), 此时月球必须也在 -x.
        月球角速度 omega_moon (顺行), 在 t_flight 内转过 omega_moon * t_flight.
        故发射时刻月球应位于:
            phi_required = pi - omega_moon * t_flight     (从 +x 起, 顺行方向)
        即月球须 *超前* 出发方向该角度, 使其 t_flight 后正好到达 -x.
        """
        return math.pi - self.omega_moon * t_flight

    def actual_phase_angle(self, t_sec):
        """t_sec 时刻月球相对出发方向 (+x) 的实际相位角 [rad]。"""
        return (self.omega_moon * t_sec) % (2.0 * math.pi)

    def _deviation(self, t_sec, t_flight):
        """实际相位角与所需相位角的偏差 (归一化到 (-pi, pi])。"""
        return _wrap_pi(self.actual_phase_angle(t_sec) - self.required_phase_angle(t_flight))

    def launch_window(self, start_date, days=60, step_hours=1.0):
        """发射窗口分析。

        参数
        -----
        start_date : 起始日期 (date/datetime/str); 视为月球在 +x 的参考历元.
        days       : 搜索天数
        step_hours : 网格步长 (小时)

        返回 dict:
            candidates : list, 每项含 date, day_offset, required_phase,
                         actual_phase, deviation(度), C3, flight_time_days
            windows    : list, 局部最优窗口 (scipy 精化)
            best       : 全局最优窗口
        """
        start = _to_date(start_date)
        ht = self.hohmann_transfer()
        t_flight = ht["t_flight"]
        phi_req = self.required_phase_angle(t_flight)
        C3 = ht["C3_moon"]
        ft_days = t_flight / DAY2SEC

        # 网格扫描
        step_sec = step_hours * 3600.0
        total_sec = days * DAY2SEC
        n = int(total_sec / step_sec) + 1
        ts = np.linspace(0.0, total_sec, n)

        candidates = []
        devs = []
        for t in ts:
            dev = self._deviation(t, t_flight)
            devs.append(dev)
            candidates.append({
                "date": start + datetime.timedelta(seconds=float(t)),
                "day_offset": t / DAY2SEC,
                "required_phase_deg": math.degrees(phi_req),
                "actual_phase_deg": math.degrees(self.actual_phase_angle(t)),
                "deviation_deg": math.degrees(dev),
                "C3": C3,
                "flight_time_days": ft_days,
            })
        devs = np.array(devs)
        abs_dev = np.abs(devs)

        # 找局部极小 (|dev| 的局部最小), 用 scipy 精化
        windows = []
        i = 0
        while i < len(ts):
            # 跳过到下一个局部极小
            if 0 < i < len(ts) - 1 and abs_dev[i] <= abs_dev[i - 1] and abs_dev[i] <= abs_dev[i + 1]:
                # 精化
                t_lo = ts[max(i - 1, 0)]
                t_hi = ts[min(i + 1, len(ts) - 1)]
                res = minimize_scalar(
                    lambda tt: abs(self._deviation(tt, t_flight)),
                    bounds=(t_lo, t_hi), method="bounded",
                    options={"xatol": 60.0},  # 1 分钟精度
                )
                t_opt = float(res.x)
                dev_opt = self._deviation(t_opt, t_flight)
                windows.append({
                    "date": start + datetime.timedelta(seconds=t_opt),
                    "day_offset": t_opt / DAY2SEC,
                    "deviation_deg": math.degrees(dev_opt),
                    "required_phase_deg": math.degrees(phi_req),
                    "actual_phase_deg": math.degrees(self.actual_phase_angle(t_opt)),
                    "C3": C3,
                    "flight_time_days": ft_days,
                })
                i += 1
            else:
                i += 1

        # 全局最优 (再用 scipy 在全区间精化)
        res_global = minimize_scalar(
            lambda tt: abs(self._deviation(tt, t_flight)),
            bounds=(0.0, total_sec), method="bounded",
            options={"xatol": 30.0},
        )
        t_best = float(res_global.x)
        best = {
            "date": start + datetime.timedelta(seconds=t_best),
            "day_offset": t_best / DAY2SEC,
            "deviation_deg": math.degrees(self._deviation(t_best, t_flight)),
            "required_phase_deg": math.degrees(phi_req),
            "actual_phase_deg": math.degrees(self.actual_phase_angle(t_best)),
            "C3": C3,
            "flight_time_days": ft_days,
        }

        return {
            "candidates": candidates,
            "windows": windows,
            "best": best,
            "required_phase_deg": math.degrees(phi_req),
            "flight_time_days": ft_days,
            "t_flight": t_flight,
        }

    # ------------------------------------------------------------------
    # 3. 完整轨迹设计 (patched conic)
    # ------------------------------------------------------------------
    def design_trajectory(self, launch_date, altitude_leo=200e3, altitude_lmo=100e3):
        """给定发射日期, 设计完整地月转移轨迹 (patched conic)。

        返回 dict 含:
            leo_state         : (r, v) LEO 圆轨道状态 (出发方向 = -月球到达方向)
            transfer_elements : 转移椭圆根数
            transfer_perigee  : 转移近地点 (r, v)
            transfer_apogee   : 转移远地点 (r, v) (= 月球到达处)
            perilune_state    : 近月点状态 (月心)
            dv1, dv2, dv_total
            key_times         : 关键时间节点
        """
        start = _to_date(launch_date)
        ht = self.hohmann_transfer(altitude_leo, altitude_lmo)
        t_flight = ht["t_flight"]
        r_leo = ht["r_leo"]
        r_lmo = ht["r_lmo"]

        # 月球到达时刻的位置 -> 出发方向 = -月球到达方向 (Hohmann)
        moon_arr_pos, moon_arr_vel = self._moon_state(t_flight)
        moon_arr_dir = moon_arr_pos / np.linalg.norm(moon_arr_pos)
        depart_dir = -moon_arr_dir  # 转移近地点方向 (指向出发方向)

        # LEO 圆轨道状态 (在出发方向, 速度切向 = +y 局部)
        r_leo_vec = r_leo * depart_dir
        v_leo_circ = math.sqrt(self.mu_earth / r_leo)
        # 切向方向 (与 depart_dir 正交, 顺行): 在 xy 平面, depart_dir=(cos,sin,0) -> 切向 (-sin,cos,0)
        tang = np.array([-depart_dir[1], depart_dir[0], 0.0])
        v_leo_vec = v_leo_circ * tang

        # 转移椭圆 (近地点 = depart_dir, 远地点 = moon_arr_dir)
        a_tr = ht["a_transfer"]
        e_tr = ht["e_transfer"]
        v_tr_per = ht["v_tr_perigee"]
        v_tr_apo = ht["v_tr_apogee"]
        r_tr_per_vec = r_leo_vec
        v_tr_per_vec = v_tr_per * tang
        r_tr_apo_vec = moon_arr_pos
        # 远地点速度方向: 与近地点速度反向 (椭圆), 沿 -tang
        v_tr_apo_vec = -v_tr_apo * tang

        dv1 = ht["dv1"]
        v_inf_moon = ht["v_inf_moon"]
        v_perilune = ht["v_perilune"]
        v_circ_lmo = ht["v_circ_lmo"]
        dv2 = ht["dv2"]

        # 近月点状态 (月心系): 进入双曲线, 近月点速度方向近似与 v_inf_moon 反向共线
        v_inf_moon_vec = v_tr_apo_vec - moon_arr_vel
        v_inf_dir = v_inf_moon_vec / np.linalg.norm(v_inf_moon_vec)
        # 近月点位置 (月心): 取与 v_inf 方向垂直, 在月球轨道面内
        # 简化: 近月点位于月球指向地球方向 (用于 LOI)
        r_perilune_vec_moon = -moon_arr_dir * r_lmo
        v_perilune_vec = v_perilune * (-v_inf_dir)  # 双曲线近月点速度方向 (近似)

        # 关键时间节点
        key_times = {
            "t0_leo": 0.0,
            "t_soi_entry_approx": t_flight * 0.85,   # 近似 (SOI 进入, 估算)
            "t_apogee_arrival": t_flight,
            "t_loi": t_flight,                        # 近月点制动
        }

        return {
            "launch_date": start,
            "leo_state": (r_leo_vec, v_leo_vec),
            "transfer_elements": {
                "a": a_tr, "e": e_tr,
                "i": 0.0, "raan": 0.0,
                "argp": math.atan2(depart_dir[1], depart_dir[0]),
                "nu_perigee": 0.0,
            },
            "transfer_perigee": (r_tr_per_vec, v_tr_per_vec),
            "transfer_apogee": (r_tr_apo_vec, v_tr_apo_vec),
            "moon_arrival_state": (moon_arr_pos, moon_arr_vel),
            "v_inf_moon_vec": v_inf_moon_vec,
            "v_inf_moon": float(np.linalg.norm(v_inf_moon_vec)),
            "perilune_state_moon": (r_perilune_vec_moon, v_perilune_vec),
            "v_perilune": v_perilune,
            "v_circ_lmo": v_circ_lmo,
            "dv1": dv1,
            "dv2": dv2,
            "dv_total": dv1 + dv2,
            "t_flight": t_flight,
            "key_times": key_times,
            "hohmann": ht,
        }

    # ------------------------------------------------------------------
    # 4. Porkchop 图数据
    # ------------------------------------------------------------------
    def porkchop_plot_data(
        self,
        start_date,
        days=60,
        altitude_leo=200e3,
        altitude_lmo=100e3,
        flight_time_range=(3.0, 8.0),
        n_dep=25,
        n_ft=20,
    ):
        """生成 porkchop 图网格数据 (C3 与飞行时间)。

        模型
        -----
        停泊轨道近地点固定在惯性 +x (固定定向的 LEO). 对每个
        (发射偏移 t_dep, 飞行时间 t_flight):
            r1 = r_leo * [1,0,0]                      (LEO 出发点)
            r2 = 月球在 t_dep + t_flight 时刻的位置    (到达点)
            用 Lambert 求解地心转移 (mu_earth) -> v1, v2
            dv1        = |v1| - v_leo_circ             (LEO 离轨)
            v_inf_moon = |v2 - v_moon_arrival|         (月球相对速度)
            C3_moon    = v_inf_moon^2                  (月球特征能量)
            LOI dv2    = sqrt(v_inf_moon^2 + 2 mu_moon/r_lmo) - sqrt(mu_moon/r_lmo)
            dv_total   = dv1 + dv2

        返回 dict 含:
            departure_days  : (n_dep,) 发射偏移 [天]
            flight_time_days: (n_ft,)  飞行时间 [天]
            C3_moon         : (n_dep, n_ft) 月球 C3 网格 [m^2/s^2]
            dv_total        : (n_dep, n_ft) 总 delta-V 网格 [m/s]
            dv1             : (n_dep, n_ft) LEO 离轨 dv
            dv2             : (n_dep, n_ft) LOI 制动 dv
            v_inf_moon      : (n_dep, n_ft) 月球相对速度
        """
        from .lambert import solve_lambert

        r_leo = self.R_earth + altitude_leo
        r_lmo = self.R_moon + altitude_lmo
        v_leo_circ = math.sqrt(self.mu_earth / r_leo)
        v_circ_lmo = math.sqrt(self.mu_moon / r_lmo)

        dep_days = np.linspace(0.0, days, n_dep)
        ft_days = np.linspace(flight_time_range[0], flight_time_range[1], n_ft)

        C3 = np.full((n_dep, n_ft), np.nan)
        dv_total = np.full((n_dep, n_ft), np.nan)
        dv1_grid = np.full((n_dep, n_ft), np.nan)
        dv2_grid = np.full((n_dep, n_ft), np.nan)
        vinf_grid = np.full((n_dep, n_ft), np.nan)

        r1 = np.array([r_leo, 0.0, 0.0])
        for i, td in enumerate(dep_days):
            t_dep = td * DAY2SEC
            for j, tf in enumerate(ft_days):
                t_flight = tf * DAY2SEC
                t_arr = t_dep + t_flight
                moon_pos, moon_vel = self._moon_state(t_arr)
                try:
                    v1, v2 = solve_lambert(
                        r1, moon_pos, t_flight, self.mu_earth,
                        direction="prograde",
                    )
                except Exception:
                    continue
                dv1 = float(np.linalg.norm(v1)) - v_leo_circ
                if dv1 < 0:
                    continue
                vinf_vec = v2 - moon_vel
                vinf = float(np.linalg.norm(vinf_vec))
                dv2 = math.sqrt(vinf * vinf + 2.0 * self.mu_moon / r_lmo) - v_circ_lmo
                C3[i, j] = vinf * vinf
                dv1_grid[i, j] = dv1
                dv2_grid[i, j] = dv2
                dv_total[i, j] = dv1 + dv2
                vinf_grid[i, j] = vinf

        return {
            "departure_days": dep_days,
            "flight_time_days": ft_days,
            "C3_moon": C3,
            "dv_total": dv_total,
            "dv1": dv1_grid,
            "dv2": dv2_grid,
            "v_inf_moon": vinf_grid,
            "altitude_leo": altitude_leo,
            "altitude_lmo": altitude_lmo,
        }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.physics.moon_transfer 自测 ===")
    mt = MoonTransfer()

    # Hohmann 转移: LEO 200km -> LMO 100km
    ht = mt.hohmann_transfer(altitude_leo=200e3, altitude_lmo=100e3)
    print("\n--- Hohmann 地月转移 (LEO 200km -> LMO 100km) ---")
    print(f"  转移椭圆 a = {ht['a_transfer']/1e6:.3f}e6 m, e = {ht['e_transfer']:.4f}")
    print(f"  LEO 圆速度        v_leo    = {ht['v_leo']:.2f} m/s")
    print(f"  转移近地点速度    v_tr_per = {ht['v_tr_perigee']:.2f} m/s")
    print(f"  转移远地点速度    v_tr_apo = {ht['v_tr_apogee']:.2f} m/s")
    print(f"  月球轨道速度      v_moon   = {ht['v_moon']:.2f} m/s")
    print(f"  月球相对速度      v_inf    = {ht['v_inf_moon']:.2f} m/s")
    print(f"  C3_moon           = {ht['C3_moon']:.3e} m^2/s^2")
    print(f"  dv1 (LEO 离轨)    = {ht['dv1']:.2f} m/s = {ht['dv1']/1e3:.3f} km/s")
    print(f"  dv2 (LOI 制动)    = {ht['dv2']:.2f} m/s = {ht['dv2']/1e3:.3f} km/s")
    print(f"  dv_total          = {ht['dv_total']:.2f} m/s = {ht['dv_total']/1e3:.3f} km/s")
    print(f"  飞行时间          = {ht['t_flight']/3600:.2f} h = {ht['t_flight']/86400:.3f} 天")
    # 关键校验
    assert 3.9e3 <= ht["dv_total"] <= 4.1e3, f"dv_total 应在 3.9~4.1 km/s, 得 {ht['dv_total']/1e3:.3f}"
    assert 4.5 <= ht["t_flight"] / 86400 <= 5.5, f"飞行时间应 ~5 天, 得 {ht['t_flight']/86400:.2f}"

    # 发射窗口
    print("\n--- 发射窗口 (60 天搜索) ---")
    lw = mt.launch_window(datetime.date(2026, 1, 1), days=60)
    print(f"  所需相位角 = {lw['required_phase_deg']:.2f}°")
    print(f"  飞行时间   = {lw['flight_time_days']:.3f} 天")
    print(f"  找到 {len(lw['windows'])} 个候选窗口:")
    for w in lw["windows"]:
        print(f"    {w['date']}  偏差={w['deviation_deg']:+.3f}°  "
              f"实际相位={w['actual_phase_deg']:.2f}°")
    best = lw["best"]
    print(f"  最优窗口: {best['date']}  (偏移 {best['day_offset']:.2f} 天), "
          f"偏差={best['deviation_deg']:+.4f}°")
    assert abs(best["deviation_deg"]) < 1.0, "最优窗口偏差应 < 1°"

    # 完整轨迹
    print("\n--- 完整轨迹设计 (patched conic) ---")
    traj = mt.design_trajectory(datetime.date(2026, 1, 1), altitude_leo=200e3, altitude_lmo=100e3)
    r_leo, v_leo = traj["leo_state"]
    print(f"  LEO 状态  r = {np.round(r_leo/1e3,1)} km, |v|={np.linalg.norm(v_leo):.2f} m/s")
    r_apo, v_apo = traj["transfer_apogee"]
    print(f"  远地点    r = {np.round(r_apo/1e6,2)}e6 m, |v|={np.linalg.norm(v_apo):.2f} m/s")
    print(f"  v_inf_moon (向量法) = {traj['v_inf_moon']:.2f} m/s (Hohmann {ht['v_inf_moon']:.2f})")
    print(f"  dv1={traj['dv1']/1e3:.3f} km/s, dv2={traj['dv2']/1e3:.3f} km/s, "
          f"dv_total={traj['dv_total']/1e3:.3f} km/s")
    assert abs(traj["dv_total"] - ht["dv_total"]) < 1e-3

    # Porkchop (小网格快速演示)
    print("\n--- Porkchop 数据 (15x10 网格) ---")
    pc = mt.porkchop_plot_data(datetime.date(2026, 1, 1), days=60,
                               n_dep=15, n_ft=10)
    print(f"  网格: departure {pc['departure_days'].size} x flight {pc['flight_time_days'].size}")
    valid = np.isfinite(pc["dv_total"])
    print(f"  有效点: {valid.sum()}/{pc['dv_total'].size}")
    if valid.any():
        idx = np.unravel_index(np.nanargmin(pc["dv_total"]), pc["dv_total"].shape)
        print(f"  最小总 dv = {pc['dv_total'][idx]/1e3:.3f} km/s "
              f"@ departure={pc['departure_days'][idx[0]]:.1f}d, "
              f"flight={pc['flight_time_days'][idx[1]]:.2f}d")
        print(f"  对应 C3_moon = {pc['C3_moon'][idx]:.3e} m^2/s^2, "
              f"v_inf_moon = {pc['v_inf_moon'][idx]:.2f} m/s")
        # 最优应接近 Hohmann (dv_total ~3.95 km/s, flight ~5d)
        assert abs(pc["dv_total"][idx]/1e3 - ht["dv_total"]/1e3) < 0.3

    print("\nmoon_transfer 自测全部通过.")
