"""轨道设计工作流 (Orbit design workflow)。

根据任务约束 (轨道类型、高度、倾角) 计算轨道根数、速度、周期与覆盖
特性，并验证可行性。真实调用 ``aerospace_agent.physics.kepler`` 与
``aerospace_agent.physics.orbital_maneuvers``，不重写物理公式。

支持的轨道类型：
    - ``leo``            : 近地圆轨道
    - ``geo``            : 地球静止轨道
    - ``sso``            : 太阳同步轨道
    - ``molniya``        : 莫尔尼亚 (大椭圆) 轨道
    - ``lunar_transfer`` : 地月转移轨道 (Hohmann 近似)
"""

from __future__ import annotations

import math
from typing import Any, Dict

from aerospace_agent.physics import (
    A_MOON,
    DAY2SEC,
    DEG2RAD,
    MU_EARTH,
    MU_MOON,
    R_EARTH,
    R_MOON,
    KeplerOrbit,
    elements_to_state,
    hohmann_transfer_delta_v,
    state_to_elements,
)
from aerospace_agent.physics.moon_transfer import MoonTransfer

from .base import BaseWorkflow, WorkflowResult, register_workflow


# 支持的轨道类型集合
SUPPORTED_ORBIT_TYPES = ("leo", "geo", "sso", "molniya", "lunar_transfer")


def _earth_central_angle(altitude: float, elev_min_deg: float = 10.0) -> float:
    """计算卫星对地覆盖的地心张角 [rad] (给定最小仰角)。

    推导
    -----
    卫星高度 h，地心距 r = R_earth + h，最小仰角 e_min。
    覆盖地心张角:

        lambda = arccos( (R_earth / r) * cos(e_min) ) - e_min

    覆盖面积比 = (1 - cos(lambda)) / 2  (占地球表面积比例)。
    """
    r = R_EARTH + altitude
    e_min = elev_min_deg * DEG2RAD
    ratio = (R_EARTH / r) * math.cos(e_min)
    if ratio >= 1.0:
        return 0.0
    return math.acos(ratio) - e_min


@register_workflow()
class OrbitDesignWorkflow(BaseWorkflow):
    """轨道设计工作流。

    name = 'orbit_design'
    """

    name = "orbit_design"
    description = "根据任务约束设计轨道：计算根数、速度、周期与覆盖特性并验证可行性"
    version = "1.0.0"
    required_tools = []  # 纯物理计算，不强制依赖外部工具

    steps = [
        {"name": "define_constraints", "description": "确定任务约束 (轨道类型/高度/倾角)"},
        {"name": "select_orbit_type", "description": "选择轨道类型 (leo/geo/sso/molniya/lunar_transfer)"},
        {"name": "compute_elements", "description": "计算轨道参数 (根数/速度/周期)"},
        {"name": "verify_feasibility", "description": "验证可行性 (高度/能量/覆盖)"},
        {"name": "output_design", "description": "输出设计结果"},
    ]

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> bool:
        if not super().validate_params(params):
            return False
        ot = params.get("orbit_type", "leo")
        if ot not in SUPPORTED_ORBIT_TYPES:
            return False
        alt = params.get("altitude", 0.0)
        if not isinstance(alt, (int, float)) or alt < 0:
            return False
        return True

    # ------------------------------------------------------------------
    # 各轨道类型的设计
    # ------------------------------------------------------------------
    @staticmethod
    def _design_circular(orbit_type, altitude, inclination_deg, mu=MU_EARTH):
        """圆轨道 (leo / geo / sso) 设计。"""
        a = R_EARTH + altitude
        e = 0.0
        i = inclination_deg * DEG2RAD
        # 圆轨道速度 (vis-viva: v = sqrt(mu / a))
        v = math.sqrt(mu / a)
        # 周期 (开普勒第三定律: T = 2 pi sqrt(a^3 / mu))
        T = 2.0 * math.pi * math.sqrt(a ** 3 / mu)

        # 覆盖特性
        cov_angle = _earth_central_angle(altitude, elev_min_deg=10.0)
        cov_frac = (1.0 - math.cos(cov_angle)) / 2.0

        # 轨道根数 (取近地点幅角 0，真近点角 0)
        r_vec, v_vec = elements_to_state(a, e, i, 0.0, 0.0, 0.0, mu)
        els = state_to_elements(r_vec, v_vec, mu)

        notes = {
            "leo": "近地圆轨道：低延迟、强观测，受大气阻力摄动。",
            "geo": "地球静止轨道：周期与地球自转同步 (1 恒星日)，对地静止。",
            "sso": "太阳同步轨道：RAAN 进动率匹配太阳视运动 (~0.9856°/天)，光照条件恒定。",
        }.get(orbit_type, "")

        if orbit_type == "geo":
            # 校验 GEO 周期 ~ 恒星日
            sidereal_day = 86164.0905  # s
            notes += f" 周期 {T:.1f}s (恒星日 {sidereal_day:.1f}s)。"
        if orbit_type == "sso":
            # 太阳同步所需 J2 进动率 ~ 0.9856 deg/day
            # dRAAN/dt = -3/2 * n * J2 * (R/a)^2 * cos(i)
            from aerospace_agent.physics.constants import J2_EARTH
            n = math.sqrt(mu / a ** 3)
            J2 = J2_EARTH
            d_raan = -1.5 * n * J2 * (R_EARTH / a) ** 2 * math.cos(i)  # rad/s
            d_raan_deg_day = math.degrees(d_raan) * DAY2SEC
            notes += (f" J2 RAAN 进动率 = {d_raan_deg_day:.4f}°/天 "
                      f"(太阳同步目标 ~+0.9856°/day，需 i≈98°)。")

        return {
            "orbit_type": orbit_type,
            "elements": {
                "a_m": a, "e": e, "i_deg": inclination_deg,
                "raan_deg": 0.0, "argp_deg": 0.0, "nu_deg": 0.0,
            },
            "state_vector": {
                "position_m": r_vec.tolist(),
                "velocity_m_s": v_vec.tolist(),
            },
            "velocity_m_s": v,
            "period_s": T,
            "period_min": T / 60.0,
            "altitude_m": altitude,
            "altitude_km": altitude / 1e3,
            "coverage": {
                "earth_central_angle_deg": math.degrees(cov_angle),
                "surface_coverage_fraction": cov_frac,
                "min_elevation_deg": 10.0,
            },
            "elements_full": els,
            "notes": notes,
        }

    @staticmethod
    def _design_molniya(altitude, inclination_deg):
        """莫尔尼亚 (大椭圆) 轨道设计。

        典型参数：近地点 ~600 km，远地点 ~39400 km，倾角 63.4° (临界倾角，
        防止近地点幅角进动)，周期 ~12 h。
        若 altitude < 1000km 则视作近地点高度，远地点取标准 39400km。
        """
        h_perigee = altitude if altitude < 2000e3 else 600e3
        h_apogee = 39400e3
        r_perigee = R_EARTH + h_perigee
        r_apogee = R_EARTH + h_apogee
        a = 0.5 * (r_perigee + r_apogee)
        e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        i = 63.4 * DEG2RAD if abs(inclination_deg - 51.6) < 1e-6 else inclination_deg * DEG2RAD

        # 近地点 / 远地点速度 (vis-viva)
        v_perigee = math.sqrt(MU_EARTH * (2.0 / r_perigee - 1.0 / a))
        v_apogee = math.sqrt(MU_EARTH * (2.0 / r_apogee - 1.0 / a))
        T = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH)

        r_vec, v_vec = elements_to_state(a, e, i, 0.0, math.radians(270.0), 0.0, MU_EARTH)
        els = state_to_elements(r_vec, v_vec, MU_EARTH)

        # 覆盖：远地点处速度低，长时间停留于高纬上空
        cov_angle = _earth_central_angle(h_apogee, elev_min_deg=10.0)

        return {
            "orbit_type": "molniya",
            "elements": {
                "a_m": a, "e": e, "i_deg": math.degrees(i),
                "raan_deg": 0.0, "argp_deg": 270.0, "nu_deg": 0.0,
            },
            "state_vector": {
                "position_m": r_vec.tolist(),
                "velocity_m_s": v_vec.tolist(),
            },
            "velocity_m_s": v_perigee,
            "velocity_perigee_m_s": v_perigee,
            "velocity_apogee_m_s": v_apogee,
            "period_s": T,
            "period_h": T / 3600.0,
            "perigee_altitude_km": h_perigee / 1e3,
            "apogee_altitude_km": h_apogee / 1e3,
            "coverage": {
                "earth_central_angle_at_apogee_deg": math.degrees(cov_angle),
                "note": "远地点位于北半球高纬上空，约 2/3 周期可见高纬区域。",
            },
            "elements_full": els,
            "notes": "莫尔尼亚轨道：临界倾角 63.4° (冻结近地点幅角)，远地点 39400km，"
                     "周期约 12h，适合高纬通信与侦察。",
        }

    @staticmethod
    def _design_lunar_transfer(altitude_leo, altitude_lmo):
        """地月转移轨道设计 (Hohmann 近似)。"""
        mt = MoonTransfer()
        ht = mt.hohmann_transfer(altitude_leo=altitude_leo, altitude_lmo=altitude_lmo)
        a_tr = ht["a_transfer"]
        e_tr = ht["e_transfer"]
        # 转移轨道根数 (近地点 = LEO)
        r_vec, v_vec = elements_to_state(a_tr, e_tr, 0.0, 0.0, 0.0, 0.0, MU_EARTH)
        els = state_to_elements(r_vec, v_vec, MU_EARTH)
        return {
            "orbit_type": "lunar_transfer",
            "elements": {
                "a_m": a_tr, "e": e_tr, "i_deg": 0.0,
                "raan_deg": 0.0, "argp_deg": 0.0, "nu_deg": 0.0,
            },
            "state_vector": {
                "position_m": r_vec.tolist(),
                "velocity_m_s": v_vec.tolist(),
            },
            "velocity_m_s": ht["v_tr_perigee"],
            "velocity_perigee_m_s": ht["v_tr_perigee"],
            "velocity_apogee_m_s": ht["v_tr_apogee"],
            "period_s": ht["T_transfer"],
            "flight_time_days": ht["t_flight"] / DAY2SEC,
            "delta_v": {
                "dv1_leo_m_s": ht["dv1"],
                "dv2_loi_m_s": ht["dv2"],
                "dv_total_m_s": ht["dv_total"],
                "dv_total_km_s": ht["dv_total"] / 1e3,
            },
            "C3_moon": ht["C3_moon"],
            "v_inf_moon_m_s": ht["v_inf_moon"],
            "elements_full": els,
            "notes": f"地月转移 (Hohmann 近似)：LEO {altitude_leo/1e3:.0f}km -> "
                     f"LMO {altitude_lmo/1e3:.0f}km，飞行时间 ~{ht['t_flight']/DAY2SEC:.2f} 天，"
                     f"总 delta-V ~{ht['dv_total']/1e3:.2f} km/s。",
        }

    # ------------------------------------------------------------------
    # 主执行
    # ------------------------------------------------------------------
    def execute(
        self,
        orbit_type: str = "leo",
        altitude: float = 400e3,
        inclination: float = 51.6,
        **kwargs,
    ) -> WorkflowResult:
        """执行轨道设计工作流。

        Parameters
        ----------
        orbit_type : str
            轨道类型：leo / geo / sso / molniya / lunar_transfer。
        altitude : float
            轨道高度 [m] (leo/geo/sso 为圆轨道高度；molniya 为近地点高度；
            lunar_transfer 为 LEO 停泊高度)。
        inclination : float
            轨道倾角 [deg]。
        **kwargs
            额外参数 (如 lunar_transfer 的 altitude_lmo)。
        """
        res = WorkflowResult()
        res.metadata["params"] = {
            "orbit_type": orbit_type,
            "altitude": altitude,
            "inclination": inclination,
            **kwargs,
        }

        # 步骤 1：确定任务约束
        self._log_step(res, "define_constraints", "success",
                       f"轨道类型={orbit_type}, 高度={altitude/1e3:.1f}km, "
                       f"倾角={inclination}°")

        # 步骤 2：选择轨道类型
        if orbit_type not in SUPPORTED_ORBIT_TYPES:
            self._log_step(res, "select_orbit_type", "failed",
                           f"不支持的轨道类型: {orbit_type}, 支持: {SUPPORTED_ORBIT_TYPES}")
            res.summary = f"轨道设计失败：不支持的轨道类型 '{orbit_type}'"
            return res
        self._log_step(res, "select_orbit_type", "success",
                       f"已选择轨道类型: {orbit_type}")

        # 步骤 3：计算轨道参数
        try:
            if orbit_type == "lunar_transfer":
                alt_lmo = kwargs.get("altitude_lmo", 100e3)
                design = self._design_lunar_transfer(altitude, alt_lmo)
            elif orbit_type == "molniya":
                design = self._design_molniya(altitude, inclination)
            else:
                design = self._design_circular(orbit_type, altitude, inclination)
        except Exception as exc:  # pragma: no cover - 异常路径
            self._log_step(res, "compute_elements", "failed", f"计算异常: {exc}")
            res.summary = f"轨道设计失败：{exc}"
            return res

        self._log_step(
            res, "compute_elements", "success",
            f"已计算轨道参数：a={design['elements']['a_m']/1e3:.1f}km, "
            f"e={design['elements']['e']:.4f}, "
            f"v={design['velocity_m_s']:.2f}m/s, "
            f"T={design['period_s']:.1f}s",
            data={"a_km": design["elements"]["a_m"] / 1e3,
                  "e": design["elements"]["e"],
                  "v_m_s": design["velocity_m_s"],
                  "T_s": design["period_s"]},
        )

        # 步骤 4：验证可行性
        feasible, issues = self._verify_feasibility(orbit_type, design)
        status = "success" if feasible else "warning"
        self._log_step(res, "verify_feasibility", status,
                       f"可行性={'通过' if feasible else '存疑'}; "
                       f"问题: {issues if issues else '无'}",
                       data={"feasible": feasible, "issues": issues})
        design["feasibility"] = {"feasible": feasible, "issues": issues}

        # 步骤 5：输出设计
        self._log_step(res, "output_design", "success",
                       "轨道设计结果已生成")
        res.success = True
        res.result = design
        res.summary = (
            f"轨道设计完成 [{orbit_type.upper()}]：半长轴 {design['elements']['a_m']/1e3:.1f} km，"
            f"偏心率 {design['elements']['e']:.4f}，速度 {design['velocity_m_s']:.2f} m/s，"
            f"周期 {design['period_s']:.1f} s。可行性：{'通过' if feasible else '存疑'}。"
        )
        res.metadata["orbit_type"] = orbit_type
        res.metadata["feasible"] = feasible
        return res

    # ------------------------------------------------------------------
    # 可行性验证
    # ------------------------------------------------------------------
    @staticmethod
    def _verify_feasibility(orbit_type: str, design: dict):
        """验证轨道可行性，返回 (feasible, issues)。"""
        issues = []
        feasible = True
        a = design["elements"]["a_m"]
        e = design["elements"]["e"]
        v = design["velocity_m_s"]

        # 大气层高度阈值 (LEO 应高于 ~160km)
        if orbit_type in ("leo", "sso"):
            alt = design.get("altitude_m", 0.0)
            if alt < 160e3:
                issues.append(f"高度 {alt/1e3:.1f}km 过低 (<160km)，大气阻力严重")
                feasible = False
            if alt > 2000e3:
                issues.append(f"高度 {alt/1e3:.1f}km 超出典型 LEO 范围")

        if orbit_type == "geo":
            # GEO 应在赤道 (i≈0) 且周期 ~恒星日
            if abs(design["elements"]["i_deg"]) > 5.0:
                issues.append("GEO 倾角应接近 0°，否则会产生 8 字形漂移")
            sidereal_day = 86164.0905
            if abs(design["period_s"] - sidereal_day) > 1000.0:
                issues.append(f"GEO 周期 {design['period_s']:.1f}s 偏离恒星日 {sidereal_day:.1f}s")

        if orbit_type == "sso":
            # 倾角应在 96°~102°
            if not (95.0 <= design["elements"]["i_deg"] <= 103.0):
                issues.append(f"SSO 倾角 {design['elements']['i_deg']:.1f}° 应在 96~102° 范围")

        if orbit_type == "molniya":
            if abs(design["elements"]["i_deg"] - 63.4) > 1.0:
                issues.append("Molniya 倾角应为 63.4° (临界倾角)")
            if design["period_h"] < 11.0 or design["period_h"] > 13.0:
                issues.append(f"Molniya 周期 {design['period_h']:.2f}h 应 ~12h")

        if orbit_type == "lunar_transfer":
            dv_total = design["delta_v"]["dv_total_km_s"]
            if dv_total > 4.5 or dv_total < 3.5:
                issues.append(f"地月转移总 delta-V {dv_total:.2f} km/s 异常 (期望 ~3.95)")

        # 通用：速度合理性
        if v <= 0 or v > 12e3:
            issues.append(f"轨道速度 {v:.2f} m/s 异常")
            feasible = False

        return feasible, issues


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.workflows.orbit_design 自测 ===")

    wf = OrbitDesignWorkflow()
    print(f"工作流: {wf!r}")
    print(f"步骤计划:\n  " + "\n  ".join(wf.get_plan()))
    print(f"工具可用性: {wf.check_tools()}")

    for ot, alt, inc in [
        ("leo", 400e3, 51.6),
        ("geo", 35786e3, 0.0),
        ("sso", 600e3, 97.8),
        ("molniya", 600e3, 63.4),
        ("lunar_transfer", 200e3, 0.0),
    ]:
        kwargs = {"altitude_lmo": 100e3} if ot == "lunar_transfer" else {}
        r = wf.execute(orbit_type=ot, altitude=alt, inclination=inc, **kwargs)
        d = r.result
        print(f"\n--- {ot.upper()} (success={r.success}, steps={len(r.steps_log)}) ---")
        print(f"  summary: {r.summary}")
        print(f"  a = {d['elements']['a_m']/1e3:.2f} km, e = {d['elements']['e']:.4f}, "
              f"v = {d['velocity_m_s']:.2f} m/s, T = {d['period_s']:.1f} s")
        if "coverage" in d:
            print(f"  覆盖: {d['coverage']}")
        if "delta_v" in d:
            print(f"  delta-V: {d['delta_v']}")
        print(f"  可行性: {d['feasibility']}")
        assert r.success

    # 不支持的轨道类型
    r_bad = wf.execute(orbit_type="xxx", altitude=400e3)
    assert not r_bad.success
    print(f"\n不支持类型测试: success={r_bad.success} (符合预期)")

    print("\norbit_design 自测全部通过.")
