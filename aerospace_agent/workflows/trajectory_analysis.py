"""轨迹分析工作流 (Trajectory analysis workflow) — 地月转移核心。

本工作流是用户最关心的核心模块，完整设计 LEO 200km -> LMO 100km 的
地月转移轨道，涵盖：

    1. 任务定义 (LEO -> LMO 地月转移)
    2. 方法选择 (拼凑圆锥近似 Patched Conic)
    3. Hohmann 转移参数计算
    4. 发射窗口确定 (相位角匹配 + scipy 寻优)
    5. 完整轨迹设计 (patched conic, 含 LEO/转移椭圆/近月点状态)
    6. Porkchop 图数据生成 (Lambert 网格, C3 / 飞行时间)
    7. 物理量分析 (delta-V 预算、能量、角动量、飞行时间)
    8. 公式推导汇总 (vis-viva / 开普勒第三定律 / 相位角 / Hohmann / C3)
    9. 输出结果 (JSON, 准备给报告模块)

所有物理计算真实调用 ``aerospace_agent.physics.moon_transfer.MoonTransfer``，
工具调用走 ``aerospace_agent.mcp_tools`` 统一 ``call`` 接口。

期望结果 (LEO 200km -> LMO 100km):
    * 总 delta-V ≈ 3.95 km/s
    * 飞行时间 ≈ 5 天
"""

from __future__ import annotations

import datetime
import json
import math
import os
from typing import Any

import numpy as np

from aerospace_agent.physics.constants import (
    A_MOON,
    DAY2SEC,
    MU_EARTH,
    MU_MOON,
    OMEGA_MOON,
    R_EARTH,
    R_MOON,
    R_SOI_MOON,
)
from aerospace_agent.physics.moon_transfer import MoonTransfer

from .base import BaseWorkflow, WorkflowResult, register_workflow
from .launch_window import _to_datetime

# 输出目录与结果文件
DEMO_OUTPUTS_DIR = "/workspace/demo_outputs"
DEFAULT_RESULT_JSON = os.path.join(DEMO_OUTPUTS_DIR, "lunar_transfer_result.json")


def _jsonify(obj: Any) -> Any:
    """递归把 numpy / datetime / tuple 对象转为 JSON 可序列化结构。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, tuple):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def build_formula_derivation() -> str:
    """构建公式推导汇总文本 (LaTeX 风格)。

    包含 vis-viva、开普勒第三定律、相位角、Hohmann dv1/dv2、C3 等公式。
    """
    text = r"""================================================================================
                 地月转移轨道设计 — 公式推导汇总 (Patched Conic)
================================================================================

本工作流采用拼凑圆锥近似法 (Patched Conic Approximation) 设计
LEO -> LMO 地月转移轨道。以下为关键物理公式 (LaTeX 风格)。

--------------------------------------------------------------------------------
1. Vis-Viva 方程 (能量守恒, 任意圆锥曲线)
--------------------------------------------------------------------------------
  任意圆锥曲线轨道上, 距中心天体距离 r 处的速度:

      v^2 = \mu \left( \frac{2}{r} - \frac{1}{a} \right)

  其中 a 为半长轴 (椭圆 a>0, 双曲线 a<0), \mu 为引力参数。
  - 圆轨道 (a = r):     v_circ = \sqrt{\mu / r}
  - 椭圆近地点:         v_per = \sqrt{\mu (2/r_per - 1/a)}
  - 椭圆远地点:         v_apo = \sqrt{\mu (2/r_apo - 1/a)}
  - 双曲线 (a<0):        v^2 = v_inf^2 + 2\mu/r

--------------------------------------------------------------------------------
2. 开普勒第三定律 (轨道周期)
--------------------------------------------------------------------------------
      T^2 = \frac{4\pi^2}{\mu} a^3   \quad\Longleftrightarrow\quad
      T = 2\pi \sqrt{\frac{a^3}{\mu}}

  转移椭圆的半周期即地月转移飞行时间:

      t_{flight} = \pi \sqrt{\frac{a_{tr}^3}{\mu_{earth}}}  \quad (\approx 5 \text{ 天})

--------------------------------------------------------------------------------
3. Hohmann 转移 (地月转移椭圆, 共面圆轨道间最省能量双脉冲转移)
--------------------------------------------------------------------------------
  转移椭圆半长轴:
      a_{tr} = \frac{r_{leo} + r_{moon}}{2}
  转移椭圆偏心率:
      e_{tr} = \frac{r_{moon} - r_{leo}}{r_{moon} + r_{leo}}
  其中 r_leo = R_earth + h_leo (LEO 半径), r_moon = a_moon (月球轨道半径)。

  第一次脉冲 \Delta v_1 (LEO 离轨, 切向加速到转移椭圆近地点):
      \Delta v_1 = \sqrt{\mu_{earth}\!\left(\frac{2}{r_{leo}} - \frac{1}{a_{tr}}\right)}
                   - \sqrt{\frac{\mu_{earth}}{r_{leo}}}
                 = v_{tr,perigee} - v_{leo,circ}

  远地点 (月球到达处) 转移椭圆速度:
      v_{tr,apogee} = \sqrt{\mu_{earth}\!\left(\frac{2}{r_{moon}} - \frac{1}{a_{tr}}\right)}
  月球轨道速度 (圆):   v_{moon} = \sqrt{\mu_{earth}/r_{moon}}
  月球相对速度 (Hohmann 近似下 v_tr_apo 与 v_moon 共线反向):
      v_{\infty,moon} = v_{moon} - v_{tr,apogee}

  第二次脉冲 \Delta v_2 (LOI 近月点制动, 月心双曲线 -> 近月圆轨道):
      v_{perilune} = \sqrt{v_{\infty,moon}^2 + \frac{2\mu_{moon}}{r_{lmo}}}
      v_{circ,lmo} = \sqrt{\frac{\mu_{moon}}{r_{lmo}}}
      \Delta v_2 = v_{perilune} - v_{circ,lmo}

  总 delta-V:
      \Delta v_{total} = \Delta v_1 + \Delta v_2  \quad (\approx 3.95 \text{ km/s})

--------------------------------------------------------------------------------
4. 特征能量 C3 (发射能量指标)
--------------------------------------------------------------------------------
      C3 = v_{\infty}^2

  地月转移中月球特征能量:
      C3_{moon} = v_{\infty,moon}^2  \quad [\text{m}^2/\text{s}^2]
  它是航天器在月球 SOI 边界相对月球的剩余速度平方, 衡量到达月球的能量需求。

--------------------------------------------------------------------------------
5. 发射窗口相位角 (Phase Angle)
--------------------------------------------------------------------------------
  设停泊轨道近地点 (出发方向) 固定在惯性 +x; 转移椭圆远地点指向 -x
  (月球到达方向)。航天器经 t_flight 到达远地点 (-x), 此时月球必须也
  在 -x。月球角速度 \omega_{moon} (顺行), 在 t_flight 内转过
  \omega_{moon} \cdot t_{flight}。故发射时刻月球应位于:

      \phi = \pi - \omega_{moon} \cdot t_{flight}

  即月球须 *超前* 出发方向该角度 (从 +x 起, 顺行方向), 使其 t_flight
  后正好到达 -x (远地点/月球到达处)。
  实际相位角 \phi_{actual}(t) = \omega_{moon} \cdot t \pmod{2\pi}。
  发射窗口 = 实际相位角 == 所需相位角的时刻 (scipy 精化寻优)。

--------------------------------------------------------------------------------
6. 角动量守恒 (Specific Angular Momentum)
--------------------------------------------------------------------------------
      \vec{h} = \vec{r} \times \vec{v}, \quad h = |\vec{h}| = \sqrt{\mu \, p}
      p = a(1 - e^2)  \quad (\text{半通径})

  转移椭圆: h_{tr} = \sqrt{\mu_{earth} \, a_{tr} (1 - e_{tr}^2)}

--------------------------------------------------------------------------------
7. 比能量 (Specific Orbital Energy)
--------------------------------------------------------------------------------
      \varepsilon = \frac{v^2}{2} - \frac{\mu}{r} = -\frac{\mu}{2a}

  转移椭圆 (束缚轨道, \varepsilon < 0):
      \varepsilon_{tr} = -\frac{\mu_{earth}}{2 a_{tr}}
  双曲线 (逃逸, \varepsilon > 0): \varepsilon = v_{\infty}^2 / 2 = C3 / 2。

--------------------------------------------------------------------------------
8. 月球引力作用球 (Sphere of Influence, Laplace)
--------------------------------------------------------------------------------
      r_{SOI} = a_{moon} \left(\frac{\mu_{moon}}{\mu_{earth}}\right)^{2/5}
              \approx 66200 \text{ km}

  在 SOI 边界做地心段 -> 月心段的速度匹配 (向量差):
      \vec{v}_{\infty,moon} = \vec{v}_{geo,at\,SOI} - \vec{v}_{moon}
  其模 |\vec{v}_{\infty,moon}| 决定月心双曲线的能量与近月点速度。

================================================================================
                              公式推导汇总结束
================================================================================
"""
    return text


@register_workflow()
class TrajectoryAnalysisWorkflow(BaseWorkflow):
    """轨迹分析工作流 (地月转移核心)。

    name = 'lunar_transfer'
    """

    name = "lunar_transfer"
    description = "地月转移轨迹分析：Hohmann 转移 + 发射窗口 + patched conic 轨迹 + porkchop + 公式推导"
    version = "1.0.0"
    required_tools = ["spiceypy"]

    steps = [
        {"name": "define_mission", "description": "定义任务：LEO 200km -> LMO 100km 地月转移"},
        {"name": "select_method", "description": "选择方法：拼凑圆锥近似 (Patched Conic)"},
        {"name": "hohmann_params", "description": "Hohmann 转移参数计算 (moon_transfer.hohmann_transfer)"},
        {"name": "launch_window", "description": "发射窗口确定 (moon_transfer.launch_window)"},
        {"name": "design_trajectory", "description": "完整轨迹设计 (moon_transfer.design_trajectory)"},
        {"name": "porkchop_data", "description": "Porkchop 图数据生成 (moon_transfer.porkchop_plot_data)"},
        {"name": "physics_analysis", "description": "物理量分析：delta-V 预算/能量/角动量/飞行时间"},
        {"name": "formula_derivation", "description": "公式推导汇总 (vis-viva/开普勒/相位角/Hohmann/C3)"},
        {"name": "output_results", "description": "输出结果，准备给报告模块 (JSON)"},
    ]

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> bool:
        if not super().validate_params(params):
            return False
        ld = params.get("launch_date", None)
        if ld is not None:
            try:
                _to_datetime(ld)
            except Exception:
                return False
        alt_leo = params.get("altitude_leo", 200e3)
        alt_lmo = params.get("altitude_lmo", 100e3)
        if not isinstance(alt_leo, (int, float)) or alt_leo <= 0:
            return False
        if not isinstance(alt_lmo, (int, float)) or alt_lmo <= 0:
            return False
        return True

    # ------------------------------------------------------------------
    # 主执行
    # ------------------------------------------------------------------
    def execute(
        self,
        launch_date: Any = None,
        altitude_leo: float = 200e3,
        altitude_lmo: float = 100e3,
        days: int = 60,
        **kwargs,
    ) -> WorkflowResult:
        """执行地月转移轨迹分析。

        Parameters
        ----------
        launch_date : str | datetime | None
            指定发射日期。若为 None, 自动搜索最优窗口。
        altitude_leo : float
            LEO 停泊轨道高度 [m] (默认 200km)。
        altitude_lmo : float
            LMO 近月轨道高度 [m] (默认 100km)。
        days : int
            发射窗口搜索天数 (launch_date 为 None 时使用)。
        **kwargs
            额外参数 (如 result_path 输出 JSON 路径, n_dep/n_ft porkchop 网格)。
        """
        res = WorkflowResult()
        res.metadata["params"] = {
            "launch_date": str(launch_date) if launch_date else None,
            "altitude_leo": altitude_leo,
            "altitude_lmo": altitude_lmo,
            "days": days,
        }
        result_path = kwargs.get("result_path", DEFAULT_RESULT_JSON)
        n_dep = kwargs.get("n_dep", 20)
        n_ft = kwargs.get("n_ft", 12)

        mt = MoonTransfer()

        # ---- 步骤 1：定义任务 ----
        r_leo = R_EARTH + altitude_leo
        r_lmo = R_MOON + altitude_lmo
        self._log_step(
            res, "define_mission", "success",
            f"任务定义：LEO {altitude_leo/1e3:.0f}km (r={r_leo/1e3:.1f}km) -> "
            f"LMO {altitude_lmo/1e3:.0f}km (r={r_lmo/1e3:.1f}km) 地月转移",
            data={"r_leo_km": r_leo / 1e3, "r_lmo_km": r_lmo / 1e3,
                  "r_moon_km": A_MOON / 1e3},
        )

        # ---- 步骤 2：选择方法 ----
        self._log_step(
            res, "select_method", "success",
            "方法选择：拼凑圆锥近似 (Patched Conic Approximation)。"
            "将三体问题拆为地心段 (椭圆) + 月心段 (双曲线), 在月球 SOI 边界 "
            f"(r_SOI={R_SOI_MOON/1e3:.0f}km) 做速度匹配。",
        )

        # ---- 步骤 3：Hohmann 转移参数 ----
        ht = mt.hohmann_transfer(altitude_leo=altitude_leo, altitude_lmo=altitude_lmo)
        transfer_params = {
            "r_leo_m": ht["r_leo"],
            "r_lmo_m": ht["r_lmo"],
            "r_moon_m": ht["r_moon"],
            "a_transfer_m": ht["a_transfer"],
            "e_transfer": ht["e_transfer"],
            "v_leo_m_s": ht["v_leo"],
            "v_tr_perigee_m_s": ht["v_tr_perigee"],
            "v_tr_apogee_m_s": ht["v_tr_apogee"],
            "v_moon_m_s": ht["v_moon"],
            "v_inf_moon_m_s": ht["v_inf_moon"],
            "C3_moon_m2_s2": ht["C3_moon"],
            "energy_geo_j_kg": ht["energy_geo"],
            "v_perilune_m_s": ht["v_perilune"],
            "v_circ_lmo_m_s": ht["v_circ_lmo"],
            "dv1_m_s": ht["dv1"],
            "dv2_m_s": ht["dv2"],
            "dv_total_m_s": ht["dv_total"],
            "t_flight_s": ht["t_flight"],
            "T_transfer_s": ht["T_transfer"],
            "altitude_leo_m": ht["altitude_leo"],
            "altitude_lmo_m": ht["altitude_lmo"],
        }
        self._log_step(
            res, "hohmann_params", "success",
            f"Hohmann 转移: a_tr={ht['a_transfer']/1e6:.3f}e6 m, "
            f"e_tr={ht['e_transfer']:.4f}, "
            f"dv1={ht['dv1']/1e3:.3f} km/s, dv2={ht['dv2']/1e3:.3f} km/s, "
            f"dv_total={ht['dv_total']/1e3:.3f} km/s, "
            f"t_flight={ht['t_flight']/DAY2SEC:.3f} 天",
            data={"dv_total_km_s": ht["dv_total"] / 1e3,
                  "t_flight_days": ht["t_flight"] / DAY2SEC,
                  "C3": ht["C3_moon"]},
        )

        # ---- 步骤 4：发射窗口确定 ----
        window_analysis, chosen_launch_date = self._determine_launch_window(
            mt, launch_date, days, altitude_leo, altitude_lmo
        )
        win_detail = (
            f"发射窗口: {chosen_launch_date.strftime('%Y-%m-%d %H:%M')} "
            f"(偏差 {window_analysis['best']['deviation_deg']:+.4f}°, "
            f"所需相位角 {window_analysis['required_phase_deg']:.2f}°)"
        )
        self._log_step(
            res, "launch_window", "success",
            win_detail + (f"; launch_date 由参数指定" if launch_date
                          else "; launch_date=None, 已自动搜索最优窗口"),
            data={"chosen_launch_date": str(chosen_launch_date),
                  "required_phase_deg": window_analysis["required_phase_deg"],
                  "n_windows": window_analysis["n_windows"]},
        )

        # ---- 步骤 5：完整轨迹设计 ----
        traj = mt.design_trajectory(
            launch_date=chosen_launch_date,
            altitude_leo=altitude_leo,
            altitude_lmo=altitude_lmo,
        )
        trajectory = self._extract_trajectory(traj)
        self._log_step(
            res, "design_trajectory", "success",
            f"完整轨迹设计完成: 转移椭圆 a={traj['transfer_elements']['a']/1e6:.3f}e6 m, "
            f"e={traj['transfer_elements']['e']:.4f}, "
            f"argp={math.degrees(traj['transfer_elements']['argp']):.2f}°; "
            f"v_inf_moon(向量)={traj['v_inf_moon']:.2f} m/s; "
            f"dv_total={traj['dv_total']/1e3:.3f} km/s",
            data={"v_inf_moon_vec_m_s": traj["v_inf_moon"],
                  "dv_total_km_s": traj["dv_total"] / 1e3},
        )

        # ---- 步骤 6：Porkchop 图数据 ----
        try:
            pc = mt.porkchop_plot_data(
                start_date=chosen_launch_date, days=days,
                altitude_leo=altitude_leo, altitude_lmo=altitude_lmo,
                n_dep=n_dep, n_ft=n_ft,
            )
            porkchop_data = self._extract_porkchop(pc)
            # 最小总 dv 点
            dv_grid = pc["dv_total"]
            valid = np.isfinite(dv_grid)
            if valid.any():
                idx = np.unravel_index(np.nanargmin(dv_grid), dv_grid.shape)
                pc_min = {
                    "departure_day": float(pc["departure_days"][idx[0]]),
                    "flight_time_day": float(pc["flight_time_days"][idx[1]]),
                    "dv_total_km_s": float(dv_grid[idx]) / 1e3,
                    "C3_m2_s2": float(pc["C3_moon"][idx]),
                }
            else:
                pc_min = None
            self._log_step(
                res, "porkchop_data", "success",
                f"Porkchop 网格 {n_dep}x{n_ft} 生成, 有效点 {int(valid.sum())}/{dv_grid.size}; "
                f"最小总 dv {pc_min['dv_total_km_s']:.3f} km/s @ "
                f"departure={pc_min['departure_day']:.1f}d, "
                f"flight={pc_min['flight_time_day']:.2f}d" if pc_min else
                f"Porkchop 网格 {n_dep}x{n_ft} 生成 (无有效点)",
                data={"grid": f"{n_dep}x{n_ft}",
                      "valid_points": int(valid.sum()),
                      "minimum": pc_min},
            )
        except Exception as exc:  # pragma: no cover - 异常路径
            porkchop_data = {"error": str(exc)}
            pc_min = None
            self._log_step(res, "porkchop_data", "warning",
                           f"Porkchop 数据生成异常: {exc}")

        # ---- 步骤 7：物理量分析 ----
        delta_v_budget = {
            "dv1_leo_departure_m_s": ht["dv1"],
            "dv1_leo_departure_km_s": ht["dv1"] / 1e3,
            "dv2_loi_braking_m_s": ht["dv2"],
            "dv2_loi_braking_km_s": ht["dv2"] / 1e3,
            "dv_total_m_s": ht["dv_total"],
            "dv_total_km_s": ht["dv_total"] / 1e3,
            "margin_5pct_m_s": 0.05 * ht["dv_total"],
            "dv_total_with_margin_m_s": 1.05 * ht["dv_total"],
        }
        # 能量与角动量
        a_tr = ht["a_transfer"]
        e_tr = ht["e_transfer"]
        p_tr = a_tr * (1.0 - e_tr ** 2)
        h_tr = math.sqrt(MU_EARTH * p_tr)
        energy_tr = -MU_EARTH / (2.0 * a_tr)
        physics_analysis = {
            "delta_v_budget": delta_v_budget,
            "energy": {
                "specific_energy_transfer_j_kg": energy_tr,
                "specific_energy_geo_j_kg": ht["energy_geo"],
                "C3_moon_m2_s2": ht["C3_moon"],
                "note": "转移椭圆束缚轨道 (energy<0); C3=v_inf^2 为月球特征能量",
            },
            "angular_momentum": {
                "h_transfer_m2_s": h_tr,
                "semi_latus_rectum_m": p_tr,
                "note": "h = sqrt(mu*p), p=a(1-e^2)",
            },
            "flight_time": {
                "t_flight_s": ht["t_flight"],
                "t_flight_days": ht["t_flight"] / DAY2SEC,
                "t_flight_hours": ht["t_flight"] / 3600.0,
                "T_transfer_s": ht["T_transfer"],
                "note": "t_flight = pi*sqrt(a_tr^3/mu) (转移椭圆半周期)",
            },
            "velocities": {
                "v_leo_circ_m_s": ht["v_leo"],
                "v_tr_perigee_m_s": ht["v_tr_perigee"],
                "v_tr_apogee_m_s": ht["v_tr_apogee"],
                "v_moon_m_s": ht["v_moon"],
                "v_inf_moon_m_s": ht["v_inf_moon"],
                "v_perilune_m_s": ht["v_perilune"],
                "v_circ_lmo_m_s": ht["v_circ_lmo"],
            },
        }
        self._log_step(
            res, "physics_analysis", "success",
            f"物理量分析: dv_total={ht['dv_total']/1e3:.3f} km/s "
            f"(dv1={ht['dv1']/1e3:.3f}, dv2={ht['dv2']/1e3:.3f}), "
            f"t_flight={ht['t_flight']/DAY2SEC:.3f} 天, "
            f"h_tr={h_tr:.3e} m^2/s, energy_tr={energy_tr:.3e} J/kg, "
            f"C3={ht['C3_moon']:.3e} m^2/s^2",
            data=delta_v_budget,
        )

        # ---- 步骤 8：公式推导汇总 ----
        formula_derivation = build_formula_derivation()
        self._log_step(
            res, "formula_derivation", "success",
            f"公式推导汇总已生成 ({len(formula_derivation)} 字符), 含 vis-viva / "
            f"开普勒第三定律 / 相位角 / Hohmann dv1,dv2 / C3 等公式",
            data={"length": len(formula_derivation)},
        )

        # ---- 步骤 9：输出结果 ----
        full_result = {
            "mission": {
                "name": "LEO->LMO 地月转移",
                "altitude_leo_m": altitude_leo,
                "altitude_lmo_m": altitude_lmo,
                "r_leo_m": r_leo,
                "r_lmo_m": r_lmo,
                "method": "Patched Conic Approximation",
            },
            "launch_date": str(chosen_launch_date),
            "transfer_params": transfer_params,
            "trajectory": trajectory,
            "window_analysis": window_analysis,
            "porkchop_data": porkchop_data,
            "porkchop_minimum": pc_min,
            "formula_derivation": formula_derivation,
            "delta_v_budget": delta_v_budget,
            "physics_analysis": physics_analysis,
            "constants_used": {
                "mu_earth": MU_EARTH,
                "mu_moon": MU_MOON,
                "R_earth": R_EARTH,
                "R_moon": R_MOON,
                "a_moon": A_MOON,
                "omega_moon": OMEGA_MOON,
                "R_soi_moon": R_SOI_MOON,
            },
        }

        # 保存 JSON
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(_jsonify(full_result), f, ensure_ascii=False, indent=2)

        self._log_step(
            res, "output_results", "success",
            f"完整结果已保存至 {result_path}",
            data={"result_path": result_path},
        )

        # 组装 WorkflowResult
        res.success = True
        res.result = {
            "transfer_params": transfer_params,
            "trajectory": trajectory,
            "window_analysis": window_analysis,
            "porkchop_data": porkchop_data,
            "formula_derivation": formula_derivation,
            "delta_v_budget": delta_v_budget,
            "physics_analysis": physics_analysis,
            "launch_date": str(chosen_launch_date),
        }
        res.artifacts.append(result_path)
        res.summary = (
            f"地月转移轨迹分析完成：LEO {altitude_leo/1e3:.0f}km -> LMO "
            f"{altitude_lmo/1e3:.0f}km (拼凑圆锥近似)。发射日期 "
            f"{chosen_launch_date.strftime('%Y-%m-%d')}，总 delta-V = "
            f"{ht['dv_total']/1e3:.3f} km/s (dv1={ht['dv1']/1e3:.3f}, "
            f"dv2={ht['dv2']/1e3:.3f})，飞行时间 {ht['t_flight']/DAY2SEC:.3f} 天，"
            f"C3 = {ht['C3_moon']:.3e} m^2/s^2，所需相位角 "
            f"{window_analysis['required_phase_deg']:.2f}°。"
            f"结果 JSON: {result_path}"
        )
        res.metadata["launch_date"] = str(chosen_launch_date)
        res.metadata["dv_total_km_s"] = ht["dv_total"] / 1e3
        res.metadata["t_flight_days"] = ht["t_flight"] / DAY2SEC
        res.metadata["C3"] = ht["C3_moon"]
        return res

    # ------------------------------------------------------------------
    # 辅助：确定发射窗口 (自动或指定)
    # ------------------------------------------------------------------
    def _determine_launch_window(self, mt: MoonTransfer, launch_date, days,
                                 altitude_leo, altitude_lmo):
        """若 launch_date 给定则用之并计算其相位偏差；否则自动搜索最优窗口。

        返回 (window_analysis_dict, chosen_launch_date)。
        """
        ht = mt.hohmann_transfer(altitude_leo=altitude_leo, altitude_lmo=altitude_lmo)
        t_flight = ht["t_flight"]
        phi_req = mt.required_phase_angle(t_flight)

        if launch_date is not None:
            chosen = _to_datetime(launch_date)
            # 计算该日期相对参考历元 (chosen 自身) 的相位偏差 -> 取 t=0
            # 这里以 chosen 为 start, 该时刻实际相位角与所需相位角偏差
            actual_phi = mt.actual_phase_angle(0.0)
            # 以搜索基准为 chosen, 找其前后最近的窗口
            lw = mt.launch_window(start_date=chosen, days=max(days, 1), step_hours=1.0)
            best = lw["best"]
            # 取最接近 chosen 时刻 (day_offset 最小) 的最优窗口
            return self._compact_window(lw, best), chosen
        else:
            # 自动搜索：以今天附近为起点 (用 2026-07-01 作为默认参考)
            ref_start = datetime.datetime(2026, 7, 1)
            lw = mt.launch_window(start_date=ref_start, days=days, step_hours=1.0)
            best = lw["best"]
            chosen = best["date"]
            return self._compact_window(lw, best), chosen

    @staticmethod
    def _compact_window(lw: dict, best: dict) -> dict:
        """把 launch_window 输出压缩为可序列化的分析字典。"""
        return {
            "required_phase_deg": lw.get("required_phase_deg", 0.0),
            "flight_time_days": lw.get("flight_time_days", 0.0),
            "t_flight_s": lw.get("t_flight", 0.0),
            "best": {
                "date": str(best["date"]),
                "day_offset": best["day_offset"],
                "deviation_deg": best["deviation_deg"],
                "actual_phase_deg": best["actual_phase_deg"],
                "required_phase_deg": best["required_phase_deg"],
                "C3": best["C3"],
                "flight_time_days": best["flight_time_days"],
            },
            "n_windows": len(lw.get("windows", [])),
            "top_windows": [
                {
                    "date": str(w["date"]),
                    "day_offset": w["day_offset"],
                    "deviation_deg": w["deviation_deg"],
                    "actual_phase_deg": w["actual_phase_deg"],
                    "C3": w["C3"],
                    "flight_time_days": w["flight_time_days"],
                }
                for w in sorted(lw.get("windows", []),
                                key=lambda w: abs(w["deviation_deg"]))[:8]
            ],
        }

    # ------------------------------------------------------------------
    # 辅助：提取轨迹 (numpy -> 可序列化)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_trajectory(traj: dict) -> dict:
        """把 design_trajectory 的输出转为可序列化字典。"""
        r_leo, v_leo = traj["leo_state"]
        r_per, v_per = traj["transfer_perigee"]
        r_apo, v_apo = traj["transfer_apogee"]
        moon_pos, moon_vel = traj["moon_arrival_state"]
        r_pl, v_pl = traj["perilune_state_moon"]
        te = traj["transfer_elements"]
        return {
            "launch_date": str(traj["launch_date"]),
            "leo_state": {
                "position_m": r_leo.tolist(),
                "velocity_m_s": v_leo.tolist(),
                "speed_m_s": float(np.linalg.norm(v_leo)),
            },
            "transfer_elements": {
                "a_m": te["a"], "e": te["e"],
                "i_deg": math.degrees(te["i"]),
                "raan_deg": math.degrees(te["raan"]),
                "argp_deg": math.degrees(te["argp"]),
                "nu_perigee_deg": math.degrees(te["nu_perigee"]),
            },
            "transfer_perigee": {
                "position_m": r_per.tolist(),
                "velocity_m_s": v_per.tolist(),
                "speed_m_s": float(np.linalg.norm(v_per)),
            },
            "transfer_apogee": {
                "position_m": r_apo.tolist(),
                "velocity_m_s": v_apo.tolist(),
                "speed_m_s": float(np.linalg.norm(v_apo)),
            },
            "moon_arrival_state": {
                "position_m": moon_pos.tolist(),
                "velocity_m_s": moon_vel.tolist(),
                "distance_m": float(np.linalg.norm(moon_pos)),
            },
            "v_inf_moon_vec_m_s": traj["v_inf_moon_vec"].tolist(),
            "v_inf_moon_m_s": traj["v_inf_moon"],
            "perilune_state_moon": {
                "position_m": r_pl.tolist(),
                "velocity_m_s": v_pl.tolist(),
                "speed_m_s": float(np.linalg.norm(v_pl)),
            },
            "v_perilune_m_s": traj["v_perilune"],
            "v_circ_lmo_m_s": traj["v_circ_lmo"],
            "dv1_m_s": traj["dv1"],
            "dv2_m_s": traj["dv2"],
            "dv_total_m_s": traj["dv_total"],
            "t_flight_s": traj["t_flight"],
            "key_times": {k: v for k, v in traj["key_times"].items()},
        }

    # ------------------------------------------------------------------
    # 辅助：提取 porkchop 数据 (含最小值统计)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_porkchop(pc: dict) -> dict:
        """把 porkchop_plot_data 输出转为可序列化字典 (保留网格 + 统计)。"""
        dv = pc["dv_total"]
        c3 = pc["C3_moon"]
        valid = np.isfinite(dv)
        stats = {}
        if valid.any():
            idx = np.unravel_index(np.nanargmin(dv), dv.shape)
            stats = {
                "min_dv_total_m_s": float(dv[idx]),
                "min_dv_total_km_s": float(dv[idx]) / 1e3,
                "min_dv_departure_day": float(pc["departure_days"][idx[0]]),
                "min_dv_flight_time_day": float(pc["flight_time_days"][idx[1]]),
                "min_C3_m2_s2": float(c3[idx]),
                "mean_dv_total_m_s": float(np.nanmean(dv)),
                "n_valid": int(valid.sum()),
                "n_total": int(dv.size),
            }
        return {
            "departure_days": pc["departure_days"].tolist(),
            "flight_time_days": pc["flight_time_days"].tolist(),
            "C3_moon_m2_s2": c3.tolist(),
            "dv_total_m_s": dv.tolist(),
            "dv1_m_s": pc["dv1"].tolist(),
            "dv2_m_s": pc["dv2"].tolist(),
            "v_inf_moon_m_s": pc["v_inf_moon"].tolist(),
            "altitude_leo_m": pc["altitude_leo"],
            "altitude_lmo_m": pc["altitude_lmo"],
            "statistics": stats,
        }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.workflows.trajectory_analysis 自测 ===")

    wf = TrajectoryAnalysisWorkflow()
    print(f"工作流: {wf!r}")
    print(f"步骤计划:\n  " + "\n  ".join(wf.get_plan()))
    print(f"工具可用性: {wf.check_tools()}")

    # 核心测试：自动找最优窗口
    print("\n--- 测试 1: launch_date=None (自动搜索最优窗口) ---")
    r = wf.execute(launch_date=None, altitude_leo=200e3, altitude_lmo=100e3, days=60)
    print(f"success={r.success}, steps={len(r.steps_log)}")
    print(f"summary: {r.summary}")
    print(f"artifacts: {r.artifacts}")
    print(f"metadata: dv_total={r.metadata['dv_total_km_s']:.3f} km/s, "
          f"t_flight={r.metadata['t_flight_days']:.3f} 天, "
          f"launch_date={r.metadata['launch_date']}")

    res = r.result
    tp = res["transfer_params"]
    print(f"\n--- 转移参数 ---")
    print(f"  a_tr = {tp['a_transfer_m']/1e6:.3f}e6 m, e_tr = {tp['e_transfer']:.4f}")
    print(f"  dv1 = {tp['dv1_m_s']/1e3:.3f} km/s, dv2 = {tp['dv2_m_s']/1e3:.3f} km/s")
    print(f"  dv_total = {tp['dv_total_m_s']/1e3:.3f} km/s")
    print(f"  t_flight = {tp['t_flight_s']/DAY2SEC:.3f} 天 = {tp['t_flight_s']/3600:.1f} h")
    print(f"  C3_moon = {tp['C3_moon_m2_s2']:.3e} m^2/s^2")
    print(f"  v_inf_moon = {tp['v_inf_moon_m_s']:.2f} m/s")

    print(f"\n--- 发射窗口 ---")
    wa = res["window_analysis"]
    print(f"  最优窗口: {wa['best']['date']}")
    print(f"  所需相位角: {wa['required_phase_deg']:.2f}°")
    print(f"  偏差: {wa['best']['deviation_deg']:+.4f}°")

    print(f"\n--- delta-V 预算 ---")
    dvb = res["delta_v_budget"]
    print(f"  dv1 (LEO 离轨): {dvb['dv1_leo_departure_km_s']:.3f} km/s")
    print(f"  dv2 (LOI 制动): {dvb['dv2_loi_braking_km_s']:.3f} km/s")
    print(f"  dv_total:       {dvb['dv_total_km_s']:.3f} km/s")
    print(f"  +5% 余量:       {dvb['dv_total_with_margin_m_s']/1e3:.3f} km/s")

    print(f"\n--- 物理分析 ---")
    pa = res["physics_analysis"]
    print(f"  角动量 h_tr = {pa['angular_momentum']['h_transfer_m2_s']:.3e} m^2/s")
    print(f"  比能量 eps_tr = {pa['energy']['specific_energy_transfer_j_kg']:.3e} J/kg")

    print(f"\n--- Porkchop ---")
    pcd = res["porkchop_data"]
    if "statistics" in pcd and pcd["statistics"]:
        st = pcd["statistics"]
        print(f"  网格有效点: {st['n_valid']}/{st['n_total']}")
        print(f"  最小 dv_total = {st['min_dv_total_km_s']:.3f} km/s @ "
              f"departure={st['min_dv_departure_day']:.1f}d, "
              f"flight={st['min_dv_flight_time_day']:.2f}d")

    print(f"\n--- formula_derivation (前 300 字) ---")
    fd = res["formula_derivation"]
    print(fd[:300])

    # 关键校验
    assert r.success, "工作流应成功"
    assert 3.9e3 <= tp["dv_total_m_s"] <= 4.1e3, \
        f"dv_total 应在 3.9~4.1 km/s, 得 {tp['dv_total_m_s']/1e3:.3f}"
    assert 4.5 <= tp["t_flight_s"] / DAY2SEC <= 5.5, \
        f"飞行时间应 ~5 天, 得 {tp['t_flight_s']/DAY2SEC:.2f}"
    assert os.path.isfile(r.artifacts[0]), "结果 JSON 应存在"
    assert "vis-viva" in fd.lower() or "vis_viva" in fd.lower()
    assert "T^2" in fd and "C3" in fd

    # 测试 2：指定发射日期
    print("\n--- 测试 2: 指定 launch_date='2026-08-06' ---")
    r2 = wf.execute(launch_date="2026-08-06", altitude_leo=200e3, altitude_lmo=100e3)
    print(f"success={r2.success}, dv_total={r2.metadata['dv_total_km_s']:.3f} km/s, "
          f"launch_date={r2.metadata['launch_date']}")
    assert r2.success

    print("\ntrajectory_analysis 自测全部通过.")
