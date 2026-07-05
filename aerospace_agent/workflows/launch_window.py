"""发射窗口分析工作流 (Launch window analysis workflow)。

针对地月转移任务，扫描日期范围寻找最优发射窗口（相位角匹配）。

核心物理
--------
* 用 :class:`aerospace_agent.physics.moon_transfer.MoonTransfer.launch_window`
  扫描发射窗口（基于 Hohmann 转移相位角匹配 + scipy 精化）。
* 用 ``aerospace_agent.mcp_tools.spiceypy_tool`` 获取月球星历（真实模式
  调用 SPICE，不可用时回退到解析公式），用于参考校验。

输出
----
* 最优窗口日期、相位角、C3、飞行时间
* 候选窗口列表
* CSV 文件 (保存到 ``/workspace/demo_outputs/``)
"""

from __future__ import annotations

import csv
import datetime
import math
import os
from typing import Any, Dict, List, Optional

from aerospace_agent.physics.constants import DAY2SEC
from aerospace_agent.physics.moon_transfer import MoonTransfer

from .base import BaseWorkflow, WorkflowResult, register_workflow

# 输出目录
DEMO_OUTPUTS_DIR = "/workspace/demo_outputs"


def _to_datetime(d) -> datetime.datetime:
    """把 date/datetime/str 转为 datetime.datetime。"""
    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, datetime.date):
        return datetime.datetime(d.year, d.month, d.day)
    if isinstance(d, str):
        return datetime.datetime.fromisoformat(d)
    raise TypeError(f"不支持日期类型: {type(d)}")


def datetime_to_jd(dt: datetime.datetime) -> float:
    """Gregorian datetime -> 儒略日 (JD)。

    采用标准算法 (Meeus, Astronomical Algorithms)。
    """
    year = dt.year
    month = dt.month
    # 包含小数天的日
    day = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return float(jd)


@register_workflow()
class LaunchWindowWorkflow(BaseWorkflow):
    """发射窗口分析工作流。

    name = 'launch_window'
    """

    name = "launch_window"
    description = "地月转移发射窗口分析：扫描日期范围，基于相位角匹配寻找最优发射窗口"
    version = "1.0.0"
    required_tools = ["spiceypy"]

    steps = [
        {"name": "get_moon_ephemeris", "description": "确定目标 (月球) 星历"},
        {"name": "compute_phase_angle", "description": "计算所需相位角"},
        {"name": "scan_date_range", "description": "扫描日期范围"},
        {"name": "evaluate_windows", "description": "评估每个候选窗口"},
        {"name": "output_best_window", "description": "输出最优窗口"},
    ]

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> bool:
        if not super().validate_params(params):
            return False
        start = params.get("start_date", "2026-07-01")
        try:
            _to_datetime(start)
        except Exception:
            return False
        days = params.get("days", 60)
        if not isinstance(days, (int, float)) or days <= 0:
            return False
        return True

    # ------------------------------------------------------------------
    # 主执行
    # ------------------------------------------------------------------
    def execute(
        self,
        start_date: str = "2026-07-01",
        days: int = 60,
        altitude_leo: float = 200e3,
        altitude_lmo: float = 100e3,
        **kwargs,
    ) -> WorkflowResult:
        """执行发射窗口分析。

        Parameters
        ----------
        start_date : str
            搜索起始日期 (ISO 格式, 如 '2026-07-01')。
        days : int
            搜索天数。
        altitude_leo : float
            LEO 停泊轨道高度 [m]。
        altitude_lmo : float
            LMO 近月轨道高度 [m]。
        **kwargs
            额外参数 (如 step_hours 网格步长, output_dir 输出目录)。
        """
        res = WorkflowResult()
        res.metadata["params"] = {
            "start_date": start_date, "days": days,
            "altitude_leo": altitude_leo, "altitude_lmo": altitude_lmo,
        }
        output_dir = kwargs.get("output_dir", DEMO_OUTPUTS_DIR)
        step_hours = kwargs.get("step_hours", 1.0)

        start_dt = _to_datetime(start_date)

        # 步骤 1：确定月球星历 (用 spiceypy_tool, 回退也可)
        moon_ephem = self._get_moon_ephemeris(start_dt)
        tool_source = moon_ephem.get("_source", "fallback")
        self._log_step(
            res, "get_moon_ephemeris", "success",
            f"月球星历来源: {tool_source}; 起始日期 {start_dt.date()} 月球距离 "
            f"{moon_ephem.get('distance_km', float('nan')):.1f} km",
            data={"source": tool_source,
                  "moon_distance_km": moon_ephem.get("distance_km"),
                  "moon_speed_km_s": moon_ephem.get("speed_km_s")},
        )

        # 步骤 2 & 3：用 MoonTransfer 计算相位角并扫描日期范围
        mt = MoonTransfer()
        # 先获取 Hohmann 转移参数以确定所需相位角与飞行时间
        ht = mt.hohmann_transfer(altitude_leo=altitude_leo, altitude_lmo=altitude_lmo)
        t_flight = ht["t_flight"]
        phi_req_deg = math.degrees(mt.required_phase_angle(t_flight))
        C3 = ht["C3_moon"]
        ft_days = t_flight / DAY2SEC

        self._log_step(
            res, "compute_phase_angle", "success",
            f"所需月球相位角 = {phi_req_deg:.2f}°, 飞行时间 = {ft_days:.3f} 天, "
            f"C3 = {C3:.3e} m^2/s^2",
            data={"required_phase_deg": phi_req_deg,
                  "flight_time_days": ft_days, "C3": C3},
        )

        try:
            lw = mt.launch_window(
                start_date=start_dt, days=days, step_hours=step_hours,
            )
        except Exception as exc:  # pragma: no cover - 异常路径
            self._log_step(res, "scan_date_range", "failed", f"扫描异常: {exc}")
            res.summary = f"发射窗口扫描失败: {exc}"
            return res

        n_cand = len(lw["candidates"])
        n_win = len(lw["windows"])
        self._log_step(
            res, "scan_date_range", "success",
            f"扫描 {days} 天 (步长 {step_hours}h): {n_cand} 个候选点, "
            f"{n_win} 个局部最优窗口",
            data={"n_candidates": n_cand, "n_windows": n_win},
        )

        # 步骤 4：评估每个候选窗口 (取偏差最小的前若干个)
        best = lw["best"]
        # 局部窗口按偏差绝对值排序
        sorted_windows = sorted(lw["windows"], key=lambda w: abs(w["deviation_deg"]))
        top_windows = sorted_windows[:min(10, len(sorted_windows))]

        self._log_step(
            res, "evaluate_windows", "success",
            f"最优窗口: {best['date']} (偏移 {best['day_offset']:.2f} 天), "
            f"偏差 {best['deviation_deg']:+.4f}°, 实际相位 {best['actual_phase_deg']:.2f}°",
            data={"best_date": str(best["date"]),
                  "best_deviation_deg": best["deviation_deg"],
                  "top_windows_count": len(top_windows)},
        )

        # 步骤 5：输出最优窗口 + 保存 CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(
            output_dir,
            f"launch_window_{start_dt.strftime('%Y%m%d')}_{days}d.csv",
        )
        self._save_candidates_csv(lw["candidates"], csv_path)

        self._log_step(
            res, "output_best_window", "success",
            f"最优窗口已确定，候选列表已保存至 {csv_path}",
            data={"csv_path": csv_path},
        )

        # 组装结果
        best_date = best["date"]
        result = {
            "start_date": str(start_dt.date()),
            "days_scanned": days,
            "altitude_leo_m": altitude_leo,
            "altitude_lmo_m": altitude_lmo,
            "required_phase_deg": phi_req_deg,
            "flight_time_days": ft_days,
            "C3_m2_s2": C3,
            "best_window": {
                "date": str(best_date),
                "day_offset": best["day_offset"],
                "deviation_deg": best["deviation_deg"],
                "actual_phase_deg": best["actual_phase_deg"],
                "required_phase_deg": best["required_phase_deg"],
                "C3": best["C3"],
                "flight_time_days": best["flight_time_days"],
            },
            "candidate_windows": [
                {
                    "date": str(w["date"]),
                    "day_offset": w["day_offset"],
                    "deviation_deg": w["deviation_deg"],
                    "actual_phase_deg": w["actual_phase_deg"],
                    "C3": w["C3"],
                    "flight_time_days": w["flight_time_days"],
                }
                for w in top_windows
            ],
            "n_candidates": n_cand,
            "n_local_windows": n_win,
            "moon_ephemeris_source": tool_source,
            "moon_state_at_start": {
                "distance_km": moon_ephem.get("distance_km"),
                "speed_km_s": moon_ephem.get("speed_km_s"),
                "ecliptic_longitude_deg": moon_ephem.get("ecliptic_longitude_deg"),
            },
        }

        res.success = True
        res.result = result
        res.artifacts.append(csv_path)
        res.summary = (
            f"发射窗口分析完成：起始 {start_dt.date()}，扫描 {days} 天，"
            f"找到 {n_win} 个候选窗口。最优发射窗口为 {best_date.strftime('%Y-%m-%d %H:%M')}，"
            f"相位偏差 {best['deviation_deg']:+.4f}°，所需相位角 {phi_req_deg:.2f}°，"
            f"C3={C3:.3e} m^2/s^2，飞行时间 {ft_days:.2f} 天。"
            f"候选窗口 CSV 已保存: {csv_path}"
        )
        res.metadata["best_date"] = str(best_date)
        res.metadata["tool_source"] = tool_source
        return res

    # ------------------------------------------------------------------
    # 辅助：获取月球星历 (走 mcp_tools 统一接口)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_moon_ephemeris(start_dt: datetime.datetime) -> dict:
        """通过 spiceypy_tool 获取起始时刻月球状态 (回退模式亦可)。"""
        jd = datetime_to_jd(start_dt)
        try:
            from aerospace_agent.mcp_tools import get_tool
            tool = get_tool("spiceypy")
            if tool is None:
                # 直接实例化回退
                from aerospace_agent.mcp_tools.spiceypy_tool import SpiceypyTool
                tool = SpiceypyTool()
            resp = tool.call("get_moon_state", epoch=jd)
            if resp.get("success"):
                data = resp["result"]
                data["_source"] = resp.get("source", "fallback")
                return data
            # 失败则回退到 MoonTransfer 圆轨道
        except Exception:
            pass
        # 最终回退：用 MoonTransfer 圆轨道近似
        mt = MoonTransfer()
        pos, vel = mt._moon_state(0.0)
        return {
            "position_km": (pos / 1e3).tolist(),
            "velocity_km_s": (vel / 1e3).tolist(),
            "distance_km": float(pos[0] / 1e3),
            "speed_km_s": float(abs(vel[1]) / 1e3),
            "_source": "fallback_moon_transfer",
            "epoch_jd": jd,
            "frame": "J2000",
        }

    # ------------------------------------------------------------------
    # 辅助：保存候选窗口到 CSV
    # ------------------------------------------------------------------
    @staticmethod
    def _save_candidates_csv(candidates: List[dict], path: str) -> None:
        """将候选窗口列表保存为 CSV。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fieldnames = [
            "date", "day_offset", "required_phase_deg",
            "actual_phase_deg", "deviation_deg", "C3", "flight_time_days",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in candidates:
                writer.writerow({
                    "date": str(c["date"]),
                    "day_offset": f"{c['day_offset']:.6f}",
                    "required_phase_deg": f"{c['required_phase_deg']:.4f}",
                    "actual_phase_deg": f"{c['actual_phase_deg']:.4f}",
                    "deviation_deg": f"{c['deviation_deg']:.4f}",
                    "C3": f"{c['C3']:.6e}",
                    "flight_time_days": f"{c['flight_time_days']:.6f}",
                })


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.workflows.launch_window 自测 ===")

    wf = LaunchWindowWorkflow()
    print(f"工作流: {wf!r}")
    print(f"步骤计划:\n  " + "\n  ".join(wf.get_plan()))
    print(f"工具可用性: {wf.check_tools()}")

    r = wf.execute(start_date="2026-07-01", days=60,
                   altitude_leo=200e3, altitude_lmo=100e3)
    print(f"\nsuccess={r.success}, steps={len(r.steps_log)}")
    print(f"summary: {r.summary}")
    print(f"artifacts: {r.artifacts}")

    d = r.result
    print(f"\n--- 最优窗口 ---")
    print(f"  日期: {d['best_window']['date']}")
    print(f"  偏移: {d['best_window']['day_offset']:.3f} 天")
    print(f"  相位偏差: {d['best_window']['deviation_deg']:+.4f}°")
    print(f"  实际相位: {d['best_window']['actual_phase_deg']:.2f}°")
    print(f"  所需相位: {d['best_window']['required_phase_deg']:.2f}°")
    print(f"  C3: {d['best_window']['C3']:.3e} m^2/s^2")
    print(f"  飞行时间: {d['best_window']['flight_time_days']:.3f} 天")
    print(f"\n  候选窗口数: {d['n_candidates']}, 局部最优数: {d['n_local_windows']}")
    print(f"  Top 候选窗口 (前5):")
    for w in d["candidate_windows"][:5]:
        print(f"    {w['date']}  偏差={w['deviation_deg']:+.4f}°  "
              f"相位={w['actual_phase_deg']:.2f}°")

    assert r.success
    assert abs(d["best_window"]["deviation_deg"]) < 1.0, "最优窗口偏差应 < 1°"
    assert os.path.isfile(r.artifacts[0]), "CSV 文件应存在"
    print(f"\nCSV 文件大小: {os.path.getsize(r.artifacts[0])} bytes")

    print("\nlaunch_window 自测全部通过.")
