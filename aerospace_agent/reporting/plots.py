"""绘图模块 (Plotting module) — 地月转移轨道可视化。

:class:`Plotter` 提供统一的绘图接口, 所有图保存为 PNG (默认 dpi=150)。

数据兼容性
----------
Plotter 各方法同时兼容两种输入格式:
1. ``MoonTransfer`` 原始输出 (numpy 数组, 键名如 ``a_transfer`` / ``dv1``)
2. ``WorkflowResult.result`` 子字典 (JSON 可序列化, SI 后缀键名如
   ``a_transfer_m`` / ``dv1_m_s``, 列表而非数组)

方法
----
    * :meth:`plot_transfer_trajectory`  3D 地月转移轨迹
    * :meth:`plot_delta_v_budget`       Δv 预算柱状图
    * :meth:`plot_porkchop`             Porkchop 等高线图
    * :meth:`plot_launch_windows`       发射窗口相位角图
    * :meth:`plot_orbit_elements`       轨道元素对比
    * :meth:`plot_energy_analysis`      能量分析
    * :meth:`plot_mission_timeline`     任务时间线甘特图

字体: 优先 ``Noto Sans CJK SC`` / ``WenQuanYi Micro Hei``; 找不到则全英文。
颜色: 地球蓝 #4A90D9, 月球灰 #AAAAAA, 转移红 #E74C3C, 机动绿 #7BC47F。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端, 无 DISPLAY 也可
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (注册 3D 投影)

from ..physics.constants import (
    A_MOON, DAY2SEC, MU_EARTH, MU_MOON, R_EARTH, R_MOON, R_SOI_MOON,
)


# ---------------------------------------------------------------------------
# 字体探测 (模块级, 只执行一次)
# ---------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "SimHei", "Microsoft YaHei", "PingFang SC", "Source Han Sans SC",
]


def _detect_cjk_font() -> Optional[str]:
    """探测可用的 CJK 字体名; 无则返回 None。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


_CJK_FONT = _detect_cjk_font()
_CJK_AVAILABLE = _CJK_FONT is not None

if _CJK_AVAILABLE:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 颜色方案 (航天工程报告风格)
# ---------------------------------------------------------------------------
COLOR_EARTH = "#4A90D9"
COLOR_MOON = "#AAAAAA"
COLOR_TRANSFER = "#E74C3C"
COLOR_MANEUVER = "#7BC47F"
COLOR_LEO = "#5DADE2"
COLOR_LMO = "#BB8FCE"
COLOR_SOI = "#F39C12"
COLOR_GRID = "#CCCCCC"


# ---------------------------------------------------------------------------
# 灵活取值辅助 (兼容两种键名格式)
# ---------------------------------------------------------------------------
def _first(d: Dict, *keys, default=None):
    """从 dict 中取第一个存在的键值。"""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default


def _to_array(v) -> np.ndarray:
    """把 list / numpy / 标量 转为一维 numpy 数组。"""
    return np.asarray(v, dtype=float)


def _to_array2d(v) -> np.ndarray:
    """把嵌套 list / 2D numpy 转为二维 numpy 数组。"""
    return np.asarray(v, dtype=float)


def _extract_pos(item) -> Optional[np.ndarray]:
    """从多种格式提取位置向量 [x, y, z]。

    支持:
    - None
    - dict: {"position_m": [...]} / {"position": [...]} / {"r": [...]}
    - (r, v) 元组 (raw MoonTransfer): 取第一个元素
    - 一维数组 [x, y, z]
    """
    if item is None:
        return None
    if isinstance(item, dict):
        for k in ("position_m", "position", "r", "pos"):
            if k in item:
                return np.asarray(item[k], dtype=float).ravel()[:3]
        return None
    arr = np.asarray(item, dtype=float)
    if arr.ndim == 2:  # (r, v) 元组
        return arr[0].ravel()[:3]
    if arr.ndim == 1 and arr.size >= 3:
        return arr[:3]
    return None


class Plotter:
    """统一绘图接口, 所有图保存到指定路径 (PNG, dpi=150)。

    参数
    -----
    output_dir : 默认输出目录, 方法未指定 output_path 时使用
    dpi        : 输出分辨率
    """

    EARTH = COLOR_EARTH
    MOON = COLOR_MOON
    TRANSFER = COLOR_TRANSFER
    MANEUVER = COLOR_MANEUVER
    LEO = COLOR_LEO
    LMO = COLOR_LMO
    SOI = COLOR_SOI

    def __init__(self, output_dir: str = "/workspace/demo_outputs", dpi: int = 150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)
        self.cjk_available = _CJK_AVAILABLE
        self.cjk_font = _CJK_FONT

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _resolve_path(self, output_path: Optional[str], default_name: str) -> str:
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            return output_path
        return os.path.join(self.output_dir, default_name)

    def _save(self, fig, path: str) -> str:
        try:
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        except (MemoryError, ValueError):
            # bbox_inches="tight" 在极端注释位置时可能爆炸, 回退到普通保存
            fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return os.path.abspath(path)

    def _title(self, cn: str, en: str) -> str:
        if self.cjk_available:
            return f"{cn}\n{en}"
        return en

    @staticmethod
    def _build_timeline(t_flight_s: float, key_times: Optional[dict] = None) -> list:
        """从飞行时间构造任务时间线 (供 plot_mission_timeline 用)。"""
        t_soi = (key_times or {}).get("t_soi_entry_approx", t_flight_s * 0.85)
        tf = t_flight_s / DAY2SEC
        ts = t_soi / DAY2SEC
        return [
            {"phase": "LEO Parking", "phase_cn": "LEO 停泊",
             "start_day": 0.0, "end_day": 0.0, "duration_day": 0.0,
             "color": COLOR_EARTH},
            {"phase": "TLI Maneuver", "phase_cn": "TLI 机动",
             "start_day": 0.0, "end_day": 0.02, "duration_day": 0.02,
             "color": COLOR_MANEUVER},
            {"phase": "Trans-Lunar Coast", "phase_cn": "转移飞行",
             "start_day": 0.02, "end_day": ts,
             "duration_day": max(ts - 0.02, 0.01), "color": COLOR_TRANSFER},
            {"phase": "SOI Crossing", "phase_cn": "SOI 穿越",
             "start_day": ts, "end_day": tf,
             "duration_day": max(tf - ts, 0.01), "color": COLOR_SOI},
            {"phase": "LOI Burn", "phase_cn": "LOI 制动",
             "start_day": tf, "end_day": tf + 0.02, "duration_day": 0.02,
             "color": COLOR_MANEUVER},
            {"phase": "LMO Operations", "phase_cn": "LMO 运行",
             "start_day": tf + 0.02, "end_day": tf + 10.0,
             "duration_day": 10.0 - 0.02, "color": COLOR_LMO},
        ]

    # ------------------------------------------------------------------
    # 1. 3D 地月转移轨迹
    # ------------------------------------------------------------------
    def plot_transfer_trajectory(
        self, transfer_params: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """3D 地月转移轨迹图。

        画地球(蓝球)、月球轨道(灰虚线)、转移椭圆(红实线)、LEO 停泊轨道、
        LMO, 标注 TLI 点、LOI 点、SOI 边界。

        参数兼容: ``MoonTransfer.design_trajectory`` 输出, 或
        ``WorkflowResult.result["trajectory"]`` 字典 (可含 hohmann 参数)。
        """
        # --- 提取轨道参数 (兼容两种格式) ---
        a_tr = _first(transfer_params, "a_transfer", "a_transfer_m", "a_m",
                      default=A_MOON / 2 + R_EARTH)
        e_tr = _first(transfer_params, "e_transfer", "e", default=0.0)
        r_leo = _first(transfer_params, "r_leo", "r_leo_m", default=R_EARTH + 200e3)
        r_lmo = _first(transfer_params, "r_lmo", "r_lmo_m", default=R_MOON + 100e3)
        r_moon = _first(transfer_params, "r_moon", "r_moon_m", default=A_MOON)

        # transfer_elements (兼容嵌套 dict 与扁平)
        te = _first(transfer_params, "transfer_elements", default={})
        argp = _first(te, "argp", "argp_deg", default=0.0)
        # argp_deg 是度, argp 是弧度
        if "argp_deg" in te and "argp" not in te:
            import math
            argp = float(argp) * math.pi / 180.0

        # 位置向量
        perigee = _extract_pos(_first(transfer_params, "transfer_perigee", default=None))
        apogee = _extract_pos(_first(transfer_params, "transfer_apogee", default=None))
        moon_arr = _extract_pos(
            _first(transfer_params, "moon_arrival_state", default=None))

        # 回退: 用 argp 方向构造
        if perigee is None:
            perigee = np.array([r_leo * np.cos(argp), r_leo * np.sin(argp), 0.0])
        if moon_arr is None:
            moon_arr = np.array([r_moon * np.cos(argp + np.pi),
                                  r_moon * np.sin(argp + np.pi), 0.0])
        if apogee is None:
            apogee = moon_arr

        # --- 转移椭圆参数化 (真近点角 0 -> pi) ---
        nu = np.linspace(0, np.pi, 200)
        p = a_tr * (1 - e_tr ** 2)
        r_nu = p / (1 + e_tr * np.cos(nu))
        x_tr = r_nu * np.cos(nu + argp)
        y_tr = r_nu * np.sin(nu + argp)
        z_tr = np.zeros_like(x_tr)

        # --- 月球轨道 (圆, 灰虚线) ---
        theta_m = np.linspace(0, 2 * np.pi, 200)
        x_m = A_MOON * np.cos(theta_m)
        y_m = A_MOON * np.sin(theta_m)

        # --- LEO 停泊轨道 ---
        leo_dir = perigee / np.linalg.norm(perigee)
        tang = np.array([-leo_dir[1], leo_dir[0], 0.0])
        leo_theta = np.linspace(0, 2 * np.pi, 100)
        x_leo = r_leo * (leo_dir[0] * np.cos(leo_theta) + tang[0] * np.sin(leo_theta))
        y_leo = r_leo * (leo_dir[1] * np.cos(leo_theta) + tang[1] * np.sin(leo_theta))

        # --- LMO (月球处小圆) ---
        lmo_theta = np.linspace(0, 2 * np.pi, 60)
        x_lmo = moon_arr[0] + r_lmo * np.cos(lmo_theta)
        y_lmo = moon_arr[1] + r_lmo * np.sin(lmo_theta)

        # --- SOI 边界 ---
        soi_theta = np.linspace(0, 2 * np.pi, 80)
        x_soi = moon_arr[0] + R_SOI_MOON * np.cos(soi_theta)
        y_soi = moon_arr[1] + R_SOI_MOON * np.sin(soi_theta)

        # --- 绘图 ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # 地球 (蓝色球, 单位 km)
        u_s, v_s = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 30))
        ax.plot_surface(
            (R_EARTH / 1e3) * np.outer(np.cos(u_s), np.sin(v_s)),
            (R_EARTH / 1e3) * np.outer(np.sin(u_s), np.sin(v_s)),
            (R_EARTH / 1e3) * np.outer(np.ones_like(u_s), np.cos(v_s)),
            color=self.EARTH, alpha=0.55, linewidth=0)

        # 月球 (灰色小球, 单位 km)
        u_m, v_m = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20))
        ax.plot_surface(
            moon_arr[0] / 1e3 + (R_MOON / 1e3) * np.outer(np.cos(u_m), np.sin(v_m)),
            moon_arr[1] / 1e3 + (R_MOON / 1e3) * np.outer(np.sin(u_m), np.sin(v_m)),
            (R_MOON / 1e3) * np.outer(np.ones_like(u_m), np.cos(v_m)),
            color=self.MOON, alpha=0.7, linewidth=0)

        ax.plot(x_m / 1e3, y_m / 1e3, np.zeros_like(x_m) / 1e3,
                color=self.MOON, ls="--", lw=1.2, label="Moon Orbit")
        ax.plot(x_tr / 1e3, y_tr / 1e3, z_tr / 1e3,
                color=self.TRANSFER, lw=2.2, label="Transfer Ellipse")
        ax.plot(x_leo / 1e3, y_leo / 1e3, np.zeros_like(x_leo) / 1e3,
                color=self.LEO, lw=1.5, label="LEO Parking")
        ax.plot(x_lmo / 1e3, y_lmo / 1e3, np.zeros_like(x_lmo) / 1e3,
                color=self.LMO, lw=1.5, label="LMO")
        ax.plot(x_soi / 1e3, y_soi / 1e3, np.zeros_like(x_soi) / 1e3,
                color=self.SOI, ls=":", lw=1.5, label="Moon SOI")

        # TLI / LOI 点
        ax.scatter(*[perigee[i] / 1e3 for i in range(3)],
                   color=self.MANEUVER, s=80, marker="^", zorder=5,
                   label="TLI (Trans-Lunar Injection)")
        ax.scatter(*[moon_arr[i] / 1e3 for i in range(3)],
                   color=self.MANEUVER, s=80, marker="v", zorder=5,
                   label="LOI (Lunar Orbit Insertion)")

        tli_lbl = "TLI" if not self.cjk_available else "TLI 机动点"
        loi_lbl = "LOI" if not self.cjk_available else "LOI 制动点"
        soi_lbl = "SOI" if not self.cjk_available else "SOI 边界"
        ax.text(perigee[0] / 1e3, perigee[1] / 1e3, perigee[2] / 1e3 + 3000,
                tli_lbl, color=self.MANEUVER, fontsize=9, fontweight="bold")
        ax.text(moon_arr[0] / 1e3, moon_arr[1] / 1e3, moon_arr[2] / 1e3 + 5000,
                loi_lbl, color=self.MANEUVER, fontsize=9, fontweight="bold")
        ax.text(moon_arr[0] / 1e3 + R_SOI_MOON / 1e3 * 0.6,
                moon_arr[1] / 1e3, 0, soi_lbl, color=self.SOI, fontsize=8)

        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        ax.set_title(self._title("地月转移轨迹 (3D)", "Earth-Moon Transfer Trajectory (3D)"))
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        max_r = A_MOON / 1e3 * 1.1
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_zlim(-max_r * 0.3, max_r * 0.3)
        try:
            ax.set_box_aspect((1, 1, 0.4))
        except Exception:
            pass

        path = self._resolve_path(output_path, "transfer_trajectory.png")
        return self._save(fig, path)

    # ------------------------------------------------------------------
    # 2. Δv 预算柱状图
    # ------------------------------------------------------------------
    def plot_delta_v_budget(
        self, delta_v_budget: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """Δv 预算柱状图 (TLI / LOI / 总), 单位 km/s。

        兼容: ``WorkflowResult.result["delta_v_budget"]`` (dv1_leo_departure_*)
        或 Hohmann dict (dv1 / dv2 / dv_total) 或 {TLI, LOI, total}。
        """
        def _kms(*keys):
            for k in keys:
                if k in delta_v_budget:
                    v = float(delta_v_budget[k])
                    return v / 1e3 if k.endswith("_m_s") or v > 100 else v
            return 0.0

        dv_tli = _kms("TLI", "dv1", "dv1_leo_departure_m_s", "dv1_leo_departure_km_s")
        dv_loi = _kms("LOI", "dv2", "dv2_loi_braking_m_s", "dv2_loi_braking_km_s")
        dv_total = _kms("total", "dv_total", "dv_total_m_s", "dv_total_km_s")
        # 若只有 m/s 值但没找到 km/s, 重新判断
        if dv_total == 0.0:
            dv_total = dv_tli + dv_loi

        margin = _first(delta_v_budget, "margin_5pct_m_s", default=None)
        dv_with_margin = _first(delta_v_budget, "dv_total_with_margin_m_s", default=None)

        fig, ax = plt.subplots(figsize=(8, 5.5))

        labels_en = ["TLI\n(LEO Departure)", "LOI\n(Lunar Capture)", "Total"]
        labels_cn = ["TLI\n(LEO 离轨)", "LOI\n(近月制动)", "总计"]
        use_labels = labels_cn if self.cjk_available else labels_en
        values = [dv_tli, dv_loi, dv_total]
        colors = [self.MANEUVER, self.SOI, self.TRANSFER]

        bars = ax.bar(use_labels, values, color=colors, edgecolor="white",
                      width=0.55, zorder=3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.3f} km/s", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.axhline(dv_tli + dv_loi, color="gray", ls="--", lw=1, alpha=0.6,
                   label=f"TLI + LOI = {dv_tli + dv_loi:.3f} km/s")
        if dv_with_margin is not None:
            wm = float(dv_with_margin) / 1e3
            ax.axhline(wm, color=self.LMO, ls=":", lw=1.2, alpha=0.7,
                       label=f"+5% margin = {wm:.3f} km/s")

        ax.set_ylabel(r"$\Delta v$ [km/s]")
        ax.set_title(self._title("Δv 预算", "Delta-V Budget"))
        ax.grid(True, axis="y", alpha=0.4, color=COLOR_GRID)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values + [dv_tli + dv_loi]) * 1.25)
        ax.legend(fontsize=9)

        path = self._resolve_path(output_path, "delta_v_budget.png")
        return self._save(fig, path)

    # ------------------------------------------------------------------
    # 3. Porkchop 等高线图
    # ------------------------------------------------------------------
    def plot_porkchop(
        self, porkchop_data: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """Porkchop 等高线图 (C3 能量 + 总 Δv)。

        兼容: ``MoonTransfer.porkchop_plot_data`` 输出 (numpy 数组, C3_moon)
        或 ``WorkflowResult.result["porkchop_data"]`` (列表, C3_moon_m2_s2)。
        """
        dep_days = _to_array(_first(porkchop_data, "departure_days", default=[]))
        ft_days = _to_array(_first(porkchop_data, "flight_time_days", default=[]))
        C3 = _to_array2d(_first(porkchop_data, "C3_moon", "C3_moon_m2_s2", default=[]))
        dv_total = _to_array2d(_first(porkchop_data, "dv_total", "dv_total_m_s",
                                      default=[]))

        if dep_days.size == 0 or ft_days.size == 0 or C3.size == 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No porkchop data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
            path = self._resolve_path(output_path, "porkchop.png")
            return self._save(fig, path)

        X, Y = np.meshgrid(dep_days, ft_days, indexing="ij")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # C3
        ax1 = axes[0]
        C3_plot = C3 / 1e6  # -> km^2/s^2
        lv = np.linspace(np.nanmin(C3_plot), np.nanmax(C3_plot), 20)
        cs1 = ax1.contourf(X, Y, C3_plot, levels=lv, cmap="plasma")
        ax1.contour(X, Y, C3_plot, levels=lv, colors="k", linewidths=0.4, alpha=0.4)
        cb1 = fig.colorbar(cs1, ax=ax1)
        cb1.set_label(r"$C_3$ [km$^2$/s$^2$]")
        ax1.set_xlabel("Departure Day Offset")
        ax1.set_ylabel("Flight Time [day]")
        ax1.set_title(self._title("月球特征能量 C3", "Lunar C3 Characteristic Energy"))

        # Δv
        ax2 = axes[1]
        dv_plot = dv_total / 1e3  # -> km/s
        lv2 = np.linspace(np.nanmin(dv_plot), np.nanmax(dv_plot), 20)
        cs2 = ax2.contourf(X, Y, dv_plot, levels=lv2, cmap="viridis")
        ax2.contour(X, Y, dv_plot, levels=lv2, colors="k", linewidths=0.4, alpha=0.4)
        cb2 = fig.colorbar(cs2, ax=ax2)
        cb2.set_label(r"Total $\Delta v$ [km/s]")
        ax2.set_xlabel("Departure Day Offset")
        ax2.set_ylabel("Flight Time [day]")
        ax2.set_title(self._title("总 Δv", "Total Delta-V"))

        # 最优窗口
        if np.isfinite(dv_total).any():
            idx = np.unravel_index(np.nanargmin(dv_total), dv_total.shape)
            best_dep = dep_days[idx[0]]
            best_ft = ft_days[idx[1]]
            best_dv = dv_total[idx] / 1e3
            for ax in axes:
                ax.scatter([best_dep], [best_ft], color="red", s=90,
                           marker="*", zorder=5, edgecolors="white", linewidths=1.2)
            ax2.annotate(
                f"Optimal\n({best_dep:.1f}d, {best_ft:.2f}d)\n"
                f"$\\Delta v$={best_dv:.3f} km/s",
                xy=(best_dep, best_ft),
                xytext=(50, 40), textcoords="offset points",
                fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))

        fig.suptitle(self._title("Porkchop 发射窗口分析",
                                  "Porkchop Launch Window Analysis"),
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = self._resolve_path(output_path, "porkchop.png")
        return self._save(fig, path)

    # ------------------------------------------------------------------
    # 4. 发射窗口相位角图
    # ------------------------------------------------------------------
    def plot_launch_windows(
        self, window_analysis: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """发射窗口相位角偏差图。

        兼容: ``MoonTransfer.launch_window`` 输出 (含 candidates / windows / best)
        或 ``WorkflowResult.result["window_analysis"]`` (含 top_windows / best)。
        若有 candidates 则画连续曲线, 否则用 top_windows 离散点。
        """
        candidates = window_analysis.get("candidates", [])
        windows = window_analysis.get("windows", window_analysis.get("top_windows", []))
        best = window_analysis.get("best")

        fig, ax = plt.subplots(figsize=(11, 5.5))

        if candidates:
            # 连续曲线 (raw 格式)
            dates = [c["day_offset"] for c in candidates]
            devs = [c["deviation_deg"] for c in candidates]
            ax.plot(dates, devs, color=self.EARTH, lw=1.2, alpha=0.85,
                    label="Phase Deviation")
            ax.fill_between(dates, devs, 0, alpha=0.12, color=self.EARTH)
            ax.axhline(0, color="gray", lw=0.8)
            for w in windows:
                ax.scatter([w["day_offset"]], [w["deviation_deg"]],
                           color=self.MANEUVER, s=45, zorder=4,
                           edgecolors="white", linewidths=0.6)
        elif windows:
            # 离散点 (compact 格式)
            ws = sorted(windows, key=lambda w: w.get("day_offset", 0))
            days = [w["day_offset"] for w in ws]
            devs = [w["deviation_deg"] for w in ws]
            ax.plot(days, devs, color=self.EARTH, lw=1.0, alpha=0.6,
                    marker="o", ls="-", label="Top Windows")
            ax.axhline(0, color="gray", lw=0.8)

        # 最优窗口 (红点)
        if best:
            ax.scatter([best["day_offset"]], [best["deviation_deg"]],
                       color="red", s=130, marker="*", zorder=6,
                       edgecolors="white", linewidths=1.2,
                       label=f"Best Window ({best['day_offset']:.2f}d)")
            date_str = str(best.get("date", ""))[:10]
            ax.annotate(
                f"Best: {date_str}\n$\\Delta\\varphi$={best['deviation_deg']:+.3f}°",
                xy=(best["day_offset"], best["deviation_deg"]),
                xytext=(45, 35), textcoords="offset points",
                fontsize=9, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

        # 保证 y 轴有合理可视范围 (避免偏差接近 0 退化)
        all_devs = []
        if candidates:
            all_devs = [c["deviation_deg"] for c in candidates]
        elif windows:
            all_devs = [w["deviation_deg"] for w in windows]
        if best:
            all_devs.append(best["deviation_deg"])
        if all_devs:
            ymin, ymax = min(all_devs), max(all_devs)
            span = max(abs(ymin), abs(ymax), 1.0)
            ax.set_ylim(min(ymin - span * 0.2, -span * 0.3),
                        max(ymax + span * 0.2, span * 0.3))

        ax.set_xlabel("Day Offset from Epoch")
        ax.set_ylabel("Phase Angle Deviation [deg]")
        ax.set_title(self._title("发射窗口相位角分析",
                                  "Launch Window Phase Angle Analysis"))
        ax.grid(True, alpha=0.4, color=COLOR_GRID)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=9)

        req = window_analysis.get("required_phase_deg")
        ft = window_analysis.get("flight_time_days")
        if req is not None:
            ax.text(0.02, 0.97,
                    f"Required phase = {req:.2f}°\n"
                    f"Flight time = {ft:.3f} d" if ft else f"Required phase = {req:.2f}°",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

        path = self._resolve_path(output_path, "launch_windows.png")
        return self._save(fig, path)

    # ------------------------------------------------------------------
    # 5. 轨道元素对比图
    # ------------------------------------------------------------------
    def plot_orbit_elements(
        self, orbit_elements: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """轨道元素对比图 (半长轴 / 偏心率 / 倾角 / 周期)。

        参数: dict 含 LEO / Transfer / LMO 子字典, 每个含
        semi_major_axis_km / eccentricity / inclination_deg / period_min。
        也接受直接含这些键的扁平 dict (自动构造)。
        """
        # 若传入的是 transfer_params, 自动构造三组轨道元素
        if "LEO" not in orbit_elements and "a_transfer_m" in orbit_elements:
            import math
            a_tr = orbit_elements["a_transfer_m"]
            r_leo = orbit_elements["r_leo_m"]
            r_lmo = orbit_elements["r_lmo_m"]
            T_tr = orbit_elements.get("T_transfer_s", 2 * math.pi * (a_tr**3/MU_EARTH)**0.5)
            orbit_elements = {
                "LEO": {"semi_major_axis_km": r_leo / 1e3, "eccentricity": 0.0,
                        "inclination_deg": 28.5,
                        "period_min": 2*math.pi*(r_leo**3/MU_EARTH)**0.5/60},
                "Transfer": {"semi_major_axis_km": a_tr / 1e3,
                             "eccentricity": orbit_elements["e_transfer"],
                             "inclination_deg": 28.5, "period_min": T_tr / 60},
                "LMO": {"semi_major_axis_km": r_lmo / 1e3, "eccentricity": 0.0,
                        "inclination_deg": 0.0,
                        "period_min": 2*math.pi*(r_lmo**3/MU_MOON)**0.5/60},
            }

        bodies = ["LEO", "Transfer", "LMO"]
        bodies_cn = ["LEO 停泊", "转移椭圆", "LMO 月轨"]
        use_bodies = bodies_cn if self.cjk_available else bodies

        def _g(b, k, d=0.0):
            return orbit_elements.get(b, {}).get(k, d)

        sma = [_g(b, "semi_major_axis_km") for b in bodies]
        ecc = [_g(b, "eccentricity") for b in bodies]
        inc = [_g(b, "inclination_deg") for b in bodies]
        per = [_g(b, "period_min") for b in bodies]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        colors = [self.LEO, self.TRANSFER, self.LMO]

        def _bar(ax, vals, title_cn, title_en, ylabel, fmt="{:.1f}"):
            bars = ax.bar(use_bodies, vals, color=colors, edgecolor="white", width=0.55)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt.format(val), ha="center", va="bottom", fontsize=9,
                        fontweight="bold")
            ax.set_title(self._title(title_cn, title_en))
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.4, color=COLOR_GRID)
            ax.set_axisbelow(True)

        _bar(axes[0, 0], sma, "半长轴", "Semi-Major Axis", "a [km]", "{:.0f}")
        _bar(axes[0, 1], ecc, "偏心率", "Eccentricity", "e", "{:.4f}")
        _bar(axes[1, 0], inc, "轨道倾角", "Inclination", "i [deg]", "{:.1f}")
        _bar(axes[1, 1], per, "轨道周期", "Orbital Period", "T [min]", "{:.1f}")

        fig.suptitle(self._title("轨道元素对比", "Orbital Elements Comparison"),
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = self._resolve_path(output_path, "orbit_elements.png")
        return self._save(fig, path)

    # ------------------------------------------------------------------
    # 6. 能量分析图
    # ------------------------------------------------------------------
    def plot_energy_analysis(
        self, transfer_params: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """能量分析图 (比能量沿轨迹变化 + C3)。

        兼容: Hohmann dict (a_transfer / e_transfer / ...) 或
        WorkflowResult transfer_params (a_transfer_m / e_transfer / ...)。
        """
        a_tr = _first(transfer_params, "a_transfer", "a_transfer_m",
                      default=A_MOON / 2 + R_EARTH)
        e_tr = _first(transfer_params, "e_transfer", "e", default=0.0)
        v_inf_moon = _first(transfer_params, "v_inf_moon", "v_inf_moon_m_s", default=0.0)
        C3_moon = _first(transfer_params, "C3_moon", "C3_moon_m2_s2", default=0.0)
        energy_geo = _first(transfer_params, "energy_geo", "energy_geo_j_kg",
                            default=-MU_EARTH / (2 * a_tr))
        v_per = _first(transfer_params, "v_tr_perigee", "v_tr_perigee_m_s", default=0.0)
        v_apo = _first(transfer_params, "v_tr_apogee", "v_tr_apogee_m_s", default=0.0)
        v_moon = _first(transfer_params, "v_moon", "v_moon_m_s", default=0.0)

        # 沿转移椭圆计算能量分量
        nu = np.linspace(0, np.pi, 200)
        p = a_tr * (1 - e_tr ** 2)
        r_nu = p / (1 + e_tr * np.cos(nu))
        v_nu = np.sqrt(MU_EARTH * (2.0 / r_nu - 1.0 / a_tr))

        KE = 0.5 * v_nu ** 2
        PE = -MU_EARTH / r_nu
        E_total = KE + PE
        t_frac = nu / np.pi

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # 左: 比能量
        ax1 = axes[0]
        ax1.plot(t_frac, KE / 1e6, color=self.TRANSFER, lw=2, label=r"$\varepsilon_k = v^2/2$")
        ax1.plot(t_frac, PE / 1e6, color=self.EARTH, lw=2, label=r"$\varepsilon_p = -\mu/r$")
        ax1.plot(t_frac, E_total / 1e6, color=self.MANEUVER, lw=2, ls="--",
                 label=r"$\varepsilon_{tot}=$" + f"{energy_geo/1e6:.3f} MJ/kg")
        ax1.axhline(0, color="gray", lw=0.8)
        ax1.set_xlabel("Normalized Flight Time (perigee $\\to$ apogee)")
        ax1.set_ylabel("Specific Energy [MJ/kg]")
        ax1.set_title(self._title("比能量沿转移轨迹", "Specific Energy Along Transfer"))
        ax1.grid(True, alpha=0.4, color=COLOR_GRID)
        ax1.set_axisbelow(True)
        ax1.legend(fontsize=9)

        # 右: 半径/速度
        ax2 = axes[1]
        ax2.plot(t_frac, r_nu / 1e6, color=self.EARTH, lw=2, label="Radius r")
        ax2.set_xlabel("Normalized Flight Time (perigee $\\to$ apogee)")
        ax2.set_ylabel(r"Radius [$\times 10^6$ m]", color=self.EARTH)
        ax2.tick_params(axis="y", labelcolor=self.EARTH)

        ax2b = ax2.twinx()
        ax2b.plot(t_frac, v_nu / 1e3, color=self.TRANSFER, lw=2, label="Velocity v")
        ax2b.set_ylabel("Velocity [km/s]", color=self.TRANSFER)
        ax2b.tick_params(axis="y", labelcolor=self.TRANSFER)

        ax2.set_title(self._title("半径与速度变化", "Radius & Velocity Profile"))
        info = (
            f"C3 (Moon) = {C3_moon/1e6:.3f} km$^2$/s$^2$\n"
            f"v_inf (Moon) = {v_inf_moon:.2f} m/s\n"
            f"v_perigee = {v_per/1e3:.3f} km/s\n"
            f"v_apogee  = {v_apo/1e3:.3f} km/s"
        )
        ax2.text(0.03, 0.97, info, transform=ax2.transAxes, fontsize=9,
                 va="top", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#FFF8E1", alpha=0.9,
                           ec=self.SOI))
        ax2.grid(True, alpha=0.4, color=COLOR_GRID)
        ax2.set_axisbelow(True)

        fig.suptitle(self._title("能量分析", "Energy Analysis"),
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        path = self._resolve_path(output_path, "energy_analysis.png")
        return self._save(fig, path)

    # ------------------------------------------------------------------
    # 7. 任务时间线甘特图
    # ------------------------------------------------------------------
    def plot_mission_timeline(
        self, result: Any, output_path: Optional[str] = None
    ) -> str:
        """任务时间线甘特图。

        参数: ``WorkflowResult`` (含 result 字典, 从中取 t_flight / key_times
        构造时间线), 或已构造的 timeline 列表, 或含 timeline 的对象。
        """
        timeline = None
        # WorkflowResult 对象
        if hasattr(result, "result") and isinstance(getattr(result, "result"), dict):
            data = result.result
            t_flight = _first(data.get("transfer_params", {}),
                              "t_flight", "t_flight_s", default=0.0)
            key_times = data.get("trajectory", {}).get("key_times", {})
            timeline = self._build_timeline(float(t_flight), key_times)
        elif hasattr(result, "timeline"):
            timeline = result.timeline
        elif isinstance(result, list):
            timeline = result
        elif isinstance(result, dict):
            timeline = result.get("timeline")
            if timeline is None:
                t_flight = _first(result, "t_flight", "t_flight_s", default=0.0)
                timeline = self._build_timeline(float(t_flight),
                                                result.get("key_times"))

        if not timeline:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.text(0.5, 0.5, "No timeline data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="gray")
            path = self._resolve_path(output_path, "mission_timeline.png")
            return self._save(fig, path)

        n = len(timeline)
        fig, ax = plt.subplots(figsize=(12, 0.6 * n + 2.5))

        for i, phase in enumerate(timeline):
            start = phase.get("start_day", 0.0)
            dur = phase.get("duration_day",
                            phase.get("end_day", start) - start)
            color = phase.get("color", self.EARTH)
            label_en = phase.get("phase", f"Phase {i}")
            label_cn = phase.get("phase_cn", label_en)
            label = f"{label_cn} / {label_en}" if self.cjk_available else label_en

            ax.barh(i, dur, left=start, height=0.55, color=color,
                    edgecolor="white", alpha=0.9)
            ax.text(start, i, " " + label, va="center", ha="left",
                    fontsize=9, fontweight="bold",
                    color="white" if dur > 1.5 else "black")
            if dur > 0.05:
                ax.text(start + dur, i, f" {dur:.2f}d", va="center", ha="left",
                        fontsize=8, color="dimgray")

        ax.set_yticks(range(n))
        ax.set_yticklabels(["" for _ in range(n)])
        ax.set_xlabel("Mission Elapsed Time [day]")
        ax.set_title(self._title("任务时间线", "Mission Timeline (Gantt)"))
        ax.grid(True, axis="x", alpha=0.4, color=COLOR_GRID)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        max_day = max(p.get("end_day", p.get("start_day", 0)) for p in timeline)
        ax.set_xlim(-0.5, max_day * 1.05 + 1)
        fig.tight_layout()

        path = self._resolve_path(output_path, "mission_timeline.png")
        return self._save(fig, path)


# ---------------------------------------------------------------------------
# 自测 (模拟数据 — 用真实 MoonTransfer 计算)
# ---------------------------------------------------------------------------
def _mock_data():
    """构造模拟数据: 同时返回 raw 格式与 WorkflowResult 格式以测试兼容性。"""
    from ..physics import MoonTransfer
    import datetime
    mt = MoonTransfer()
    ht = mt.hohmann_transfer(200e3, 100e3)
    traj = mt.design_trajectory(datetime.date(2026, 1, 1), 200e3, 100e3)
    pc = mt.porkchop_plot_data(datetime.date(2026, 1, 1), days=60, n_dep=20, n_ft=15)
    lw = mt.launch_window(datetime.date(2026, 1, 1), days=60)
    return mt, ht, traj, pc, lw


if __name__ == "__main__":
    print("=== aerospace_agent.reporting.plots 自测 (模拟数据) ===")
    plotter = Plotter()
    print(f"CJK 字体: {plotter.cjk_font} (可用={plotter.cjk_available})")
    print(f"输出目录: {plotter.output_dir}")

    mt, ht, traj, pc, lw = _mock_data()

    print("\n[1/7] 3D 转移轨迹 (raw design_trajectory) ...")
    p1 = plotter.plot_transfer_trajectory(traj)
    print(f"  -> {p1}")

    print("[2/7] Δv 预算 (raw hohmann) ...")
    p2 = plotter.plot_delta_v_budget(ht)
    print(f"  -> {p2}")

    print("[3/7] Porkchop (raw, 20x15) ...")
    p3 = plotter.plot_porkchop(pc)
    print(f"  -> {p3}")

    print("[4/7] 发射窗口 (raw launch_window) ...")
    p4 = plotter.plot_launch_windows(lw)
    print(f"  -> {p4}")

    print("[5/7] 轨道元素 (auto from hohmann) ...")
    p5 = plotter.plot_orbit_elements(ht)
    print(f"  -> {p5}")

    print("[6/7] 能量分析 (raw hohmann) ...")
    p6 = plotter.plot_energy_analysis(ht)
    print(f"  -> {p6}")

    print("[7/7] 任务时间线 (从 t_flight 构造) ...")
    p7 = plotter.plot_mission_timeline({"t_flight": ht["t_flight"],
                                        "key_times": traj["key_times"]})
    print(f"  -> {p7}")

    # 测试 WorkflowResult 格式兼容性
    print("\n--- WorkflowResult 格式兼容性测试 ---")
    from ..workflows.trajectory_analysis import TrajectoryAnalysisWorkflow
    wf = TrajectoryAnalysisWorkflow()
    wr = wf.execute(launch_date=None, altitude_leo=200e3, altitude_lmo=100e3,
                    days=60, n_dep=12, n_ft=8)
    data = wr.result
    tp = data["transfer_params"]
    # 转移轨迹: 用 trajectory dict (含 position_m)
    traj_wr = dict(data["trajectory"])
    traj_wr.update({"a_transfer_m": tp["a_transfer_m"], "e_transfer": tp["e_transfer"],
                    "r_leo_m": tp["r_leo_m"], "r_lmo_m": tp["r_lmo_m"]})
    p1b = plotter.plot_transfer_trajectory(traj_wr,
        output_path=os.path.join(plotter.output_dir, "wr_transfer_trajectory.png"))
    print(f"  transfer (WR fmt) -> {os.path.basename(p1b)}")
    p2b = plotter.plot_delta_v_budget(data["delta_v_budget"],
        output_path=os.path.join(plotter.output_dir, "wr_delta_v_budget.png"))
    print(f"  delta_v (WR fmt) -> {os.path.basename(p2b)}")
    p3b = plotter.plot_porkchop(data["porkchop_data"],
        output_path=os.path.join(plotter.output_dir, "wr_porkchop.png"))
    print(f"  porkchop (WR fmt) -> {os.path.basename(p3b)}")
    p4b = plotter.plot_launch_windows(data["window_analysis"],
        output_path=os.path.join(plotter.output_dir, "wr_launch_windows.png"))
    print(f"  windows (WR fmt) -> {os.path.basename(p4b)}")
    p5b = plotter.plot_orbit_elements(tp,
        output_path=os.path.join(plotter.output_dir, "wr_orbit_elements.png"))
    print(f"  orbit (WR fmt) -> {os.path.basename(p5b)}")
    p6b = plotter.plot_energy_analysis(tp,
        output_path=os.path.join(plotter.output_dir, "wr_energy_analysis.png"))
    print(f"  energy (WR fmt) -> {os.path.basename(p6b)}")
    p7b = plotter.plot_mission_timeline(wr,
        output_path=os.path.join(plotter.output_dir, "wr_mission_timeline.png"))
    print(f"  timeline (WR fmt) -> {os.path.basename(p7b)}")

    print("\nplots 自测全部通过, 14 张图已生成 (raw + WorkflowResult 各 7).")
