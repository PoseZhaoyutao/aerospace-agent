"""HTML 报告生成器 (Self-contained HTML report generator)。

:class:`ReportGenerator` 把 :class:`WorkflowResult` 转换为一份自包含的
HTML 报告: 先用 :class:`Plotter` 生成全部 PNG 图, 再以 base64 data URI
内嵌, 配合 MathJax CDN 渲染 LaTeX 公式, 浏览器打开即可查看。

数据来源
--------
``WorkflowResult.result`` 是主输出字典, 含:
    transfer_params / trajectory / window_analysis / porkchop_data /
    delta_v_budget / physics_analysis / launch_date / formula_derivation

报告结构 (10 节)
----------------
1. 标题页  2. 执行摘要  3. 任务设计  4. 物理基础与公式推导
5. 发射窗口分析  6. 轨迹设计  7. Δv 预算  8. 任务时间线
9. 工具与验证  10. 结论与建议

风格: 深色专业航天工程报告, 响应式, 内联 CSS, MathJax CDN。
"""

from __future__ import annotations

import base64
import datetime
import html
import os
from typing import Any, Dict, List, Optional, Sequence

from .plots import Plotter
from .formulas import FORMULAS, DERIVATIONS, get_formula_latex, get_derivation
from ..physics.constants import DAY2SEC


# ---------------------------------------------------------------------------
# 内联 CSS (深色航天工程报告风格)
# ---------------------------------------------------------------------------
_CSS = """
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0;
  background: #0d1117;
  color: #e6edf3;
  font-family: -apple-system, "Segoe UI", "Noto Sans", "Noto Sans CJK SC",
               "WenQuanYi Micro Hei", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.7;
}
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
header.title-page {
  text-align: center; padding: 60px 24px 40px;
  background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
  border-bottom: 3px solid #4A90D9;
  border-radius: 12px 12px 0 0;
}
header.title-page h1 {
  font-size: 2.4em; margin: 0 0 12px; color: #58a6ff;
  letter-spacing: 1px;
}
header.title-page .subtitle { color: #8b949e; font-size: 1.1em; }
header.title-page .meta {
  margin-top: 24px; display: flex; justify-content: center; gap: 32px;
  flex-wrap: wrap; color: #c9d1d9;
}
header.title-page .meta div { text-align: center; }
header.title-page .meta .label { font-size: 0.75em; color: #6e7681;
  text-transform: uppercase; letter-spacing: 1px; }
header.title-page .meta .value { font-size: 1.1em; font-weight: 600; }

section {
  background: #161b22; margin: 20px 0; padding: 28px;
  border-radius: 10px; border: 1px solid #30363d;
}
section h2 {
  color: #58a6ff; border-bottom: 2px solid #30363d;
  padding-bottom: 10px; margin-top: 0; font-size: 1.5em;
}
section h3 { color: #7ee787; margin-top: 24px; font-size: 1.15em; }
section h4 { color: #d2a8ff; margin-top: 18px; }

table {
  border-collapse: collapse; width: 100%; margin: 14px 0;
  font-size: 0.92em;
}
th, td {
  border: 1px solid #30363d; padding: 9px 12px; text-align: left;
}
th { background: #21262d; color: #58a6ff; font-weight: 600; }
tr:nth-child(even) td { background: #0d1117; }
td.num, th.num { text-align: right; font-family: "SFMono-Regular",
  Consolas, "Liberation Mono", Menlo, monospace; }

.figure { text-align: center; margin: 20px 0; }
.figure img { max-width: 100%; border-radius: 8px;
  border: 1px solid #30363d; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
.figure .caption { color: #8b949e; font-size: 0.88em; margin-top: 8px;
  font-style: italic; }

.formula-block {
  background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
  padding: 16px; margin: 12px 0; overflow-x: auto;
}
.formula-name { color: #d2a8ff; font-weight: 600; margin-bottom: 6px; }
.derivation {
  background: #161b22; border-left: 3px solid #7ee787;
  padding: 10px 14px; margin: 8px 0; font-size: 0.9em;
  white-space: pre-wrap; color: #c9d1d9; font-family: monospace;
}

.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px; margin: 16px 0;
}
.kpi-card {
  background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
  padding: 16px; text-align: center;
}
.kpi-card .kpi-value { font-size: 1.8em; font-weight: 700; color: #58a6ff;
  font-family: monospace; }
.kpi-card .kpi-label { font-size: 0.8em; color: #8b949e;
  text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

.badge {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 0.78em; font-weight: 600;
}
.badge.ok { background: #1a4731; color: #7ee787; }
.badge.no { background: #4a1f1f; color: #ff7b72; }
.badge.info { background: #1f2a4a; color: #58a6ff; }

.note {
  background: #1a1f2e; border-left: 4px solid #f0883e; padding: 12px 16px;
  margin: 12px 0; border-radius: 4px; color: #d29922;
}

footer {
  text-align: center; color: #6e7681; font-size: 0.85em;
  padding: 24px; border-top: 1px solid #30363d; margin-top: 24px;
}

@media (max-width: 720px) {
  .container { padding: 12px; }
  section { padding: 16px; }
  header.title-page h1 { font-size: 1.7em; }
}

mjx-container { color: #e6edf3 !important; }
"""


class ReportGenerator:
    """自包含 HTML 报告生成器。

    参数
    -----
    output_dir : Plotter 默认输出目录 (临时图片存放)
    """

    def __init__(self, output_dir: str = "/workspace/demo_outputs"):
        self.plotter = Plotter(output_dir=output_dir)
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _embed_image(path: str) -> str:
        """把图片文件编码为 base64 data URI (供 HTML 内嵌)。"""
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _format_table(
        data: Sequence[Sequence[Any]], headers: Sequence[str],
        num_cols: Optional[Sequence[int]] = None,
    ) -> str:
        """生成 HTML 表格。"""
        num_cols = set(num_cols or [])
        th = "".join(
            f'<th class="{"num" if i in num_cols else ""}">{html.escape(str(h))}</th>'
            for i, h in enumerate(headers))
        rows = []
        for row in data:
            cells = "".join(
                f'<td class="{"num" if i in num_cols else ""}">{html.escape(str(c))}</td>'
                for i, c in enumerate(row))
            rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    @staticmethod
    def _figure(img_uri: str, caption: str) -> str:
        return (f'<div class="figure">'
                f'<img src="{img_uri}" alt="{html.escape(caption)}">'
                f'<div class="caption">{html.escape(caption)}</div></div>')

    @staticmethod
    def _fmt(v, fmt_spec=".3f"):
        try:
            return format(float(v), fmt_spec)
        except (TypeError, ValueError):
            return str(v)

    def _get_tools_status(self) -> List[Dict[str, Any]]:
        """获取 MCP 工具可用性状态 (失败则返回默认列表)。"""
        try:
            from ..mcp_tools.registry import get_status_summary
            return get_status_summary()
        except Exception:
            return [
                {"name": "orekit", "library": "orekit", "available": False,
                 "source": "fallback", "methods": []},
                {"name": "spiceypy", "library": "spiceypy", "available": False,
                 "source": "fallback", "methods": []},
                {"name": "basilisk", "library": "Basilisk", "available": False,
                 "source": "fallback", "methods": []},
                {"name": "astropy", "library": "astropy", "available": False,
                 "source": "fallback", "methods": []},
            ]

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def generate_lunar_transfer_report(
        self,
        result: Any,
        output_path: str = "/workspace/reports/lunar_transfer_report.html",
    ) -> str:
        """生成自包含 HTML 报告, 返回报告文件绝对路径。

        参数
        -----
        result      : :class:`WorkflowResult` (TrajectoryAnalysisWorkflow 输出)
        output_path : HTML 输出路径
        """
        data = result.result if hasattr(result, "result") else result
        tp = data.get("transfer_params", {})

        # --- 1. 生成所有图 ---
        imgs: Dict[str, Optional[str]] = {}
        plot_specs: List[tuple] = [
            ("transfer", lambda: self.plotter.plot_transfer_trajectory(
                self._merge_traj(data))),
            ("delta_v", lambda: self.plotter.plot_delta_v_budget(
                data.get("delta_v_budget", {}))),
            ("porkchop", lambda: self.plotter.plot_porkchop(
                data.get("porkchop_data", {}))),
            ("launch_windows", lambda: self.plotter.plot_launch_windows(
                data.get("window_analysis", {}))),
            ("orbit_elements", lambda: self.plotter.plot_orbit_elements(tp)),
            ("energy", lambda: self.plotter.plot_energy_analysis(tp)),
            ("timeline", lambda: self.plotter.plot_mission_timeline(result)),
        ]
        for key, fn in plot_specs:
            try:
                imgs[key] = self._embed_image(fn())
            except Exception as e:
                imgs[key] = None
                print(f"[report] 警告: 生成图 {key} 失败: {e}")

        # --- 2. 组装 HTML ---
        parts: List[str] = []
        parts.append('<!DOCTYPE html><html lang="zh-CN"><head>')
        parts.append('<meta charset="UTF-8">')
        parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        mission_name = "Earth-Moon Lunar Transfer Mission"
        parts.append(f"<title>{html.escape(mission_name)} — 任务分析报告</title>")
        parts.append(
            '<script>MathJax = {tex: {inlineMath: [["$","$"],["\\\\(","\\\\)"]], '
            'displayMath: [["$$","$$"],["\\\\[","\\\\]"]]}, '
            'svg: {fontCache: "global"}};</script>')
        parts.append(
            '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" '
            'id="MathJax-script" async></script>')
        parts.append(f"<style>{_CSS}</style>")
        parts.append("</head><body><div class='container'>")

        parts.append(self._section_title_page(result, data))
        parts.append(self._section_executive_summary(result, data))
        parts.append(self._section_mission_design(result, data))
        parts.append(self._section_physics_formulas(result, data))
        parts.append(self._section_launch_windows(result, data, imgs))
        parts.append(self._section_trajectory_design(result, data, imgs))
        parts.append(self._section_delta_v_budget(result, data, imgs))
        parts.append(self._section_timeline(result, data, imgs))
        parts.append(self._section_tools_validation(result, data))
        parts.append(self._section_conclusion(result, data))

        parts.append(
            f"<footer>Generated by aerospace_agent.reporting v1.0.0 &middot; "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &middot; "
            f"MathJax CDN &middot; Self-contained HTML</footer>")
        parts.append("</div></body></html>")

        html_content = "\n".join(parts)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return os.path.abspath(output_path)

    @staticmethod
    def _merge_traj(data: Dict) -> Dict:
        """合并 trajectory + transfer_params, 供 plot_transfer_trajectory 用。"""
        traj = dict(data.get("trajectory", {}))
        tp = data.get("transfer_params", {})
        traj.update({
            "a_transfer_m": tp.get("a_transfer_m"),
            "e_transfer": tp.get("e_transfer"),
            "r_leo_m": tp.get("r_leo_m"),
            "r_lmo_m": tp.get("r_lmo_m"),
            "r_moon_m": tp.get("r_moon_m"),
        })
        return traj

    # ------------------------------------------------------------------
    # 各章节
    # ------------------------------------------------------------------
    def _section_title_page(self, result, data) -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        launch = data.get("launch_date") or result.metadata.get("launch_date", now)
        success = result.success if hasattr(result, "success") else True
        version = "1.0.0"
        return f"""
<header class="title-page">
  <h1>Earth-Moon Lunar Transfer Mission</h1>
  <div class="subtitle">地月转移轨道分析报告 / Lunar Transfer Trajectory Analysis Report</div>
  <div class="meta">
    <div><div class="label">Launch Date / 发射日期</div>
         <div class="value">{html.escape(str(launch)[:16])}</div></div>
    <div><div class="label">Report Date / 报告日期</div>
         <div class="value">{now}</div></div>
    <div><div class="label">Version / 版本</div>
         <div class="value">{version}</div></div>
    <div><div class="label">Status / 状态</div>
         <div class="value">{'<span class="badge ok">SUCCESS</span>' if success else '<span class="badge no">FAILED</span>'}</div></div>
  </div>
</header>"""

    def _section_executive_summary(self, result, data) -> str:
        tp = data.get("transfer_params", {})
        dvb = data.get("delta_v_budget", {})
        wa = data.get("window_analysis", {})
        meta = result.metadata if hasattr(result, "metadata") else {}
        best = wa.get("best", {})

        dv_total = meta.get("dv_total_km_s", tp.get("dv_total_m_s", 0) / 1e3)
        t_flight = meta.get("t_flight_days", tp.get("t_flight_s", 0) / DAY2SEC)
        C3 = meta.get("C3", tp.get("C3_moon_m2_s2", 0))
        best_date = str(best.get("date", "N/A"))[:10]

        s = f"""
<section>
  <h2>1. Executive Summary / 执行摘要</h2>
  <h3>Mission Objective / 任务目标</h3>
  <p>设计一条从近地停泊轨道 (LEO) 经 Hohmann 转移到达近月轨道 (LMO) 的地月转移
  轨道, 完成发射窗口分析、轨迹设计、Δv 预算与任务时间线规划。Design a Hohmann
  transfer trajectory from LEO parking orbit to LMO, including launch window
  analysis, trajectory design, delta-V budget, and mission timeline.</p>
  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-value">{self._fmt(dv_total)} km/s</div>
         <div class="kpi-label">Total Δv / 总速度增量</div></div>
    <div class="kpi-card"><div class="kpi-value">{self._fmt(t_flight)} d</div>
         <div class="kpi-label">Flight Time / 飞行时间</div></div>
    <div class="kpi-card"><div class="kpi-value">{self._fmt(C3/1e6)} km²/s²</div>
         <div class="kpi-label">C3 (Moon) / 月球特征能量</div></div>
    <div class="kpi-card"><div class="kpi-value">{html.escape(best_date)}</div>
         <div class="kpi-label">Best Window / 最优窗口</div></div>
  </div>
  <h3>Key Results / 关键结果</h3>"""
        dv1 = dvb.get("dv1_leo_departure_km_s", tp.get("dv1_m_s", 0) / 1e3)
        dv2 = dvb.get("dv2_loi_braking_km_s", tp.get("dv2_m_s", 0) / 1e3)
        rows = [
            ["Total Δv / 总 Δv", f"{dv_total:.3f} km/s"],
            ["TLI Δv (LEO departure / 离轨)", f"{dv1:.3f} km/s"],
            ["LOI Δv (Lunar capture / 制动)", f"{dv2:.3f} km/s"],
            ["Flight time / 飞行时间", f"{t_flight:.3f} day"],
            ["C3 (Moon) / 月球特征能量", f"{C3/1e6:.3f} km²/s²"],
            ["Transfer semi-major axis / 转移半长轴",
             f"{tp.get('a_transfer_m',0)/1e6:.3f}×10⁶ m"],
            ["Transfer eccentricity / 转移偏心率",
             f"{tp.get('e_transfer',0):.4f}"],
            ["Best window deviation / 最优窗口偏差",
             f"{best.get('deviation_deg',0):+.4f}°"],
        ]
        s += self._format_table(rows, ["Parameter / 参数", "Value / 值"], num_cols=[1])
        summary = result.summary if hasattr(result, "summary") else ""
        if summary:
            s += f'<div class="note"><strong>Workflow Summary / 工作流摘要:</strong> {html.escape(summary)}</div>'
        s += "</section>"
        return s

    def _section_mission_design(self, result, data) -> str:
        tp = data.get("transfer_params", {})
        traj = data.get("trajectory", {})
        te = traj.get("transfer_elements", {})
        rows = [
            ["LEO Parking Orbit / LEO 停泊轨道",
             f"{tp.get('altitude_leo_m',0)/1e3:.0f} km",
             f"{tp.get('r_leo_m',0)/1e3:.1f} km",
             "28.5°",
             f"{2*3.14159265358979*(tp.get('r_leo_m',1)**3/3.986e14)**0.5/60:.2f} min"],
            ["Transfer Ellipse / 转移椭圆",
             "—",
             f"a={tp.get('a_transfer_m',0)/1e6:.3f}×10⁶ m",
             f"{te.get('i_deg',28.5):.1f}°",
             f"{tp.get('T_transfer_s',0)/3600:.2f} h"],
            ["LMO / 近月轨道",
             f"{tp.get('altitude_lmo_m',0)/1e3:.0f} km",
             f"{tp.get('r_lmo_m',0)/1e3:.1f} km",
             "0.0°",
             f"{2*3.14159265358979*(tp.get('r_lmo_m',1)**3/4.9049e12)**0.5/60:.2f} min"],
        ]
        return f"""
<section>
  <h2>2. Mission Design / 任务设计</h2>
  <p>本设计采用 Hohmann 转移近似 (patched conic), 月球绕地做圆周运动。
  The design uses Hohmann transfer approximation (patched conic) with the
  Moon on a circular orbit.</p>
  {self._format_table(rows,
    ["Orbit / 轨道", "Altitude / 高度", "Radius / 半径", "Inclination / 倾角", "Period / 周期"],
    num_cols=[1,2,3,4])}
  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-value">{tp.get('v_leo_m_s',0)/1e3:.3f}</div>
         <div class="kpi-label">v_LEO [km/s]</div></div>
    <div class="kpi-card"><div class="kpi-value">{tp.get('v_tr_perigee_m_s',0)/1e3:.3f}</div>
         <div class="kpi-label">v_perigee [km/s]</div></div>
    <div class="kpi-card"><div class="kpi-value">{tp.get('v_tr_apogee_m_s',0)/1e3:.3f}</div>
         <div class="kpi-label">v_apogee [km/s]</div></div>
    <div class="kpi-card"><div class="kpi-value">{tp.get('v_moon_m_s',0)/1e3:.3f}</div>
         <div class="kpi-label">v_moon [km/s]</div></div>
  </div>
</section>"""

    def _section_physics_formulas(self, result, data) -> str:
        formula_keys = [
            ("vis_viva", "Vis-Viva Equation / 活力公式"),
            ("kepler_third", "Kepler's Third Law / 开普勒第三定律"),
            ("hohmann_semi_major", "Hohmann Semi-Major Axis / 转移半长轴"),
            ("hohmann_dv1", "TLI: First Impulse / 第一脉冲"),
            ("hohmann_dv2", "LOI: Second Impulse / 第二脉冲"),
            ("hohmann_transfer_time", "Transfer Flight Time / 转移飞行时间"),
            ("phase_angle", "Launch Window Phase Angle / 相位角条件"),
            ("specific_energy", "Specific Orbital Energy / 比能量"),
            ("C3", "Characteristic Energy C3 / 特征能量"),
            ("soi", "Sphere of Influence / 引力作用球"),
            ("perilune_velocity", "Perilune Velocity / 近月点速度"),
            ("delta_v_capture", "LOI Capture Δv / 捕获制动"),
        ]
        blocks = []
        for key, title in formula_keys:
            latex = get_formula_latex(key)
            deriv = get_derivation(key)
            blocks.append(f"""
    <div class="formula-block">
      <div class="formula-name">{html.escape(title)}</div>
      <div>$$ {latex} $$</div>
      <div class="derivation">{html.escape(deriv)}</div>
    </div>""")
        return f"""
<section>
  <h2>3. Physical Foundations &amp; Formula Derivation / 物理基础与公式推导</h2>
  <p>地月转移轨道设计的核心物理公式如下, 由 MathJax 渲染 (需联网加载 CDN)。
  Core physical formulas for Earth-Moon transfer design, rendered by MathJax.</p>
  {''.join(blocks)}
</section>"""

    def _section_launch_windows(self, result, data, imgs) -> str:
        wa = data.get("window_analysis", {})
        best = wa.get("best", {})
        top = wa.get("top_windows", [])
        rows = []
        for w in top[:8]:
            rows.append([
                html.escape(str(w.get("date", ""))[:16]),
                f"{w.get('day_offset',0):.2f}",
                f"{w.get('deviation_deg',0):+.3f}°",
                f"{w.get('actual_phase_deg',0):.2f}°",
                f"{w.get('C3',0)/1e6:.3f}",
                f"{w.get('flight_time_days',0):.3f}",
            ])
        table_html = self._format_table(
            rows,
            ["Window Date / 窗口日期", "Day Offset", "Deviation",
             "Actual Phase", "C3 [km²/s²]", "Flight [day]"],
            num_cols=[1, 2, 3, 4, 5])
        porkchop_fig = self._figure(imgs["porkchop"],
            "Figure 5.1 Porkchop 等高线图 (C3 与总 Δv)") if imgs.get("porkchop") else ""
        lw_fig = self._figure(imgs["launch_windows"],
            "Figure 5.2 发射窗口相位角偏差图") if imgs.get("launch_windows") else ""
        return f"""
<section>
  <h2>4. Launch Window Analysis / 发射窗口分析</h2>
  <p>发射窗口由月球相位角匹配条件确定: 月球须在 t_flight 后到达转移椭圆远地点。
  Launch windows are determined by lunar phase-angle matching.</p>
  <div class="note"><strong>Required phase angle / 所需相位角:</strong>
  {wa.get('required_phase_deg',0):.2f}° &nbsp;|&nbsp;
  <strong>Flight time / 飞行时间:</strong> {wa.get('flight_time_days',0):.3f} day &nbsp;|&nbsp;
  <strong>Windows found / 找到窗口数:</strong> {wa.get('n_windows',0)}</div>
  {table_html}
  <div class="note"><strong>Best window / 最优窗口:</strong>
  {html.escape(str(best.get('date','N/A'))[:16])} &nbsp;|&nbsp;
  deviation = {best.get('deviation_deg',0):+.4f}°</div>
  {porkchop_fig}
  {lw_fig}
</section>"""

    def _section_trajectory_design(self, result, data, imgs) -> str:
        traj_fig = self._figure(imgs["transfer"],
            "Figure 6.1 地月转移 3D 轨迹 (地球/月球轨道/转移椭圆/LEO/LMO/SOI)") \
            if imgs.get("transfer") else ""
        oe_fig = self._figure(imgs["orbit_elements"],
            "Figure 6.2 轨道元素对比 (LEO / Transfer / LMO)") \
            if imgs.get("orbit_elements") else ""
        en_fig = self._figure(imgs["energy"],
            "Figure 6.3 能量分析 (比能量沿轨迹 + 速度/半径)") \
            if imgs.get("energy") else ""
        return f"""
<section>
  <h2>5. Trajectory Design / 轨迹设计</h2>
  <p>采用 patched conic (拼凑圆锥) 近似: 地心段转移椭圆 + 月心段双曲线捕获。
  Uses patched-conic approximation: geocentric transfer ellipse +
  selenocentric hyperbolic capture.</p>
  {traj_fig}
  {oe_fig}
  {en_fig}
</section>"""

    def _section_delta_v_budget(self, result, data, imgs) -> str:
        dvb = data.get("delta_v_budget", {})
        tp = data.get("transfer_params", {})
        rows = [
            ["TLI / Trans-Lunar Injection (LEO 离轨)",
             f"{dvb.get('dv1_leo_departure_km_s', tp.get('dv1_m_s',0)/1e3):.3f}",
             f"{dvb.get('dv1_leo_departure_m_s', tp.get('dv1_m_s',0)):.1f}",
             "切向加速, 进入转移椭圆"],
            ["LOI / Lunar Orbit Insertion (近月制动)",
             f"{dvb.get('dv2_loi_braking_km_s', tp.get('dv2_m_s',0)/1e3):.3f}",
             f"{dvb.get('dv2_loi_braking_m_s', tp.get('dv2_m_s',0)):.1f}",
             "双曲线近月点制动, 捕获至 LMO"],
            ["Total / 总计",
             f"{dvb.get('dv_total_km_s', tp.get('dv_total_m_s',0)/1e3):.3f}",
             f"{dvb.get('dv_total_m_s', tp.get('dv_total_m_s',0)):.1f}",
             "TLI + LOI"],
            ["+5% Margin / 含余量",
             f"{dvb.get('dv_total_with_margin_m_s',0)/1e3:.3f}",
             f"{dvb.get('dv_total_with_margin_m_s',0):.1f}",
             "工程余量 (TCM/容差)"],
        ]
        table_html = self._format_table(
            rows, ["Maneuver / 机动", "Δv [km/s]", "Δv [m/s]", "Note / 说明"],
            num_cols=[1, 2])
        dv_fig = self._figure(imgs["delta_v"], "Figure 7.1 Δv 预算柱状图") \
            if imgs.get("delta_v") else ""
        return f"""
<section>
  <h2>6. Delta-V Budget / Δv 预算</h2>
  {table_html}
  {dv_fig}
</section>"""

    def _section_timeline(self, result, data, imgs) -> str:
        tp = data.get("transfer_params", {})
        traj = data.get("trajectory", {})
        key_times = traj.get("key_times", {})
        t_flight = tp.get("t_flight_s", 0) / DAY2SEC
        t_soi = key_times.get("t_soi_entry_approx", t_flight * 0.85 * DAY2SEC) / DAY2SEC
        rows = [
            ["LEO 停泊 / LEO Parking", "0.00", f"{0.00:.2f}", "0.00"],
            ["TLI 机动 / TLI Maneuver", "0.00", f"{0.02:.2f}", "0.02"],
            ["转移飞行 / Trans-Lunar Coast", f"{0.02:.2f}", f"{t_soi:.2f}",
             f"{max(t_soi-0.02,0.01):.2f}"],
            ["SOI 穿越 / SOI Crossing", f"{t_soi:.2f}", f"{t_flight:.2f}",
             f"{max(t_flight-t_soi,0.01):.2f}"],
            ["LOI 制动 / LOI Burn", f"{t_flight:.2f}", f"{t_flight+0.02:.2f}", "0.02"],
            ["LMO 运行 / LMO Operations", f"{t_flight+0.02:.2f}", f"{t_flight+10:.2f}",
             f"{10-0.02:.2f}"],
        ]
        table_html = self._format_table(
            rows, ["Phase / 阶段", "Start [day]", "End [day]", "Duration [day]"],
            num_cols=[1, 2, 3])
        tl_fig = self._figure(imgs["timeline"], "Figure 8.1 任务时间线甘特图") \
            if imgs.get("timeline") else ""
        return f"""
<section>
  <h2>7. Mission Timeline / 任务时间线</h2>
  {table_html}
  {tl_fig}
</section>"""

    def _section_tools_validation(self, result, data) -> str:
        tools = self._get_tools_status()
        rows = []
        for t in tools:
            avail = t.get("available", False)
            badge = '<span class="badge ok">Available</span>' if avail \
                else '<span class="badge no">Fallback</span>'
            methods = ", ".join(t.get("methods", [])[:4])
            rows.append([
                f"<code>{html.escape(t.get('name',''))}</code>",
                f"<code>{html.escape(t.get('library',''))}</code>",
                badge,
                html.escape(methods),
            ])
        rows.append([
            "<code>numpy/scipy</code>", "<code>numpy/scipy</code>",
            '<span class="badge ok">Available</span>',
            "数值计算/寻优 (本工作流核心)"])
        rows.append([
            "<code>matplotlib</code>", "<code>matplotlib</code>",
            '<span class="badge ok">Available</span>',
            "绘图 (reporting 子包)"])
        table_html = self._format_table(
            rows, ["Tool / 工具", "Library / 依赖", "Status / 状态", "Methods / 方法"])
        return f"""
<section>
  <h2>8. Tools &amp; Validation / 工具与验证</h2>
  <p>本工作流可对接多种航天动力学工具 (MCP 风格)。核心计算由 numpy/scipy 完成,
  高精度场景可切换至 orekit / spiceypy / basilisk。The workflow integrates
  multiple astrodynamics tools via MCP; core computation uses numpy/scipy.</p>
  {table_html}
  <h3>Validation Method / 验证方法</h3>
  <div class="derivation">1. Hohmann 转移 Δv 总量应在 3.9~4.1 km/s (LEO 200km -> LMO 100km)
2. 转移飞行时间应 ~5 天 (Hohmann 半周期)
3. 最优窗口相位角偏差 < 1°
4. 月球 SOI 半径 ~66200 km (Laplace 公式)
5. C3 (Moon) 与 v_inf_moon 平方一致 (能量守恒)
6. Porkchop 最优 Δv 应接近 Hohmann 解 (~3.95 km/s)</div>
</section>"""

    def _section_conclusion(self, result, data) -> str:
        tp = data.get("transfer_params", {})
        meta = result.metadata if hasattr(result, "metadata") else {}
        wa = data.get("window_analysis", {})
        best = wa.get("best", {})
        dv_total = meta.get("dv_total_km_s", 0)
        t_flight = meta.get("t_flight_days", 0)
        best_date = str(best.get("date", "N/A"))[:10]
        return f"""
<section>
  <h2>9. Conclusion &amp; Recommendations / 结论与建议</h2>
  <h3>Conclusion / 结论</h3>
  <p>本报告完成了从 LEO {tp.get('altitude_leo_m',0)/1e3:.0f} km 停泊轨道
  到 LMO {tp.get('altitude_lmo_m',0)/1e3:.0f} km 的 Hohmann 地月转移轨道
  设计, 总 Δv = {dv_total:.3f} km/s, 飞行时间 {t_flight:.3f} 天,
  最优发射窗口 {html.escape(best_date)}。
  The report completes a Hohmann Earth-Moon transfer design with total
  Δv = {dv_total:.3f} km/s and flight time {t_flight:.3f} days.</p>
  <h3>Recommendations / 建议</h3>
  <div class="derivation">1. 实际任务应增加 2~3 次中途修正 (TCM) 机动, 预留 ~50~100 m/s 余量
2. 考虑月球轨道倾角 (5.145°) 与节点进动, 可用精确历表 (SPICE/orekit) 复算窗口
3. LOI 可拆分为多次制动 (降低单次点火负荷, 但增加总 Δv)
4. 如需自由返回轨道, 应改用月球借力或低能转移 (BLT)
5. 推进剂预算应按 Tsiolkovsky 方程 + 边际系数 (1.1~1.2) 计算
6. 高保真验证建议用 basilisk/orekit 做完整三体数值积分</div>
</section>"""

    # ------------------------------------------------------------------
    # 纯文本摘要 (供 CLI 输出)
    # ------------------------------------------------------------------
    def generate_summary_text(self, result: Any) -> str:
        """生成纯文本摘要 (供 CLI 输出)。"""
        data = result.result if hasattr(result, "result") else result
        tp = data.get("transfer_params", {})
        dvb = data.get("delta_v_budget", {})
        wa = data.get("window_analysis", {})
        best = wa.get("best", {})
        meta = result.metadata if hasattr(result, "metadata") else {}
        summary = result.summary if hasattr(result, "summary") else ""

        dv_total = meta.get("dv_total_km_s", tp.get("dv_total_m_s", 0) / 1e3)
        dv1 = dvb.get("dv1_leo_departure_km_s", tp.get("dv1_m_s", 0) / 1e3)
        dv2 = dvb.get("dv2_loi_braking_km_s", tp.get("dv2_m_s", 0) / 1e3)
        t_flight = meta.get("t_flight_days", tp.get("t_flight_s", 0) / DAY2SEC)
        C3 = meta.get("C3", tp.get("C3_moon_m2_s2", 0))
        v_inf = tp.get("v_inf_moon_m_s", 0)

        lines = [
            "=" * 64,
            "  Earth-Moon Lunar Transfer Mission",
            "  地月转移轨道分析 — Summary",
            "=" * 64,
            f"  Status             : {'SUCCESS' if result.success else 'FAILED'}",
            f"  Best launch window : {best.get('date', 'N/A')}",
            f"  Launch date        : {data.get('launch_date', 'N/A')}",
            "",
            "  --- Key Results ---",
            f"  Total Δv           : {dv_total:.3f} km/s",
            f"    TLI (LEO dep.)   : {dv1:.3f} km/s",
            f"    LOI (capture)    : {dv2:.3f} km/s",
            f"  Flight time        : {t_flight:.3f} day",
            f"  C3 (Moon)          : {C3/1e6:.3f} km^2/s^2",
            f"  v_inf (Moon)       : {v_inf:.2f} m/s",
            "",
            "  --- Transfer Orbit ---",
            f"  Semi-major axis    : {tp.get('a_transfer_m',0)/1e6:.3f} x10^6 m",
            f"  Eccentricity       : {tp.get('e_transfer',0):.4f}",
            f"  Period             : {tp.get('T_transfer_s',0)/3600:.3f} h",
            "",
            "  --- Launch Window ---",
            f"  Required phase     : {wa.get('required_phase_deg',0):.2f} deg",
            f"  Best deviation     : {best.get('deviation_deg',0):+.4f} deg",
            f"  Windows found      : {wa.get('n_windows',0)}",
            "",
            "  --- Key Formulas ---",
        ]
        for key, label in [("vis_viva", "Vis-viva"), ("hohmann_dv1", "TLI Δv"),
                           ("hohmann_dv2", "LOI Δv"), ("phase_angle", "Phase angle"),
                           ("C3", "C3")]:
            lines.append(f"  {label:<16}: {get_formula_latex(key)}")
        lines.append("")
        if summary:
            lines.append("  --- Workflow Summary ---")
            lines.append(f"  {summary}")
        lines.append("=" * 64)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 自测 (模拟数据 — 用真实 TrajectoryAnalysisWorkflow)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.reporting.report 自测 (模拟数据) ===")
    from ..workflows.trajectory_analysis import TrajectoryAnalysisWorkflow

    wf = TrajectoryAnalysisWorkflow()
    result = wf.execute(launch_date=None, altitude_leo=200e3, altitude_lmo=100e3,
                        days=60, n_dep=12, n_ft=8)
    print(f"WorkflowResult 就绪: success={result.success}")

    gen = ReportGenerator()
    print("\n--- 纯文本摘要 ---")
    print(gen.generate_summary_text(result))

    print("\n--- 生成 HTML 报告 ---")
    report_path = gen.generate_lunar_transfer_report(result)
    print(f"\n报告已生成: {report_path}")
    size_kb = os.path.getsize(report_path) / 1024
    print(f"文件大小: {size_kb:.1f} KB")
    with open(report_path, "r", encoding="utf-8") as f:
        head = f.read(200)
    print(f"前 200 字符:\n{head}")
    print("\nreport 自测通过.")
