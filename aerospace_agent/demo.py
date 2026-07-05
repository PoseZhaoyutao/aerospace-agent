"""地月转移轨道端到端演示编排模块。

本模块把整条链路串起来，验证 Agent 各模块协同工作：

    1. 装配默认 Agent（LLM + CEO 上下文管理 + 记忆 + 工具 + 工作流 + RAG）
    2. Agent ReAct 循环处理"设计地月转移轨道"任务（展示 Agent 框架可用）
    3. 直接调用真实 TrajectoryAnalysisWorkflow（含完整物理：Hohmann / 拼凑圆锥 /
       Lambert / 发射窗口 / Porkchop / 公式推导）
    4. 调用 BasiliskVisualizationWorkflow 生成 3D 轨迹可视化
    5. 调用 Plotter 生成 7 张分析图
    6. 调用 ReportGenerator 生成自包含 HTML 报告（图片 base64 内嵌 + MathJax 公式）
    7. 汇总所有产物路径

用法：
    python -m aerospace_agent.demo
    aerospace-agent demo
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

# 演示产物目录
DEMO_OUTPUTS_DIR = "/workspace/demo_outputs"
REPORTS_DIR = "/workspace/reports"


def _ensure_dirs() -> None:
    """确保产物目录存在。"""
    Path(DEMO_OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)


def _print_header(title: str) -> None:
    """打印分节标题。"""
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_step(idx: int, total: int, msg: str) -> None:
    """打印步骤进度。"""
    print(f"\n[{idx}/{total}] {msg}")


# ----------------------------------------------------------------------
# 阶段 1：Agent ReAct 演示
# ----------------------------------------------------------------------
def run_agent_react(task: str = "设计地月转移轨道") -> str:
    """运行 Agent 的 ReAct 循环（MockLLM），展示 Agent 框架可用。

    Returns:
        Agent 最终答案文本。
    """
    _print_header("阶段 1：Agent ReAct 循环演示")
    from .core.agent import create_default_agent

    agent = create_default_agent(max_steps=6, force_mock=True)
    print(f"任务: {task}")
    print(f"LLM: {agent.llm.__class__.__name__}")
    print(f"原生工具: {list(agent.tools.keys())}")
    print(f"MCP 工具: {list(agent.mcp_tools.keys())}")
    print(f"工作流: {list(agent.workflows.keys())}")
    result = agent.run(task)
    return result


# ----------------------------------------------------------------------
# 阶段 2：真实工作流执行（完整物理）
# ----------------------------------------------------------------------
def run_trajectory_workflow(
    altitude_leo: float = 200e3,
    altitude_lmo: float = 100e3,
    days: int = 60,
) -> Any:
    """运行真实 TrajectoryAnalysisWorkflow（地月转移核心物理）。

    Returns:
        WorkflowResult 对象。
    """
    _print_header("阶段 2：TrajectoryAnalysisWorkflow 完整物理计算")
    from .workflows.registry import get_workflow

    wf = get_workflow("lunar_transfer")
    if wf is None:
        raise RuntimeError("lunar_transfer 工作流未注册")
    print(f"工作流: {wf.name} - {wf.description}")
    print(f"参数: altitude_leo={altitude_leo}m, altitude_lmo={altitude_lmo}m, "
          f"搜索天数={days}")
    print(f"计划步骤: {wf.get_plan()}")
    result = wf.execute(
        altitude_leo=altitude_leo,
        altitude_lmo=altitude_lmo,
        days=days,
    )
    # 打印关键结果
    if result.success:
        r = result.result
        tp = r.get("transfer_params", {})
        dv = r.get("delta_v_budget", {})
        win = r.get("window_analysis", {})
        print("\n--- 关键结果 ---")
        print(f"总 delta-V : {dv.get('dv_total_km_s', '?')} km/s")
        print(f"  TLI dv1  : {dv.get('dv1_leo_departure_km_s', '?')} km/s")
        print(f"  LOI dv2  : {dv.get('dv2_loi_braking_km_s', '?')} km/s")
        print(f"飞行时间   : {win.get('flight_time_days', '?')} 天")
        print(f"C3 能量    : {tp.get('C3_moon_m2_s2', '?')} m^2/s^2")
        best = win.get("best")
        if isinstance(best, dict):
            print(f"最优窗口   : {best.get('date', best.get('datetime', '?'))}  "
                  f"偏差={best.get('deviation_deg', '?')}°")
        print(f"产物: {result.artifacts}")
    else:
        print(f"[工作流失败] {result.summary}")
    return result


# ----------------------------------------------------------------------
# 阶段 3：Basilisk 可视化
# ----------------------------------------------------------------------
def run_basilisk_viz(result_path: str) -> List[str]:
    """运行 Basilisk 可视化工作流，生成 3D/2D 轨迹图。

    Returns:
        生成的图片路径列表。
    """
    _print_header("阶段 3：Basilisk 轨迹可视化")
    from .workflows.registry import get_workflow

    wf = get_workflow("basilisk_viz")
    if wf is None:
        print("[跳过] basilisk_viz 工作流未注册")
        return []
    out_3d = os.path.join(DEMO_OUTPUTS_DIR, "trajectory_3d.png")
    out_2d = os.path.join(DEMO_OUTPUTS_DIR, "trajectory_2d.png")
    print(f"输入: {result_path}")
    print(f"输出: {out_3d}, {out_2d}")
    viz_result = wf.execute(
        result_path=result_path,
        output_path=out_3d,
        output_path_2d=out_2d,
    )
    print(f"成功: {viz_result.success}")
    print(f"说明: {viz_result.summary}")
    return viz_result.artifacts


# ----------------------------------------------------------------------
# 阶段 4：报告图表生成
# ----------------------------------------------------------------------
def run_plotter(result: Any) -> List[str]:
    """调用 Plotter 生成全部 7 张分析图。

    Returns:
        图片路径列表。
    """
    _print_header("阶段 4：Plotter 报告图表生成")
    from .reporting.plots import Plotter

    plotter = Plotter(output_dir=DEMO_OUTPUTS_DIR)
    # result 是 WorkflowResult；Plotter 接受其 .result 字典
    data = result.result if hasattr(result, "result") else result
    paths: List[str] = []
    jobs = [
        ("transfer_trajectory", "plot_transfer_trajectory",
         ["transfer_trajectory.png"]),
        ("delta_v_budget", "plot_delta_v_budget",
         ["delta_v_budget.png"]),
        ("porkchop", "plot_porkchop", ["porkchop.png"]),
        ("launch_windows", "plot_launch_windows",
         ["launch_windows.png"]),
        ("orbit_elements", "plot_orbit_elements",
         ["orbit_elements.png"]),
        ("energy_analysis", "plot_energy_analysis",
         ["energy_analysis.png"]),
        ("mission_timeline", "plot_mission_timeline",
         ["mission_timeline.png"]),
    ]
    for label, method_name, filenames in jobs:
        method = getattr(plotter, method_name, None)
        if method is None:
            print(f"  [跳过] {label}: 方法不存在")
            continue
        try:
            method(data)
            for fn in filenames:
                p = os.path.join(DEMO_OUTPUTS_DIR, fn)
                if os.path.exists(p):
                    paths.append(p)
                    print(f"  [OK] {label} -> {p}")
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] {label}: {e}")
    return paths


# ----------------------------------------------------------------------
# 阶段 5：HTML 报告生成
# ----------------------------------------------------------------------
def run_report_generator(result: Any) -> str:
    """调用 ReportGenerator 生成自包含 HTML 报告。

    Returns:
        报告文件路径。
    """
    _print_header("阶段 5：ReportGenerator HTML 报告生成")
    from .reporting.report import ReportGenerator

    gen = ReportGenerator()
    out_path = os.path.join(REPORTS_DIR, "lunar_transfer_report.html")
    print(f"输出: {out_path}")
    final_path = gen.generate_lunar_transfer_report(result, output_path=out_path)
    size_kb = os.path.getsize(final_path) / 1024
    print(f"报告大小: {size_kb:.1f} KB")
    print(f"自包含: 图片 base64 内嵌 + MathJax 公式渲染")
    return final_path


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def run_full_demo(task: str = "设计地月转移轨道") -> Dict[str, Any]:
    """运行完整端到端演示。

    Returns:
        包含各阶段产物路径的字典。
    """
    _ensure_dirs()
    start = time.time()
    _print_header("航天导航控制 Agent — 地月转移轨道端到端演示")
    print(f"任务: {task}")
    print(f"产物目录: {DEMO_OUTPUTS_DIR} / {REPORTS_DIR}")

    artifacts: Dict[str, Any] = {}

    # 阶段 1：Agent ReAct（展示 Agent 框架）
    try:
        react_answer = run_agent_react(task)
        artifacts["react_answer"] = react_answer
    except Exception as e:  # noqa: BLE001
        print(f"[阶段 1 失败] {e}")
        artifacts["react_answer"] = f"<失败: {e}>"

    # 阶段 2：真实物理工作流
    try:
        wf_result = run_trajectory_workflow()
        artifacts["workflow_result"] = {
            "success": wf_result.success,
            "summary": wf_result.summary,
            "artifacts": wf_result.artifacts,
        }
        result_json = os.path.join(DEMO_OUTPUTS_DIR,
                                   "lunar_transfer_result.json")
        if os.path.exists(result_json):
            artifacts["result_json"] = result_json
    except Exception as e:  # noqa: BLE001
        print(f"[阶段 2 失败] {e}")
        wf_result = None

    # 阶段 3：Basilisk 可视化
    if wf_result is not None and wf_result.success:
        try:
            viz_paths = run_basilisk_viz(
                artifacts.get("result_json",
                              os.path.join(DEMO_OUTPUTS_DIR,
                                           "lunar_transfer_result.json")))
            artifacts["basilisk_plots"] = viz_paths
        except Exception as e:  # noqa: BLE001
            print(f"[阶段 3 失败] {e}")

    # 阶段 4：报告图表
    if wf_result is not None and wf_result.success:
        try:
            plot_paths = run_plotter(wf_result)
            artifacts["analysis_plots"] = plot_paths
        except Exception as e:  # noqa: BLE001
            print(f"[阶段 4 失败] {e}")

    # 阶段 5：HTML 报告
    if wf_result is not None and wf_result.success:
        try:
            report_path = run_report_generator(wf_result)
            artifacts["html_report"] = report_path
        except Exception as e:  # noqa: BLE001
            print(f"[阶段 5 失败] {e}")

    # 汇总
    elapsed = time.time() - start
    _print_header("演示完成 — 产物汇总")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"工作流成功: {artifacts.get('workflow_result', {}).get('success', False)}")
    print("\n产物清单:")
    for k, v in artifacts.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for p in v:
                print(f"    - {p}")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            preview = str(v)
            if len(preview) > 120:
                preview = preview[:120] + " ..."
            print(f"  {k}: {preview}")

    # 关键产物提示
    if "html_report" in artifacts:
        print(f"\n>>> 最终报告: {artifacts['html_report']}")
    if "result_json" in artifacts:
        print(f">>> 完整结果: {artifacts['result_json']}")

    return artifacts


def main() -> None:
    """命令行入口。"""
    run_full_demo()


if __name__ == "__main__":
    main()
