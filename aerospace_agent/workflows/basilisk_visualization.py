"""Basilisk 可视化工作流 (Basilisk visualization workflow)。

加载轨迹数据 (从内存 / JSON 结果文件)，调用 Basilisk 工具生成 3D 轨迹
可视化，并额外生成地月系 XY 平面的 2D 投影图。

工具调用统一走 ``aerospace_agent.mcp_tools.basilisk_tool.BasiliskTool.call``
的 ``visualize_trajectory`` 方法 (Basilisk 不可用时回退到 matplotlib 3D)。
2D 投影图由本工作流用 matplotlib 直接绘制 (地月系俯视)。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from aerospace_agent.physics import (
    A_MOON,
    DAY2SEC,
    MU_EARTH,
    R_EARTH,
    R_MOON,
    propagate_orbit,
)
from aerospace_agent.physics.moon_transfer import MoonTransfer

from .base import BaseWorkflow, WorkflowResult, register_workflow

DEMO_OUTPUTS_DIR = "/workspace/demo_outputs"


@register_workflow()
class BasiliskVisualizationWorkflow(BaseWorkflow):
    """Basilisk 可视化工作流。

    name = 'basilisk_viz'
    """

    name = "basilisk_viz"
    description = "加载轨迹数据，调用 Basilisk 生成 3D 可视化，并绘制地月系 2D 投影图"
    version = "1.0.0"
    required_tools = ["basilisk"]

    steps = [
        {"name": "load_trajectory", "description": "加载轨迹数据 (内存/JSON)"},
        {"name": "call_basilisk", "description": "调用 Basilisk 工具 (visualize_trajectory)"},
        {"name": "generate_3d", "description": "生成 3D 可视化并保存"},
        {"name": "generate_2d", "description": "生成 2D 投影图 (XY 平面, 地月系)"},
    ]

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    def validate_params(self, params: dict) -> bool:
        if not super().validate_params(params):
            return False
        # 至少提供 trajectory_data 或 result_path 之一 (否则用默认)
        return True

    # ------------------------------------------------------------------
    # 主执行
    # ------------------------------------------------------------------
    def execute(
        self,
        trajectory_data: Any = None,
        result_path: Optional[str] = None,
        output_path: str = os.path.join(DEMO_OUTPUTS_DIR, "trajectory_3d.png"),
        **kwargs,
    ) -> WorkflowResult:
        """执行可视化工作流。

        Parameters
        ----------
        trajectory_data : array_like | dict | None
            轨迹数据。可为 (N,3)/(N,6) 数组、含 'states' 的 dict、或 None。
            若为 None 且 result_path 给定, 从 JSON 加载并传播生成完整轨迹。
        result_path : str | None
            trajectory_analysis 工作流产出的 JSON 路径。若给定, 从中提取
            转移轨道近地点状态并用二体传播生成完整轨迹。
        output_path : str
            3D 可视化 PNG 输出路径。
        **kwargs
            额外参数 (如 output_path_2d 2D 图路径, n_points 传播点数)。
        """
        res = WorkflowResult()
        res.metadata["params"] = {
            "result_path": result_path,
            "output_path": output_path,
        }
        output_path_2d = kwargs.get("output_path_2d")
        if output_path_2d is None:
            # 默认 2D 路径：把 3D 路径中的 "3d" 替换为 "2d"
            base, ext = os.path.splitext(output_path)
            if "3d" in base:
                output_path_2d = base.replace("3d", "2d") + ext
            else:
                output_path_2d = base + "_2d" + ext
        n_points = kwargs.get("n_points", 200)

        # ---- 步骤 1：加载轨迹数据 ----
        positions, meta = self._resolve_trajectory(
            trajectory_data, result_path, n_points
        )
        if positions is None or len(positions) == 0:
            self._log_step(res, "load_trajectory", "failed",
                           "无法获取轨迹数据 (trajectory_data 与 result_path 均无效)")
            res.summary = "可视化失败：无轨迹数据"
            return res

        self._log_step(
            res, "load_trajectory", "success",
            f"已加载轨迹: {len(positions)} 个点, 来源={meta.get('source','unknown')}; "
            f"起点 |r|={np.linalg.norm(positions[0])/1e3:.1f}km, "
            f"终点 |r|={np.linalg.norm(positions[-1])/1e3:.1f}km",
            data={"n_points": len(positions),
                  "source": meta.get("source"),
                  "start_r_km": float(np.linalg.norm(positions[0]) / 1e3),
                  "end_r_km": float(np.linalg.norm(positions[-1]) / 1e3)},
        )

        # ---- 步骤 2 & 3：调用 Basilisk 工具生成 3D 可视化 ----
        viz_resp = self._call_basilisk_visualize(positions, output_path)
        if not viz_resp.get("success"):
            self._log_step(res, "call_basilisk", "failed",
                           f"Basilisk 可视化失败: {viz_resp.get('error')}")
            res.summary = f"3D 可视化失败: {viz_resp.get('error')}"
            return res

        self._log_step(
            res, "call_basilisk", "success",
            f"Basilisk 工具调用成功 (source={viz_resp['source']}); "
            f"3D 图已生成",
            data={"source": viz_resp["source"],
                  "message": viz_resp.get("message")},
        )

        self._log_step(
            res, "generate_3d", "success",
            f"3D 轨迹图已保存: {output_path}",
            data={"output_path": output_path,
                  "file_exists": os.path.isfile(output_path)},
        )
        res.artifacts.append(output_path)

        # ---- 步骤 4：生成 2D 投影图 (XY 平面, 地月系) ----
        try:
            self._generate_2d_projection(positions, output_path_2d, meta)
            self._log_step(
                res, "generate_2d", "success",
                f"2D 投影图已保存: {output_path_2d}",
                data={"output_path_2d": output_path_2d,
                      "file_exists": os.path.isfile(output_path_2d)},
            )
            res.artifacts.append(output_path_2d)
        except Exception as exc:  # pragma: no cover - 异常路径
            self._log_step(res, "generate_2d", "warning",
                           f"2D 投影图生成失败: {exc}")

        # 组装结果
        res.success = True
        res.result = {
            "n_points": len(positions),
            "trajectory_source": meta.get("source"),
            "basilisk_source": viz_resp["source"],
            "output_3d": output_path,
            "output_2d": output_path_2d if os.path.isfile(output_path_2d) else None,
            "start_position_m": positions[0].tolist(),
            "end_position_m": positions[-1].tolist(),
        }
        res.summary = (
            f"轨迹可视化完成：{len(positions)} 个点，3D 图 (Basilisk "
            f"source={viz_resp['source']}) 保存至 {output_path}"
            + (f"，2D 地月系投影图保存至 {output_path_2d}"
               if os.path.isfile(output_path_2d) else "")
        )
        res.metadata["basilisk_source"] = viz_resp["source"]
        return res

    # ------------------------------------------------------------------
    # 辅助：解析/生成轨迹数据
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_trajectory(trajectory_data, result_path, n_points):
        """返回 (positions (N,3) ndarray, meta dict)。

        优先级：
            1. trajectory_data 直接可用 (array / dict with 'states')
            2. result_path JSON -> 提取近地点状态 -> 二体传播
            3. 都没有 -> 用 MoonTransfer 生成默认地月转移轨迹
        """
        # 1) 直接给定的轨迹数据
        if trajectory_data is not None:
            arr = BasiliskVisualizationWorkflow._coerce_positions(trajectory_data)
            if arr is not None and len(arr) > 0:
                return arr, {"source": "trajectory_data_arg"}

        # 2) 从 JSON 结果文件加载并传播
        if result_path is not None and os.path.isfile(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                traj = data.get("trajectory", {})
                perigee = traj.get("transfer_perigee", {})
                r0 = perigee.get("position_m")
                v0 = perigee.get("velocity_m_s")
                t_flight = traj.get("t_flight_s") or data.get(
                    "transfer_params", {}
                ).get("t_flight_s")
                if r0 and v0 and t_flight:
                    r0 = np.asarray(r0, dtype=float)
                    v0 = np.asarray(v0, dtype=float)
                    times = np.linspace(0.0, t_flight, n_points)
                    states = propagate_orbit(r0, v0, MU_EARTH, times)
                    return states[:, :3], {
                        "source": "json_propagated",
                        "result_path": result_path,
                        "t_flight_s": t_flight,
                    }
            except Exception:
                pass

        # 3) 默认：用 MoonTransfer 生成地月转移轨迹
        mt = MoonTransfer()
        traj = mt.design_trajectory(
            launch_date="2026-08-06", altitude_leo=200e3, altitude_lmo=100e3
        )
        r0, v0 = traj["transfer_perigee"]
        t_flight = traj["t_flight"]
        times = np.linspace(0.0, t_flight, n_points)
        states = propagate_orbit(r0, v0, MU_EARTH, times)
        return states[:, :3], {
            "source": "default_moon_transfer",
            "t_flight_s": t_flight,
        }

    @staticmethod
    def _coerce_positions(data) -> Optional[np.ndarray]:
        """把多种轨迹格式统一为 (N,3) ndarray。"""
        if isinstance(data, dict):
            data = data.get("states", data.get("trajectory",
                                                data.get("position", data.get("positions"))))
        if data is None:
            return None
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, arr.shape[0])
        if arr.ndim != 2:
            return None
        return arr[:, :3]

    # ------------------------------------------------------------------
    # 辅助：调用 Basilisk 工具 (统一 call 接口)
    # ------------------------------------------------------------------
    @staticmethod
    def _call_basilisk_visualize(positions: np.ndarray, output_path: str) -> dict:
        """通过 mcp_tools 统一接口调用 Basilisk visualize_trajectory。"""
        try:
            from aerospace_agent.mcp_tools import get_tool
            tool = get_tool("basilisk")
            if tool is None:
                from aerospace_agent.mcp_tools.basilisk_tool import BasiliskTool
                tool = BasiliskTool()
            return tool.call(
                "visualize_trajectory",
                trajectory_data=positions,
                output_path=output_path,
            )
        except Exception as exc:
            return {
                "success": False,
                "source": "fallback",
                "error": str(exc),
                "message": f"Basilisk 工具调用异常: {exc}",
            }

    # ------------------------------------------------------------------
    # 辅助：生成 2D 投影图 (XY 平面, 地月系)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_2d_projection(positions: np.ndarray, output_path: str,
                                meta: dict) -> None:
        """绘制地月系 XY 平面 2D 投影图。

        内容：
            - 地球 (原点圆, R_earth)
            - 月球轨道 (圆, a_moon)
            - 转移轨迹 (XY 投影)
            - LEO 出发点 / 月球到达点
            - 月球到达时刻月球位置 (由轨迹终点近似)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 转 km
        pos_km = positions / 1e3
        earth_r_km = R_EARTH / 1e3
        moon_orbit_km = A_MOON / 1e3
        moon_r_km = R_MOON / 1e3

        fig, ax = plt.subplots(figsize=(9, 9))

        # 地球
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.fill(earth_r_km * np.cos(theta), earth_r_km * np.sin(theta),
                color="#4a90d9", alpha=0.4, label="地球")
        # 月球轨道
        ax.plot(moon_orbit_km * np.cos(theta), moon_orbit_km * np.sin(theta),
                color="#888888", ls="--", lw=1, label="月球轨道")

        # 转移轨迹 (XY 投影)
        ax.plot(pos_km[:, 0], pos_km[:, 1], color="#1f77b4", lw=2,
                label="转移轨迹 (XY 投影)")
        # 出发点 (近地点)
        ax.scatter([pos_km[0, 0]], [pos_km[0, 1]], color="green", s=80,
                   zorder=5, label="LEO 出发 (近地点)")
        # 到达点 (远地点/月球到达)
        ax.scatter([pos_km[-1, 0]], [pos_km[-1, 1]], color="red", s=80,
                   zorder=5, label="月球到达 (远地点)")
        # 月球 (在到达点附近绘制)
        ax.scatter([pos_km[-1, 0]], [pos_km[-1, 1]], color="#cccccc", s=200,
                   zorder=4, edgecolor="black")
        ax.scatter([pos_km[-1, 0]], [pos_km[-1, 1]], color="#cccccc", s=200,
                   zorder=4, label="月球")

        ax.set_aspect("equal")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title("地月转移轨迹 — XY 平面投影 (地月系俯视)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

        # 坐标范围
        max_r = max(np.max(np.abs(pos_km)), moon_orbit_km) * 1.15
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== aerospace_agent.workflows.basilisk_visualization 自测 ===")

    wf = BasiliskVisualizationWorkflow()
    print(f"工作流: {wf!r}")
    print(f"步骤计划:\n  " + "\n  ".join(wf.get_plan()))
    print(f"工具可用性: {wf.check_tools()}")

    # 测试 1：从 lunar_transfer_result.json 加载 (若存在)
    json_path = os.path.join(DEMO_OUTPUTS_DIR, "lunar_transfer_result.json")
    if os.path.isfile(json_path):
        print(f"\n--- 测试 1: 从 JSON 加载 ({json_path}) ---")
        r = wf.execute(result_path=json_path,
                       output_path=os.path.join(DEMO_OUTPUTS_DIR, "trajectory_3d.png"))
    else:
        print("\n--- 测试 1: 默认轨迹 (MoonTransfer) ---")
        r = wf.execute(output_path=os.path.join(DEMO_OUTPUTS_DIR, "trajectory_3d.png"))

    print(f"success={r.success}, steps={len(r.steps_log)}")
    print(f"summary: {r.summary}")
    print(f"artifacts: {r.artifacts}")
    print(f"result: {r.result}")

    assert r.success
    for art in r.artifacts:
        assert os.path.isfile(art), f"文件应存在: {art}"
        print(f"  {art}: {os.path.getsize(art)} bytes")

    # 测试 2：直接传入轨迹数据 (array)
    print("\n--- 测试 2: 直接传入轨迹数组 ---")
    mt = MoonTransfer()
    traj = mt.design_trajectory(launch_date="2026-08-06")
    r0, v0 = traj["transfer_perigee"]
    times = np.linspace(0.0, traj["t_flight"], 100)
    states = propagate_orbit(r0, v0, MU_EARTH, times)
    r2 = wf.execute(
        trajectory_data=states[:, :3],
        output_path=os.path.join(DEMO_OUTPUTS_DIR, "trajectory_3d_v2.png"),
    )
    print(f"success={r2.success}, artifacts={r2.artifacts}")
    assert r2.success

    print("\nbasilisk_visualization 自测全部通过.")
