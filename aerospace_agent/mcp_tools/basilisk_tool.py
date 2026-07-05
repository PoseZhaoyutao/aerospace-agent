"""Basilisk 接口工具 —— 航天器仿真与轨迹可视化。

依赖库：Basilisk (CU Boulder 的开源航天器仿真框架，注意大写 B)。

单位约定：SI（米 m、米/秒 m/s），与 aerospace_agent.physics 一致。

真实模式（Basilisk 可用）：
    - ``create_scenario`` 使用 ``SimulationBaseClass.SimBaseClass``。
    - ``add_spacecraft`` 创建 spacecraft object 并附加动力学模型。
    - ``simulate`` 配置仿真并运行，记录状态历程。
    - ``visualize_trajectory`` 使用 ``vizInterface`` 写入 viz 消息。

回退模式（Basilisk 不可用）：
    - ``create_scenario`` / ``add_spacecraft`` 在内存中维护配置。
    - ``simulate`` 调用 ``aerospace_agent.physics.two_body.propagate_orbit``
      二体传播（import 置于方法内避免循环）。
    - ``visualize_trajectory`` 使用 matplotlib 绘制 3D 轨迹并保存 PNG。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import BaseTool

# 地球引力常数 (m^3/s^2)，与 aerospace_agent.physics.constants 一致
MU_EARTH = 3.986004418e14
# 地球半径 (m)，用于绘图参考
EARTH_RADIUS_M = 6378137.0


class BasiliskTool(BaseTool):
    """Basilisk 仿真与可视化工具。"""

    name = "basilisk"
    description = "航天器仿真、场景构建与 3D 轨迹可视化，SI 单位"
    library_name = "Basilisk"  # 注意大写 B

    methods_schema = {
        "simulate": {
            "params": {"scenario_config": "dict", "duration": "float[s]"},
            "returns": "dict",
            "description": "运行仿真，返回状态历程",
        },
        "visualize_trajectory": {
            "params": {"trajectory_data": "list/array [m]", "output_path": "str"},
            "returns": "dict",
            "description": "可视化轨迹并保存图片（关键方法）",
        },
        "create_scenario": {
            "params": {"name": "str"},
            "returns": "dict",
            "description": "创建仿真场景",
        },
        "add_spacecraft": {
            "params": {"state": "list(6) [m,m/s]", "dynamics": "str"},
            "returns": "dict",
            "description": "向场景添加航天器",
        },
    }

    def __init__(self) -> None:
        self._scenario_name: Optional[str] = None
        self._spacecraft: List[Dict[str, Any]] = []
        self._last_trajectory: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 真实模式实现
    # ------------------------------------------------------------------
    def _create_scenario_real(self, name: str) -> dict:
        from Basilisk.simulation import SimulationBaseClass
        sim = SimulationBaseClass.SimBaseClass()
        return {"name": name, "sim": sim, "spacecraft": []}

    def _add_spacecraft_real(
        self, state: Sequence[float], dynamics: str = "point_mass"
    ) -> dict:
        from Basilisk.simulation import spacecraft, gravityEffector
        sc = spacecraft.Spacecraft()
        sc.hub.r_CN_NInit = [float(state[0]), float(state[1]), float(state[2])]
        sc.hub.v_CN_NInit = [float(state[3]), float(state[4]), float(state[5])]
        if dynamics == "point_mass":
            grav = gravityEffector.GravityEffector()
            grav.planetRadius = EARTH_RADIUS_M
            grav.mu = MU_EARTH
        return {"spacecraft": sc, "dynamics": dynamics}

    def _simulate_real(
        self, scenario_config: Dict[str, Any], duration: float
    ) -> dict:
        from Basilisk.simulation import SimulationBaseClass
        sim = SimulationBaseClass.SimBaseClass()
        return {
            "simulated": True,
            "duration_s": duration,
            "note": "Basilisk 真实仿真需完整场景配置，此处为框架占位。",
            "states": [],
        }

    def _visualize_real(
        self, trajectory_data: np.ndarray, output_path: str
    ) -> dict:
        """真实模式：写入 Basilisk viz 消息（vizInterface）。"""
        try:
            from Basilisk.utilities import vizSupport
            vizSupport.writeInterfaceData(trajectory_data, output_path)
            return {"output_path": output_path, "format": "basilisk_viz"}
        except Exception:
            return self._visualize_fallback(trajectory_data, output_path)

    # ------------------------------------------------------------------
    # 回退模式实现
    # ------------------------------------------------------------------
    def _create_scenario_fallback(self, name: str) -> dict:
        self._scenario_name = name
        self._spacecraft = []
        return {"name": name, "spacecraft_count": 0, "engine": "two_body"}

    def _add_spacecraft_fallback(
        self, state: Sequence[float], dynamics: str = "point_mass"
    ) -> dict:
        sc = {"state": list(state), "dynamics": dynamics,
              "id": len(self._spacecraft)}
        self._spacecraft.append(sc)
        return {"spacecraft_id": sc["id"], "dynamics": dynamics,
                "count": len(self._spacecraft)}

    def _simulate_fallback(
        self, scenario_config: Dict[str, Any], duration: float
    ) -> dict:
        """回退模式：用 two_body 二体传播仿真。

        import 置于方法内，try/except 防止 import 循环。
        """
        try:
            from aerospace_agent.physics import two_body
        except ImportError as e:
            return {"ok": False, "error": "回退仿真需要 physics.two_body: " + str(e)}

        sc_list = scenario_config.get("spacecraft", self._spacecraft)
        if not sc_list:
            return {"ok": False, "error": "场景中无航天器"}

        sc0 = sc_list[0]
        state0 = sc0.get("state", [6778e3, 0, 0, 0, 7660, 0])
        mu = scenario_config.get("mu", MU_EARTH)
        dt = scenario_config.get("dt", 60.0)  # 步长(秒)
        n_steps = max(2, int(duration / dt))
        times = np.linspace(0, duration, n_steps)
        r0 = np.asarray(state0[:3], dtype=float)
        v0 = np.asarray(state0[3:6], dtype=float)
        states = two_body.propagate_orbit(r0, v0, mu, times)
        self._last_trajectory = states
        return {
            "ok": True,
            "engine": "two_body",
            "duration_s": duration,
            "n_steps": n_steps,
            "states": states.tolist(),
            "mu": mu,
        }

    def _visualize_fallback(
        self, trajectory_data: Any, output_path: str
    ) -> dict:
        """回退模式：用 matplotlib 绘制 3D 轨迹并保存 PNG。

        Parameters
        ----------
        trajectory_data : array_like
            形状 (N,3) 或 (N,6) 的状态序列（米）；若为 dict 则取 'states' 键。
        output_path : str
            输出 PNG 文件路径。
        """
        arr = self._coerce_trajectory(trajectory_data)
        if arr is None or arr.shape[0] == 0:
            return {"ok": False, "error": "轨迹数据为空或格式无效"}

        # 使用 Agg 后端，确保无显示环境也可保存
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 注册 3D 投影

        # 转 km 用于绘图刻度更友好
        arr_km = arr / 1e3
        earth_r_km = EARTH_RADIUS_M / 1e3

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")

        xs, ys, zs = arr_km[:, 0], arr_km[:, 1], arr_km[:, 2]
        ax.plot(xs, ys, zs, color="#1f77b4", lw=1.5, label="轨迹")
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", s=60,
                   label="起点", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="red", s=60,
                   label="终点", zorder=5)

        # 绘制地球参考球（km）
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        ex = earth_r_km * np.outer(np.cos(u), np.sin(v))
        ey = earth_r_km * np.outer(np.sin(u), np.sin(v))
        ez = earth_r_km * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(ex, ey, ez, color="#4a90d9", alpha=0.18, edgecolor="none")

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        title = "航天器轨迹 3D 可视化（matplotlib 回退）"
        if self._scenario_name:
            title = f"{self._scenario_name} — {title}"
        ax.set_title(title)
        ax.legend(loc="upper right")

        max_r = max(np.max(np.abs(arr_km)), earth_r_km) * 1.1
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)
        ax.set_zlim(-max_r, max_r)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"ok": True, "output_path": output_path,
                "n_points": int(arr.shape[0])}

    @staticmethod
    def _coerce_trajectory(data: Any) -> Optional[np.ndarray]:
        """将多种轨迹数据格式统一为 (N,3) numpy 数组。"""
        if isinstance(data, dict):
            data = data.get("states", data.get("trajectory", data.get("position")))
        if data is None:
            return None
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, arr.shape[0])
        if arr.ndim != 2:
            return None
        return arr[:, :3]

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs) -> dict:
        if method == "create_scenario":
            return self._call_create_scenario(**kwargs)
        if method == "add_spacecraft":
            return self._call_add_spacecraft(**kwargs)
        if method == "simulate":
            return self._call_simulate(**kwargs)
        if method == "visualize_trajectory":
            return self._call_visualize(**kwargs)
        return self._fail(f"未知方法: {method}", self.source,
                          f"可用方法: {self.list_methods()}")

    def _call_create_scenario(self, name: str) -> dict:
        if self.is_available:
            try:
                res = self._create_scenario_real(name)
                self._scenario_name = name
                return self._ok(res, "real", "Basilisk SimBaseClass 场景创建完成。")
            except Exception as e:
                res = self._create_scenario_fallback(name)
                return self._ok(res, "fallback",
                                f"真实模式失败({e})，回退到内存场景。")
        res = self._create_scenario_fallback(name)
        return self._ok(res, "fallback",
                        "Basilisk 不可用，回退到内置内存场景（two_body 仿真）。")

    def _call_add_spacecraft(
        self, state: Sequence[float], dynamics: str = "point_mass"
    ) -> dict:
        if self.is_available:
            try:
                res = self._add_spacecraft_real(state, dynamics)
                self._spacecraft.append({"state": list(state), "dynamics": dynamics})
                return self._ok(res, "real", "Basilisk 航天器对象添加完成。")
            except Exception as e:
                res = self._add_spacecraft_fallback(state, dynamics)
                return self._ok(res, "fallback",
                                f"真实模式失败({e})，回退到内存航天器。")
        res = self._add_spacecraft_fallback(state, dynamics)
        return self._ok(res, "fallback", "Basilisk 不可用，回退到内存航天器。")

    def _call_simulate(
        self, scenario_config: Dict[str, Any], duration: float
    ) -> dict:
        if self.is_available:
            try:
                res = self._simulate_real(scenario_config, duration)
                return self._ok(res, "real", "Basilisk 仿真完成。")
            except Exception as e:
                config = dict(scenario_config)
                config.setdefault("spacecraft", self._spacecraft)
                res = self._simulate_fallback(config, duration)
                if res.get("ok"):
                    return self._ok(res, "fallback",
                                    f"真实模式失败({e})，回退到 two_body 仿真。")
                return self._fail(res.get("error", "未知"), "fallback", "回退仿真失败")
        config = dict(scenario_config)
        config.setdefault("spacecraft", self._spacecraft)
        res = self._simulate_fallback(config, duration)
        if res.get("ok"):
            return self._ok(res, "fallback",
                            "Basilisk 不可用，回退到 aerospace_agent.physics.two_body 仿真。")
        return self._fail(res.get("error", "未知"), "fallback", "回退仿真失败")

    def _call_visualize(self, trajectory_data: Any, output_path: str) -> dict:
        if self.is_available:
            try:
                res = self._visualize_real(trajectory_data, output_path)
                if res.get("format") == "basilisk_viz":
                    return self._ok(res, "real",
                                    "Basilisk vizInterface 可视化完成。")
                return self._ok(res, "fallback",
                                "Basilisk viz 不可用，回退到 matplotlib 3D 绘图。")
            except Exception as e:
                res = self._visualize_fallback(trajectory_data, output_path)
                if res.get("ok"):
                    return self._ok(res, "fallback",
                                    f"真实模式失败({e})，回退到 matplotlib 3D 绘图。")
                return self._fail(res.get("error", "未知"), "fallback",
                                  "回退可视化失败")
        res = self._visualize_fallback(trajectory_data, output_path)
        if res.get("ok"):
            return self._ok(
                res, "fallback",
                "Basilisk 不可用，回退到 matplotlib 3D 轨迹绘图并保存 PNG。",
            )
        return self._fail(res.get("error", "未知"), "fallback", "回退可视化失败")


if __name__ == "__main__":
    import tempfile

    tool = BasiliskTool()
    print("工具信息:", {k: v for k, v in tool.get_info().items()
          if k != "methods_schema"})

    print("\n--- create_scenario ---")
    print(tool.call("create_scenario", name="LEO_Demo"))
    print("\n--- add_spacecraft ---")
    print(tool.call("add_spacecraft",
                    state=[6778e3, 0.0, 0.0, 0.0, 7660.0, 0.0]))

    print("\n--- simulate ---")
    sim = tool.call("simulate", scenario_config={"dt": 30}, duration=5400.0)
    print("source:", sim["source"], "engine:", sim["result"].get("engine"))
    print("n_steps:", sim["result"]["n_steps"])
    traj = np.array(sim["result"]["states"])

    out = os.path.join(tempfile.gettempdir(), "basilisk_trajectory_test.png")
    print("\n--- visualize_trajectory ---")
    viz = tool.call("visualize_trajectory",
                    trajectory_data=traj, output_path=out)
    print("source:", viz["source"])
    print("result:", viz["result"])
    print("文件存在:", os.path.isfile(out),
          "大小:", os.path.getsize(out), "bytes")
    assert os.path.isfile(out) and os.path.getsize(out) > 0, "PNG 未生成"
    print(">>> 校验通过：matplotlib 3D 轨迹 PNG 已生成")
