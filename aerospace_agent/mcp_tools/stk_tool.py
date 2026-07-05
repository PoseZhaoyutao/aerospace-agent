"""STK 接口工具 —— AGI STK 桌面应用的 COM 自动化接口。

依赖：comtypes + STK Application（仅 Windows）。

STK (Systems Tool Kit) 是 AGI 公司的商业航天任务分析软件，通过
comtypes.client 连接其 COM Application 对象实现自动化。

可用性检测：
    1. 尝试 import comtypes.client
    2. 尝试通过 comtypes 创建 STK Application 对象（检查 STK 是否安装）

方法：
    - connect()：连接 STK Application
    - create_scenario(name, start, stop)：创建场景
    - add_satellite(name, state)：添加卫星
    - compute_access(from_obj, to_obj)：计算可见性
    - generate_report(report_type)：生成报告

回退策略：
    - STK 为 Windows 专属商业软件，无内置物理回退。
    - 所有方法返回 source='unavailable'，但提供 STK 脚本模板文本
      （可保存为 .py 在装有 STK 的 Windows 机器上运行）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import BaseTool


# STK 脚本模板（Python + comtypes）
_STK_SCRIPT_TEMPLATE = '''"""
STK 自动化脚本模板（由 aerospace_agent.mcp_tools.StkTool 生成）。
运行环境：Windows + STK + comtypes (pip install comtypes)。
"""
import comtypes.client
from datetime import datetime, timedelta

# 启动/连接 STK
uiApplication = comtypes.client.CreateObject("STK11.Application")
uiApplication.Visible = True
uiApplication.UserControl = True
root = uiApplication.Personality2

# 创建场景
root.NewScenario("{scenario_name}")
scenario = root.CurrentScenario
scenario.SetTimePeriod("{start}", "{stop}")

# 添加卫星（J2000 ECI 笛卡尔状态）
satellite = scenario.Children.New(18, "{sat_name}")  # 18 = eSatellite
satellite.SetStateType("Cartesian")  # Cartesian 状态类型
satellite.Cartesian.X = {x}
satellite.Cartesian.Y = {y}
satellite.Cartesian.Z = {z}
satellite.Cartesian.VX = {vx}
satellite.Cartesian.VY = {vy}
satellite.Cartesian.VZ = {vz}
satellite.Cartesian.CoordinateSystem = "J2000"
satellite.Propagate()

# 计算可见性 (Access)
access = {from_obj}.GetAccessToObject({to_obj})
access.ComputeAccess()

# 生成报告
report = scenario.GetReport("Access Summary")
report.Execute()

print("STK 场景构建完成。")
'''


class StkTool(BaseTool):
    """STK 桌面应用 COM 自动化工具。"""

    name = "stk"
    description = "AGI STK 场景/卫星/可见性/报告自动化（Windows + comtypes）"
    library_name = "comtypes"  # 实际依赖 comtypes.client；STK 本身为独立应用

    methods_schema = {
        "connect": {
            "params": {},
            "returns": "dict",
            "description": "连接 STK Application",
        },
        "create_scenario": {
            "params": {"name": "str", "start": "str", "stop": "str"},
            "returns": "dict",
            "description": "创建 STK 场景",
        },
        "add_satellite": {
            "params": {"name": "str", "state": "list(6)"},
            "returns": "dict",
            "description": "添加卫星并设置状态",
        },
        "compute_access": {
            "params": {"from_obj": "str", "to_obj": "str"},
            "returns": "dict",
            "description": "计算两对象间可见性",
        },
        "generate_report": {
            "params": {"report_type": "str"},
            "returns": "dict",
            "description": "生成 STK 报告",
        },
    }

    def __init__(self) -> None:
        self._root = None  # STK root 对象（真实模式）
        self._connected = False
        self._stk_version: Optional[str] = None

    # ------------------------------------------------------------------
    # 可用性检测（覆写：检测 comtypes + STK）
    # ------------------------------------------------------------------
    def _check_available(self) -> bool:
        cache_key = "stk_runtime:comtypes"
        if cache_key in BaseTool._availability_cache:
            return BaseTool._availability_cache[cache_key]

        available = False
        try:
            import comtypes.client
            # 尝试创建 STK Application（可能因未安装 STK 而失败）
            # 尝试常见版本号 11-13
            for ver in (11, 12, 13):
                try:
                    app = comtypes.client.CreateObject(f"STK{ver}.Application")
                    self._stk_version = str(ver)
                    available = True
                    break
                except Exception:
                    continue
        except ImportError:
            available = False

        BaseTool._availability_cache[cache_key] = available
        return available

    # ------------------------------------------------------------------
    # 真实模式实现
    # ------------------------------------------------------------------
    def _connect_real(self) -> dict:
        import comtypes.client
        ver = self._stk_version or "11"
        ui_app = comtypes.client.CreateObject(f"STK{ver}.Application")
        ui_app.Visible = True
        ui_app.UserControl = True
        self._root = ui_app.Personality2
        self._connected = True
        return {"connected": True, "version": ver}

    def _create_scenario_real(
        self, name: str, start: str, stop: str
    ) -> dict:
        if not self._connected:
            self._connect_real()
        self._root.NewScenario(name)
        scenario = self._root.CurrentScenario
        scenario.SetTimePeriod(start, stop)
        return {"scenario": name, "start": start, "stop": stop}

    def _add_satellite_real(
        self, name: str, state: Sequence[float]
    ) -> dict:
        scenario = self._root.CurrentScenario
        sat = scenario.Children.New(18, name)  # 18 = eSatellite
        sat.SetStateType("Cartesian")
        c = sat.Cartesian
        c.X, c.Y, c.Z = float(state[0]), float(state[1]), float(state[2])
        c.VX, c.VY, c.VZ = float(state[3]), float(state[4]), float(state[5])
        c.CoordinateSystem = "J2000"
        sat.Propagate()
        return {"satellite": name, "state": list(state)}

    def _compute_access_real(
        self, from_obj: str, to_obj: str
    ) -> dict:
        scenario = self._root.CurrentScenario
        from_o = scenario.Children[from_obj]
        to_o = scenario.Children[to_obj]
        access = from_o.GetAccessToObject(to_o)
        access.ComputeAccess()
        # 获取 access 数据
        access_results = access.GetAccessData()
        return {"from": from_obj, "to": to_obj,
                "access": str(access_results)}

    def _generate_report_real(self, report_type: str) -> dict:
        scenario = self._root.CurrentScenario
        report = scenario.GetReport(report_type)
        report.Execute()
        return {"report_type": report_type, "status": "executed"}

    # ------------------------------------------------------------------
    # 脚本模板生成（回退模式下仍可生成）
    # ------------------------------------------------------------------
    def _generate_script_template(
        self, scenario_name: str = "MyScenario",
        sat_name: str = "Satellite1",
        state: Optional[Sequence[float]] = None,
        from_obj: str = "Satellite1", to_obj: str = "Facility1",
        start: str = "1 Jan 2025 00:00:00",
        stop: str = "2 Jan 2025 00:00:00",
    ) -> str:
        """生成 STK Python 自动化脚本模板。"""
        if state is None:
            state = [6778e3, 0, 0, 0, 7660, 0]
        return _STK_SCRIPT_TEMPLATE.format(
            scenario_name=scenario_name, sat_name=sat_name,
            x=float(state[0]), y=float(state[1]), z=float(state[2]),
            vx=float(state[3]), vy=float(state[4]), vz=float(state[5]),
            from_obj=from_obj, to_obj=to_obj,
            start=start, stop=stop,
        )

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs) -> dict:
        if method == "connect":
            return self._call_connect(**kwargs)
        if method == "create_scenario":
            return self._call_create_scenario(**kwargs)
        if method == "add_satellite":
            return self._call_add_satellite(**kwargs)
        if method == "compute_access":
            return self._call_compute_access(**kwargs)
        if method == "generate_report":
            return self._call_generate_report(**kwargs)
        return self._fail(f"未知方法: {method}", self.source,
                          f"可用方法: {self.list_methods()}")

    def _call_connect(self) -> dict:
        if not self.is_available:
            script = self._generate_script_template()
            return {
                "success": False,
                "source": "unavailable",
                "result": None,
                "error": "STK 需要 Windows + comtypes + STK Application，当前环境不可用。",
                "message": (
                    "STK 不可用。已附带 STK Python 自动化脚本模板，"
                    "可在装有 STK 的 Windows 机器上运行。"
                    "安装 comtypes: pip install comtypes。"
                ),
            }
        try:
            res = self._connect_real()
            return self._ok(res, "real", "STK Application 连接成功。")
        except Exception as e:
            return self._fail(str(e), "real", "STK 连接失败")

    def _call_create_scenario(
        self, name: str, start: str, stop: str
    ) -> dict:
        if not self.is_available:
            return self._unavailable(
                "create_scenario", "comtypes + STK",
                install_hint="STK 为 Windows 商业软件。可调用 connect() 获取脚本模板。"
            )
        try:
            res = self._create_scenario_real(name, start, stop)
            return self._ok(res, "real", f"STK 场景 {name} 创建完成。")
        except Exception as e:
            return self._fail(str(e), "real", "场景创建失败")

    def _call_add_satellite(
        self, name: str, state: Sequence[float]
    ) -> dict:
        if not self.is_available:
            return self._unavailable(
                "add_satellite", "comtypes + STK",
                install_hint="STK 为 Windows 商业软件。可调用 connect() 获取脚本模板。"
            )
        try:
            res = self._add_satellite_real(name, state)
            return self._ok(res, "real", f"STK 卫星 {name} 添加完成。")
        except Exception as e:
            return self._fail(str(e), "real", "卫星添加失败")

    def _call_compute_access(
        self, from_obj: str, to_obj: str
    ) -> dict:
        if not self.is_available:
            return self._unavailable(
                "compute_access", "comtypes + STK",
                install_hint="可见性计算需 STK 运行时。可调用 connect() 获取脚本模板。"
            )
        try:
            res = self._compute_access_real(from_obj, to_obj)
            return self._ok(res, "real", f"STK Access {from_obj}->{to_obj} 计算完成。")
        except Exception as e:
            return self._fail(str(e), "real", "可见性计算失败")

    def _call_generate_report(self, report_type: str) -> dict:
        if not self.is_available:
            return self._unavailable(
                "generate_report", "comtypes + STK",
                install_hint="报告生成需 STK 运行时。可调用 connect() 获取脚本模板。"
            )
        try:
            res = self._generate_report_real(report_type)
            return self._ok(res, "real", f"STK 报告 {report_type} 生成完成。")
        except Exception as e:
            return self._fail(str(e), "real", "报告生成失败")

    def get_script_template(self, **kwargs) -> str:
        """便捷方法：获取 STK 脚本模板文本（回退模式下也可用）。"""
        return self._generate_script_template(**kwargs)


if __name__ == "__main__":
    tool = StkTool()
    print("工具信息:", {k: v for k, v in tool.get_info().items()
          if k != "methods_schema"})

    print("\n--- connect (回退/不可用) ---")
    r = tool.call("connect")
    print("source:", r["source"], "success:", r["success"])
    print("message:", r["message"])

    print("\n--- create_scenario (回退/不可用) ---")
    print(tool.call("create_scenario", name="Test", start="1 Jan 2025",
                    stop="2 Jan 2025"))

    print("\n--- add_satellite (回退/不可用) ---")
    print(tool.call("add_satellite", name="Sat1",
                    state=[6778e3, 0, 0, 0, 7660, 0]))

    print("\n--- compute_access (回退/不可用) ---")
    print(tool.call("compute_access", from_obj="Sat1", to_obj="Facility1"))

    print("\n--- generate_report (回退/不可用) ---")
    print(tool.call("generate_report", report_type="Access Summary"))

    print("\n--- STK 脚本模板（前 30 行）---")
    script = tool.get_script_template()
    print("\n".join(script.split("\n")[:30]))
