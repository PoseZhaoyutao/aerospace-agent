"""GMAT 接口工具 —— 任务设计与优化。

GMAT (General Mission Analysis Tool) 是 NASA 开源的独立桌面应用，
并非 Python 包，因此本工具的可用性检测通过以下方式：
    1. 环境变量 ``GMAT_PATH``
    2. 常见安装路径（Windows / Linux / macOS）

方法：
    - run_script(script_path)：执行 GMAT 脚本（真实模式调用 GMAT 二进制）
    - optimize(mission_params, objective)：任务参数优化（仅真实模式）
    - generate_script(mission_spec)：从规范生成 GMAT 脚本文本（真实/回退均可用）

回退策略：
    - generate_script 始终可用（生成脚本文本，不依赖 GMAT 运行时）
    - run_script / optimize 在 GMAT 不可用时返回 source='unavailable'
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .base import BaseTool


# GMAT 可执行文件常见名
_GMAT_BIN_NAMES = ["gmat", "GmatR2022a", "GmatConsole", "gmat-console"]
# 常见安装路径（按平台）
_GMAT_SEARCH_PATHS = [
    "/opt/gmat",
    "/usr/local/gmat",
    "/Applications/GMAT",
    "C:/Program Files/NASA/GMAT",
    "C:/Program Files (x86)/NASA/GMAT",
    os.path.expanduser("~/gmat"),
]


class GmatTool(BaseTool):
    """GMAT 任务设计与优化工具。"""

    name = "gmat"
    description = "GMAT 任务设计、脚本生成与优化（独立应用，非 Python 包）"
    library_name = "gmat"  # 仅用于元信息；实际检测见 _check_available

    methods_schema = {
        "run_script": {
            "params": {"script_path": "str"},
            "returns": "dict",
            "description": "执行 GMAT 脚本文件",
        },
        "optimize": {
            "params": {"mission_params": "dict", "objective": "str"},
            "returns": "dict",
            "description": "优化任务参数（需 GMAT 运行时）",
        },
        "generate_script": {
            "params": {"mission_spec": "dict"},
            "returns": "str",
            "description": "从任务规范生成 GMAT 脚本文本",
        },
    }

    def __init__(self) -> None:
        self._gmat_path: Optional[str] = None

    # ------------------------------------------------------------------
    # 可用性检测（覆写：GMAT 不是 Python 包）
    # ------------------------------------------------------------------
    def _check_available(self) -> bool:
        cache_key = f"gmat_runtime:{self.library_name}"
        if cache_key in BaseTool._availability_cache:
            return BaseTool._availability_cache[cache_key]

        available = False
        gmat_path = None

        # 1. 环境变量 GMAT_PATH
        env_path = os.environ.get("GMAT_PATH")
        if env_path and os.path.exists(env_path):
            gmat_path = env_path
            available = True

        # 2. 常见安装路径
        if not available:
            for base in _GMAT_SEARCH_PATHS:
                if os.path.isdir(base):
                    # 在路径下查找可执行文件
                    for bin_name in _GMAT_BIN_NAMES:
                        candidate = os.path.join(base, "bin", bin_name)
                        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                            gmat_path = candidate
                            available = True
                            break
                        candidate2 = os.path.join(base, bin_name)
                        if os.path.isfile(candidate2) and os.access(candidate2, os.X_OK):
                            gmat_path = candidate2
                            available = True
                            break
                    if available:
                        break

        # 3. PATH 中查找
        if not available:
            for bin_name in _GMAT_BIN_NAMES:
                found = shutil.which(bin_name)
                if found:
                    gmat_path = found
                    available = True
                    break

        self._gmat_path = gmat_path
        BaseTool._availability_cache[cache_key] = available
        return available

    # ------------------------------------------------------------------
    # 真实模式实现
    # ------------------------------------------------------------------
    def _run_script_real(self, script_path: str) -> dict:
        """真实模式：调用 GMAT 二进制执行脚本。"""
        if not os.path.isfile(script_path):
            return {"ok": False, "error": f"脚本文件不存在: {script_path}"}
        if not self._gmat_path:
            return {"ok": False, "error": "未定位到 GMAT 可执行文件"}

        try:
            # GMAT 命令行批处理模式：-b 运行脚本后退出
            cmd = [self._gmat_path, "-b", "-r", script_path]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _optimize_real(
        self, mission_params: Dict[str, Any], objective: str
    ) -> dict:
        """真实模式：构造优化脚本并运行 GMAT 优化器。"""
        spec = {
            "spacecraft": {
                "state": mission_params.get("initial_state", [7100, 0, 0, 0, 7.5, 0]),
            },
            "objective": objective,
            "optimization_vars": mission_params.get("variables", {}),
        }
        script = self.generate_script_text(spec, mode="optimize")
        # 写入临时脚本并运行
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".script", delete=False, dir="/data/user/work"
        )
        try:
            tmp.write(script)
            tmp.close()
            run_res = self._run_script_real(tmp.name)
            return {"script": script, "run": run_res}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 脚本生成（真实/回退均可用）
    # ------------------------------------------------------------------
    def generate_script_text(
        self, mission_spec: Dict[str, Any], mode: str = "propagate"
    ) -> str:
        """从任务规范生成 GMAT 脚本文本。

        Parameters
        ----------
        mission_spec : dict
            任务规范，可包含:
                - spacecraft.state: [x,y,z,vx,vy,vz] (km, km/s)
                - spacecraft.epoch: UTC 字符串
                - duration: 传播时长（天）
                - objective: 优化目标（mode='optimize' 时）
                - optimization_vars: 优化变量
        mode : str
            'propagate' 或 'optimize'。
        """
        sc = mission_spec.get("spacecraft", {})
        state = sc.get("state", [7100.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        epoch = sc.get("epoch", "01 Jan 2000 12:00:00.000")
        duration = mission_spec.get("duration", 1.0)  # 天

        lines: List[str] = []
        lines.append("% GMAT 脚本（由 aerospace_agent.mcp_tools.GmatTool 生成）")
        lines.append("% 模式: " + mode)
        lines.append("")

        # 航天器
        lines.append("%----------------------------------------")
        lines.append("%---------- Spacecraft")
        lines.append("%----------------------------------------")
        lines.append("Create Spacecraft(DefaultSC);")
        lines.append("DefaultSC.DateFormat = UTCGregorian;")
        lines.append(f"DefaultSC.Epoch = '{epoch}';")
        lines.append("DefaultSC.CoordinateSystem = EarthEarthJ2K;")
        lines.append("DefaultSC.DisplayStateType = Cartesian;")
        lines.append(f"DefaultSC.X = {float(state[0]):.6f};")
        lines.append(f"DefaultSC.Y = {float(state[1]):.6f};")
        lines.append(f"DefaultSC.Z = {float(state[2]):.6f};")
        lines.append(f"DefaultSC.VX = {float(state[3]):.6f};")
        lines.append(f"DefaultSC.VY = {float(state[4]):.6f};")
        lines.append(f"DefaultSC.VZ = {float(state[5]):.6f};")
        lines.append("")

        # 力模型
        lines.append("%----------------------------------------")
        lines.append("%---------- ForceModel")
        lines.append("%----------------------------------------")
        lines.append("Create ForceModel(DefaultFM);")
        lines.append("DefaultFM.CentralBody = Earth;")
        lines.append("DefaultFM.PointMasses = {Earth};")
        lines.append("DefaultFM.Drag = None;")
        lines.append("DefaultFM.SRP = Off;")
        lines.append("")

        # 传播器
        lines.append("%----------------------------------------")
        lines.append("%---------- Propagators")
        lines.append("%----------------------------------------")
        lines.append("Create Propagator(DefaultProp);")
        lines.append("DefaultProp.FM = DefaultFM;")
        lines.append("DefaultProp.Type = RungeKutta89;")
        lines.append("DefaultProp.InitialStepSize = 60;")
        lines.append("")

        # 优化（仅 optimize 模式）
        if mode == "optimize":
            lines.append("%----------------------------------------")
            lines.append("%---------- Optimizers")
            lines.append("%----------------------------------------")
            lines.append("Create VF13ad(DefaultOpt);")
            lines.append("DefaultOpt.ShowStatus = true;")
            lines.append("DefaultOpt.ReportStyle = Normal;")
            lines.append("DefaultOpt.ReportFile = 'GmatOptReport.txt';")
            lines.append("DefaultOpt.MaximumIterations = 100;")
            lines.append("")
            objective = mission_spec.get("objective", "")
            for var_name, bounds in mission_spec.get(
                "optimization_vars", {}
            ).items():
                lo, hi = bounds if isinstance(bounds, (list, tuple)) else (None, None)
                lines.append(f"DefaultOpt.AddVariable {var_name} {lo} {hi};")
            if objective:
                lines.append(f"% 目标: {objective}")
            lines.append("")

        # 报告
        lines.append("%----------------------------------------")
        lines.append("%---------- Reports")
        lines.append("%----------------------------------------")
        lines.append("Create ReportFile(DefaultRF);")
        lines.append("DefaultRF.Filename = 'GmatReport.txt';")
        lines.append("DefaultRF.Precision = 16;")
        lines.append(
            "DefaultRF.Add = {DefaultSC.UTCJulian, DefaultSC.X, DefaultSC.Y, "
            "DefaultSC.Z, DefaultSC.VX, DefaultSC.VY, DefaultSC.VZ};"
        )
        lines.append("")

        # 任务序列
        lines.append("%----------------------------------------")
        lines.append("%---------- Mission Sequence")
        lines.append("%----------------------------------------")
        lines.append("BeginMissionSequence;")
        if mode == "optimize":
            lines.append("Optimize DefaultOpt(DefaultRF) {objective};")
        lines.append(
            f"Propagate 'Propagate' DefaultProp(DefaultSC) "
            f"{{DefaultSC.ElapsedDays = {float(duration)}}};"
        )
        lines.append("Report DefaultRF DefaultSC.UTCJulian DefaultSC.X "
                     "DefaultSC.Y DefaultSC.Z;")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def call(self, method: str, **kwargs) -> dict:
        if method == "generate_script":
            return self._call_generate_script(**kwargs)
        if method == "run_script":
            return self._call_run_script(**kwargs)
        if method == "optimize":
            return self._call_optimize(**kwargs)
        return self._fail(f"未知方法: {method}", self.source,
                          f"可用方法: {self.list_methods()}")

    def _call_generate_script(self, mission_spec: Dict[str, Any]) -> dict:
        """生成脚本：真实/回退均可用（纯文本生成，不依赖 GMAT 运行时）。"""
        try:
            mode = mission_spec.get("mode", "propagate")
            script = self.generate_script_text(mission_spec, mode=mode)
            source = "real" if self.is_available else "fallback"
            msg = ("GMAT 脚本生成完成。" +
                   ("已检测到 GMAT 运行时，可直接 run_script。"
                    if self.is_available else
                    "未检测到 GMAT 运行时；脚本可保存后用 GMAT 应用执行。"))
            return self._ok({"script": script, "mode": mode}, source, msg)
        except Exception as e:
            return self._fail(str(e), "fallback", "脚本生成失败")

    def _call_run_script(self, script_path: str) -> dict:
        if not self.is_available:
            return self._unavailable(
                "run_script", "GMAT",
                install_hint=(
                    "请安装 GMAT (https://gmat.gsfc.nasa.gov/) 并设置环境变量 "
                    "GMAT_PATH 指向安装目录，或将 gmat 加入 PATH。"
                ),
            )
        res = self._run_script_real(script_path)
        ok = res.get("ok", False)
        if ok:
            return self._ok(res, "real", "GMAT 脚本执行完成。")
        return self._fail(res.get("error", "未知错误"), "real", "GMAT 执行失败")

    def _call_optimize(
        self, mission_params: Dict[str, Any], objective: str
    ) -> dict:
        if not self.is_available:
            return self._unavailable(
                "optimize", "GMAT",
                install_hint=(
                    "任务优化需要 GMAT 运行时。可先用 generate_script 生成优化脚本，"
                    "再在 GMAT 中手动运行。"
                ),
            )
        try:
            mission_params.setdefault("objective", objective)
            res = self._optimize_real(mission_params, objective)
            if res.get("run", {}).get("ok", False):
                return self._ok(res, "real", "GMAT 优化完成。")
            return self._fail(
                res.get("run", {}).get("error", "未知错误"),
                "real", "GMAT 优化执行失败",
            )
        except Exception as e:
            return self._fail(str(e), "real", "GMAT 优化异常")


if __name__ == "__main__":
    tool = GmatTool()
    print("工具信息:", {k: v for k, v in tool.get_info().items()
          if k != "methods_schema"})
    print("GMAT 路径:", tool._gmat_path)
    print("可用:", tool.is_available)

    print("\n--- generate_script (回退) ---")
    spec = {
        "spacecraft": {"state": [6778, 0, 0, 0, 7.66, 0],
                       "epoch": "01 Jan 2000 12:00:00.000"},
        "duration": 2.0,
    }
    r = tool.call("generate_script", mission_spec=spec)
    print("source:", r["source"])
    print(r["result"]["script"])

    print("\n--- run_script (回退/不可用) ---")
    print(tool.call("run_script", script_path="/tmp/nonexist.script"))

    print("\n--- optimize (回退/不可用) ---")
    print(tool.call("optimize", mission_params={"variables": {"DefaultSC.VY": [7.0, 8.0]}},
                    objective="minimize perigee"))
