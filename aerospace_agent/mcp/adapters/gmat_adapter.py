"""GMAT 适配器 — 基于 NASA GMAT 的任务分析与轨道传播引擎。

第一性原理：
  1. GMAT 以脚本驱动，通过可执行文件执行 .script 文件，本适配器负责脚本生成与子进程调度。
  2. 不在 Python 顶层 import GMAT——而是通过 GMAT_PATH 定位二进制并 subprocess 执行。
  3. 输入输出始终经 Canonical Model 转换，GMAT 内部坐标系/单位对外不可见。
  4. 不可用（无 GMAT_PATH 或二进制缺失）时返回结构化结果，绝不崩溃。
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

from aerospace_agent.local_runtime import run_command

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:
    from ..schemas import ForceModel, GroundStation, OrbitState, PropagatorConfig


class GMATAdapter(BaseAdapter):
    """GMAT 引擎适配器。

    能力：propagate_orbit / run_script / compute_ground_access
    依赖：GMAT 可执行文件（通过 GMAT_PATH 环境变量定位）
    """

    engine_name: str = "gmat"
    _capabilities: Set[str] = {
        "propagate_orbit", "run_script", "compute_ground_access",
    }

    def __init__(self):
        super().__init__()
        self._gmat_bin: Optional[str] = None

    # ------------------------------------------------------------------
    # 通用辅助
    # ------------------------------------------------------------------
    def _guard(self, operation: str):
        """可用性闸门：调用 _require_available()，不可用时返回 unavailable_result。"""
        try:
            self._require_available()
            return None
        except AdapterError:
            return self.unavailable_result(operation)

    def _error_result(self, operation: str, reason: str) -> dict:
        return {"status": "error", "engine": self.engine_name,
                "operation": operation, "reason": reason}

    def _todo_result(self, operation: str, message: str = "") -> dict:
        return {"status": "todo", "engine": self.engine_name,
                "operation": operation, "message": message}

    # ------------------------------------------------------------------
    # 资源定位
    # ------------------------------------------------------------------
    def _find_gmat_binary(self) -> Optional[str]:
        """定位 GMAT 可执行文件：优先 GMAT_PATH 环境变量。"""
        if self._gmat_bin and Path(self._gmat_bin).exists():
            return self._gmat_bin
        env_path = os.environ.get("GMAT_PATH")
        if not env_path:
            return None
        p = Path(env_path)
        # GMAT_PATH 可指向可执行文件本身或 bin 目录
        candidates = [p] if p.is_file() else []
        if p.is_dir():
            candidates = [p / "bin" / n for n in ("gmat", "gmat.exe", "GmatConsole")]
            candidates += [p / n for n in ("gmat", "gmat.exe", "GmatConsole")]
        for c in candidates:
            if c.exists():
                self._gmat_bin = str(c)
                return self._gmat_bin
        return None

    # ------------------------------------------------------------------
    # 契约实现
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 GMAT 可执行文件是否可定位。绝不抛异常。"""
        try:
            return self._find_gmat_binary() is not None
        except Exception:
            return False

    def version(self) -> str:
        """返回 GMAT 版本字符串，不可用时返回 'unavailable'。绝不抛异常。"""
        bin_path = self._find_gmat_binary()
        if bin_path is None:
            return "unavailable"
        try:
            # 尝试从版本文件读取，回退到目录名
            ver_file = Path(bin_path).parent / "version.txt"
            if ver_file.exists():
                return ver_file.read_text(errors="ignore").strip() or "unknown"
            return Path(bin_path).parent.parent.name
        except Exception:
            return "unknown"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法
    # ------------------------------------------------------------------
    def run_script(self, script_text: str = "", script_path: str = "",
                   workspace: str = "") -> dict:
        """执行 GMAT 脚本：写入工作区并以子进程运行 GMAT。"""
        unavail = self._guard("run_script")
        if unavail is not None:
            return unavail
        bin_path = self._find_gmat_binary()
        try:
            ws = Path(workspace) if workspace else Path(tempfile.gettempdir())
            ws.mkdir(parents=True, exist_ok=True)
            if script_path:
                sp = Path(script_path)
            else:
                sp = ws / "gmat_run.script"
            sp.write_text(script_text, encoding="utf-8")
            proc = run_command(
                [bin_path, "-r", str(sp), "-x"],
                timeout=600,
            )
            return {
                "status": "success" if proc.ok else "failed",
                "engine": self.engine_name, "operation": "run_script",
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "script_path": str(sp),
            }
        except Exception as exc:  # 子进程异常不向上抛
            return self._error_result("run_script", str(exc))

    def propagate_orbit(self, initial_state, force_model, config) -> dict:
        """轨道传播——生成 GMAT 脚本并调用 run_script，解析报告回写 Canonical。"""
        unavail = self._guard("propagate_orbit")
        if unavail is not None:
            return unavail
        try:
            # --- 提取初始状态（兼容 OrbitState 对象与 dict）---
            if isinstance(initial_state, dict):
                pos = initial_state.get("position_m")
                vel = initial_state.get("velocity_mps")
                epoch_obj = initial_state.get("epoch", {})
                epoch_val = (epoch_obj.get("value")
                             if isinstance(epoch_obj, dict)
                             else str(epoch_obj))
                frame_obj = initial_state.get("frame", {})
                frame_name = (frame_obj.get("name", "Earth")
                              if isinstance(frame_obj, dict)
                              else "Earth")
            else:
                pos = initial_state.position_m
                vel = initial_state.velocity_mps
                epoch_val = initial_state.epoch.value
                frame_name = (initial_state.frame.name.value
                              if hasattr(initial_state.frame, "name")
                              else "Earth")

            if pos is None or vel is None:
                return self._error_result(
                    "propagate_orbit",
                    "initial_state 缺少 position_m 或 velocity_mps")

            # --- 解析力学模型（兼容 dict 与 ForceModel 对象）---
            if isinstance(force_model, dict):
                gravity = force_model.get("gravity", "point_mass")
                sh_degree = force_model.get("degree", 0)
                sh_order = force_model.get("order", sh_degree)
            else:
                gravity = getattr(force_model, "gravity", "point_mass")
                sh_degree = getattr(force_model, "degree", 0)
                sh_order = getattr(force_model, "order", sh_degree)

            # --- 解析配置 ---
            duration_s = float(getattr(config, "duration_s", 86400.0))
            out_step_raw = getattr(config, "output_step_s", None)
            step_s = (float(out_step_raw) if out_step_raw
                      else float(getattr(config, "step_s", 60.0)))
            mu = float(getattr(config, "mu", 3.986004418e14))

            # --- 单位换算 SI → GMAT (km, km/s, days) ---
            pos_km = [float(pos[i]) / 1000.0 for i in range(3)]
            vel_kms = [float(vel[i]) / 1000.0 for i in range(3)]
            duration_days = duration_s / 86400.0

            # --- ISO 历元 → GMAT UTCGregorian ---
            gmat_epoch = self._iso_to_gmat_utc(str(epoch_val))

            # --- 力模型行 ---
            force_lines = self._build_force_lines(gravity, sh_degree,
                                                  sh_order)

            # --- 临时工作区与输出文件 ---
            with tempfile.TemporaryDirectory(prefix="gmat_prop_") as ws:
                output_file = str(Path(ws) / "orbit_report.txt")
                script = self._build_prop_script(
                    gmat_epoch, pos_km, vel_kms, force_lines,
                    step_s, duration_days, output_file)
                result = self.run_script(script_text=script, workspace=ws)
                # run_script 非 success 直接透传
                if result.get("status") != "success":
                    return result
                content = ""
                try:
                    content = Path(output_file).read_text(
                        encoding="utf-8", errors="ignore")
                except Exception:
                    pass
                state_history = self._parse_orbit_report(content)

            return {
                "status": "success",
                "state_history": state_history,
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "propagator_type": "numerical",
                    "units": "SI (m, m/s, s)",
                    "frame": frame_name,
                    "step_count": len(state_history),
                    "duration_s": duration_s,
                    "output_step_s": step_s,
                    "mu": mu,
                    "force_model": gravity,
                },
            }
        except Exception as exc:
            return self._error_result("propagate_orbit", f"传播失败: {exc}")

    def compute_ground_access(self, orbit_state, station, start_epoch,
                              stop_epoch, min_elevation_deg: float) -> dict:
        """地面可见性计算——生成 GMAT AER 报告脚本并解析可见窗口。"""
        unavail = self._guard("compute_ground_access")
        if unavail is not None:
            return unavail
        try:
            # --- 提取轨道状态（兼容 OrbitState 对象与 dict）---
            if isinstance(orbit_state, dict):
                pos = orbit_state.get("position_m")
                vel = orbit_state.get("velocity_mps")
                epoch_obj = orbit_state.get("epoch", {})
                epoch_val = (epoch_obj.get("value")
                             if isinstance(epoch_obj, dict)
                             else str(epoch_obj))
            else:
                pos = orbit_state.position_m
                vel = orbit_state.velocity_mps
                epoch_val = orbit_state.epoch.value

            if pos is None or vel is None:
                return self._error_result(
                    "compute_ground_access",
                    "orbit_state 缺少 position_m 或 velocity_mps")

            # --- 提取地面站（兼容 GroundStation 对象与 dict）---
            if isinstance(station, dict):
                st_name = station.get("name", "DefaultStation")
                lat = station.get("latitude_deg", 0.0)
                lon = station.get("longitude_deg", 0.0)
                alt = station.get("altitude_m", 0.0)
            else:
                st_name = getattr(station, "name", "DefaultStation")
                lat = getattr(station, "latitude_deg", 0.0)
                lon = getattr(station, "longitude_deg", 0.0)
                alt = getattr(station, "altitude_m", 0.0)

            # --- 解析起止历元 → datetime ---
            start_dt = self._epoch_to_datetime(start_epoch)
            stop_dt = self._epoch_to_datetime(stop_epoch)
            duration_s = (stop_dt - start_dt).total_seconds()
            if duration_s <= 0:
                return self._error_result(
                    "compute_ground_access",
                    "stop_epoch 必须晚于 start_epoch")

            # --- 单位换算 ---
            pos_km = [float(pos[i]) / 1000.0 for i in range(3)]
            vel_kms = [float(vel[i]) / 1000.0 for i in range(3)]
            duration_days = duration_s / 86400.0
            gmat_epoch = self._iso_to_gmat_utc(str(epoch_val))
            # GMAT 标识符需为字母数字下划线
            safe_name = "".join(
                c if c.isalnum() else "_" for c in str(st_name)
            ) or "Station"
            if safe_name[0].isdigit():
                safe_name = "S_" + safe_name

            # --- 临时工作区与输出文件 ---
            with tempfile.TemporaryDirectory(prefix="gmat_access_") as ws:
                output_file = str(Path(ws) / "aer_report.txt")
                script = self._build_access_script(
                    gmat_epoch, pos_km, vel_kms, safe_name,
                    float(lat), float(lon), float(alt),
                    duration_days, output_file)
                result = self.run_script(script_text=script, workspace=ws)
                # run_script 非 success 直接透传
                if result.get("status") != "success":
                    return result
                content = ""
                try:
                    content = Path(output_file).read_text(
                        encoding="utf-8", errors="ignore")
                except Exception:
                    pass
                windows = self._parse_aer_report(
                    content, float(min_elevation_deg), start_dt)

            return {
                "status": "success",
                "access_windows": windows,
                "total_windows": len(windows),
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "station": st_name,
                    "min_elevation_deg": float(min_elevation_deg),
                    "units": "SI (deg, s)",
                },
            }
        except Exception as exc:
            return self._error_result("compute_ground_access",
                                      f"可见性计算失败: {exc}")

    # ------------------------------------------------------------------
    # 脚本生成 / 报告解析辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _iso_to_gmat_utc(iso_str: str) -> str:
        """ISO '2020-01-01T12:00:00' → GMAT UTCGregorian '01 Jan 2020 12:00:00.000'。"""
        from datetime import datetime
        months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        cleaned = str(iso_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return (f"{dt.day:02d} {months[dt.month - 1]} {dt.year} "
                f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}."
                f"{dt.microsecond // 1000:03d}")

    @staticmethod
    def _epoch_to_datetime(epoch):
        """从 Epoch 对象/dict/字符串解析为 datetime（带 UTC 时区）。"""
        from datetime import datetime, timezone
        if isinstance(epoch, dict):
            val = str(epoch.get("value"))
        elif hasattr(epoch, "value"):
            val = str(epoch.value)
        else:
            val = str(epoch)
        cleaned = val.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(cleaned)
        except Exception:
            dt = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _build_force_lines(gravity: str, sh_degree, sh_order) -> str:
        """根据 Canonical 力学模型生成 GMAT ForceModel 配置行。"""
        lines = ["DefaultFM.CentralBody = Earth;"]
        if gravity == "spherical_harmonics":
            deg = max(int(sh_degree), 2)
            ord_ = max(int(sh_order), 2)
            lines.append(
                f"DefaultFM.SphericalHarmonics.Earth.Degree = {deg};")
            lines.append(
                f"DefaultFM.SphericalHarmonics.Earth.Order = {ord_};")
        else:
            lines.append("DefaultFM.PointMass.Earth = Yes;")
        return "\n".join(lines)

    @staticmethod
    def _build_prop_script(gmat_epoch, pos_km, vel_kms, force_lines,
                           step_s, duration_days, output_file) -> str:
        """生成 GMAT 轨道传播脚本。"""
        lines = [
            "% GMAT Script: Orbit Propagation",
            "Create Spacecraft DefaultSC;",
            "DefaultSC.Epoch.Format = 'UTCGregorian';",
            f"DefaultSC.Epoch.UTCGregorian = '{gmat_epoch}';",
            f"DefaultSC.X = {pos_km[0]};",
            f"DefaultSC.Y = {pos_km[1]};",
            f"DefaultSC.Z = {pos_km[2]};",
            f"DefaultSC.VX = {vel_kms[0]};",
            f"DefaultSC.VY = {vel_kms[1]};",
            f"DefaultSC.VZ = {vel_kms[2]};",
            "DefaultSC.CoordinateSystem = 'EarthMJ2000Eq';",  # K5-M12: GMAT 标准坐标系名
            "",
            "Create ForceModel DefaultFM;",
            force_lines,
            "",
            "Create Propagator DefaultProp;",
            "DefaultProp.FM = DefaultFM;",
            "DefaultProp.Type = 'PrinceDormand78';",
            f"DefaultProp.InitialStepSize = {step_s};",
            f"DefaultProp.MaxStep = {step_s};",
            "",
            "Create ReportFile orbit_report;",
            f"orbit_report.Filename = '{output_file}';",
            ("orbit_report.Add = {DefaultSC.ElapsedDays, DefaultSC.X, "
             "DefaultSC.Y, DefaultSC.Z, DefaultSC.VX, DefaultSC.VY, "
             "DefaultSC.VZ};"),
            "",
            "BeginMissionSequence",
            (f"Propagate DefaultProp(DefaultSC) "
             f"{{DefaultSC.ElapsedDays = {duration_days}}};"),
            ("Report orbit_report DefaultSC.ElapsedDays DefaultSC.X "
             "DefaultSC.Y DefaultSC.Z DefaultSC.VX DefaultSC.VY "
             "DefaultSC.VZ;"),
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_access_script(gmat_epoch, pos_km, vel_kms, station_name,
                             lat, lon, alt, duration_days,
                             output_file) -> str:
        """生成 GMAT 地面站可见性（AER）脚本。"""
        lines = [
            "% GMAT Script: Ground Station Access",
            "Create Spacecraft DefaultSC;",
            "DefaultSC.Epoch.Format = 'UTCGregorian';",
            f"DefaultSC.Epoch.UTCGregorian = '{gmat_epoch}';",
            f"DefaultSC.X = {pos_km[0]};",
            f"DefaultSC.Y = {pos_km[1]};",
            f"DefaultSC.Z = {pos_km[2]};",
            f"DefaultSC.VX = {vel_kms[0]};",
            f"DefaultSC.VY = {vel_kms[1]};",
            f"DefaultSC.VZ = {vel_kms[2]};",
            "DefaultSC.CoordinateSystem = 'EarthMJ2000Eq';",  # K5-M12: GMAT 标准坐标系名
            "",
            f"Create GroundStation {station_name};",
            f"{station_name}.HorizonReference = 'Sphere';",
            f"{station_name}.Latitude = {lat};",
            f"{station_name}.Longitude = {lon};",
            f"{station_name}.Altitude = {alt};",
            "",
            "Create ForceModel DefaultFM;",
            "DefaultFM.CentralBody = Earth;",
            "DefaultFM.PointMass.Earth = Yes;",
            "",
            "Create Propagator DefaultProp;",
            "DefaultProp.FM = DefaultFM;",
            "DefaultProp.Type = 'PrinceDormand78';",
            "DefaultProp.InitialStepSize = 60;",
            "DefaultProp.MaxStep = 60;",
            "",
            "Create ReportFile aer_report;",
            f"aer_report.Filename = '{output_file}';",
            "",
            "BeginMissionSequence",
            (f"Propagate DefaultProp(DefaultSC) "
             f"{{DefaultSC.ElapsedDays = {duration_days}}};"),
            (f"Report aer_report DefaultSC.ElapsedDays {station_name}.Azimuth "
             f"{station_name}.Elevation {station_name}.Range;"),
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_orbit_report(content: str) -> list:
        """解析 GMAT 轨道报告 CSV（ElapsedDays,X,Y,Z,VX,VY,VZ）→ Canonical state_history。

        单位换算 GMAT (km, km/s, days) → SI (m, m/s, s)。
        """
        history = []
        if not content:
            return history
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            # 跳过表头（含非数字 token）
            try:
                vals = [float(parts[i]) for i in range(7)]
            except ValueError:
                continue
            elapsed_days, x, y, z, vx, vy, vz = vals
            history.append({
                "position_m": [x * 1000.0, y * 1000.0, z * 1000.0],
                "velocity_mps": [vx * 1000.0, vy * 1000.0, vz * 1000.0],
                "time_s": elapsed_days * 86400.0,
            })
        return history

    @staticmethod
    def _parse_aer_report(content: str, min_el: float, base_dt) -> list:
        """解析 GMAT AER 报告（ElapsedDays,Azimuth,Elevation,Range）→ 可见窗口列表。

        仰角 >= min_el 的连续时段聚合为一个窗口，时间基于 base_dt 推算。
        """
        from datetime import timedelta
        samples = []  # (elapsed_days, elevation_deg)
        if content:
            for raw in content.splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    elapsed_days = float(parts[0])
                    # 列序: ElapsedDays, Azimuth, Elevation, Range
                    elevation = float(parts[2])
                except (ValueError, IndexError):
                    continue
                samples.append((elapsed_days, elevation))

        if not samples:
            return []

        windows = []
        in_window = False
        win_start_days = None
        max_el = -90.0

        for elapsed_days, el in samples:
            visible = el >= min_el
            if visible and not in_window:
                in_window = True
                win_start_days = elapsed_days
                max_el = el
            elif visible and in_window:
                max_el = max(max_el, el)
            elif not visible and in_window:
                in_window = False
                windows.append((win_start_days, elapsed_days, max_el))
                max_el = -90.0
        if in_window:
            windows.append((win_start_days, samples[-1][0], max_el))

        result = []
        for start_days, stop_days, max_elev in windows:
            start_dt = base_dt + timedelta(seconds=start_days * 86400.0)
            stop_dt = base_dt + timedelta(seconds=stop_days * 86400.0)
            duration = (stop_dt - start_dt).total_seconds()
            result.append({
                "start": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "stop": stop_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "max_elevation_deg": round(max_elev, 3),
                "duration_s": round(duration, 1),
            })
        return result
