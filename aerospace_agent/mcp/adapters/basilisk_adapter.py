"""Basilisk 适配器 — 基于 Basilisk 的航天器动力学与姿态仿真引擎。

第一性原理：
  1. Basilisk（BSK）是模块化 6-DOF 仿真器，强项在姿态控制与多体动力学集成。
  2. 仿真通过 SimulationParameters + 任务模块（dynamics/FSW）组装，非单函数调用。
  3. 本适配器封装 BSK.utilities 构建仿真任务，输出回写 Canonical Model。
  4. 懒加载 basilisk，未安装时返回结构化结果，绝不崩溃。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Set

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:
    from ..schemas import AttitudeState, ForceModel, OrbitState, PropagatorConfig


class BasiliskAdapter(BaseAdapter):
    """Basilisk 引擎适配器。

    能力：propagate_orbit / attitude_control
    依赖：basilisk（pip install basilisk）+ Basilisk.utilities 模块
    """

    engine_name: str = "basilisk"
    _capabilities: Set[str] = {"propagate_orbit", "attitude_control"}

    def __init__(self):
        super().__init__()

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
    # 契约实现
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """检测 basilisk 是否安装。绝不抛异常。"""
        try:
            import basilisk  # noqa: F401
            return True
        except Exception:
            return False

    def version(self) -> str:
        """返回 basilisk.__version__，不可用时返回 'unavailable'。绝不抛异常。"""
        try:
            import basilisk
            return getattr(basilisk, "__version__", "unknown")
        except Exception:
            return "unavailable"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法
    # ------------------------------------------------------------------
    def propagate_orbit(self, initial_state, force_model, config) -> dict:
        """轨道传播——Basilisk 仿真任务（spacecraft + dynamics 模块）。"""
        unavail = self._guard("propagate_orbit")
        if unavail is not None:
            return unavail
        try:
            from Basilisk.utilities import SimulationBaseClass, macros  # noqa: F401
            from Basilisk.simulation import spacecraft  # noqa: F401
            # --- 额外依赖 ---
            from Basilisk.utilities import unitTestSupport  # noqa: F401
            from Basilisk.utilities.orbitalMotion import elem2rv  # noqa: F401
            import numpy as np

            # --- 从 OrbitState 提取初值 (SI 单位) ---
            r = initial_state.position_m
            v = initial_state.velocity_mps
            if r is None or v is None:
                return self._error_result(
                    "propagate_orbit", "缺少 position_m 或 velocity_mps")
            r_vec = np.array(r, dtype=float)
            v_vec = np.array(v, dtype=float)

            # --- 传播参数 (mu 不在 PropagatorConfig 中, 用 getattr 兜底) ---
            duration_s = float(getattr(config, "duration_s", 3600.0))
            output_step_s = getattr(config, "output_step_s", None)
            mu = float(getattr(config, "mu", 3.986004418e14))
            step_s = float(getattr(config, "step_s", 60.0))
            if step_s <= 0:
                step_s = 60.0

            # --- 构建 Basilisk 仿真任务 ---
            scSim = SimulationBaseClass.SimBaseClass()
            simTaskName = "scTask"
            simProcess = scSim.CreateNewProcess("dynProcess")
            simTimeStep = macros.sec2nano(step_s)
            simProcess.addTask(scSim.CreateNewTask(simTaskName, simTimeStep))

            # --- 创建 spacecraft 模块并设初值 ---
            scObject = spacecraft.Spacecraft()
            scObject.ModelTag = "bsk_sc"
            scObject.hub.r_CN_NInit = r_vec
            scObject.hub.v_CN_NInit = v_vec
            scObject.hub.mu = mu
            scSim.AddModelToTask(simTaskName, scObject)

            # --- 配置数据记录 (按 output_step_s 采样) ---
            use_sampling = bool(output_step_s and output_step_s > 0)
            if use_sampling:
                sample_nano = macros.sec2nano(float(output_step_s))
                scSim.AddVariableForLogging(
                    "bsk_sc.scStateOutMsg.r_BN_N", sample_nano, 0, 0)
                scSim.AddVariableForLogging(
                    "bsk_sc.scStateOutMsg.v_BN_N", sample_nano, 0, 0)

            # --- 运行仿真 ---
            simTime = duration_s
            simulationTime = macros.sec2nano(simTime)
            scSim.InitializeSimulation()
            scSim.ConfigureStopTime(simulationTime)
            scSim.ExecuteSimulation()

            # --- 提取状态历史 ---
            state_history = []
            if use_sampling:
                try:
                    r_log = scSim.GetLogVariableData(
                        "bsk_sc.scStateOutMsg.r_BN_N")
                    v_log = scSim.GetLogVariableData(
                        "bsk_sc.scStateOutMsg.v_BN_N")
                    n = min(len(r_log), len(v_log))
                    for i in range(n):
                        t_s = float(r_log[i, 0]) * 1e-9  # ns → s
                        state_history.append({
                            "position_m": [float(x) for x in r_log[i, 1:4]],
                            "velocity_mps": [float(x) for x in v_log[i, 1:4]],
                            "time_s": t_s,
                        })
                except Exception:
                    state_history = []

            # --- 终态兜底 (无采样或日志为空) ---
            if not state_history:
                try:
                    final_msg = scObject.scStateOutMsg.read()
                    state_history.append({
                        "position_m": [float(x) for x in final_msg.r_BN_N],
                        "velocity_mps": [float(x) for x in final_msg.v_BN_N],
                        "time_s": duration_s,
                    })
                except Exception as inner:
                    return self._error_result(
                        "propagate_orbit", f"提取终态失败: {inner}")

            return {
                "status": "success",
                "state_history": state_history,
                "metadata": {
                    "engine": "basilisk",
                    "engine_version": self.version(),
                    "propagator_type": "two_body",
                    "mu": mu,
                    "duration_s": duration_s,
                    "step_s": step_s,
                    "output_step_s": output_step_s,
                    "step_count": len(state_history),
                },
            }
        except Exception as exc:
            return self._error_result("propagate_orbit", str(exc))

    def attitude_control(self, initial_state, controller: str = "MRP_feedback",
                         **kwargs) -> dict:
        """姿态控制仿真——Basilisk FSW 模块（MRP 反馈/太阳指向等）。"""
        unavail = self._guard("attitude_control")
        if unavail is not None:
            return unavail
        try:
            from Basilisk.fswAlgorithms import mrpFeedback  # noqa: F401
            from Basilisk.utilities import SimulationBaseClass  # noqa: F401
            # --- 额外依赖 ---
            from Basilisk.simulation import spacecraft  # noqa: F401
            from Basilisk.utilities import macros  # noqa: F401
            import numpy as np

            # --- 从 AttitudeState 提取初值 ---
            quat = getattr(initial_state, "quaternion", None)
            omega = getattr(initial_state, "angular_velocity_radps", None)

            # 四元数 (scalar-first [q0,q1,q2,q3]) → MRP (sigma = v / (1+q0))
            if quat is not None:
                q = np.array(quat, dtype=float)
                denom = 1.0 + q[0]
                if abs(denom) < 1e-12:
                    mrp_init = np.array([0.0, 0.0, 0.0])
                else:
                    mrp_init = np.array([q[1], q[2], q[3]]) / denom
            else:
                mrp_init = np.array([0.0, 0.0, 0.0])

            if omega is not None:
                omega_bn = np.array(omega, dtype=float)
            else:
                omega_bn = np.array([0.0, 0.0, 0.0])

            # --- MRP 反馈控制参数 (默认 K=3.5, P=35.0) ---
            K = float(kwargs.get("K", 3.5))
            P = float(kwargs.get("P", 35.0))
            duration_s = float(kwargs.get("duration_s", 600.0))
            step_s = float(kwargs.get("step_s", 0.1))
            if step_s <= 0:
                step_s = 0.1
            sample_step_s = float(kwargs.get("output_step_s", step_s))

            # --- 构建 Basilisk 仿真任务 ---
            scSim = SimulationBaseClass.SimBaseClass()
            simTaskName = "fswTask"
            simProcess = scSim.CreateNewProcess("fswProcess")
            simTimeStep = macros.sec2nano(step_s)
            simProcess.addTask(scSim.CreateNewTask(simTaskName, simTimeStep))

            # --- spacecraft 模块 (姿态动力学) ---
            scObject = spacecraft.Spacecraft()
            scObject.ModelTag = "bsk_sc"
            scObject.hub.r_CN_NInit = np.array([0.0, 0.0, 0.0])
            scObject.hub.v_CN_NInit = np.array([0.0, 0.0, 0.0])
            scObject.hub.sigma_BNInit = mrp_init
            scObject.hub.omega_BN_BInit = omega_bn
            scObject.hub.mHub = 100.0
            scObject.hub.IHubPntBc_B = np.diag([10.0, 10.0, 10.0])
            scSim.AddModelToTask(simTaskName, scObject)

            # --- mrpFeedback 控制模块 ---
            mrpWrap = mrpFeedback.mrpFeedback()
            mrpWrap.ModelTag = "mrpFeedback"
            mrpWrap.K = K
            mrpWrap.P = P
            if hasattr(mrpWrap, "integralLimit"):
                mrpWrap.integralLimit = 0.1
            if hasattr(mrpWrap, "ki"):
                mrpWrap.ki = 0.0
            scSim.AddModelToTask(simTaskName, mrpWrap)

            # --- 配置姿态数据记录 ---
            sample_nano = macros.sec2nano(sample_step_s)
            scSim.AddVariableForLogging(
                "bsk_sc.scStateOutMsg.sigma_BN_B", sample_nano, 0, 0)

            # --- 运行仿真 ---
            simulationTime = macros.sec2nano(duration_s)
            scSim.InitializeSimulation()
            scSim.ConfigureStopTime(simulationTime)
            scSim.ExecuteSimulation()

            # --- 提取姿态历史 (MRP → 四元数, scalar-first) ---
            attitude_history = []
            try:
                sigma_log = scSim.GetLogVariableData(
                    "bsk_sc.scStateOutMsg.sigma_BN_B")
                for i in range(len(sigma_log)):
                    t_s = float(sigma_log[i, 0]) * 1e-9  # ns → s
                    mrp_i = [float(x) for x in sigma_log[i, 1:4]]
                    s_sq = sum(c * c for c in mrp_i)
                    q0 = (1.0 - s_sq) / (1.0 + s_sq)
                    qv = [2.0 * c / (1.0 + s_sq) for c in mrp_i]
                    attitude_history.append({
                        "mrp": mrp_i,
                        "quaternion": [q0, qv[0], qv[1], qv[2]],
                        "time_s": t_s,
                    })
            except Exception:
                pass

            # --- 终态兜底 ---
            if not attitude_history:
                try:
                    final_msg = scObject.scStateOutMsg.read()
                    mrp_final = [float(x) for x in final_msg.sigma_BN_B]
                    s_sq = sum(c * c for c in mrp_final)
                    q0 = (1.0 - s_sq) / (1.0 + s_sq)
                    qv = [2.0 * c / (1.0 + s_sq) for c in mrp_final]
                    attitude_history.append({
                        "mrp": mrp_final,
                        "quaternion": [q0, qv[0], qv[1], qv[2]],
                        "time_s": duration_s,
                    })
                except Exception as inner:
                    return self._error_result(
                        "attitude_control", f"提取姿态终态失败: {inner}")

            return {
                "status": "success",
                "controller": controller,
                "attitude_history": attitude_history,
                "metadata": {
                    "engine": "basilisk",
                    "engine_version": self.version(),
                    "controller_type": "MRP_feedback",
                    "K": K,
                    "P": P,
                    "duration_s": duration_s,
                    "step_s": step_s,
                    "step_count": len(attitude_history),
                },
            }
        except Exception as exc:
            return self._error_result("attitude_control", str(exc))
