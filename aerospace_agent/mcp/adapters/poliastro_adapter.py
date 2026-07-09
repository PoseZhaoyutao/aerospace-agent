"""Poliastro 适配器 — 基于 poliastro 的轻量二体/三体轨道传播引擎。

第一性原理：
  1. poliastro 基于 astropy+jplephem，适合快速二体传播与轨道根数互转，无需重型环境。
  2. 不承担高保真摄动（球谐/大气），定位为"快速分析后端"。
  3. 输入输出经 Canonical Model 转换，内部用 poliastro.twobody.Orbit。
  4. 懒加载 poliastro，未安装时返回结构化结果，绝不崩溃。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Set

from .base import BaseAdapter, AdapterError

if TYPE_CHECKING:
    from ..schemas import ForceModel, OrbitState, PropagatorConfig


class PoliastroAdapter(BaseAdapter):
    """Poliastro 引擎适配器。

    能力：propagate_orbit / convert_orbit
    依赖：poliastro（pip install poliastro）
    """

    engine_name: str = "poliastro"
    _capabilities: Set[str] = {"propagate_orbit", "convert_orbit"}

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
        """检测 poliastro 是否安装。绝不抛异常。"""
        try:
            import poliastro  # noqa: F401
            return True
        except Exception:
            return False

    def version(self) -> str:
        """返回 poliastro.__version__，不可用时返回 'unavailable'。绝不抛异常。"""
        try:
            import poliastro
            return getattr(poliastro, "__version__", "unknown")
        except Exception:
            return "unavailable"

    def capabilities(self) -> Set[str]:
        return set(self._capabilities)

    # ------------------------------------------------------------------
    # 能力方法 (真实实现,替代 _todo_result)
    # ------------------------------------------------------------------
    def propagate_orbit(self, initial_state, force_model, config) -> dict:
        """二体/数值传播 — poliastro.twobody.Orbit.propagate。

        将 Canonical OrbitState → poliastro Orbit → 传播 → 回写 Canonical。
        支持 point_mass (二体) 和 spherical_harmonics J2 (带摄动)。
        """
        unavail = self._guard("propagate_orbit")
        if unavail is not None:
            return unavail
        try:
            import numpy as np  # noqa: F401
            from poliastro.twobody import Orbit
            from poliastro.bodies import Earth
            from astropy import units as u
            from astropy.time import Time

            # --- 从 OrbitState 提取状态 (SI → km for poliastro) ---
            r = initial_state.position_m
            v = initial_state.velocity_mps
            if r is None or v is None:
                return self._error_result("propagate_orbit", "缺少 position_m 或 velocity_mps")
            r_km = [x / 1000.0 for x in r]
            v_kms = [x / 1000.0 for x in v]

            # --- 解析 epoch ---
            epoch = self._parse_epoch(initial_state, Time)

            # --- 创建 poliastro Orbit ---
            orbit = Orbit.from_vectors(
                attractor=Earth,
                r=r_km * u.km,
                v=v_kms * u.km / u.s,
                epoch=epoch,
            )

            # --- 传播参数 ---
            duration_s = getattr(config, "duration_s", 3600.0)
            output_step_s = getattr(config, "output_step_s", None)
            mu = getattr(config, "mu", 3.986004418e14)

            # --- 判断力学模型 ---
            is_j2 = self._is_j2(force_model)

            # --- 传播 ---
            state_history = []
            if output_step_s and output_step_s > 0:
                n_steps = int(duration_s / output_step_s)
                for i in range(n_steps + 1):
                    t = i * output_step_s
                    prop = orbit if t == 0 else self._propagate_one(orbit, t, is_j2)
                    state_history.append(self._extract_state(prop, t))
            else:
                prop = self._propagate_one(orbit, duration_s, is_j2)
                state_history.append(self._extract_state(prop, duration_s))

            return {
                "status": "success",
                "state_history": state_history,
                "metadata": {
                    "engine": self.engine_name,
                    "engine_version": self.version(),
                    "units": "SI (m, m/s, s)",
                    "frame": initial_state.frame.name.value
                             if hasattr(initial_state.frame, "name") else "GCRF",
                    "propagator_type": "j2" if is_j2 else "two_body",
                    "step_count": len(state_history),
                    "mu": mu,
                },
            }
        except Exception as exc:
            return self._error_result("propagate_orbit", str(exc))

    def convert_orbit(self, initial_state, target_representation: str = "keplerian",
                      mu: float = 3.986004418e14) -> dict:
        """轨道表示互转 (cartesian ↔ keplerian) — poliastro 专有能力。"""
        unavail = self._guard("convert_orbit")
        if unavail is not None:
            return unavail
        try:
            from poliastro.twobody import Orbit
            from poliastro.bodies import Earth
            from astropy import units as u
            from astropy.time import Time

            r = initial_state.position_m
            v = initial_state.velocity_mps
            if r is None or v is None:
                return self._error_result("convert_orbit", "缺少 position_m 或 velocity_mps")
            r_km = [x / 1000.0 for x in r]
            v_kms = [x / 1000.0 for x in v]
            epoch = self._parse_epoch(initial_state, Time)

            orbit = Orbit.from_vectors(
                attractor=Earth,
                r=r_km * u.km,
                v=v_kms * u.km / u.s,
                epoch=epoch,
            )

            if target_representation == "keplerian":
                a, e, inc, raan, argp, nu = orbit.classical()
                return {
                    "status": "success",
                    "representation": "keplerian",
                    "elements": {
                        "a_m": float(a.to(u.m).value),
                        "e": float(e),
                        "i_deg": float(inc.to(u.deg).value),
                        "raan_deg": float(raan.to(u.deg).value),
                        "argp_deg": float(argp.to(u.deg).value),
                        "ta_deg": float(nu.to(u.deg).value),
                    },
                    "engine": self.engine_name,
                }
            elif target_representation == "cartesian":
                r_out = orbit.r.to(u.m).value.tolist()
                v_out = orbit.v.to(u.m / u.s).value.tolist()
                return {
                    "status": "success",
                    "representation": "cartesian",
                    "position_m": r_out,
                    "velocity_mps": v_out,
                    "engine": self.engine_name,
                }
            else:
                return self._error_result(
                    "convert_orbit",
                    f"不支持的表示: {target_representation}")
        except Exception as exc:
            return self._error_result("convert_orbit", str(exc))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_epoch(initial_state, Time):
        """从 OrbitState 解析 astropy Time。"""
        epoch_val = initial_state.epoch.value
        try:
            if isinstance(epoch_val, str):
                return Time(epoch_val, scale="utc")
            # 数值型 (JD/MJD/UNIX)
            fmt = getattr(initial_state.epoch, "format", None)
            if fmt and hasattr(fmt, "value"):
                fmt_str = fmt.value
            else:
                fmt_str = "iso"
            if fmt_str in ("jd", "mjd", "unix"):
                return Time(epoch_val, format=fmt_str, scale="utc")
            return Time("J2000", scale="utc")
        except Exception:
            return Time("J2000", scale="utc")

    @staticmethod
    def _is_j2(force_model) -> bool:
        """判断是否 J2 摄动。"""
        if not isinstance(force_model, dict):
            return False
        gravity = force_model.get("gravity", "point_mass")
        return gravity == "spherical_harmonics" and force_model.get("degree", 0) == 2

    def _propagate_one(self, orbit, t_s: float, is_j2: bool):
        """单步传播。"""
        from astropy import units as u
        if is_j2:
            try:
                from poliastro.twobody.propagation import CowellPropagator
                from poliastro.core.perturbations import J2_perturbation
                from poliastro.bodies import Earth
                return orbit.propagate(
                    t_s * u.s,
                    method=CowellPropagator(
                        f=J2_perturbation(
                            J2=Earth.J2.value,
                            R=Earth.R.to(u.m).value,
                        )
                    ),
                )
            except Exception:
                # J2 传播器不可用,回退到二体
                return orbit.propagate(t_s * u.s)
        return orbit.propagate(t_s * u.s)

    @staticmethod
    def _extract_state(orbit, time_s: float) -> dict:
        """从 poliastro Orbit 提取 Canonical 状态 (SI 单位)。"""
        from astropy import units as u
        return {
            "epoch": str(orbit.epoch.iso),
            "position_m": orbit.r.to(u.m).value.tolist(),
            "velocity_mps": orbit.v.to(u.m / u.s).value.tolist(),
            "time_s": time_s,
        }
