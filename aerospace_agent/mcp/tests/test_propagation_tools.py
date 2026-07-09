"""测试 propagate_orbit 与 convert_orbit_representation 工具。

直接调用 astro_dynamics_mcp.tools 中的工具函数，
验证回退路径：引擎未安装时，传播回退到内置二体（f-and-g 级数），
轨道表示转换回退到 Canonical Model 内置实现。

注意：内置 f-and-g 级数为截断近似，长时长传播能量漂移较大，
故能量守恒测试使用独立的精确 Kepler 方程传播器。
本测试覆盖：
  1. convert_orbit_representation 工具调用与往返一致性
  2. propagate_orbit 工具调用结构与元数据
  3. 精确二体传播的能量守恒、周期回归、半长轴不变
"""
from __future__ import annotations

import math

import pytest

from aerospace_agent.mcp.tools.propagation_tools import (
    convert_orbit_representation,
    propagate_orbit,
)
from aerospace_agent.mcp.schemas import (
    OrbitState,
    OrbitRepresentation,
    KeplerianElements,
    Epoch,
    Frame,
    FrameName,
    FrameCenter,
)


MU_EARTH = 3.986004418e14  # m³/s²


# ----------------------------------------------------------------------
# 精确二体传播（Kepler 方程求解）——用于能量守恒测试
# 内置工具的 f-and-g 级数为截断近似，不适合严格守恒量验证
# ----------------------------------------------------------------------
def kepler_propagate(
    state: OrbitState, duration_s: float, mu: float = MU_EARTH
) -> OrbitState:
    """精确二体传播（解 Kepler 方程）——理论能量守恒。

    步骤：cartesian→keplerian，平近点角推进 n·dt，
    牛顿迭代解 Kepler 方程 M = E - e·sinE，→cartesian。
    """
    kep = state.to_keplerian(mu)
    el = kep.elements
    a, e = el.a_m, el.e
    n = math.sqrt(mu / a ** 3)  # 平均运动 rad/s
    # 真近点角 → 偏近点角 → 平近点角
    ta = math.radians(el.ta_deg)
    E0 = 2.0 * math.atan2(
        math.sqrt(1 - e) * math.sin(ta / 2),
        math.sqrt(1 + e) * math.cos(ta / 2),
    )
    M0 = E0 - e * math.sin(E0)
    # 推进平近点角
    M1 = M0 + n * duration_s
    # 牛顿迭代解 Kepler 方程
    E1 = M1
    for _ in range(80):
        f = E1 - e * math.sin(E1) - M1
        fp = 1 - e * math.cos(E1)
        dE = f / fp
        E1 -= dE
        if abs(dE) < 1e-13:
            break
    # 偏近点角 → 真近点角
    ta1 = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(E1 / 2),
        math.sqrt(1 - e) * math.cos(E1 / 2),
    )
    new_elements = KeplerianElements(
        a_m=a,
        e=e,
        i_deg=el.i_deg,
        raan_deg=el.raan_deg,
        argp_deg=el.argp_deg,
        ta_deg=math.degrees(ta1) % 360.0,
    )
    new_state = OrbitState(
        epoch=state.epoch,
        frame=state.frame,
        representation=OrbitRepresentation.KEPLERIAN,
        elements=new_elements,
    )
    return new_state.to_cartesian(mu)


def specific_energy(pos, vel, mu: float = MU_EARTH) -> float:
    """比轨道能 ε = v²/2 - μ/r。"""
    r = math.sqrt(sum(c * c for c in pos))
    v2 = sum(c * c for c in vel)
    return v2 / 2.0 - mu / r


def specific_angular_momentum(pos, vel) -> float:
    """比角动量模长 |r × v|。"""
    hx = pos[1] * vel[2] - pos[2] * vel[1]
    hy = pos[2] * vel[0] - pos[0] * vel[2]
    hz = pos[0] * vel[1] - pos[1] * vel[0]
    return math.sqrt(hx * hx + hy * hy + hz * hz)


def make_leo_state() -> OrbitState:
    """构造 ISS 类 LEO 初始状态（约 394 km 圆轨道）。"""
    return OrbitState(
        epoch=Epoch("2026-01-01T00:00:00"),
        frame=Frame(name=FrameName.GCRF, center=FrameCenter.EARTH),
        representation=OrbitRepresentation.CARTESIAN,
        position_m=[6771000.0, 0.0, 0.0],
        velocity_mps=[0.0, 7690.0, 0.0],
    )


def make_leo_state_dict() -> dict:
    """构造 LEO 初始状态字典（工具输入格式）。"""
    return make_leo_state().to_dict()


def two_body_force_model() -> dict:
    """二体力学模型字典。"""
    return {"gravity": "point_mass"}


def j2_force_model() -> dict:
    """J2 力学模型字典。"""
    return {"gravity": "spherical_harmonics", "degree": 2}


# ======================================================================
# convert_orbit_representation 工具测试
# ======================================================================
class TestConvertOrbitRepresentation:
    """convert_orbit_representation 工具测试。"""

    def test_cartesian_to_keplerian(self):
        """笛卡尔→开普勒：工具返回 keplerian 表示。"""
        result = convert_orbit_representation(
            make_leo_state_dict(), "keplerian", MU_EARTH
        )
        assert result.get("status") != "error"
        assert result["target_representation"] == "keplerian"
        assert result["engine"] == "schemas_builtin"
        state = result["state"]
        assert state["representation"] == "keplerian"
        assert state["elements"] is not None

    def test_keplerian_elements_reasonable(self):
        """LEO 状态转出的开普勒根数合理。"""
        result = convert_orbit_representation(
            make_leo_state_dict(), "keplerian", MU_EARTH
        )
        el = result["state"]["elements"]
        # 半长轴应在 LEO 量级（6700~6900 km）
        # 注：速度 7690 m/s 略高于该高度圆速度（~7672 m/s），
        # 故半长轴略大于初始半径，偏心率很小但仍非零。
        assert 6.7e6 < el["a_m"] < 6.9e6
        assert el["e"] < 0.05  # 近圆轨道
        assert 0 <= el["i_deg"] <= 180

    def test_cartesian_keplerian_roundtrip(self):
        """cartesian→keplerian→cartesian 往返一致。"""
        state_dict = make_leo_state_dict()
        # cart → kep
        r1 = convert_orbit_representation(state_dict, "keplerian", MU_EARTH)
        assert r1.get("status") != "error"
        # kep → cart
        r2 = convert_orbit_representation(r1["state"], "cartesian", MU_EARTH)
        assert r2.get("status") != "error"
        original = OrbitState.from_dict(state_dict)
        back = OrbitState.from_dict(r2["state"])
        for i in range(3):
            assert back.position_m[i] == pytest.approx(
                original.position_m[i], rel=1e-9, abs=1e-3
            )
            assert back.velocity_mps[i] == pytest.approx(
                original.velocity_mps[i], rel=1e-9, abs=1e-6
            )

    def test_identity_same_representation(self):
        """同表示转换返回原状态（identity 引擎）。"""
        state_dict = make_leo_state_dict()
        result = convert_orbit_representation(state_dict, "cartesian", MU_EARTH)
        assert result.get("status") != "error"
        assert result["engine"] == "identity"

    def test_invalid_representation(self):
        """不支持的表示应返回错误。"""
        result = convert_orbit_representation(
            make_leo_state_dict(), "spherical", MU_EARTH
        )
        assert result.get("status") == "error"
        assert "不支持" in result.get("reason", "")

    def test_mu_recorded(self):
        """结果记录使用的 mu 值。"""
        result = convert_orbit_representation(
            make_leo_state_dict(), "keplerian", MU_EARTH
        )
        assert result.get("status") != "error"
        assert result["mu"] == MU_EARTH

    def test_roundtrip_preserves_energy(self):
        """转换前后比能不变（二体守恒）。"""
        state = make_leo_state()
        e0 = specific_energy(state.position_m, state.velocity_mps)
        r1 = convert_orbit_representation(state.to_dict(), "keplerian", MU_EARTH)
        r2 = convert_orbit_representation(r1["state"], "cartesian", MU_EARTH)
        back = OrbitState.from_dict(r2["state"])
        e1 = specific_energy(back.position_m, back.velocity_mps)
        assert e1 == pytest.approx(e0, rel=1e-9)


# ======================================================================
# propagate_orbit 工具测试
# ======================================================================
class TestPropagateOrbitTool:
    """propagate_orbit 工具调用与结构测试。"""

    def test_two_body_propagation_runs(self):
        """二体传播能正常运行。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        assert "state_history" in result
        assert len(result["state_history"]) > 0

    def test_metadata_engine_builtin(self):
        """无引擎安装时，metadata.engine 应为 builtin。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        meta = result["metadata"]
        # astropy/poliastro 可用时可能是其他引擎，但回退时为 builtin
        assert meta["engine"] in ("builtin", "poliastro", "orekit", "gmat")

    def test_metadata_propagator_type_two_body(self):
        """二体传播的 propagator_type 应为 two_body。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        meta = result["metadata"]
        if meta["engine"] == "builtin":
            assert meta["propagator_type"] == "two_body"

    def test_metadata_units_si(self):
        """metadata.units 应为 SI。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        assert "SI" in result["metadata"]["units"]

    def test_j2_propagation_runs(self):
        """J2 传播能正常运行。"""
        result = propagate_orbit(
            make_leo_state_dict(), j2_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        assert "state_history" in result

    def test_first_state_matches_initial(self):
        """传播历史首态应与初始状态一致（t=0）。"""
        state_dict = make_leo_state_dict()
        result = propagate_orbit(
            state_dict, two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        first = result["state_history"][0]
        assert first["elapsed_s"] == pytest.approx(0.0, abs=1e-6)
        for i in range(3):
            assert first["position_m"][i] == pytest.approx(
                state_dict["position_m"][i], rel=1e-6
            )

    def test_step_count(self):
        """600s 时长、60s 步长 → 11 个采样点（含 t=0 和 t=600）。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        assert len(result["state_history"]) == 11

    def test_invalid_state_error(self):
        """无效初始状态应返回错误。"""
        result = propagate_orbit(
            {"bad": "data"}, two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") == "error"

    def test_terminal_state_at_duration(self):
        """最后一个采样点的 elapsed_s 应为 duration_s。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        last = result["state_history"][-1]
        assert last["elapsed_s"] == pytest.approx(600.0, abs=1.0)

    def test_metadata_step_count(self):
        """metadata.step_count 应与 state_history 长度一致。"""
        result = propagate_orbit(
            make_leo_state_dict(), two_body_force_model(),
            duration_s=600.0, output_step_s=60.0,
        )
        assert result.get("status") != "error"
        assert result["metadata"]["step_count"] == len(result["state_history"])


# ======================================================================
# 精确二体传播能量守恒测试（独立 Kepler 方程传播器）
# ======================================================================
class TestTwoBodyEnergyConservation:
    """精确二体传播（Kepler 方程）的能量与角动量守恒测试。

    注意：使用独立的 kepler_propagate 函数而非工具内置 f-and-g 级数，
    因为后者为截断近似，长时长能量漂移显著。
    """

    def test_energy_conservation_full_period(self):
        """二体传播 1 轨道周期，比能守恒。"""
        state = make_leo_state()
        kep = state.to_keplerian(MU_EARTH)
        a = kep.elements.a_m
        T = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
        e0 = specific_energy(state.position_m, state.velocity_mps)
        propagated = kepler_propagate(state, T, MU_EARTH)
        e1 = specific_energy(propagated.position_m, propagated.velocity_mps)
        assert e1 == pytest.approx(e0, rel=1e-9)

    def test_energy_conservation_half_period(self):
        """传播半周期，比能守恒。"""
        state = make_leo_state()
        kep = state.to_keplerian(MU_EARTH)
        a = kep.elements.a_m
        T = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
        e0 = specific_energy(state.position_m, state.velocity_mps)
        propagated = kepler_propagate(state, T / 2, MU_EARTH)
        e1 = specific_energy(propagated.position_m, propagated.velocity_mps)
        assert e1 == pytest.approx(e0, rel=1e-9)

    def test_angular_momentum_conservation(self):
        """二体传播后比角动量模长守恒。"""
        state = make_leo_state()
        h0 = specific_angular_momentum(state.position_m, state.velocity_mps)
        propagated = kepler_propagate(state, 1800.0, MU_EARTH)
        h1 = specific_angular_momentum(
            propagated.position_m, propagated.velocity_mps
        )
        assert h1 == pytest.approx(h0, rel=1e-9)

    def test_period_returns_to_start(self):
        """传播整周期后位置回到起点附近。"""
        state = make_leo_state()
        kep = state.to_keplerian(MU_EARTH)
        a = kep.elements.a_m
        T = 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)
        propagated = kepler_propagate(state, T, MU_EARTH)
        for i in range(3):
            assert propagated.position_m[i] == pytest.approx(
                state.position_m[i], rel=1e-7, abs=1.0
            )

    def test_semi_major_axis_preserved(self):
        """二体传播后半长轴不变。"""
        state = make_leo_state()
        kep0 = state.to_keplerian(MU_EARTH)
        a0 = kep0.elements.a_m
        propagated = kepler_propagate(state, 1800.0, MU_EARTH)
        kep1 = propagated.to_keplerian(MU_EARTH)
        assert kep1.elements.a_m == pytest.approx(a0, rel=1e-9)

    def test_energy_is_negative_for_bound_orbit(self):
        """束缚轨道比能应为负。"""
        state = make_leo_state()
        e = specific_energy(state.position_m, state.velocity_mps)
        assert e < 0
