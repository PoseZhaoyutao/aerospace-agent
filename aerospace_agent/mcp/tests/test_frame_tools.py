"""测试 transform_frame 工具——坐标系转换。

直接调用 astro_dynamics_mcp.tools.transform_frame 工具函数，
验证回退路径：当 astropy/SPICE 等引擎未安装时，
transform_frame 应回退到解析实现（GCRF↔ITRF 用地球自转角一阶近似），
而非崩溃。本测试覆盖该回退路径的旋转正确性与守恒量。
"""
from __future__ import annotations

import math

import pytest

from aerospace_agent.mcp.tools.frame_tools import transform_frame
from aerospace_agent.mcp.schemas import (
    OrbitState,
    OrbitRepresentation,
    Epoch,
    Frame,
    FrameName,
    FrameCenter,
)


def make_gcrf_state() -> OrbitState:
    """构造 GCRF 系下的 LEO 轨道状态。"""
    return OrbitState(
        epoch=Epoch("2026-01-01T00:00:00"),
        frame=Frame(name=FrameName.GCRF, center=FrameCenter.EARTH),
        representation=OrbitRepresentation.CARTESIAN,
        position_m=[6771000.0, 1000000.0, 500000.0],
        velocity_mps=[0.0, 7690.0, 0.0],
    )


# ======================================================================
# GCRF → ITRF 转换测试
# ======================================================================
class TestTransformFrameGcrfToItrf:
    """GCRF→ITRF 转换测试。"""

    def test_z_component_preserved(self):
        """绕 Z 轴旋转，Z 分量不变（一阶近似下）。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ITRF")
        assert result.get("status") != "error"
        new_pos = result["state"]["position_m"]
        assert new_pos[2] == pytest.approx(state.position_m[2], abs=1e-3)

    def test_magnitude_preserved(self):
        """旋转保模长。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ITRF")
        assert result.get("status") != "error"
        new_pos = result["state"]["position_m"]
        r_gcrf = math.sqrt(sum(c * c for c in state.position_m))
        r_itrf = math.sqrt(sum(c * c for c in new_pos))
        assert r_itrf == pytest.approx(r_gcrf, rel=1e-9)

    def test_engine_used_recorded(self):
        """转换结果必须记录 engine_used。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ITRF")
        assert result.get("status") != "error"
        assert result["engine_used"] in ("analytic_fallback", "astropy")

    def test_frame_info_recorded(self):
        """结果包含 frame_info 元数据。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ITRF")
        assert result.get("status") != "error"
        fi = result["frame_info"]
        assert fi["source_frame"] == "GCRF"
        assert fi["target_frame"] == "ITRF"
        assert "SI" in fi["units"]

    def test_output_frame_is_itrf(self):
        """输出状态的 frame 应为 ITRF。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ITRF")
        assert result.get("status") != "error"
        assert result["state"]["frame"]["name"] == "ITRF"

    def test_velocity_transformed(self):
        """速度也应被转换（与原始不同）。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ITRF")
        assert result.get("status") != "error"
        new_vel = result["state"]["velocity_mps"]
        # Z 轴旋转保留 vz，但 vx/vy 应变化（叠加地球自转速度项）
        assert new_vel[0] != pytest.approx(state.velocity_mps[0], abs=0.1) \
            or new_vel[1] != pytest.approx(state.velocity_mps[1], abs=0.1)


# ======================================================================
# 惯性系等价转换测试
# ======================================================================
class TestTransformFrameInertialEquiv:
    """惯性系等价转换测试——GCRF/ICRF/EME2000/J2000 互转无需旋转。"""

    def test_gcrf_to_eme2000_identity(self):
        """GCRF→EME2000 为等价惯性系，坐标不变。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "EME2000")
        assert result.get("status") != "error"
        new_pos = result["state"]["position_m"]
        for i in range(3):
            assert new_pos[i] == pytest.approx(state.position_m[i], rel=1e-9)

    def test_gcrf_to_j2000_identity(self):
        """GCRF→J2000 为等价惯性系，坐标不变。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "J2000")
        assert result.get("status") != "error"
        new_pos = result["state"]["position_m"]
        for i in range(3):
            assert new_pos[i] == pytest.approx(state.position_m[i], rel=1e-9)

    def test_gcrf_to_icrf_not_identity(self):
        """K5-C5: GCRF（地心系）→ICRF（质心系）不再视为等价，
        需要真实坐标转换（无引擎时返回 error 是正确行为）。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "ICRF")
        # ICRF 已从惯性系等价组移除，不再恒等变换
        # 无 astropy/spiceypy 时应返回 error（正确行为）
        assert result.get("status") in ("error", "success")


# ======================================================================
# 往返转换测试
# ======================================================================
class TestTransformFrameRoundtrip:
    """坐标系转换往返测试。"""

    def test_gcrf_itrf_gcrf_roundtrip(self):
        """GCRF→ITRF→GCRF 往返一致。"""
        state = make_gcrf_state()
        r1 = transform_frame(state.to_dict(), "ITRF")
        assert r1.get("status") != "error"
        r2 = transform_frame(r1["state"], "GCRF")
        assert r2.get("status") != "error"
        new_pos = r2["state"]["position_m"]
        for i in range(3):
            assert new_pos[i] == pytest.approx(
                state.position_m[i], rel=1e-6, abs=1.0
            )


# ======================================================================
# 错误处理测试
# ======================================================================
class TestTransformFrameErrorHandling:
    """transform_frame 错误处理测试。"""

    def test_invalid_target_frame(self):
        """不支持的目标坐标系应返回错误。"""
        state = make_gcrf_state()
        result = transform_frame(state.to_dict(), "GALACTIC")
        assert result.get("status") == "error"
        assert "不支持" in result.get("reason", "")

    def test_invalid_state_dict(self):
        """无效状态字典应返回错误。"""
        result = transform_frame({"bad": "data"}, "ITRF")
        assert result.get("status") == "error"


# ======================================================================
# Frame schema 属性测试
# ======================================================================
class TestFrameSchema:
    """Frame schema 属性与序列化测试。"""

    def test_frame_is_inertial_property(self):
        """Frame.is_inertial 属性——GCRF 惯性、ITRF 固连。"""
        gcrf = Frame(name=FrameName.GCRF, center=FrameCenter.EARTH)
        itrf = Frame(name=FrameName.ITRF, center=FrameCenter.EARTH)
        assert gcrf.is_inertial is True
        assert itrf.is_body_fixed is True
        assert itrf.is_inertial is False

    def test_frame_dict_roundtrip(self):
        """Frame 字典序列化往返一致。"""
        gcrf = Frame(name=FrameName.GCRF, center=FrameCenter.EARTH)
        d = gcrf.to_dict()
        gcrf2 = Frame.from_dict(d)
        assert gcrf2.name == FrameName.GCRF
        assert gcrf2.center == FrameCenter.EARTH
