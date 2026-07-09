"""测试 convert_time 工具——时间尺度与格式转换。

直接调用 astro_dynamics_mcp.tools.convert_time 工具函数，
验证回退路径：当 astropy/SPICE 等引擎未安装时，
convert_time 应回退到 Python 内置 datetime（仅 UTC），
而非崩溃。本测试覆盖该回退路径的精度与一致性，
以及 astropy 可用时的完整跨尺度转换。
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aerospace_agent.mcp.tools.time_tools import convert_time
from aerospace_agent.mcp.schemas import Epoch, TimeScale, TimeFormat


# ======================================================================
# 基本格式转换测试
# ======================================================================
class TestConvertTimeBasic:
    """convert_time 基本 UTC 格式转换测试。"""

    def test_iso_to_jd_known_epoch(self):
        """已知历元 ISO→JD：2026-01-01T00:00:00 → 2461041.5。"""
        result = convert_time(
            "2026-01-01T00:00:00",
            from_scale="UTC", from_format="ISO",
            to_scale="UTC", to_format="JD",
        )
        assert result.get("status") != "error"
        jd = result["output"]["value"]
        assert jd == pytest.approx(2461041.5, abs=1e-3)

    def test_iso_to_jd_j2000(self):
        """J2000 历元 ISO→JD：2000-01-01T12:00:00 → 2451545.0。"""
        result = convert_time(
            "2000-01-01T12:00:00",
            from_scale="UTC", from_format="ISO",
            to_scale="UTC", to_format="JD",
        )
        assert result.get("status") != "error"
        jd = result["output"]["value"]
        assert jd == pytest.approx(2451545.0, abs=1e-3)

    def test_iso_to_mjd(self):
        """ISO→MJD：2026-01-01T00:00:00 → 61041.0。"""
        result = convert_time(
            "2026-01-01T00:00:00",
            from_scale="UTC", from_format="ISO",
            to_scale="UTC", to_format="MJD",
        )
        assert result.get("status") != "error"
        mjd = result["output"]["value"]
        assert mjd == pytest.approx(61041.0, abs=1e-3)

    def test_iso_to_unix(self):
        """ISO→UNIX 时间戳。"""
        result = convert_time(
            "2026-01-01T00:00:00",
            from_scale="UTC", from_format="ISO",
            to_scale="UTC", to_format="UNIX",
        )
        assert result.get("status") != "error"
        unix = result["output"]["value"]
        expected = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        assert unix == pytest.approx(expected, abs=1.0)

    def test_jd_to_iso_roundtrip(self):
        """JD→ISO 往返：先 ISO→JD，再 JD→ISO，应回到原值。"""
        r1 = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "JD")
        assert r1.get("status") != "error"
        jd = r1["output"]["value"]
        r2 = convert_time(jd, "UTC", "JD", "UTC", "ISO")
        assert r2.get("status") != "error"
        iso_out = r2["output"]["value"]
        assert "2026-01-01T00:00:00" in str(iso_out)

    def test_mjd_to_jd_consistency(self):
        """MJD→JD 转换：JD = MJD + 2400000.5。"""
        r = convert_time(61041.0, "UTC", "MJD", "UTC", "JD")
        assert r.get("status") != "error"
        jd = r["output"]["value"]
        assert jd == pytest.approx(61041.0 + 2400000.5, abs=1e-3)

    def test_iso_to_jd_at_noon(self):
        """正午时刻 ISO→JD 应比 0 时大 0.5 天。"""
        r0 = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "JD")
        r12 = convert_time("2026-01-01T12:00:00", "UTC", "ISO", "UTC", "JD")
        assert r0.get("status") != "error"
        assert r12.get("status") != "error"
        assert r12["output"]["value"] - r0["output"]["value"] == pytest.approx(0.5, abs=1e-6)


# ======================================================================
# 引擎标识测试
# ======================================================================
class TestConvertTimeEngineUsed:
    """convert_time 引擎标识与元数据测试。"""

    def test_engine_used_recorded(self):
        """转换结果必须记录 engine_used。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "JD")
        assert result.get("status") != "error"
        assert result["engine_used"] in ("datetime_builtin", "astropy")

    def test_engine_for_utc_to_utc(self):
        """UTC→UTC 转换：astropy 不可用时回退 datetime_builtin。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "JD")
        assert result.get("status") != "error"
        assert result["engine_used"] is not None

    def test_output_scale_and_format_recorded(self):
        """输出包含 scale 和 format 标签。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "JD")
        assert result.get("status") != "error"
        assert result["output"]["scale"] == "UTC"
        assert result["output"]["format"] == "JD"

    def test_input_recorded(self):
        """结果包含输入元数据。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "JD")
        assert result.get("status") != "error"
        assert result["input"]["value"] == "2026-01-01T00:00:00"
        assert result["input"]["scale"] == "UTC"
        assert result["input"]["format"] == "ISO"


# ======================================================================
# 错误处理测试
# ======================================================================
class TestConvertTimeErrorHandling:
    """convert_time 错误处理测试。"""

    def test_invalid_scale(self):
        """无效时间尺度应返回错误。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "GALACTIC", "ISO")
        assert result.get("status") == "error"

    def test_invalid_format(self):
        """无效格式应返回错误。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "UTC", "STARDATE")
        assert result.get("status") == "error"

    def test_utc_to_tdb_without_astropy(self):
        """无 astropy 时 UTC→TDB 应优雅报错（而非崩溃）。"""
        result = convert_time("2026-01-01T00:00:00", "UTC", "ISO", "TDB", "JD")
        # astropy 可用时转换成功；否则返回 error
        if result.get("status") == "error":
            assert "astropy" in result.get("reason", "").lower() or \
                   "不可用" in result.get("reason", "")
        else:
            assert result["output"]["scale"] == "TDB"

    def test_invalid_iso_input(self):
        """无效 ISO 字符串应返回错误。"""
        result = convert_time("not-a-date", "UTC", "ISO", "UTC", "JD")
        assert result.get("status") == "error"


# ======================================================================
# Epoch schema 一致性测试
# ======================================================================
class TestEpochSchema:
    """Epoch schema 序列化一致性测试。"""

    def test_epoch_roundtrip(self):
        """Epoch schema 序列化/反序列化往返一致。"""
        e = Epoch("2026-01-01T00:00:00", TimeScale.UTC, TimeFormat.ISO)
        d = e.to_dict()
        e2 = Epoch.from_dict(d)
        assert e2.value == e.value
        assert e2.scale == TimeScale.UTC
        assert e2.format == TimeFormat.ISO

    def test_epoch_iso_utc_passthrough(self):
        """已为 ISO UTC 的 Epoch 直接转字符串应原样返回。"""
        e = Epoch("2026-01-01T00:00:00", TimeScale.UTC, TimeFormat.ISO)
        assert e.to_iso_utc() == "2026-01-01T00:00:00"
