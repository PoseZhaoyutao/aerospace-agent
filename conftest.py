"""全局共享测试夹具（conftest.py）。

所有测试模块均可使用此文件中定义的 fixture，无需重复导入。
常用 fixture:
  - mock_agent   : 使用 MockLLM 的完整 Agent 实例
  - mock_llm     : MockLLM 实例
  - orbit_state_leo : 标准 LEO 轨道状态（400km 圆轨道）
  - tmp_data_dir : 临时数据目录（测试后自动清理）
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

import pytest

# 确保包可导入
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Workspace-local substitute for pytest's ACL-sensitive tmpdir plugin.

    The managed Windows runner rejects the default temporary-directory
    cleanup.  Each test receives an isolated, retained path below the project
    root, whose node id is sanitized for Windows file-name rules.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    root = _PROJECT_ROOT / ".pytest-artifacts" / f"{safe_name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# 环境隔离
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """每个测试自动使用临时目录，避免污染真实数据。"""
    monkeypatch.setenv("AEROSPACE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AEROSPACE_LOG_LEVEL", "WARNING")
    yield


# ---------------------------------------------------------------------------
# LLM 夹具
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm():
    """MockLLM 实例——离线可用，不依赖网络。"""
    from aerospace_agent.core.llm_interface import MockLLM
    return MockLLM()


@pytest.fixture
def mock_agent(mock_llm) -> "Any":
    """使用 MockLLM 的完整 Agent 实例。

    装配全部默认组件（工具、RAG、MCP 工具），但 LLM 强制为 MockLLM。
    """
    from aerospace_agent.core.agent import create_default_agent
    agent = create_default_agent(force_mock=True, max_steps=5)
    return agent


# ---------------------------------------------------------------------------
# 轨道状态夹具
# ---------------------------------------------------------------------------
@pytest.fixture
def orbit_state_leo():
    """标准 LEO 轨道状态：400km 圆轨道，倾角 51.6°。"""
    from aerospace_agent.mcp.schemas import (
        OrbitState, KeplerianElements, Epoch, Frame, FrameName, FrameCenter,
    )
    return OrbitState(
        representation="keplerian",
        keplerian=KeplerianElements(
            semi_major_axis=6778137.0,  # 400km altitude
            eccentricity=0.0,
            inclination=0.9,  # ~51.6°
            raan=0.0,
            arg_perigee=0.0,
            true_anomaly=0.0,
        ),
        epoch=Epoch(tai="2026-01-01T00:00:00"),
        frame=Frame(name=FrameName.GCRF, center=FrameCenter.EARTH),
    )


@pytest.fixture
def orbit_state_geo():
    """标准 GEO 轨道状态：35786km 圆轨道，倾角 0°。"""
    from aerospace_agent.mcp.schemas import (
        OrbitState, KeplerianElements, Epoch, Frame, FrameName, FrameCenter,
    )
    return OrbitState(
        representation="keplerian",
        keplerian=KeplerianElements(
            semi_major_axis=42164137.0,  # ~35786km altitude
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            arg_perigee=0.0,
            true_anomaly=0.0,
        ),
        epoch=Epoch(tai="2026-01-01T00:00:00"),
        frame=Frame(name=FrameName.GCRF, center=FrameCenter.EARTH),
    )


# ---------------------------------------------------------------------------
# 临时目录夹具
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """临时数据目录，测试后自动清理。"""
    d = tmp_path / "aerospace_data"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# RAG 夹具
# ---------------------------------------------------------------------------
@pytest.fixture
def rag_instance(tmp_data_dir):
    """独立 RAG 实例（使用临时目录，不加载已有索引）。"""
    from aerospace_agent.rag.aerospace_rag import AerospaceRAG
    rag = AerospaceRAG(data_dir=str(tmp_data_dir), autoload=False,
                       auto_default_knowledge=False)
    return rag


# ---------------------------------------------------------------------------
# 物理常量
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def physics_constants():
    """常用物理常量。"""
    return {
        "MU_EARTH": 3.986004418e14,   # m³/s²
        "R_EARTH": 6378137.0,          # m
        "J2": 1.08263e-3,
        "OMEGA_EARTH": 7.2921159e-5,   # rad/s
        "AU": 1.495978707e11,          # m
        "MOON_DIST": 3.844e8,          # m
    }
