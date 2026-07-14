"""Workspace-local test fixtures for restricted Windows runners.

All temporary data is centralized below ``.test-artifacts/pytest`` (or the
``AEROSPACE_TEST_ARTIFACT_ROOT`` override) so a successful acceptance run can
remove it as one unit.
"""
from __future__ import annotations

from pathlib import Path
import os
import re
from uuid import uuid4
from typing import Any

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    base = Path(
        os.environ.get(
            "AEROSPACE_TEST_ARTIFACT_ROOT",
            str(Path.cwd() / ".test-artifacts" / "pytest"),
        )
    )
    root = base / f"{safe_name}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep legacy subsystem state inside the current test artifact root."""

    monkeypatch.setenv("AEROSPACE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AEROSPACE_LOG_LEVEL", "WARNING")


@pytest.fixture
def mock_llm():
    from aerospace_agent.core.llm_interface import MockLLM

    return MockLLM()


@pytest.fixture
def mock_agent(mock_llm) -> Any:
    from aerospace_agent.core.agent import create_default_agent

    return create_default_agent(force_mock=True, max_steps=5)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "aerospace_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def rag_instance(tmp_data_dir: Path):
    from aerospace_agent.rag.aerospace_rag import AerospaceRAG

    return AerospaceRAG(
        data_dir=str(tmp_data_dir),
        autoload=False,
        auto_default_knowledge=False,
    )
