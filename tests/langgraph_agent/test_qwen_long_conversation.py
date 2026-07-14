"""Long-running live checks for session memory retention and compression."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService
from aerospace_agent.langgraph_agent.agent_core.session_memory import SessionMemoryService
from aerospace_agent.langgraph_agent.config import load_settings


@dataclass(frozen=True)
class _Endpoint:
    endpoint: str
    model: str


@pytest.fixture(scope="session")
def live_endpoint() -> _Endpoint:
    endpoint = os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("AEROSPACE_LOCAL_LLM_MODEL", "qwythos")
    try:
        with urllib.request.urlopen(f"{endpoint}/models", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        pytest.skip(f"local model endpoint unavailable: {exc}")
    models = {str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)}
    if model not in models:
        pytest.skip(f"configured model is not advertised: {model!r} not in {sorted(models)}")
    return _Endpoint(endpoint, model)


def _agent(endpoint: _Endpoint, workspace: Path) -> LangGraphAerospaceAgent:
    return LangGraphAerospaceAgent(
        settings=load_settings(workspace=workspace),
        llm_endpoint=endpoint.endpoint,
        model_name=endpoint.model,
        checkpoint_backend="sqlite",
        checkpoint_db_path=workspace / "data" / "langgraph" / "long-conversation.sqlite",
    )


@pytest.mark.qwen3
@pytest.mark.integration
def test_live_long_conversation_retains_memory_through_compaction_and_restart(
    live_endpoint: _Endpoint,
    tmp_path: Path,
):
    project_service = ProjectIdentityService(tmp_path)
    project = project_service.initialize()
    first = _agent(live_endpoint, tmp_path)
    try:
        first_result = first.run(
            "Remember: spacecraft dry mass is 120 kg; optical navigation is primary.",
            thread_id="long-memory",
        )
        assert first_result.status == "success"
        # Twenty continuation turns exercise durable memory beyond a short
        # prompt window while keeping the fixture bounded for CI.
        for index in range(1, 21):
            result = first.run(
                f"Continue the engineering discussion with short item {index}.",
                thread_id="long-memory",
            )
            assert result.status == "success"
        memory = SessionMemoryService(
            project_service.session_db_path,
            project_id=project.project_id,
            thread_id="long-memory",
            checkpoint_validator=lambda checkpoint, _hash: checkpoint.project_id == project.project_id,
        )
        persisted = memory.search("dry mass")
        assert any("120 kg" in item.content for item in persisted)
    finally:
        first.close()

    restarted = _agent(live_endpoint, tmp_path)
    try:
        recalled = restarted.run("What is the spacecraft dry mass?", thread_id="long-memory")
        isolated = restarted.run("What is the spacecraft dry mass?", thread_id="other-thread")
        assert recalled.status == "success"
        assert isolated.status == "success"
        a_messages = restarted.get_conversation_state("long-memory").values["messages"]
        b_messages = restarted.get_conversation_state("other-thread").values["messages"]
        assert any(
            "[SESSION MEMORY]" in str(getattr(item, "content", ""))
            and "120 kg" in str(getattr(item, "content", ""))
            for item in a_messages
        )
        assert all("120 kg" not in str(getattr(item, "content", "")) for item in b_messages)
    finally:
        restarted.close()
