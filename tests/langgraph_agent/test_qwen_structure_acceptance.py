"""Live structural checks against the already-running local model endpoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
from aerospace_agent.langgraph_agent.agent_core.git_service import GitService
from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService
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
        models = {str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)}
    except Exception as exc:
        pytest.skip(f"local model endpoint unavailable: {exc}")
    if model not in models:
        pytest.skip(f"configured model is not advertised: {model!r} not in {sorted(models)}")
    return _Endpoint(endpoint, model)


def _agent(endpoint: _Endpoint, workspace: Path, *, checkpoint_backend: str = "memory"):
    return LangGraphAerospaceAgent(
        settings=load_settings(workspace=workspace),
        llm_endpoint=endpoint.endpoint,
        model_name=endpoint.model,
        checkpoint_backend=checkpoint_backend,
        checkpoint_db_path=workspace / "data" / "langgraph" / "live-checkpoints.sqlite",
    )


def _tool_payload(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


@pytest.mark.qwen3
@pytest.mark.integration
def test_live_context_persists_after_restart_and_stays_thread_scoped(live_endpoint: _Endpoint, tmp_path: Path):
    project = ProjectIdentityService(tmp_path).initialize()
    first = _agent(live_endpoint, tmp_path, checkpoint_backend="sqlite")
    try:
        result = first.run("Remember: DRY_MASS_120_KG", thread_id="memory-a")
        assert result.status == "success"
    finally:
        first.close()

    second = _agent(live_endpoint, tmp_path, checkpoint_backend="sqlite")
    try:
        restored = second.run("Recall the prior marker", thread_id="memory-a")
        isolated = second.run("Hello", thread_id="memory-b")
        assert restored.status == "success"
        assert isolated.status == "success"
        a_messages = second.get_conversation_state("memory-a").values["messages"]
        b_messages = second.get_conversation_state("memory-b").values["messages"]
        assert any("DRY_MASS_120_KG" in str(getattr(item, "content", "")) for item in a_messages)
        assert all("DRY_MASS_120_KG" not in str(getattr(item, "content", "")) for item in b_messages)
        assert second.services.project_id == project.project_id
    finally:
        second.close()


@pytest.mark.qwen3
@pytest.mark.integration
def test_live_file_terminal_and_web_tools_cross_the_core_boundary(live_endpoint: _Endpoint, tmp_path: Path):
    ProjectIdentityService(tmp_path).initialize()
    (tmp_path / "AGENTS.md").write_text("Repository Guidelines\n", encoding="utf-8")
    agent = _agent(live_endpoint, tmp_path)
    try:
        file_result = agent.run(
            "file.read",
            thread_id="live-file",
            context={
                "requested_tool_name": "file.read",
                "parsed_arguments": {"path": "AGENTS.md"},
                "arguments_validated": True,
            },
        )
        terminal_result = agent.run(
            "terminal.run",
            thread_id="live-terminal",
            context={
                "requested_tool_name": "terminal.run",
                "parsed_arguments": {"argv": ["python", "--version"]},
                "arguments_validated": True,
            },
        )
        web_result = agent.run(
            "web.search",
            thread_id="live-web",
            context={
                "requested_tool_name": "web.search",
                "parsed_arguments": {"query": "NASA Artemis program", "max_results": 3},
                "arguments_validated": True,
            },
        )
        browser_open = agent.run(
            "browser.open",
            thread_id="live-browser",
            context={
                "requested_tool_name": "browser.open",
                "parsed_arguments": {"url": "https://example.com"},
                "arguments_validated": True,
            },
        )
        browser_page = _tool_payload(browser_open.tool_results[0])["result"]["page_id"]
        browser_extract = agent.run(
            "browser.extract",
            thread_id="live-browser",
            context={
                "requested_tool_name": "browser.extract",
                "parsed_arguments": {"page_id": browser_page, "max_chars": 4000},
                "arguments_validated": True,
            },
        )
        assert file_result.status == terminal_result.status == web_result.status == browser_open.status == browser_extract.status == "success"
        assert file_result.metrics["capability_route"] == "direct_execution"
        terminal_tool = _tool_payload(terminal_result.tool_results[0])
        web_tool = _tool_payload(web_result.tool_results[0])
        assert terminal_tool["status"] == "success"
        assert web_tool["status"] == "success"
        assert web_tool["result"]["provider"]
        assert web_tool["result"]["results"]
        browser_tool = _tool_payload(browser_open.tool_results[0])
        assert browser_tool["status"] == "success"
        assert browser_tool["result"]["page_id"]
        assert _tool_payload(browser_extract.tool_results[0])["status"] == "success"
    finally:
        agent.close()


@pytest.mark.qwen3
@pytest.mark.integration
def test_live_git_status_precedes_scoped_checkpoint_and_rollback(live_endpoint: _Endpoint, tmp_path: Path):
    if shutil.which("git") is None:
        pytest.skip("git executable is not installed")
    environment = os.environ.copy()
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_EDITOR": "true"})
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, env=environment, check=True)
    subprocess.run(["git", "config", "user.email", "agent@example.invalid"], cwd=tmp_path, env=environment, check=True)
    subprocess.run(["git", "config", "user.name", "Agent"], cwd=tmp_path, env=environment, check=True)
    (tmp_path / "tracked.txt").write_text("v0\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "tracked.txt"], cwd=tmp_path, env=environment, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "initial"], cwd=tmp_path, env=environment, check=True)
    ProjectIdentityService(tmp_path).initialize()

    agent = _agent(live_endpoint, tmp_path)
    try:
        status = agent.run(
            "git.status",
            thread_id="live-git",
            context={
                "requested_tool_name": "git.status",
                "parsed_arguments": {"paths": None},
                "arguments_validated": True,
            },
        )
        assert status.status == "success"
        assert _tool_payload(status.tool_results[0])["status"] == "success"
    finally:
        agent.close()

    service = GitService(tmp_path)
    (tmp_path / "tracked.txt").write_text("v1\n", encoding="utf-8")
    checkpoint = service.create_checkpoint(
        message="live checkpoint",
        paths=["tracked.txt"],
        confirmation_consumed=True,
    )
    assert checkpoint.status == "success"
    (tmp_path / "tracked.txt").write_text("v2\n", encoding="utf-8")
    restored = service.restore_paths(paths=["tracked.txt"], confirmation_consumed=True)
    assert restored.status == "success"
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "v1\n"


@pytest.mark.qwen3
@pytest.mark.integration
def test_live_complex_request_builds_executes_and_reviews_task_plan(live_endpoint: _Endpoint, tmp_path: Path):
    ProjectIdentityService(tmp_path).initialize()
    (tmp_path / "AGENTS.md").write_text("Repository Guidelines\n", encoding="utf-8")
    agent = _agent(live_endpoint, tmp_path)
    try:
        result = agent.run(
            "complex task plan: read AGENTS.md then run python --version",
            thread_id="live-complex",
        )
        debug_state = agent.get_conversation_state("live-complex").values
        assert result.status == "success", {
            "errors": result.errors,
            "status": debug_state.get("status"),
            "termination_reason": debug_state.get("termination_reason"),
            "task_plan": debug_state.get("task_plan"),
            "plan_execution": debug_state.get("plan_execution"),
            "review_result": debug_state.get("review_result"),
        }
        assert result.metrics["capability_route"] == "complex_task"
        state = agent.get_conversation_state("live-complex").values
        assert state["task_plan"]["steps"]
        assert state["plan_execution"]["status"] == "completed"
        assert state["review_result"]["status"] == "passed"
        assert {item["tool_name"] for item in state["tool_results"]} >= {"file.read", "terminal.run"}
    finally:
        agent.close()
