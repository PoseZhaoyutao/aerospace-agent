from __future__ import annotations

from pathlib import Path
import re
from dataclasses import replace
from typing import Any, Callable
from uuid import uuid4
import os
import urllib.request

import pytest

from langchain_core.messages import HumanMessage

from aerospace_agent.langgraph_agent.graph import ServiceBundle
from aerospace_agent.langgraph_agent.schema import ActionType, Decision, ToolCallRequest
from aerospace_agent.langgraph_agent.state import create_initial_state


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Use workspace-local paths when this directory is run standalone."""
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip process-backed stdio checks unless explicitly enabled.

    The managed Windows runner denies creation of the named pipes used by
    MCP's subprocess transport.  The product still returns a structured
    ``MCPUnavailableError``; set ``AEROSPACE_RUN_STDIO_TESTS=1`` on a normal
    workstation to execute the real transport tests.
    """
    if os.environ.get("AEROSPACE_RUN_STDIO_TESTS", "").lower() in {"1", "true", "yes"}:
        return
    skip = pytest.mark.skip(reason="MCP stdio subprocess transport is disabled in the managed runner")
    for item in items:
        if "stdio_gateway" in item.name or "sync_gateway_call" in item.name:
            item.add_marker(skip)
        if item.name == "test_missing_llm_is_structured_error_without_auto_spawn":
            endpoint = os.environ.get("AEROSPACE_LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
            try:
                with urllib.request.urlopen(f"{endpoint}/models", timeout=0.3):
                    item.add_marker(pytest.mark.skip(reason="local Qwen endpoint is available; missing-LLM branch is not applicable"))
            except Exception:
                pass


class RulePlanner:
    """Small deterministic planner double used by graph runtime tests."""

    def __init__(self, decision: Decision | Callable[[dict[str, Any]], Decision] | None = None):
        self.decision = decision
        self.calls = 0

    def plan(self, state: dict[str, Any]) -> Decision:
        self.calls += 1
        if callable(self.decision):
            return self.decision(state)
        if self.decision is not None:
            return self.decision
        return Decision(action=ActionType.RESPOND, rationale="No tool is required")


class DeterministicGateway:
    def __init__(self, handler: Callable[[ToolCallRequest], Any] | None = None):
        self.handler = handler or (lambda request: {"ok": True, "tool": request.tool_name})

    def call_tool(self, request: ToolCallRequest):
        return self.handler(request)


def initial_input(message: str, *, thread_id: str = "test", max_steps: int = 15):
    state = create_initial_state(thread_id=thread_id, max_steps=max_steps)
    state["messages"] = [HumanMessage(content=message)]
    return state


def _tool_decision(name: str = "check_engine_availability") -> Decision:
    return Decision(
        action=ActionType.CALL_TOOL,
        rationale="Call the requested availability check",
        tool_request=ToolCallRequest(tool_name=name, arguments={}),
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def mcp_settings(workspace: Path):
    from aerospace_agent.langgraph_agent.config import load_settings

    return load_settings(workspace=workspace).mcp


@pytest.fixture
def knowledge_service(tmp_path: Path):
    from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService

    service = KnowledgeService(workspace=tmp_path)
    service.initialize_seed_wiki()
    return service


@pytest.fixture
def evolution_service(workspace: Path):
    from aerospace_agent.langgraph_agent.services.evolution import EvolutionService

    return EvolutionService(workspace=workspace, data_dir=workspace / "data" / "langgraph" / "evolution")


@pytest.fixture
def make_services(tmp_path: Path, knowledge_service):
    def factory(*, planner=None, gateway=None):
        return ServiceBundle(
            knowledge=knowledge_service,
            context=None,
            planner=planner or RulePlanner(),
            mcp_gateway=gateway or DeterministicGateway(),
            llm=None,
            model_name="deterministic-test",
            endpoint="test://endpoint",
        )

    return factory


@pytest.fixture
def services(make_services):
    return make_services()


@pytest.fixture
def services_with_repeating_planner(make_services):
    return make_services(planner=RulePlanner(_tool_decision()))


@pytest.fixture
def services_with_failing_tool(make_services):
    def fail(request):
        raise RuntimeError("engine unavailable")

    return make_services(
        planner=RulePlanner(_tool_decision()),
        gateway=DeterministicGateway(fail),
    )


@pytest.fixture
def failing_checkpointer():
    """Saver double that fails every durable write."""
    from langgraph.checkpoint.memory import InMemorySaver

    class FailingCheckpointer(InMemorySaver):
        def put(self, *args, **kwargs):
            raise OSError("checkpoint write failed")

        def put_writes(self, *args, **kwargs):
            raise OSError("checkpoint write failed")

    return FailingCheckpointer()


@pytest.fixture
def agent_factory(workspace, make_services):
    """Build agents with the same settings/services contract as production."""
    from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
    from aerospace_agent.langgraph_agent.config import load_settings

    def factory(*, services=None, interrupt_before=None, checkpoint_backend="memory",
                checkpoint_db_path=None, checkpointer=None, **kwargs):
        return LangGraphAerospaceAgent(
            settings=load_settings(workspace=workspace),
            services=services or make_services(),
            interrupt_before=interrupt_before,
            checkpoint_backend=checkpoint_backend,
            checkpoint_db_path=checkpoint_db_path,
            checkpointer=checkpointer,
            **kwargs,
        )

    return factory
