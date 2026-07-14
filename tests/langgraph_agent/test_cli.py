from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "start_langgraph_agent.py"


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), "--workspace", str(tmp_path), "--json", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_cli_agent_receives_composed_context_and_gateway_object(tmp_path: Path, monkeypatch):
    """The launcher delegates runtime construction instead of assembling tools."""
    from aerospace_agent.langgraph_agent.graph import ServiceBundle
    from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService
    import aerospace_agent.langgraph_agent.services.runtime as runtime_module

    class Gateway:
        closed = False

        def list_tools(self):
            return []

        def call_tool(self, request):
            raise AssertionError(f"unexpected tool call: {request}")

        def close(self):
            self.closed = True

    gateway = Gateway()
    context = object()
    knowledge = KnowledgeService(workspace=tmp_path)
    calls: list[tuple[object, Path, bool, bool, bool]] = []
    evolution = object()

    class Factory:
        def __init__(self, settings, *, project_root, allow_degraded_mcp,
                     mock_llm, check_llm_endpoint):
            calls.append((settings, project_root, allow_degraded_mcp, mock_llm, check_llm_endpoint))

        def create(self):
            return SimpleNamespace(
                bundle=ServiceBundle(
                    knowledge=knowledge,
                    context=context,
                    evolution=evolution,
                    mcp_gateway=gateway,
                    llm=None,
                    model_name="mock",
                    endpoint="",
                    runtime_warnings=("explicit degraded-mode warning",),
                ),
                knowledge=knowledge,
                evolution=evolution,
            )

    monkeypatch.setattr(runtime_module, "RuntimeServicesFactory", Factory)
    launcher = import_module("start_langgraph_agent")
    settings = launcher._load_settings(tmp_path, None)

    agent = launcher._create_agent(settings, tmp_path, mock=True)
    try:
        assert calls == [(settings, tmp_path, True, True, True)]
        assert agent.services.context is context
        assert agent.services.mcp_gateway is gateway
        assert not isinstance(agent.services.mcp_gateway, dict)
        assert agent.evolution_service is evolution
        result = agent.run("What is two-body dynamics?")
        assert "explicit degraded-mode warning" in result.warnings
    finally:
        agent.close()


def test_cli_defers_llm_construction_to_runtime_factory(tmp_path: Path, monkeypatch):
    """Endpoint probing belongs to the composition root, not the launcher."""
    from aerospace_agent.langgraph_agent.graph import ServiceBundle
    from aerospace_agent.langgraph_agent.services.knowledge import KnowledgeService
    import aerospace_agent.langgraph_agent.agent as agent_module
    import aerospace_agent.langgraph_agent.services.runtime as runtime_module

    knowledge = KnowledgeService(workspace=tmp_path)

    class Gateway:
        closed = False

        def list_tools(self):
            return []

        def call_tool(self, request):
            raise AssertionError(f"unexpected tool call: {request}")

        def close(self):
            self.closed = True

    class Factory:
        def __init__(self, *_args, **_kwargs):
            pass

        def create(self):
            return SimpleNamespace(
                bundle=ServiceBundle(knowledge=knowledge, context=object(), mcp_gateway=Gateway()),
                knowledge=knowledge,
                evolution=object(),
            )

    class UnexpectedClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("the CLI must not construct an LLM client")

    monkeypatch.setattr(runtime_module, "RuntimeServicesFactory", Factory)
    monkeypatch.setattr(agent_module, "SimpleLLMClient", UnexpectedClient)
    launcher = import_module("start_langgraph_agent")
    settings = launcher._load_settings(tmp_path, None)

    agent = launcher._create_agent(settings, tmp_path, mock=False, check_endpoint=False)
    agent.close()


def test_json_mock_task_is_single_document_with_citations(tmp_path: Path):
    result = run_cli(
        tmp_path,
        "--mock",
        "--task",
        "cite evidence for two-body central gravity",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["citations"]
    assert payload["metrics"]["model_name"] == "mock"
    assert payload["metrics"]["endpoint"] == ""
    assert result.stdout.strip().count("\n") >= 0


def test_json_stream_task_uses_verified_agent_output(tmp_path: Path):
    result = run_cli(
        tmp_path,
        "--mock",
        "--stream",
        "--task",
        "two-body central gravity",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["chunks"]
    assert "".join(payload["chunks"]) == payload["answer"]
    assert payload["answer"]


def test_mock_work_task_reports_missing_model_planner_as_partial(tmp_path: Path):
    result = run_cli(
        tmp_path,
        "--mock",
        "--task",
        "Propagate this orbit for 24 hours",
    )

    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial"
    assert any("planner" in item.lower() for item in payload["warnings"])
    assert payload["citations"] == []


def test_knowledge_actions_are_json_and_graph_paths_absolute(tmp_path: Path):
    initialized = run_cli(tmp_path, "--init-knowledge")
    assert initialized.returncode == 0, initialized.stderr
    assert json.loads(initialized.stdout)["created"] == 6

    status = run_cli(tmp_path, "--knowledge-status")
    assert json.loads(status.stdout)["wiki_pages"] == 6

    graph = run_cli(tmp_path, "--knowledge-graph", "reports/graph.html")
    payload = json.loads(graph.stdout)
    assert Path(payload["html_path"]).is_absolute()
    assert Path(payload["json_path"]).is_absolute()


def test_missing_llm_is_structured_error_without_auto_spawn(tmp_path: Path):
    result = run_cli(tmp_path, "--task", "hello")
    assert result.returncode == 2
    assert json.loads(result.stdout)["error_code"] == "LLM_UNAVAILABLE"
    assert "spawn" not in result.stderr.lower()


def test_export_schemas_writes_protocol_and_core_tool_contracts(tmp_path: Path):
    result = run_cli(tmp_path, "--export-schemas")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    expected = {
        "agent-input-v1.json",
        "agent-output-v1.json",
        "decision-v1.json",
        "evidence-item-v1.json",
        "tool-call-request-v1.json",
        "tool-call-response-v1.json",
        "evolution-proposal-v1.json",
        "evolution-record-v1.json",
        "validation-result-v1.json",
        "core-tool-catalog-v1.json",
    }
    assert {Path(item).name for item in payload["files"]} == expected
    assert all(Path(item).is_absolute() for item in payload["files"])
    catalog_path = next(
        Path(item) for item in payload["files"] if Path(item).name == "core-tool-catalog-v1.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "1.0"
    assert len(catalog["tools"]) >= 30


def test_evolve_actions_without_verified_proposal_are_noop(tmp_path: Path):
    evolved = run_cli(tmp_path, "--mock", "--evolve", "missing-thread")
    assert evolved.returncode == 0, evolved.stderr
    payload = json.loads(evolved.stdout)
    assert payload["status"] == "no_op"
    due = run_cli(tmp_path, "--mock", "--evolve-due")
    assert due.returncode == 0, due.stderr
    payload = json.loads(due.stdout)
    assert payload["status"] == "no_op"
    assert payload["reason"] == "no_checkpoint_thread"


def test_evolve_due_evaluates_real_thread_context(tmp_path: Path):
    task = run_cli(tmp_path, "--mock", "--thread", "scheduled", "--task", "two-body central gravity")
    assert task.returncode == 0, task.stderr
    due = run_cli(tmp_path, "--mock", "--evolve-due", "scheduled")
    assert due.returncode == 0, due.stderr
    payload = json.loads(due.stdout)
    assert payload["thread_id"] == "scheduled"
    assert payload["turn_count"] == 1
    assert payload["status"] == "no_op"
    assert payload["reason"] in {"not_idle", "insufficient_context"}


def test_internal_scheduler_cli_create_list_cancel_and_run_due(tmp_path: Path):
    initialized = run_cli(tmp_path, "--init-project")
    assert initialized.returncode == 0, initialized.stderr

    created = run_cli(
        tmp_path,
        "--thread",
        "thread-a",
        "--schedule-reminder",
        "2030-01-01T00:00:00+08:00",
        "future reminder",
    )
    assert created.returncode == 0, created.stderr
    job = json.loads(created.stdout)["job"]
    assert job["status"] == "scheduled"
    assert job["thread_id"] == "thread-a"

    listed = run_cli(tmp_path, "--thread", "thread-a", "--schedule-list")
    jobs = json.loads(listed.stdout)["jobs"]
    assert [item["job_id"] for item in jobs] == [job["job_id"]]

    cancelled = run_cli(
        tmp_path,
        "--schedule-version",
        str(job["version"]),
        "--schedule-cancel",
        job["job_id"],
    )
    assert cancelled.returncode == 0, cancelled.stderr
    assert json.loads(cancelled.stdout)["job"]["status"] == "cancelled"

    due = run_cli(
        tmp_path,
        "--thread",
        "thread-a",
        "--schedule-reminder",
        "2020-01-01T00:00:00+08:00",
        "due reminder",
    )
    assert due.returncode == 0, due.stderr
    run_due = run_cli(tmp_path, "--schedule-run-due")
    assert run_due.returncode == 0, run_due.stderr
    payload = json.loads(run_due.stdout)
    assert payload["status"] == "succeeded"
    assert payload["message"] == "due reminder"
