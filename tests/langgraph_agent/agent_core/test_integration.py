from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent import LangGraphAerospaceAgent
from aerospace_agent.langgraph_agent.agent_core.capabilities import CapabilityRegistry
from aerospace_agent.langgraph_agent.agent_core.dag import DAGExecutionOutcome
from aerospace_agent.langgraph_agent.agent_core.execution import (
    AuthorizedExecutor,
    ExecutionRegistry,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityManifest,
    CheckResult,
    PlanExecutionState,
    PlanStepExecutionState,
    ReviewResult,
    ToolResult,
)
from aerospace_agent.langgraph_agent.agent_core.planning import build_task_plan
from aerospace_agent.langgraph_agent.agent_core.project_memory import ProjectIdentityService
from aerospace_agent.langgraph_agent.agent_core.routing import CapabilityRouter
from aerospace_agent.langgraph_agent.agent_core import runtime as runtime_module
from aerospace_agent.langgraph_agent.agent_core.session_memory import SessionMemoryService
from aerospace_agent.langgraph_agent.agent_core.tools import CoreToolCatalog, CoreToolServices
from aerospace_agent.langgraph_agent.config import load_settings
from aerospace_agent.langgraph_agent.graph import ServiceBundle


def _router() -> CapabilityRouter:
    registry = CapabilityRegistry(
        [
            CapabilityManifest(
                capability_id="basic.files",
                version="1.0.0",
                category="basic",
                status="available",
                intents=["file"],
                tool_names=["file.read"],
                risk_level="read_only",
                source="aerospace_agent.mcp.tools.core_tool_adapters",
            )
        ]
    )
    return CapabilityRouter(registry)


class _EchoLLM:
    def chat(self, prompt: str, **_kwargs) -> str:
        return f"echo:{prompt.rsplit('User:', 1)[-1].strip()}"


class _RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    def chat(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.system_prompts.append(str(kwargs.get("system_prompt", "")))
        return "recorded response"


class _CountingKnowledge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, top_k: int = 5):
        self.calls.append(query)
        return [
            {
                "page_path": "knowledge/source.md",
                "excerpt": "The two-body acceleration is -mu r / |r|^3.",
                "score": 1.0,
                "source_hash": "a" * 64,
            }
        ][:top_k]


def test_natural_language_preference_is_extracted_as_user_stated_memory() -> None:
    assert runtime_module.AgentCoreRuntime._extract_memories("以后都称呼我爸爸") == [
        ("preference", "user_stated", "以后都称呼我爸爸")
    ]


class _DirectExecutor:
    def execute_route(self, *, route, arguments, state):
        return ToolResult(
            status="success",
            result={"path": arguments["path"], "content": "ok"},
            audit_id="audit:direct",
            operation_id=f"op:{state['run_id']}",
            recovery_class="read_only",
        )


def _plan(project_id: str, thread_id: str, root_run_id: str):
    return build_task_plan(
        {
            "plan_id": f"plan:{root_run_id}",
            "project_id": project_id,
            "thread_id": thread_id,
            "root_run_id": root_run_id,
            "goal": {
                "objective": "read one file",
                "success_criteria": ["file result reviewed"],
            },
            "selected_capabilities": ["basic.files"],
            "steps": [
                {
                    "step_id": "read",
                    "title": "read",
                    "description": "read a project file",
                    "executor_type": "basic_tool",
                    "capability": "basic.files",
                    "tool_name": "file.read",
                    "inputs": {"path": "README.md"},
                    "expected_outputs": ["content"],
                    "verification": [
                        {
                            "check_id": "check:read",
                            "description": "read completed",
                            "method": "tool",
                            "acceptance_rule": "successful result",
                        }
                    ],
                }
            ],
            "execution_snapshot": {
                "capability_snapshots": [
                    {
                        "capability_id": "basic.files",
                        "version": "1.0.0",
                        "manifest_sha256": "1" * 64,
                    }
                ],
                "registry_snapshot_sha256": "2" * 64,
                "captured_at": datetime.now(UTC).isoformat(),
            },
            "created_at": datetime.now(UTC).isoformat(),
        }
    )


class _TaskPlanner:
    def create_task_plan(self, *, route, state):
        del route
        return {
            "task_plan": _plan(
                state["project_id"],
                state["thread_id"],
                state["root_run_id"],
            ),
            "retrieval_request": None,
        }


class _DAG:
    def execute(self, plan):
        result = ToolResult(
            status="success",
            result={"content": "ok"},
            audit_id="audit:plan",
            operation_id="op:plan",
            recovery_class="read_only",
        )
        state = PlanExecutionState(
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            root_run_id=plan.root_run_id,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            step_states=[
                PlanStepExecutionState(
                    step_id="read",
                    status="completed",
                    attempts=1,
                    last_input_hash="3" * 64,
                    last_output_refs=[result.audit_id],
                    last_checkpoint_id="checkpoint:after",
                )
            ],
            updated_at=datetime.now(UTC).isoformat(),
        )
        return DAGExecutionOutcome(
            status="completed",
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            state=state,
            step_results={"read": result},
        )


class _Reviewer:
    def review_outcome(self, *, plan, outcome, evidence):
        del evidence
        return ReviewResult(
            review_id="review:1",
            project_id=plan.project_id,
            thread_id=plan.thread_id,
            plan_id=plan.plan_id,
            root_run_id=plan.root_run_id,
            plan_sha256=plan.plan_sha256,
            status="passed",
            goal_satisfied=True,
            boundary_compliant=True,
            constraints_satisfied=True,
            checkpoint_valid=True,
            evidence_sufficient=True,
            tool_execution_safe=True,
            checks=[
                CheckResult(
                    check_id="check:read",
                    passed=True,
                    severity="info",
                    message="checked",
                    evidence_refs=[outcome.step_results["read"].audit_id],
                )
            ],
            recommended_action="respond",
            confidence=1.0,
            reviewed_at=datetime.now(UTC).isoformat(),
        )


def _services(project_id: str, *, knowledge=None) -> ServiceBundle:
    return ServiceBundle(
        knowledge=knowledge,
        llm=_EchoLLM(),
        model_name="deterministic",
        endpoint="test://local",
        project_id=project_id,
        capability_router=_router(),
        direct_executor=_DirectExecutor(),
        task_plan_service=_TaskPlanner(),
        dag_executor=_DAG(),
        review_service=_Reviewer(),
    )


@pytest.mark.parametrize(
    ("message", "context", "expected"),
    [
        ("你好", None, "conversation"),
        ("为什么二体加速度采用这个形式？", None, "knowledge_qa"),
        (
            "file.read",
            {
                "requested_tool_name": "file.read",
                "parsed_arguments": {"path": "README.md"},
                "arguments_validated": True,
            },
            "direct_execution",
        ),
        ("这是复杂任务，请分解并规划步骤", None, "complex_task"),
        ("请查询会话记忆", None, "memory_operation"),
        ("git status", None, "project_operation"),
        ("处理一下", None, "clarify"),
    ],
)
def test_initialized_agent_core_exposes_all_seven_routes(
    tmp_path: Path,
    message: str,
    context: dict | None,
    expected: str,
):
    project_service = ProjectIdentityService(tmp_path)
    project = project_service.initialize()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=_services(str(project.project_id)),
        checkpoint_backend="memory",
    )
    try:
        output = agent.run(message, thread_id=f"route-{expected}", context=context)
        assert output.metrics["capability_route"] == expected
        if expected == "direct_execution":
            assert output.tool_results[0].result["content"] == "ok"
        if expected == "complex_task":
            snapshot = agent.get_conversation_state("route-complex_task")
            assert snapshot.values["task_plan"]["schema_version"] == "1.0"
            assert snapshot.values["review_result"]["status"] == "passed"
            assert agent.services.plan_execution_verifier.has_plan_for_run(
                project_id=project.project_id,
                thread_id="route-complex_task",
                root_run_id=snapshot.values["root_run_id"],
            )
    finally:
        agent.close()


def test_core_rag_denial_wins_and_positive_request_consumes_one_root_budget(tmp_path: Path):
    project = ProjectIdentityService(tmp_path).initialize()
    knowledge = _CountingKnowledge()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=_services(str(project.project_id), knowledge=knowledge),
        checkpoint_backend="memory",
    )
    try:
        denied = agent.run(
            "为什么是二体加速度？不要来源，也不要核实",
            thread_id="rag-denied",
        )
        assert denied.metrics["retrieval_reason"] == "explicit_no_retrieval"
        assert knowledge.calls == []

        used = agent.run("为什么是二体加速度？请给出来源", thread_id="rag-used")
        assert used.metrics["retrieval_reason"] == "explicit_evidence_request"
        assert len(knowledge.calls) == 1
        run = agent.services.execution_run_store.get(used.metrics["run_id"])
        assert run.retrieval_budget == 1
        assert run.retrieval_state == "consumed"
    finally:
        agent.close()


def test_initialized_sqlite_checkpoint_restarts_same_thread_without_cross_thread_leakage(
    tmp_path: Path,
):
    project = ProjectIdentityService(tmp_path).initialize()
    settings = load_settings(workspace=tmp_path)
    checkpoint = tmp_path / "data" / "langgraph" / "core-checkpoints.sqlite"
    first = LangGraphAerospaceAgent(
        settings=settings,
        services=_services(str(project.project_id)),
        checkpoint_backend="sqlite",
        checkpoint_db_path=checkpoint,
    )
    try:
        first.run("你好", thread_id="thread-a")
    finally:
        first.close()

    restarted = LangGraphAerospaceAgent(
        settings=settings,
        services=_services(str(project.project_id)),
        checkpoint_backend="sqlite",
        checkpoint_db_path=checkpoint,
    )
    try:
        restarted.run("谢谢", thread_id="thread-a")
        restarted.run("你好", thread_id="thread-b")
        a_messages = restarted.get_conversation_state("thread-a").values["messages"]
        b_messages = restarted.get_conversation_state("thread-b").values["messages"]
        assert len(a_messages) >= 4
        assert len(b_messages) == 2
        assert all("谢谢" not in str(getattr(item, "content", "")) for item in b_messages)
    finally:
        restarted.close()


def test_restarted_thread_rehydrates_explicit_session_memory_into_model_prompt(tmp_path: Path):
    project = ProjectIdentityService(tmp_path).initialize()
    settings = load_settings(workspace=tmp_path)
    checkpoint = tmp_path / "data" / "langgraph" / "core-checkpoints.sqlite"
    first_llm = _RecordingLLM()
    first = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=first_llm, model_name="deterministic"),
        checkpoint_backend="sqlite",
        checkpoint_db_path=checkpoint,
    )
    try:
        first.run("记住: 航天器干质量为120千克", thread_id="memory-thread")
    finally:
        first.close()

    second_llm = _RecordingLLM()
    restarted = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=second_llm, model_name="deterministic"),
        checkpoint_backend="sqlite",
        checkpoint_db_path=checkpoint,
    )
    try:
        output = restarted.run("请回忆干质量", thread_id="memory-thread")
        assert output.status == "success"
        assert second_llm.prompts
        assert "航天器干质量为120千克" in second_llm.prompts[-1]
        assert "我是您航天领域共同学习进步的AI助手" in second_llm.system_prompts[-1]
        assert "Qwythos" not in second_llm.system_prompts[-1]
        assert "Empero AI" not in second_llm.system_prompts[-1]
    finally:
        restarted.close()


def test_uninitialized_project_keeps_legacy_graph_behavior(tmp_path: Path):
    services = ServiceBundle(llm=_EchoLLM(), model_name="legacy")
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=services,
        checkpoint_backend="memory",
    )
    try:
        output = agent.run("你好", thread_id="legacy")
        assert output.status == "success"
        assert "capability_route" not in output.metrics
    finally:
        agent.close()


def test_initialized_default_runtime_wires_real_core_services_with_binding_parity(
    tmp_path: Path,
):
    project = ProjectIdentityService(tmp_path).initialize()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        assert agent.services.project_id == project.project_id
        assert isinstance(agent.services.core_tool_services, CoreToolServices)
        assert isinstance(agent.services.core_tool_catalog, CoreToolCatalog)
        assert isinstance(agent.services.execution_registry, ExecutionRegistry)
        assert isinstance(agent.services.execution_service, ExecutionService)
        assert agent.services.direct_executor is not None

        entries = agent.services.core_tool_catalog.entries()
        available = {
            entry.tool_name
            for entry in entries
            if entry.manifest.status == "available"
        }
        assert available == set(agent.services.core_tool_catalog.executable_names())
        assert agent.services.core_tool_catalog.get("file.read").manifest.status == "available"
        assert agent.services.core_tool_catalog.get("git.status").manifest.status == "unavailable"
    finally:
        agent.close()


def test_initialized_runtime_wires_configured_public_web_search_provider(tmp_path: Path):
    project = ProjectIdentityService(tmp_path).initialize()
    settings = load_settings(workspace=tmp_path)
    agent = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        web = agent.services.core_tool_services.web
        assert web.search_providers
        assert web.default_search_provider == "duckduckgo"
        assert agent.services.core_tool_catalog.get("web.search").manifest.status == "available"
        assert project.project_id == agent.services.project_id
    finally:
        agent.close()


def test_turn_lifecycle_is_checkpoint_visible_and_context_is_thread_scoped(tmp_path: Path):
    ProjectIdentityService(tmp_path).initialize()
    settings = load_settings(workspace=tmp_path)
    agent = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        output = agent.run("你好", thread_id="turn-a")
        assert output.metrics["turn_state"] == "done"
        assert output.metrics["turn_state_history"] == [
            "restore",
            "compact",
            "command",
            "build",
            "run",
            "save",
            "respond",
            "done",
        ]

        snapshot = agent.get_conversation_state("turn-a")
        assert snapshot.values["turn_state"] == "done"
        assert snapshot.values["turn_state_history"][-2:] == ["respond", "done"]
    finally:
        agent.close()


def test_natural_language_preference_is_extracted_as_user_stated_memory() -> None:
    assert runtime_module.AgentCoreRuntime._extract_memories("以后都称呼我爸爸") == [
        ("preference", "user_stated", "以后都称呼我爸爸")
    ]


def test_default_core_conversation_uses_neither_tool_nor_rag_and_complex_is_honest_partial(
    tmp_path: Path,
):
    ProjectIdentityService(tmp_path).initialize()
    knowledge = _CountingKnowledge()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(
            llm=_EchoLLM(),
            knowledge=knowledge,
            model_name="deterministic",
        ),
        checkpoint_backend="memory",
    )
    try:
        ordinary = agent.run("你好", thread_id="ordinary")
        assert ordinary.status == "success"
        assert ordinary.tool_results == []
        assert knowledge.calls == []
        assert agent.services.execution_registry.audit_records() == []

        complex_output = agent.run(
            "这是复杂任务，请分解并规划步骤",
            thread_id="complex-without-planner",
        )
        complex_state = agent.get_conversation_state("complex-without-planner").values
        assert complex_output.metrics["capability_route"] == "complex_task"
        assert complex_output.status in {"partial", "error"}
        assert complex_output.status != "success"
        assert complex_state["termination_reason"] in {
            "task_plan_service_unavailable",
            "complex_execution_unavailable",
        }
        assert not complex_state.get("review_result")
    finally:
        agent.close()


def test_default_core_natural_file_and_terminal_requests_execute_read_only_tools(
    tmp_path: Path,
):
    ProjectIdentityService(tmp_path).initialize()
    (tmp_path / "AGENTS.md").write_text("Repository Guidelines\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("old\n", encoding="utf-8")
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=None, model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        file_output = agent.run("read file AGENTS.md", thread_id="natural-file")
        terminal_output = agent.run(
            "run command python --version", thread_id="natural-terminal"
        )
        write_output = agent.run(
            "write file notes.txt: new value", thread_id="natural-write"
        )

        assert file_output.metrics["capability_route"] == "direct_execution"
        assert file_output.tool_results[0].status == "success"
        assert file_output.tool_results[0].result["content"].strip() == "Repository Guidelines"
        assert terminal_output.metrics["capability_route"] == "direct_execution"
        assert terminal_output.tool_results[0].status == "success"
        assert terminal_output.tool_results[0].result["returncode"] == 0
        assert write_output.metrics["capability_route"] == "direct_execution"
        assert write_output.tool_results[0].status == "blocked"
        assert "confirmation" in str(write_output.tool_results[0].error).casefold()
        assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "old\n"
    finally:
        agent.close()


def test_default_core_file_read_crosses_authorized_executor_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ProjectIdentityService(tmp_path).initialize()
    (tmp_path / "README.md").write_text("runtime-bound", encoding="utf-8")
    seen: list[AuthorizedExecutor] = []
    execute = ExecutionService.execute

    def execute_spy(self, authorized):
        seen.append(authorized)
        return execute(self, authorized)

    monkeypatch.setattr(ExecutionService, "execute", execute_spy)
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        output = agent.run(
            "file.read",
            thread_id="real-direct",
            context={
                "requested_tool_name": "file.read",
                "parsed_arguments": {"path": "README.md"},
                "arguments_validated": True,
            },
        )
        assert output.status == "success"
        assert output.tool_results[0].result["content"] == "runtime-bound"
        assert len(seen) == 1
        assert isinstance(seen[0], AuthorizedExecutor)
        audit = agent.services.direct_executor.audit_records(thread_id="real-direct")
        assert len(audit) == 1
        assert audit[0]["executor_name"] == "file.read"
        assert audit[0]["status"] == "success"
    finally:
        agent.close()


def test_default_runtime_blocks_direct_execution_when_root_run_has_active_plan(
    tmp_path: Path,
):
    project = ProjectIdentityService(tmp_path).initialize()
    (tmp_path / "README.md").write_text("planned-only", encoding="utf-8")
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        runtime = agent.services.agent_core_runtime
        root_run_id = "root-with-active-plan"
        thread_id = "planned-thread"
        runtime.plan_execution_verifier.register_plan(
            _plan(project.project_id, thread_id, root_run_id)
        )
        route = runtime.for_thread(thread_id).router.route(
            "file.read",
            requested_tool_name="file.read",
            parsed_arguments={"path": "README.md"},
            arguments_validated=True,
        )
        result = runtime.execute_route(
            route=route,
            arguments={"path": "README.md"},
            state={
                "project_id": project.project_id,
                "thread_id": thread_id,
                "run_id": root_run_id,
                "root_run_id": root_run_id,
            },
        )
        assert result.status == "blocked"
        assert result.error is not None
        assert result.error.code == "conflict"
        assert "immutable TaskPlan" in result.error.message
    finally:
        agent.close()


def test_checkpointed_explicit_memory_restarts_same_thread_without_cross_thread_leakage(
    tmp_path: Path,
):
    project_service = ProjectIdentityService(tmp_path)
    project = project_service.initialize()
    settings = load_settings(workspace=tmp_path)
    checkpoint = tmp_path / "data" / "langgraph" / "memory-checkpoints.sqlite"
    first = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="sqlite",
        checkpoint_db_path=checkpoint,
    )
    try:
        first.run("约束：所有轨道状态使用 SI 单位", thread_id="memory-a")
        memory = SessionMemoryService(
            project_service.session_db_path,
            project_id=project.project_id,
            thread_id="memory-a",
            checkpoint_validator=lambda _checkpoint, _source_hash: True,
        ).list()
        assert [(item.kind, item.truth_status, item.content) for item in memory] == [
            ("constraint", "user_stated", "所有轨道状态使用 SI 单位")
        ]
        policy = first.services.agent_core_runtime.memory_persistence("memory-a")
        assert policy["summary_generated"] is False
        assert policy["summary_policy"] == "deterministic_explicit_memory_snapshot"
    finally:
        first.close()

    restarted = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="sqlite",
        checkpoint_db_path=checkpoint,
    )
    try:
        restarted.run("SI", thread_id="memory-a")
        restarted.run("SI", thread_id="memory-b")
        a_messages = restarted.get_conversation_state("memory-a").values["messages"]
        b_messages = restarted.get_conversation_state("memory-b").values["messages"]
        assert any(
            "[SESSION MEMORY]" in str(getattr(item, "content", ""))
            and "所有轨道状态使用 SI 单位" in str(getattr(item, "content", ""))
            for item in a_messages
        )
        assert all(
            "所有轨道状态使用 SI 单位" not in str(getattr(item, "content", ""))
            for item in b_messages
        )
    finally:
        restarted.close()


def test_preference_is_compressed_into_a_deterministic_session_summary(tmp_path: Path) -> None:
    project_service = ProjectIdentityService(tmp_path)
    project = project_service.initialize()
    settings = load_settings(workspace=tmp_path)
    first = LangGraphAerospaceAgent(
        settings=settings,
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="sqlite",
        checkpoint_db_path=tmp_path / "data" / "langgraph" / "summary-checkpoints.sqlite",
    )
    try:
        first.run("以后都称呼我爸爸", thread_id="summary-thread")
        for index in range(5):
            first.run(f"Continue the bounded engineering discussion, item {index}.", thread_id="summary-thread")
        policy = first.services.agent_core_runtime.memory_persistence("summary-thread")
        assert policy["summary_due"] is True
        assert policy["summary_generated"] is True
        session = SessionMemoryService(
            project_service.session_db_path,
            project_id=project.project_id,
            thread_id="summary-thread",
            checkpoint_validator=lambda _checkpoint, _source_hash: True,
        )
        summary = session.latest_summary()
        assert summary is not None
        assert summary.preferences == ["以后都称呼我爸爸"]
    finally:
        first.close()


def test_initialized_simple_mode_uses_agent_core_graph_and_never_wires_raw_mcp(
    tmp_path: Path,
):
    ProjectIdentityService(tmp_path).initialize()
    raw_calls: list[dict] = []

    def raw_handler(**arguments):
        raw_calls.append(arguments)
        return {"unsafe": True}

    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(
            llm=_EchoLLM(),
            model_name="deterministic",
            mcp_gateway={"raw": raw_handler},
        ),
        mode="simple",
        checkpoint_backend="memory",
    )
    try:
        output = agent.run("你好", thread_id="simple-core")
        graph_nodes = set(agent.graph.get_graph().nodes)

        assert output.metrics["capability_route"] == "conversation"
        assert "capability_route" in graph_nodes
        assert "tool_execute" not in graph_nodes
        assert raw_calls == []
    finally:
        agent.close()


@pytest.mark.parametrize("failing_component", ["execution_run_store", "agent_core_runtime"])
def test_agent_core_migration_failure_degrades_to_legacy_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_component: str,
):
    ProjectIdentityService(tmp_path).initialize()

    class MigrationFailed:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("unsupported future schema")

    if failing_component == "execution_run_store":
        monkeypatch.setattr(
            "aerospace_agent.langgraph_agent.agent_core.rag_gate.ExecutionRunStore",
            MigrationFailed,
        )
    else:
        monkeypatch.setattr(
            "aerospace_agent.langgraph_agent.agent_core.runtime.AgentCoreRuntime",
            MigrationFailed,
        )

    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(
            llm=_EchoLLM(),
            knowledge=_CountingKnowledge(),
            model_name="legacy-after-migration-failure",
        ),
        checkpoint_backend="memory",
    )
    try:
        assert agent.services.agent_core_enabled is False
        warning = agent.services.runtime_warnings[-1]
        assert warning["code"] == "migration_failed"
        assert warning["component"] == failing_component

        output = agent.run("你好", thread_id="migration-fallback")
        assert output.status == "success"
        assert "capability_route" not in output.metrics
        assert output.metrics["runtime_warnings"][-1] == warning
        assert agent.get_conversation_state("migration-fallback").values["messages"]
    finally:
        agent.close()


def test_caller_context_is_ephemeral_and_does_not_accumulate_across_turns(
    tmp_path: Path,
):
    ProjectIdentityService(tmp_path).initialize()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        supplied = {"ephemeral": "single-turn"}
        agent.run("first", thread_id="ephemeral-context", context=supplied)
        agent.run("second", thread_id="ephemeral-context", context=supplied)
        snapshot = agent.get_conversation_state("ephemeral-context")
        context_messages = [
            message
            for message in snapshot.values["messages"]
            if str(getattr(message, "content", "")).startswith("[context]")
        ]

        assert len(context_messages) == 1
        metadata = context_messages[0].additional_kwargs
        assert metadata["agent_context_ephemeral"] is True
        assert metadata["agent_context_run_id"] == snapshot.values["run_id"]
    finally:
        agent.close()


def test_default_runtime_uses_explicit_versioned_narrow_trusted_code_roots(
    tmp_path: Path,
):
    ProjectIdentityService(tmp_path).initialize()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        registry = agent.services.execution_registry
        roots = registry.trusted_code_roots
        adapter_roots = {root.import_root: root for root in roots if root.purpose == "adapter"}
        contract_roots = [root for root in roots if root.purpose == "input_contract"]

        assert set(adapter_roots) == {
            "aerospace_agent.mcp.tools",
            "aerospace_agent.integrations",
            "aerospace_agent.domains",
        }
        assert all(root.schema_version == "1.0" for root in roots)
        assert all(root.path.is_dir() for root in adapter_roots.values())
        assert len(contract_roots) == 1
        assert contract_roots[0].path.name == "system.py"
        assert contract_roots[0].path.is_file()

        snapshot = registry.snapshot("core.file.read")
        assert snapshot.trusted_code_roots_sha256 == registry.trusted_code_roots_sha256
        assert len(snapshot.trusted_code_roots_sha256) == 64

        package_root = Path(__file__).resolve().parents[3] / "aerospace_agent"
        with pytest.raises(ValueError, match="allowed adapter import root"):
            runtime_module.TrustedCodeRoot(
                root_id="whole-package",
                schema_version="1.0",
                purpose="adapter",
                path=package_root,
                import_root="aerospace_agent",
            )
    finally:
        agent.close()


def test_initialized_runtime_wires_evolution_and_acquisition_services(
    tmp_path: Path,
):
    ProjectIdentityService(tmp_path).initialize()
    agent = LangGraphAerospaceAgent(
        settings=load_settings(workspace=tmp_path),
        services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
        checkpoint_backend="memory",
    )
    try:
        services = agent.services
        assert services.evolution_candidate_service is not None
        assert services.capability_acquisition_service is not None
        assert services.integration_trust_service is not None
        assert services.evolution_candidate_service.availability()["available"] is True
        assert services.capability_acquisition_service.availability()["available"] is True
    finally:
        agent.close()


def test_non_migration_runtime_error_is_not_hidden_as_legacy_degradation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ProjectIdentityService(tmp_path).initialize()

    class BrokenRuntime:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("programming defect")

    monkeypatch.setattr(runtime_module, "AgentCoreRuntime", BrokenRuntime)

    with pytest.raises(RuntimeError, match="programming defect"):
        LangGraphAerospaceAgent(
            settings=load_settings(workspace=tmp_path),
            services=ServiceBundle(llm=_EchoLLM(), model_name="deterministic"),
            checkpoint_backend="memory",
        )
