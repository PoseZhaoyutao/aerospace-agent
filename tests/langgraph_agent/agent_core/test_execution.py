from __future__ import annotations

from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent_core.confirmation import ConfirmationService
from aerospace_agent.langgraph_agent.agent_core.execution import (
    AuthorizedExecutor,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.execution_checkpoints import (
    ExecutionCheckpointStore,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityManifest,
    GoalBoundary,
    PlanExecutionSnapshot,
    PlanStep,
    ContractModel,
    ToolResult,
    VerificationCheck,
)
from aerospace_agent.langgraph_agent.agent_core.planning import (
    PlanExecutionVerifier,
    build_task_plan,
)
from aerospace_agent.mcp.tools.environment_tools import check_engine_availability


class EngineInput(ContractModel):
    engines: list[str] | None = None


def _manifest(*, status: str = "available", risk: str = "read_only", dependencies=()):
    return CapabilityManifest(
        capability_id="engine-capability",
        version="1.0.0",
        category="basic",
        status=status,
        intents=["environment"],
        tool_names=["check_engine_availability"],
        risk_level=risk,
        required_dependencies=list(dependencies),
        source="aerospace_agent.mcp.tools.environment_tools",
    )


def _registry(
    tmp_path: Path,
    *,
    manifest: CapabilityManifest | None = None,
    confirmation_service: ConfirmationService | None = None,
    dependency_files=None,
    requires_confirmation: bool = False,
) -> ExecutionRegistry:
    registry = ExecutionRegistry(
        Path.cwd(),
        audit_database_path=tmp_path / "execution_audit.sqlite",
        confirmation_service=confirmation_service,
    )
    registry.register(
        kind="tool",
        manifest=manifest or _manifest(),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=EngineInput,
        entrypoint=(
            "aerospace_agent.mcp.tools.environment_tools.check_engine_availability"
        ),
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class="read_only",
        dependency_files=dependency_files or {},
        requires_confirmation=requires_confirmation,
    )
    return registry


def _request(**overrides) -> ExecutionRequest:
    values = {
        "kind": "tool",
        "capability_id": "engine-capability",
        "executor_name": "check_engine_availability",
        "operation_id": "operation-1",
        "arguments": {"engines": []},
    }
    values.update(overrides)
    return ExecutionRequest(**values)


def _context(registry: ExecutionRegistry, **overrides) -> ExecutionContext:
    values = {
        "project_id": "project",
        "thread_id": "thread",
        "root_run_id": "root-run",
        "workspace_root": str(Path.cwd()),
        "capability_snapshot": registry.snapshot("engine-capability"),
    }
    values.update(overrides)
    return ExecutionContext(**values)


def _checkpoint_store(registry: ExecutionRegistry) -> ExecutionCheckpointStore:
    return ExecutionCheckpointStore(
        registry._audit_database_path.with_name("execution_checkpoints.sqlite")
    )


def _service(registry: ExecutionRegistry) -> ExecutionService:
    return ExecutionService(registry, checkpoint_store=_checkpoint_store(registry))


def test_resolve_returns_sealed_non_callable_and_only_service_invokes_handler(tmp_path) -> None:
    registry = _registry(tmp_path)

    authorized = registry.resolve(_request(), _context(registry))

    assert isinstance(authorized, AuthorizedExecutor)
    assert not callable(authorized)
    assert not hasattr(authorized, "handler")
    assert registry.audit_records() == []

    result = _service(registry).execute(authorized)
    assert result.status == "success"
    assert isinstance(result.result, dict)
    assert result.result
    assert registry.audit_records()[-1]["audit_id"] == result.audit_id


def test_authorization_is_single_use(tmp_path) -> None:
    registry = _registry(tmp_path)
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)

    assert _service(registry).execute(authorized).status == "success"
    replay = _service(registry).execute(authorized)

    assert replay.status == "blocked"
    assert replay.error is not None and replay.error.code == "conflict"
    assert len(registry.audit_records()) == 2


@pytest.mark.parametrize("status", ["interface_only", "disabled", "unavailable"])
def test_non_available_capability_never_produces_authorized_executor(
    tmp_path, status: str
) -> None:
    registry = _registry(tmp_path, manifest=_manifest(status=status))

    result = registry.resolve(_request(), _context(registry))

    assert isinstance(result, ToolResult)
    assert result.status == "unavailable"
    assert result.error is not None and result.error.code == "unavailable"


def test_snapshot_mismatch_blocks_before_invocation(tmp_path) -> None:
    registry = _registry(tmp_path)
    snapshot = registry.snapshot("engine-capability").model_copy(
        update={"manifest_sha256": "f" * 64}
    )

    result = registry.resolve(
        _request(), _context(registry, capability_snapshot=snapshot)
    )

    assert isinstance(result, ToolResult)
    assert result.status == "blocked"
    assert result.error is not None and result.error.code == "conflict"


def test_dependency_hash_change_invalidates_registration(tmp_path) -> None:
    dependency = tmp_path / "dependency.lock"
    dependency.write_text("v1", encoding="utf-8")
    registry = _registry(
        tmp_path,
        manifest=_manifest(dependencies=["engine-lock"]),
        dependency_files={"engine-lock": dependency},
    )
    dependency.write_text("v2", encoding="utf-8")

    result = registry.resolve(_request(), _context(registry))

    assert isinstance(result, ToolResult)
    assert result.status == "unavailable"


def test_entrypoint_outside_allowed_import_roots_is_rejected_at_registration(tmp_path) -> None:
    registry = ExecutionRegistry(
        Path.cwd(), audit_database_path=tmp_path / "execution_audit.sqlite"
    )

    with pytest.raises(ValueError, match="allowed current-repository"):
        registry.register(
            kind="tool",
            manifest=_manifest(),
            executor_name="check_engine_availability",
            handler=check_engine_availability,
            input_model=EngineInput,
            entrypoint="neighbor_project.tools.check_engine_availability",
            adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
            recovery_class="read_only",
        )


def test_invalid_input_schema_fails_closed(tmp_path) -> None:
    registry = _registry(tmp_path)

    result = registry.resolve(
        _request(arguments={"engines": "gmat"}),
        _context(registry),
    )

    assert isinstance(result, ToolResult)
    assert result.status == "invalid_arguments"
    assert result.error is not None and result.error.code == "invalid_arguments"


def test_high_risk_resolution_requires_and_consumes_confirmation(tmp_path) -> None:
    confirmations = ConfirmationService(tmp_path / "confirmation.sqlite")
    registry = _registry(
        tmp_path,
        manifest=_manifest(risk="high_risk"),
        confirmation_service=confirmations,
    )
    request = _request()
    context = _context(registry)

    blocked = registry.resolve(request, context)
    assert isinstance(blocked, ToolResult)
    assert blocked.error is not None and blocked.error.code == "confirmation_required"

    grant = confirmations.issue(
        project_id=context.project_id,
        thread_id=context.thread_id,
        root_run_id=context.root_run_id,
        operation_id=request.operation_id,
        action_hash=registry.preview_action_hash(request, context),
    )
    confirmed = registry.resolve(
        request.model_copy(update={"confirmation_id": grant.confirmation_id}),
        context,
    )

    assert isinstance(confirmed, AuthorizedExecutor)
    assert _service(registry).execute(confirmed).status == "success"
    continuation = confirmations.continuation_checkpoint(grant.confirmation_id)
    assert continuation is not None
    assert continuation["operation_id"] == request.operation_id

    replay = registry.resolve(
        request.model_copy(update={"confirmation_id": grant.confirmation_id}),
        context,
    )
    assert isinstance(replay, ToolResult)
    assert replay.error is not None and replay.error.code == "confirmation_replayed"


def test_execution_service_records_audit_and_success_checkpoint(tmp_path) -> None:
    registry = _registry(tmp_path)
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)

    store = _checkpoint_store(registry)
    result = ExecutionService(registry, checkpoint_store=store).execute(authorized)

    checkpoint_id = f"execution:root-run:operation-1:{result.audit_id}"
    receipt, checkpointed = store.get(checkpoint_id)
    assert receipt.operation_id == "operation-1"
    assert checkpointed.status == "success"
    records = registry.audit_records()
    assert records[-1]["audit_id"] == result.audit_id
    assert records[-1]["root_run_id"] == "root-run"


def test_human_step_never_calls_a_handler_and_returns_interrupt_checkpoint(tmp_path) -> None:
    plan_verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = ExecutionRegistry(
        Path.cwd(),
        audit_database_path=tmp_path / "execution_audit.sqlite",
        plan_execution_verifier=plan_verifier,
    )
    manifest = CapabilityManifest(
        capability_id="human.review",
        version="1.0.0",
        category="project",
        status="available",
        intents=["human"],
        risk_level="read_only",
        source="aerospace_agent.mcp.tools",
    )
    registry.register(
        kind="human",
        manifest=manifest,
        executor_name="human.review",
        handler=None,
        input_model=ContractModel,
        entrypoint="aerospace_agent.mcp.tools.human",
        adapter_path=Path("aerospace_agent/mcp/tools/__init__.py"),
        recovery_class="read_only",
    )
    request = ExecutionRequest(
        kind="human",
        capability_id="human.review",
        executor_name="human.review",
        operation_id="human-operation",
        arguments={},
        origin="planned",
        step_id="human-step",
    )
    capability_snapshot = registry.snapshot("human.review")
    plan = build_task_plan(
        {
            "plan_id": "plan-1",
            "project_id": "project",
            "thread_id": "thread",
            "root_run_id": "root-run",
            "goal": GoalBoundary(objective="Obtain a human review"),
            "steps": [
                PlanStep(
                    step_id="human-step",
                    title="Human review",
                    description="Interrupt for an external answer",
                    executor_type="human",
                    capability="human.review",
                    human_instruction="Review the values",
                    expected_outputs=["answer"],
                    verification=[
                        VerificationCheck(
                            check_id="human-check",
                            description="answer exists",
                            method="human",
                            acceptance_rule="reviewer answered",
                        )
                    ],
                )
            ],
            "execution_snapshot": PlanExecutionSnapshot(
                capability_snapshots=[capability_snapshot],
                registry_snapshot_sha256="c" * 64,
                captured_at="2026-07-13T08:00:00+00:00",
            ),
            "created_at": "2026-07-13T08:00:00+00:00",
        }
    )
    plan_verifier.register_plan(plan)
    context = ExecutionContext(
        project_id="project",
        thread_id="thread",
        root_run_id="root-run",
        workspace_root=str(Path.cwd()),
        capability_snapshot=capability_snapshot,
        plan_id="plan-1",
        plan_sha256=plan.plan_sha256,
        registry_snapshot_sha256=plan.execution_snapshot.registry_snapshot_sha256,
    )
    authorized = registry.resolve(request, context)
    assert isinstance(authorized, AuthorizedExecutor)
    store = _checkpoint_store(registry)
    result = ExecutionService(registry, checkpoint_store=store).execute(authorized)

    assert result.status == "interrupted"
    _, checkpointed = store.get(
        f"execution:root-run:human-operation:{result.audit_id}"
    )
    assert checkpointed.status == "interrupted"
