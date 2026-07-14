from __future__ import annotations

from base64 import b64encode
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ConfigDict, Field, field_validator

from aerospace_agent.langgraph_agent.agent_core.approval import (
    CapabilityApprovalLedger,
    CapabilityApprovalVerifier,
    approval_signature_payload,
)
from aerospace_agent.langgraph_agent.agent_core.confirmation import ConfirmationService
from aerospace_agent.langgraph_agent.agent_core.execution import (
    AuthorizedExecutor,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.execution_checkpoints import (
    ExecutionCheckpointReceipt,
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
    WorkflowSnapshot,
)
from aerospace_agent.langgraph_agent.agent_core.planning import (
    PlanExecutionVerifier,
    build_task_plan,
)
from aerospace_agent.mcp.tools.environment_tools import check_engine_availability
from aerospace_agent.mcp.tools.workflow_tools import list_workflow_templates


class EngineInput(ContractModel):
    engines: list[str] | None = None


class DefaultEngineInput(ContractModel):
    engines: list[str] = Field(default_factory=lambda: ["gmat"])


class PassThroughInput(ContractModel):
    model_config = ConfigDict(extra="forbid", title="SameInput")
    engines: list[str]


class RewritingInput(ContractModel):
    model_config = ConfigDict(extra="forbid", title="SameInput")
    engines: list[str]

    @field_validator("engines")
    @classmethod
    def rewrite_engines(cls, _value: list[str]) -> list[str]:
        return []


def _manifest(
    *,
    category: str = "basic",
    status: str = "available",
    risk: str = "read_only",
    tool_names: list[str] | None = None,
    source: str = "aerospace_agent.mcp.tools.environment_tools",
    dependencies: list[str] | None = None,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="engine-capability",
        version="1.0.0",
        category=category,
        status=status,
        intents=["environment"],
        tool_names=tool_names if tool_names is not None else ["check_engine_availability"],
        risk_level=risk,
        required_dependencies=dependencies or [],
        source=source,
    )


def _registry(
    tmp_path: Path,
    *,
    confirmation_service: ConfirmationService | None = None,
    approval_verifier=None,
    plan_execution_verifier=None,
    clock=None,
    authorization_ttl_seconds: int = 600,
) -> ExecutionRegistry:
    options = {
        "confirmation_service": confirmation_service,
        "approval_verifier": approval_verifier,
        "plan_execution_verifier": plan_execution_verifier,
        "authorization_ttl_seconds": authorization_ttl_seconds,
    }
    if clock is not None:
        options["clock"] = clock
    return ExecutionRegistry(
        Path.cwd(),
        audit_database_path=tmp_path / "execution_audit.sqlite",
        **options,
    )


def _register_engine(
    registry: ExecutionRegistry,
    *,
    manifest: CapabilityManifest | None = None,
    input_model=EngineInput,
    recovery_class: str = "read_only",
    dependency_files=None,
) -> None:
    registry.register(
        kind="tool",
        manifest=manifest or _manifest(),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=input_model,
        entrypoint=(
            "aerospace_agent.mcp.tools.environment_tools.check_engine_availability"
        ),
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class=recovery_class,
        dependency_files=dependency_files or {},
    )


def _request(**overrides) -> ExecutionRequest:
    values = {
        "kind": "tool",
        "capability_id": "engine-capability",
        "executor_name": "check_engine_availability",
        "operation_id": "operation",
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


def _service(registry: ExecutionRegistry, store=None) -> ExecutionService:
    return ExecutionService(
        registry,
        checkpoint_store=store or _checkpoint_store(registry),
    )


def _approval_verifier(tmp_path: Path):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    ledger = CapabilityApprovalLedger(
        tmp_path / "approval.sqlite", trusted_public_keys={"operator": public}
    )
    return private, ledger, CapabilityApprovalVerifier(ledger)


def _approve(private, ledger, digest: str, record_id: str = "approval-1") -> None:
    ledger.append(
        approval_record_id=record_id,
        key_id="operator",
        digest=digest,
        signature_b64=b64encode(private.sign(approval_signature_payload(digest))).decode(),
        created_at="2026-07-13T08:00:00+00:00",
    )


def test_registration_rejects_undeclared_executor_and_forged_handler_identity(tmp_path) -> None:
    registry = _registry(tmp_path)
    undeclared = _manifest(tool_names=[])
    with pytest.raises(ValueError, match="declared"):
        _register_engine(registry, manifest=undeclared)

    registry = _registry(tmp_path / "second")
    with pytest.raises(ValueError, match="entrypoint"):
        registry.register(
            kind="tool",
            manifest=_manifest(),
            executor_name="check_engine_availability",
            handler=check_engine_availability,
            input_model=EngineInput,
            entrypoint="aerospace_agent.mcp.tools.fake.check_engine_availability",
            adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
            recovery_class="read_only",
        )

    registry = _registry(tmp_path / "third")
    with pytest.raises(ValueError, match="handler source"):
        registry.register(
            kind="tool",
            manifest=_manifest(),
            executor_name="check_engine_availability",
            handler=check_engine_availability,
            input_model=EngineInput,
            entrypoint=(
                "aerospace_agent.mcp.tools.environment_tools.check_engine_availability"
            ),
            adapter_path=Path("aerospace_agent/mcp/tools/__init__.py"),
            recovery_class="read_only",
        )


def test_manifest_dependencies_must_exactly_match_trusted_hash_inputs(tmp_path) -> None:
    registry = _registry(tmp_path)
    manifest = _manifest(dependencies=["engine-lock"])

    with pytest.raises(ValueError, match="required_dependencies"):
        _register_engine(registry, manifest=manifest)


def test_kind_specific_policy_rejects_basic_capability_builder(tmp_path) -> None:
    registry = _registry(tmp_path)

    with pytest.raises(ValueError, match="capability_builder"):
        registry.register(
            kind="capability_builder",
            manifest=_manifest(),
            executor_name="check_engine_availability",
            handler=check_engine_availability,
            input_model=EngineInput,
            entrypoint=(
                "aerospace_agent.mcp.tools.environment_tools.check_engine_availability"
            ),
            adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
            recovery_class="manual_recovery",
        )


def test_workflow_requires_signed_approval_verifier_and_locked_snapshot(tmp_path) -> None:
    workflow_manifest = CapabilityManifest(
        capability_id="workflow-capability",
        version="1.0.0",
        category="workflow",
        status="available",
        intents=["workflow"],
        risk_level="read_only",
        source="aerospace_agent.mcp.tools.workflow_tools",
    )
    snapshot = WorkflowSnapshot(
        workflow_id="workflow.demo",
        version="1.0.0",
        workflow_sha256="a" * 64,
        manifest_sha256="b" * 64,
        approval_record_id="approval-1",
    )
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="approval"):
        registry.register(
            kind="workflow",
            manifest=workflow_manifest,
            executor_name="workflow.demo",
            handler=list_workflow_templates,
            input_model=ContractModel,
            entrypoint="aerospace_agent.mcp.tools.workflow_tools.list_workflow_templates",
            adapter_path=Path("aerospace_agent/mcp/tools/workflow_tools.py"),
            recovery_class="read_only",
            workflow_snapshot=snapshot,
        )

    private, ledger, verifier = _approval_verifier(tmp_path / "approved")
    approved = _registry(tmp_path / "approved", approval_verifier=verifier)
    registration = dict(
        kind="workflow",
        manifest=workflow_manifest,
        executor_name="workflow.demo",
        handler=list_workflow_templates,
        input_model=ContractModel,
        entrypoint="aerospace_agent.mcp.tools.workflow_tools.list_workflow_templates",
        adapter_path=Path("aerospace_agent/mcp/tools/workflow_tools.py"),
        recovery_class="read_only",
        workflow_snapshot=snapshot,
    )
    digest = approved.preview_registration_digest(**registration)
    _approve(private, ledger, digest)
    approved.register(**registration)


def test_normalized_default_arguments_are_what_confirmation_hash_protects(tmp_path) -> None:
    confirmations = ConfirmationService(tmp_path / "confirmation.sqlite")
    registry = _registry(tmp_path, confirmation_service=confirmations)
    _register_engine(
        registry,
        manifest=_manifest(risk="high_risk"),
        input_model=DefaultEngineInput,
        recovery_class="manual_recovery",
    )
    request = _request(arguments={})
    context = _context(registry)
    action_hash = registry.preview_action_hash(request, context)
    grant = confirmations.issue(
        project_id="project",
        thread_id="thread",
        root_run_id="root-run",
        operation_id="operation",
        action_hash=action_hash,
    )

    authorized = registry.resolve(
        request.model_copy(update={"confirmation_id": grant.confirmation_id}),
        context,
    )

    assert isinstance(authorized, AuthorizedExecutor)
    assert _service(registry).execute(authorized).status == "success"


def test_manual_recovery_requires_confirmation_even_when_manifest_is_read_only(tmp_path) -> None:
    registry = _registry(tmp_path)
    _register_engine(registry, recovery_class="manual_recovery")

    result = registry.resolve(_request(), _context(registry))

    assert isinstance(result, ToolResult)
    assert result.error is not None and result.error.code == "confirmation_required"


def test_execute_revalidates_dependency_hash_and_authorization_expiry(tmp_path) -> None:
    current = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    dependency = tmp_path / "engine.lock"
    dependency.write_text("v1", encoding="utf-8")
    manifest = _manifest(dependencies=["engine-lock"])
    registry = _registry(tmp_path, clock=lambda: current[0], authorization_ttl_seconds=10)
    _register_engine(
        registry,
        manifest=manifest,
        dependency_files={"engine-lock": dependency},
    )
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)
    dependency.write_text("v2", encoding="utf-8")

    changed = _service(registry).execute(authorized)
    assert changed.status == "unavailable"

    dependency.write_text("v1", encoding="utf-8")
    second = registry.resolve(_request(operation_id="second"), _context(registry))
    assert isinstance(second, AuthorizedExecutor)
    current[0] += timedelta(seconds=10)
    expired = _service(registry).execute(second)
    assert expired.status == "blocked"
    assert expired.error is not None and expired.error.code == "conflict"


def test_checkpoint_store_is_mandatory_and_failure_is_structured_and_audited(tmp_path) -> None:
    registry = _registry(tmp_path)
    _register_engine(registry)
    with pytest.raises(TypeError, match="checkpoint_store"):
        ExecutionService(registry)  # type: ignore[call-arg]
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)

    class FailingCheckpointStore(ExecutionCheckpointStore):
        def write(self, *_args, **_kwargs):
            raise OSError("checkpoint unavailable")

    result = _service(
        registry,
        store=FailingCheckpointStore(tmp_path / "failing_checkpoints.sqlite"),
    ).execute(authorized)

    assert result.status == "failed"
    assert result.recovery_class == "manual_recovery"
    assert result.error is not None and "checkpoint" in result.error.message
    assert registry.audit_records()[-1]["status"] == "failed"


def test_checkpoint_receipt_without_persisted_row_is_rejected(tmp_path) -> None:
    registry = _registry(tmp_path)
    _register_engine(registry)
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)

    class ForgedReceiptStore(ExecutionCheckpointStore):
        def write(self, request, context, _result):
            return ExecutionCheckpointReceipt(
                checkpoint_id="forged",
                project_id=context.project_id,
                thread_id=context.thread_id,
                root_run_id=context.root_run_id,
                operation_id=request.operation_id,
                persisted_at="2026-07-13T08:00:00+00:00",
            )

    result = _service(
        registry,
        store=ForgedReceiptStore(tmp_path / "forged_checkpoints.sqlite"),
    ).execute(authorized)

    assert result.status == "failed"
    assert result.error is not None and "checkpoint" in result.error.message


def test_confirmation_boundary_rejects_duck_typed_noop_service(tmp_path) -> None:
    class NoOpConfirmation:
        def consume(self, **_kwargs):
            return None

    with pytest.raises(TypeError, match="concrete ConfirmationService"):
        ExecutionRegistry(
            Path.cwd(),
            audit_database_path=tmp_path / "audit.sqlite",
            confirmation_service=NoOpConfirmation(),  # type: ignore[arg-type]
        )


def test_audit_is_persistent_and_contains_required_execution_evidence(tmp_path) -> None:
    registry = _registry(tmp_path)
    _register_engine(registry)
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)
    result = _service(registry).execute(authorized)
    restarted = _registry(tmp_path)

    record = restarted.audit_records()[-1]
    assert record["audit_id"] == result.audit_id
    assert record["risk_level"] == "read_only"
    assert record["result_sha256"]
    assert record["created_at"]
    assert record["recovery_class"] == "read_only"


def test_approval_is_rechecked_at_resolve_not_only_registration(tmp_path) -> None:
    private, ledger, verifier = _approval_verifier(tmp_path)
    workflow_manifest = CapabilityManifest(
        capability_id="workflow-capability",
        version="1.0.0",
        category="workflow",
        status="available",
        intents=["workflow"],
        risk_level="read_only",
        source="aerospace_agent.mcp.tools.workflow_tools",
    )
    snapshot = WorkflowSnapshot(
        workflow_id="workflow.demo",
        version="1.0.0",
        workflow_sha256="a" * 64,
        manifest_sha256="b" * 64,
        approval_record_id="approval-1",
    )
    registry = _registry(tmp_path, approval_verifier=verifier)
    registration = dict(
        kind="workflow",
        manifest=workflow_manifest,
        executor_name="workflow.demo",
        handler=list_workflow_templates,
        input_model=ContractModel,
        entrypoint="aerospace_agent.mcp.tools.workflow_tools.list_workflow_templates",
        adapter_path=Path("aerospace_agent/mcp/tools/workflow_tools.py"),
        recovery_class="read_only",
        workflow_snapshot=snapshot,
    )
    digest = registry.preview_registration_digest(**registration)
    _approve(private, ledger, digest)
    registry.register(**registration)
    context = ExecutionContext(
        project_id="project",
        thread_id="thread",
        root_run_id="run",
        workspace_root=str(Path.cwd()),
        capability_snapshot=registry.snapshot("workflow-capability"),
        workflow_snapshot=snapshot,
    )
    ledger.revoke(
        approval_record_id="approval-1",
        revoked_at="2026-07-13T09:00:00+00:00",
        reason="revoked for test",
    )

    result = registry.resolve(
        ExecutionRequest(
            kind="workflow",
            capability_id="workflow-capability",
            executor_name="workflow.demo",
            operation_id="workflow-op",
            arguments={},
        ),
        context,
    )

    assert isinstance(result, ToolResult)
    assert result.status == "unavailable"


def test_runtime_trust_baseline_is_rechecked_at_resolve_and_execute(tmp_path) -> None:
    state = {"status": "available", "combined_digest": "c" * 64}
    registry = _registry(tmp_path)
    registration = dict(
        kind="tool",
        manifest=_manifest(),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=EngineInput,
        entrypoint="aerospace_agent.mcp.tools.environment_tools.check_engine_availability",
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class="read_only",
        runtime_trust_digest="c" * 64,
        runtime_trust_verifier=lambda: dict(state),
    )
    registry.register(**registration)
    context = _context(registry)
    request = _request()
    authorized = registry.resolve(request, context)
    assert isinstance(authorized, AuthorizedExecutor)

    state["status"] = "unavailable"
    blocked = _service(registry).execute(authorized)

    assert blocked.status == "unavailable"
    assert blocked.error is not None
    assert "runtime trust" in blocked.error.message


def test_manifest_source_cannot_be_a_forged_parent_package(tmp_path) -> None:
    registry = _registry(tmp_path)

    with pytest.raises(ValueError, match="matched import root"):
        _register_engine(registry, manifest=_manifest(source="aerospace_agent"))


def test_approval_digest_binds_schema_paths_recovery_confirmation_and_validation(tmp_path) -> None:
    registry = _registry(tmp_path)
    base = dict(
        kind="tool",
        manifest=_manifest(),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=EngineInput,
        entrypoint="aerospace_agent.mcp.tools.environment_tools.check_engine_availability",
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class="read_only",
    )
    baseline = registry.preview_registration_digest(**base)
    different_schema = registry.preview_registration_digest(
        **{**base, "input_model": DefaultEngineInput}
    )
    confirmation = registry.preview_registration_digest(
        **{**base, "requires_confirmation": True}
    )
    manual = registry.preview_registration_digest(
        **{**base, "recovery_class": "manual_recovery"}
    )
    validated_manifest = _manifest().model_copy(update={"validators": ["contract-test"]})
    evidence = registry.preview_registration_digest(
        **{
            **base,
            "manifest": validated_manifest,
            "validation_evidence": {"contract-test": "d" * 64},
        }
    )

    assert len({baseline, different_schema, confirmation, manual, evidence}) == 5
    with pytest.raises(ValueError, match="path_fields"):
        registry.preview_registration_digest(**{**base, "path_fields": ["undeclared"]})


def test_registration_digest_binds_input_validator_implementation(tmp_path) -> None:
    registry = _registry(tmp_path)
    base = {
        "kind": "tool",
        "manifest": _manifest(),
        "executor_name": "check_engine_availability",
        "handler": check_engine_availability,
        "entrypoint": "aerospace_agent.mcp.tools.environment_tools.check_engine_availability",
        "adapter_path": Path("aerospace_agent/mcp/tools/environment_tools.py"),
        "recovery_class": "read_only",
    }

    pass_digest = registry.preview_registration_digest(
        **base, input_model=PassThroughInput
    )
    rewrite_digest = registry.preview_registration_digest(
        **base, input_model=RewritingInput
    )

    assert PassThroughInput.model_json_schema() == RewritingInput.model_json_schema()
    assert pass_digest != rewrite_digest


def test_arbitrary_handler_cannot_claim_reversible_or_compensatable(tmp_path) -> None:
    registry = _registry(tmp_path)

    with pytest.raises(ValueError, match="journal-managed"):
        _register_engine(registry, recovery_class="reversible")
    with pytest.raises(ValueError, match="compensation controller"):
        _register_engine(registry, recovery_class="compensatable")


def test_planned_tool_requires_exact_namespace_step_and_executor_binding(tmp_path) -> None:
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = _registry(tmp_path, plan_execution_verifier=verifier)
    _register_engine(registry)
    snapshot = registry.snapshot("engine-capability")
    plan = build_task_plan(
        {
            "plan_id": "plan",
            "project_id": "project",
            "thread_id": "thread",
            "root_run_id": "root-run",
            "goal": GoalBoundary(objective="Check environment"),
            "steps": [
                PlanStep(
                    step_id="check",
                    title="Check",
                    description="Check engine availability",
                    executor_type="basic_tool",
                    capability="engine-capability",
                    tool_name="check_engine_availability",
                    inputs={"engines": []},
                    expected_outputs=["availability"],
                    verification=[
                        VerificationCheck(
                            check_id="schema",
                            description="valid output",
                            method="schema",
                            acceptance_rule="matches schema",
                        )
                    ],
                )
            ],
            "execution_snapshot": PlanExecutionSnapshot(
                capability_snapshots=[snapshot],
                registry_snapshot_sha256="c" * 64,
                captured_at="2026-07-13T08:00:00+00:00",
            ),
            "created_at": "2026-07-13T08:00:00+00:00",
        }
    )
    verifier.register_plan(plan)
    context = _context(
        registry,
        plan_id=plan.plan_id,
        plan_sha256=plan.plan_sha256,
        registry_snapshot_sha256=plan.execution_snapshot.registry_snapshot_sha256,
    )
    request = _request(origin="planned", step_id="check")

    authorized = registry.resolve(request, context)
    assert isinstance(authorized, AuthorizedExecutor)
    direct = registry.resolve(
        _request(operation_id="direct"),
        context,
    )
    wrong_step = registry.resolve(
        _request(operation_id="wrong", origin="planned", step_id="other"),
        context,
    )

    assert isinstance(direct, ToolResult) and direct.status == "blocked"
    assert isinstance(wrong_step, ToolResult) and wrong_step.status == "blocked"

    store = _checkpoint_store(registry)
    first_result = _service(registry, store=store).execute(authorized)
    repeated = registry.resolve(
        _request(operation_id="repeat", origin="planned", step_id="check"),
        context,
    )
    assert isinstance(repeated, AuthorizedExecutor)
    repeated_result = _service(registry, store=store).execute(repeated)
    assert repeated_result == first_result


def test_active_plan_cannot_be_bypassed_by_claiming_direct_origin(tmp_path) -> None:
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = _registry(tmp_path, plan_execution_verifier=verifier)
    _register_engine(registry)
    snapshot = registry.snapshot("engine-capability")
    plan = build_task_plan(
        {
            "plan_id": "plan-direct-bypass",
            "project_id": "project",
            "thread_id": "thread",
            "root_run_id": "root-run",
            "goal": GoalBoundary(objective="Check environment"),
            "steps": [
                PlanStep(
                    step_id="check",
                    title="Check",
                    description="Check exact requested engine",
                    executor_type="basic_tool",
                    capability="engine-capability",
                    tool_name="check_engine_availability",
                    inputs={"engines": ["planned"]},
                    expected_outputs=["availability"],
                    verification=[
                        VerificationCheck(
                            check_id="schema",
                            description="valid output",
                            method="schema",
                            acceptance_rule="matches schema",
                        )
                    ],
                )
            ],
            "execution_snapshot": PlanExecutionSnapshot(
                capability_snapshots=[snapshot],
                registry_snapshot_sha256="d" * 64,
                captured_at="2026-07-13T08:00:00+00:00",
            ),
            "created_at": "2026-07-13T08:00:00+00:00",
        }
    )
    verifier.register_plan(plan)

    direct = registry.resolve(_request(arguments={"engines": ["planned"]}), _context(registry))
    tampered = registry.resolve(
        _request(origin="planned", step_id="check", arguments={"engines": ["tampered"]}),
        _context(
            registry,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            registry_snapshot_sha256="d" * 64,
        ),
    )
    stale_snapshot = registry.resolve(
        _request(origin="planned", step_id="check", arguments={"engines": ["planned"]}),
        _context(
            registry,
            plan_id=plan.plan_id,
            plan_sha256=plan.plan_sha256,
            registry_snapshot_sha256="e" * 64,
        ),
    )

    assert isinstance(direct, ToolResult) and direct.status == "blocked"
    assert isinstance(tampered, ToolResult) and tampered.status == "blocked"
    assert isinstance(stale_snapshot, ToolResult) and stale_snapshot.status == "blocked"


def test_audit_hashes_normalized_arguments_including_defaults(tmp_path) -> None:
    registry = _registry(tmp_path)
    _register_engine(registry, input_model=DefaultEngineInput)
    authorized = registry.resolve(_request(arguments={}), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)

    _service(registry).execute(authorized)

    expected = hashlib.sha256(
        json.dumps(
            {"engines": ["gmat"]},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert registry.audit_records()[-1]["arguments_sha256"] == expected


def test_audit_database_failure_after_execution_is_structured_with_fallback(tmp_path) -> None:
    registry = _registry(tmp_path)
    _register_engine(registry)
    authorized = registry.resolve(_request(), _context(registry))
    assert isinstance(authorized, AuthorizedExecutor)
    broken_database = tmp_path / "broken-audit"
    broken_database.mkdir()
    registry._audit_database_path = broken_database

    result = _service(registry).execute(authorized)

    assert result.status == "failed"
    assert result.recovery_class == "manual_recovery"
    assert result.error is not None and "audit" in result.error.message
    assert (tmp_path / "execution_audit_fallback.jsonl").is_file()

