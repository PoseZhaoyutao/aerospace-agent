from __future__ import annotations

from pathlib import Path

from aerospace_agent.langgraph_agent.agent_core.dag import (
    CanonicalMetadataVocabulary,
    CheckpointedDAGExecutor,
)
from aerospace_agent.langgraph_agent.agent_core.execution import (
    AuthorizedExecutor,
    ExecutionRegistry,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.execution_checkpoints import (
    ExecutionCheckpointStore,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityManifest,
    ContractModel,
    GoalBoundary,
    PlanExecutionSnapshot,
    PlanStep,
    ToolError,
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


def _manifest(*, status: str = "available", risk: str = "read_only") -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="engine-capability",
        version="1.0.0",
        category="basic",
        status=status,
        intents=["environment"],
        tool_names=["check_engine_availability"],
        risk_level=risk,
        source="aerospace_agent.mcp.tools.environment_tools",
    )


def _runtime(tmp_path: Path, *, status: str = "available", risk: str = "read_only"):
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = ExecutionRegistry(
        Path.cwd(),
        audit_database_path=tmp_path / "execution_audit.sqlite",
        plan_execution_verifier=verifier,
    )
    registry.register(
        kind="tool",
        manifest=_manifest(status=status, risk=risk),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=EngineInput,
        entrypoint="aerospace_agent.mcp.tools.environment_tools.check_engine_availability",
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class="read_only",
    )
    service = ExecutionService(
        registry,
        checkpoint_store=ExecutionCheckpointStore(tmp_path / "execution-checkpoints.sqlite"),
    )
    dag = CheckpointedDAGExecutor(
        database_path=tmp_path / "dag.sqlite",
        workspace_root=Path.cwd(),
        registry=registry,
        execution_service=service,
        plan_verifier=verifier,
        metadata_vocabulary=CanonicalMetadataVocabulary(
            quantity_units={"m", "m/s", "km", "km/s"},
            frame_ids={"ICRF", "ITRF"},
            time_systems={"UTC", "TAI", "TT"},
        ),
    )
    return registry, dag


class InterruptingExecutionService(ExecutionService):
    def __init__(self, registry: ExecutionRegistry, checkpoint_store: ExecutionCheckpointStore):
        super().__init__(registry, checkpoint_store=checkpoint_store)
        self.calls = 0

    def execute(self, authorized: AuthorizedExecutor) -> ToolResult:
        self.calls += 1
        return ToolResult(
            status="interrupted",
            error=ToolError(
                code="interrupted",
                message="simulated process interruption",
                recoverability="manual_recovery",
            ),
            audit_id=f"interrupted-{self.calls}",
            operation_id=authorized.operation_id,
            recovery_class="manual_recovery",
        )


def _plan_payload(registry: ExecutionRegistry) -> dict:
    return {
        "plan_id": "plan-1",
        "project_id": "project-1",
        "thread_id": "thread-1",
        "root_run_id": "run-1",
        "goal": GoalBoundary(
            objective="Read environment status",
            in_scope=["read environment"],
            success_criteria=["availability returned"],
        ),
        "steps": [
            PlanStep(
                step_id="step-1",
                title="Check",
                description="Read engine availability",
                executor_type="basic_tool",
                capability="engine-capability",
                tool_name="check_engine_availability",
                inputs={"engines": []},
                expected_outputs=["availability"],
                verification=[
                    VerificationCheck(
                        check_id="schema",
                        description="result is valid",
                        method="schema",
                        acceptance_rule="matches output schema",
                    )
                ],
            )
        ],
        "execution_snapshot": PlanExecutionSnapshot(
            capability_snapshots=[registry.snapshot("engine-capability")],
            registry_snapshot_sha256="c" * 64,
            captured_at="2026-07-13T08:00:00+00:00",
        ),
        "created_at": "2026-07-13T08:00:00+00:00",
    }


def test_dag_checkpoints_before_and_after_and_reuses_exact_idempotency_key(tmp_path) -> None:
    registry, dag = _runtime(tmp_path)
    plan = build_task_plan(_plan_payload(registry))

    first = dag.execute(
        plan,
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )
    audit_count = len(registry.audit_records())
    second = dag.execute(
        plan,
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert first.status == "completed"
    assert second.status == "completed"
    assert second.reused_step_ids == ["step-1"]
    assert len(registry.audit_records()) == audit_count == 1
    checkpoints = dag.list_checkpoints(plan.plan_id)
    assert [item.phase for item in checkpoints] == ["before", "after"]
    assert checkpoints[0].idempotency_key.startswith(f"{plan.plan_id}:step-1:")
    assert first.state.step_states[0].last_checkpoint_id == checkpoints[-1].checkpoint_id


def test_snapshot_or_namespace_mismatch_is_invalid_plan_with_zero_dag_side_effects(tmp_path) -> None:
    registry, dag = _runtime(tmp_path)
    payload = _plan_payload(registry)
    payload["execution_snapshot"] = payload["execution_snapshot"].model_copy(
        update={
            "capability_snapshots": [
                registry.snapshot("engine-capability").model_copy(
                    update={"manifest_sha256": "f" * 64}
                )
            ]
        }
    )
    plan = build_task_plan(payload)

    mismatch = dag.execute(
        plan,
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )
    wrong_thread = dag.execute(
        build_task_plan(_plan_payload(registry)),
        project_id="project-1",
        thread_id="other-thread",
        root_run_id="run-1",
    )

    assert mismatch.status == wrong_thread.status == "invalid_plan"
    assert dag.list_checkpoints("plan-1") == []
    assert registry.audit_records() == []


def test_unavailable_capability_is_invalid_before_checkpoint_or_audit(tmp_path) -> None:
    registry, dag = _runtime(tmp_path, status="unavailable")
    outcome = dag.execute(
        build_task_plan(_plan_payload(registry)),
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert outcome.status == "invalid_plan"
    assert "unavailable" in (outcome.error or "")
    assert dag.list_checkpoints("plan-1") == []
    assert registry.audit_records() == []


def test_cycle_is_rejected_without_checkpoint_or_execution(tmp_path) -> None:
    registry, dag = _runtime(tmp_path)
    payload = _plan_payload(registry)
    first = payload["steps"][0].model_copy(update={"dependencies": ["step-2"]})
    second = payload["steps"][0].model_copy(
        update={"step_id": "step-2", "dependencies": ["step-1"]}
    )
    raw = {
        key: (value.model_dump(mode="json") if hasattr(value, "model_dump") else value)
        for key, value in payload.items()
    }
    raw["steps"] = [first.model_dump(mode="json"), second.model_dump(mode="json")]
    raw["plan_sha256"] = "0" * 64

    outcome = dag.execute(
        raw,
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert outcome.status == "invalid_plan"
    assert "acyclic" in (outcome.error or "")
    assert dag.list_checkpoints("plan-1") == []
    assert registry.audit_records() == []


def test_cross_domain_metadata_and_conversion_verification_are_preflighted(tmp_path) -> None:
    registry, dag = _runtime(tmp_path)
    payload = _plan_payload(registry)
    step_1 = payload["steps"][0].model_copy(
        update={
            "executor_type": "domain_subgraph",
            "tool_name": None,
            "domain_subgraph": "simulation",
        }
    )
    step_2 = step_1.model_copy(
        update={
            "step_id": "step-2",
            "domain_subgraph": "orbit_design",
            "dependencies": ["step-1"],
        }
    )
    payload["steps"] = [step_1, step_2]
    payload["handoffs"] = [
        {
            "source_step_id": "step-1",
            "target_step_id": "step-2",
            "source_domain": "simulation",
            "target_domain": "orbit_design",
            "reason": "transfer state",
            "required_inputs": ["position"],
            "expected_outputs": ["position"],
            "source_output_mapping": {"position": "position"},
            "target_input_mapping": {"position": "position"},
            "source_metadata": {
                "quantity_units": {"position": "furlong"},
                "frame_id": "CUSTOM",
                "time_system": "LOCAL",
                "epoch_field": "epoch",
            },
            "target_metadata": {
                "quantity_units": {"position": "km"},
                "frame_id": "ICRF",
                "time_system": "UTC",
                "epoch_field": "epoch",
            },
            "conversion": {
                "converter_capability": "engine-capability",
                "input_mapping": {"position": "position"},
                "output_mapping": {"position": "position"},
                "validation_check_id": "missing-check",
            },
        }
    ]

    outcome = dag.execute(
        build_task_plan(payload),
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert outcome.status == "invalid_plan"
    assert "canonical" in (outcome.error or "")
    assert dag.list_checkpoints("plan-1") == []
    assert registry.audit_records() == []


def test_high_risk_plan_step_cannot_omit_confirmation_declaration(tmp_path) -> None:
    registry, dag = _runtime(tmp_path)
    payload = _plan_payload(registry)
    raw_step = payload["steps"][0].model_dump(mode="json")
    raw_step.update({"risk_level": "high_risk", "requires_confirmation": False})
    payload["steps"] = [raw_step]

    outcome = dag.execute(
        payload,
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert outcome.status == "invalid_plan"
    assert "confirmation" in (outcome.error or "")
    assert dag.list_checkpoints("plan-1") == []
    assert registry.audit_records() == []


def test_high_risk_step_without_grant_stops_at_confirmation_checkpoint(tmp_path) -> None:
    registry, dag = _runtime(tmp_path, risk="high_risk")
    payload = _plan_payload(registry)
    payload["steps"] = [
        payload["steps"][0].model_copy(
            update={"risk_level": "high_risk", "requires_confirmation": True}
        )
    ]

    outcome = dag.execute(
        build_task_plan(payload),
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert outcome.status == "blocked"
    assert outcome.step_results["step-1"].error is not None
    assert outcome.step_results["step-1"].error.code == "confirmation_required"
    assert [item.phase for item in dag.list_checkpoints("plan-1")] == ["before", "after"]


def test_interrupted_write_is_not_retried_until_audited_state_inspection(tmp_path) -> None:
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = ExecutionRegistry(
        Path.cwd(),
        audit_database_path=tmp_path / "execution_audit.sqlite",
        plan_execution_verifier=verifier,
    )
    registry.register(
        kind="tool",
        manifest=_manifest(),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=EngineInput,
        entrypoint="aerospace_agent.mcp.tools.environment_tools.check_engine_availability",
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class="read_only",
    )
    service = InterruptingExecutionService(
        registry,
        ExecutionCheckpointStore(tmp_path / "execution-checkpoints.sqlite"),
    )
    dag = CheckpointedDAGExecutor(
        database_path=tmp_path / "dag.sqlite",
        workspace_root=Path.cwd(),
        registry=registry,
        execution_service=service,
        plan_verifier=verifier,
        metadata_vocabulary=CanonicalMetadataVocabulary(),
    )
    payload = _plan_payload(registry)
    payload["steps"] = [payload["steps"][0].model_copy(update={"risk_level": "project_write"})]
    plan = build_task_plan(payload)

    first = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )
    second = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )

    assert first.status == second.status == "interrupted"
    assert service.calls == 1
    assert [item.phase for item in dag.list_checkpoints(plan.plan_id)] == [
        "before",
        "after",
        "inspection_required",
    ]


def test_cross_domain_conversion_requires_named_executable_verification(tmp_path) -> None:
    registry, dag = _runtime(tmp_path)
    payload = _plan_payload(registry)
    source = payload["steps"][0].model_copy(
        update={
            "executor_type": "domain_subgraph",
            "tool_name": None,
            "domain_subgraph": "simulation",
        }
    )
    target = source.model_copy(
        update={
            "step_id": "step-2",
            "domain_subgraph": "orbit_design",
            "dependencies": ["step-1"],
        }
    )
    payload["steps"] = [source, target]
    payload["handoffs"] = [
        {
            "source_step_id": "step-1",
            "target_step_id": "step-2",
            "source_domain": "simulation",
            "target_domain": "orbit_design",
            "reason": "transfer state",
            "required_inputs": ["position"],
            "expected_outputs": ["position"],
            "source_output_mapping": {"position": "position"},
            "target_input_mapping": {"position": "position"},
            "source_metadata": {
                "quantity_units": {"position": "m"},
                "frame_id": "ITRF",
                "time_system": "UTC",
                "epoch_field": "epoch",
            },
            "target_metadata": {
                "quantity_units": {"position": "km"},
                "frame_id": "ICRF",
                "time_system": "TAI",
                "epoch_field": "epoch",
            },
            "conversion": {
                "converter_capability": "engine-capability",
                "input_mapping": {"position": "position"},
                "output_mapping": {"position": "position"},
                "validation_check_id": "not-in-plan",
            },
        }
    ]

    outcome = dag.execute(
        build_task_plan(payload),
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
    )

    assert outcome.status == "invalid_plan"
    assert "verification check" in (outcome.error or "")
    assert dag.list_checkpoints("plan-1") == []
    assert registry.audit_records() == []

