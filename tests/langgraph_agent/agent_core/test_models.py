from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityManifest,
    CapabilitySnapshot,
    ConfirmationGrant,
    CrossDomainHandoff,
    DomainDataMetadata,
    GoalBoundary,
    HandoffConversion,
    PlanExecutionSnapshot,
    PlanExecutionState,
    PlanStep,
    PlanStepExecutionState,
    TaskPlan,
    ToolCall,
    ToolError,
    ToolResult,
    VerificationCheck,
    WorkflowSnapshot,
)


SHA = "a" * 64


def _step(**overrides):
    data = {
        "step_id": "s1",
        "title": "Read input",
        "description": "Read one project file",
        "executor_type": "basic_tool",
        "capability": "core.files",
        "tool_name": "file.read",
        "expected_outputs": ["content"],
        "verification": [
            VerificationCheck(
                check_id="v1",
                description="Content was returned",
                method="schema",
                acceptance_rule="result contains content",
            )
        ],
    }
    data.update(overrides)
    return PlanStep.model_validate(data)


def _plan(*, steps=None, **overrides):
    data = {
        "plan_id": "p1",
        "project_id": "project-1",
        "thread_id": "thread-1",
        "root_run_id": "run-1",
        "goal": GoalBoundary(objective="Read a file"),
        "steps": steps or [_step()],
        "execution_snapshot": PlanExecutionSnapshot(
            capability_snapshots=[
                CapabilitySnapshot(
                    capability_id="core.files",
                    version="1.0.0",
                    manifest_sha256=SHA,
                    adapter_sha256=SHA,
                )
            ],
            registry_snapshot_sha256=SHA,
            captured_at="2026-07-13T12:00:00+08:00",
        ),
        "created_at": "2026-07-13T12:00:00+08:00",
        "plan_sha256": SHA,
    }
    data.update(overrides)
    return TaskPlan.model_validate(data)


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError):
        CapabilityManifest(
            capability_id="core.files",
            version="1.0.0",
            category="basic",
            status="available",
            intents=["read file"],
            risk_level="read_only",
            source="current_workspace",
            unexpected=True,
        )


def test_tool_result_status_and_error_must_be_consistent():
    success = {
        "status": "success",
        "result": {},
        "audit_id": "audit",
        "operation_id": "operation",
        "recovery_class": "read_only",
    }
    assert ToolResult.model_validate(success).error is None
    with pytest.raises(ValidationError):
        ToolResult.model_validate(
            {
                **success,
                "error": ToolError(
                    code="failed", message="contradiction", recoverability="manual_recovery"
                ),
            }
        )
    with pytest.raises(ValidationError):
        ToolResult.model_validate({**success, "status": "failed"})
    with pytest.raises(ValidationError):
        ToolResult.model_validate(
            {
                **success,
                "status": "timeout",
                "error": ToolError(
                    code="failed", message="wrong code", recoverability="retryable"
                ),
            }
        )


@pytest.mark.parametrize(
    ("executor_type", "fields"),
    [
        ("basic_tool", {"tool_name": "file.read"}),
        ("space_basic_tool", {"tool_name": "space.validate_state"}),
        ("workflow", {"workflow_id": "wf", "workflow_version": "1"}),
        ("domain_subgraph", {"domain_subgraph": "simulation"}),
        ("capability_builder", {"capability_gap_id": "gap-1"}),
        ("human", {"human_instruction": "Confirm values"}),
    ],
)
def test_plan_step_accepts_exact_discriminated_executor(executor_type, fields):
    references = {"tool_name": None}
    references.update(fields)
    step = _step(
        executor_type=executor_type,
        capability=f"cap.{executor_type}",
        **references,
    )
    assert step.executor_type == executor_type


def test_plan_step_rejects_missing_or_extra_executor_reference():
    with pytest.raises(ValidationError):
        _step(executor_type="workflow", tool_name=None)
    with pytest.raises(ValidationError):
        _step(workflow_id="wf", workflow_version="1")


def test_high_risk_step_requires_confirmation():
    with pytest.raises(ValidationError):
        _step(risk_level="high_risk", requires_confirmation=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_outputs": []},
        {"verification": []},
    ],
)
def test_plan_step_requires_expected_outputs_and_verification(overrides):
    with pytest.raises(ValidationError):
        _step(**overrides)


def test_task_plan_rejects_duplicate_steps_and_dependency_cycles():
    duplicate = [_step(), _step()]
    with pytest.raises(ValidationError):
        _plan(steps=duplicate)

    cyclic = [
        _step(step_id="s1", dependencies=["s2"]),
        _step(step_id="s2", dependencies=["s1"]),
    ]
    with pytest.raises(ValidationError):
        _plan(steps=cyclic)


def test_hash_fields_require_a_full_sha256_but_accept_hex_case():
    data = _plan().model_dump()
    data["plan_sha256"] = "ABC"
    with pytest.raises(ValidationError):
        TaskPlan.model_validate(data)

    data["plan_sha256"] = "A" * 64
    assert TaskPlan.model_validate(data).plan_sha256 == "A" * 64


def test_task_plan_requires_snapshots_for_referenced_capabilities_and_workflows():
    with pytest.raises(ValidationError):
        _plan(
            steps=[_step(capability="core.other")],
        )

    workflow_step = _step(
        executor_type="workflow",
        capability="core.files",
        tool_name=None,
        workflow_id="wf",
        workflow_version="1",
    )
    with pytest.raises(ValidationError):
        _plan(steps=[workflow_step])

    plan = _plan(
        steps=[workflow_step],
        execution_snapshot=PlanExecutionSnapshot(
            capability_snapshots=[
                CapabilitySnapshot(
                    capability_id="core.files",
                    version="1.0.0",
                    manifest_sha256=SHA,
                    adapter_sha256=SHA,
                )
            ],
            workflow_snapshots=[
                WorkflowSnapshot(
                    workflow_id="wf",
                    version="1",
                    workflow_sha256=SHA,
                    manifest_sha256=SHA,
                    approval_record_id="approval-1",
                )
            ],
            registry_snapshot_sha256=SHA,
            captured_at="2026-07-13T12:00:00+08:00",
        ),
    )
    assert plan.steps[0].workflow_id == "wf"


def test_tool_and_confirmation_contracts_use_spec_literals():
    call = ToolCall(
        tool_name="file.read",
        arguments={"path": "README.md"},
        run_id="run-1",
        operation_id="op-1",
    )
    error = ToolError(
        code="confirmation_required",
        message="confirmation needed",
        recoverability="not_applicable",
    )
    result = ToolResult(
        status="blocked",
        error=error,
        audit_id="audit-1",
        operation_id=call.operation_id,
        recovery_class="read_only",
    )
    grant = ConfirmationGrant(
        confirmation_id="confirm-1",
        project_id="project-1",
        thread_id="thread-1",
        root_run_id=call.run_id,
        operation_id=call.operation_id,
        action_hash=SHA,
        issued_at="2026-07-13T12:00:00+08:00",
        expires_at="2026-07-13T12:10:00+08:00",
    )
    assert result.error.code == "confirmation_required"
    assert grant.used_at is None


def test_task_plan_contains_no_mutable_step_status():
    payload = _plan().model_dump()
    payload["steps"][0]["status"] = "running"
    with pytest.raises(ValidationError):
        TaskPlan.model_validate(payload)

    state = PlanExecutionState(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="p1",
        plan_sha256=SHA,
        step_states=[
            PlanStepExecutionState(
                step_id="s1",
                status="running",
                attempts=1,
            )
        ],
        updated_at="2026-07-13T12:01:00+08:00",
    )
    assert state.step_states[0].status == "running"


def test_workflow_snapshot_requires_immutable_hashes():
    with pytest.raises(ValidationError):
        WorkflowSnapshot(
            workflow_id="wf",
            version="1",
            workflow_sha256="bad",
            manifest_sha256=SHA,
            approval_record_id="approval-1",
        )


def test_task_plan_and_nested_plan_content_are_immutable():
    plan = _plan()
    with pytest.raises(ValidationError):
        plan.plan_id = "changed"
    with pytest.raises(ValidationError):
        plan.steps[0].risk_level = "high_risk"
    with pytest.raises(TypeError):
        plan.steps[0].inputs["path"] = "changed"
    with pytest.raises((AttributeError, TypeError)):
        plan.steps.append(_step(step_id="s2"))


def test_large_legal_dag_does_not_depend_on_python_recursion_limit():
    steps = [
        _step(
            step_id=f"s{index}",
            dependencies=[f"s{index - 1}"] if index else [],
        )
        for index in range(1100)
    ]
    plan = _plan(steps=list(reversed(steps)))
    assert len(plan.steps) == 1100


def test_contracts_do_not_coerce_negative_attempt_counts():
    with pytest.raises(ValidationError):
        PlanStepExecutionState(step_id="s1", status="pending", attempts="-2")
    with pytest.raises(ValidationError):
        PlanStepExecutionState(step_id="s1", status="pending", attempts=-2)


def test_plan_inputs_reject_non_json_mutable_values():
    with pytest.raises(ValidationError):
        _step(inputs={"mutable": {"not", "json"}})
    with pytest.raises(ValidationError):
        _step(inputs={"mutable": bytearray(b"not-json")})


def test_cross_domain_handoff_requires_conversion_when_metadata_differs():
    source = DomainDataMetadata(
        quantity_units={"position": "m"},
        frame_id="GCRF",
        time_system="UTC",
        epoch_field="epoch",
    )
    target = DomainDataMetadata(
        quantity_units={"position": "km"},
        frame_id="GCRF",
        time_system="UTC",
        epoch_field="epoch",
    )
    data = {
        "source_step_id": "s1",
        "target_step_id": "s2",
        "source_domain": "orbit_design",
        "target_domain": "control_planning",
        "reason": "Transfer trajectory state",
        "source_metadata": source,
        "target_metadata": target,
    }
    with pytest.raises(ValidationError):
        CrossDomainHandoff.model_validate(data)

    data["conversion"] = HandoffConversion(
        converter_capability="space.convert_orbit_representation",
        input_mapping={"position": "position"},
        output_mapping={"position": "position"},
        validation_check_id="v-convert",
    )
    assert CrossDomainHandoff.model_validate(data).conversion is not None


def test_task_plan_validates_handoff_step_references_and_converter_snapshot():
    metadata = DomainDataMetadata(quantity_units={"position": "m"}, frame_id="GCRF")
    missing_step_handoff = CrossDomainHandoff(
        source_step_id="missing",
        target_step_id="s2",
        source_domain="orbit_design",
        target_domain="control_planning",
        reason="Transfer state",
        source_metadata=metadata,
        target_metadata=metadata,
    )
    steps = [_step(step_id="s1"), _step(step_id="s2", dependencies=["s1"])]
    with pytest.raises(ValidationError):
        _plan(steps=steps, handoffs=[missing_step_handoff])

    converted = CrossDomainHandoff(
        source_step_id="s1",
        target_step_id="s2",
        source_domain="orbit_design",
        target_domain="control_planning",
        reason="Convert state units",
        source_metadata=metadata,
        target_metadata=DomainDataMetadata(quantity_units={"position": "km"}, frame_id="GCRF"),
        conversion=HandoffConversion(
            converter_capability="space.convert_orbit_representation",
            input_mapping={"position": "position"},
            output_mapping={"position": "position"},
            validation_check_id="v1",
        ),
    )
    with pytest.raises(ValidationError):
        _plan(steps=steps, handoffs=[converted])

    snapshot = PlanExecutionSnapshot(
        capability_snapshots=[
            CapabilitySnapshot(
                capability_id="core.files",
                version="1.0.0",
                manifest_sha256=SHA,
                adapter_sha256=SHA,
            ),
            CapabilitySnapshot(
                capability_id="space.convert_orbit_representation",
                version="1.0.0",
                manifest_sha256=SHA,
                adapter_sha256=SHA,
            ),
        ],
        registry_snapshot_sha256=SHA,
        captured_at="2026-07-13T12:00:00+08:00",
    )
    assert _plan(steps=steps, handoffs=[converted], execution_snapshot=snapshot).handoffs
