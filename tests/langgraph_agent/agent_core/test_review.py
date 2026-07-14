from __future__ import annotations

from pathlib import Path

from aerospace_agent.langgraph_agent.agent_core.dag import (
    CanonicalMetadataVocabulary,
    CheckpointedDAGExecutor,
)
from aerospace_agent.langgraph_agent.agent_core.execution import (
    ExecutionRegistry,
    ExecutionService,
)
from aerospace_agent.langgraph_agent.agent_core.execution_checkpoints import ExecutionCheckpointStore
from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityManifest,
    CapabilitySnapshot,
    CheckResult,
    DomainReview,
    GoalBoundary,
    ContractModel,
    PlanExecutionSnapshot,
    PlanExecutionState,
    PlanStep,
    PlanStepExecutionState,
    ToolError,
    ToolResult,
    VerificationCheck,
)
from aerospace_agent.langgraph_agent.agent_core.planning import PlanExecutionVerifier, build_task_plan
from aerospace_agent.langgraph_agent.agent_core.review import ReviewAssessment, ReviewService
from aerospace_agent.mcp.tools.environment_tools import check_engine_availability


class EngineInput(ContractModel):
    engines: list[str] | None = None


def _plan():
    return build_task_plan(
        {
            "plan_id": "plan-1",
            "project_id": "project-1",
            "thread_id": "thread-1",
            "root_run_id": "run-1",
            "goal": GoalBoundary(
                objective="Produce verified result",
                in_scope=["environment"],
                hard_constraints=["read only"],
                success_criteria=["verified output"],
            ),
            "steps": [
                PlanStep(
                    step_id="step-1",
                    title="Check",
                    description="read",
                    executor_type="basic_tool",
                    capability="engine-capability",
                    tool_name="check_engine_availability",
                    inputs={"engines": []},
                    expected_outputs=["availability"],
                    verification=[
                        VerificationCheck(
                            check_id="schema",
                            description="valid",
                            method="schema",
                            acceptance_rule="valid",
                        )
                    ],
                )
            ],
            "execution_snapshot": PlanExecutionSnapshot(
                capability_snapshots=[
                    CapabilitySnapshot(
                        capability_id="engine-capability",
                        version="1.0.0",
                        manifest_sha256="a" * 64,
                        adapter_sha256="b" * 64,
                    )
                ],
                registry_snapshot_sha256="c" * 64,
                captured_at="2026-07-13T08:00:00+00:00",
            ),
            "created_at": "2026-07-13T08:00:00+00:00",
        }
    )


def _state(plan, **step_overrides):
    step = {
        "step_id": "step-1",
        "status": "completed",
        "attempts": 1,
        "last_input_hash": "d" * 64,
        "last_checkpoint_id": "checkpoint-1",
    }
    step.update(step_overrides)
    return PlanExecutionState(
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
        plan_id=plan.plan_id,
        plan_sha256=plan.plan_sha256,
        step_states=[PlanStepExecutionState(**step)],
        updated_at="2026-07-13T08:01:00+00:00",
    )


def _success_result() -> ToolResult:
    return ToolResult(
        status="success",
        result={"availability": True},
        audit_id="audit-1",
        operation_id="operation-1",
        recovery_class="read_only",
    )


def _assessment(**overrides) -> ReviewAssessment:
    values = {
        "goal_satisfied": True,
        "boundary_compliant": True,
        "constraints_satisfied": True,
        "evidence_sufficient": True,
        "tool_execution_safe": True,
        "checks": [
            CheckResult(
                check_id="schema",
                passed=True,
                severity="error",
                message="validated",
                evidence_refs=["audit-1"],
            )
        ],
        "verified_claims": ["availability was returned"],
        "confidence": 0.95,
    }
    values.update(overrides)
    return ReviewAssessment(**values)


def test_review_passes_only_with_exact_identity_and_all_completion_gates(tmp_path) -> None:
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    registry = ExecutionRegistry(
        Path.cwd(),
        audit_database_path=tmp_path / "audit.sqlite",
        plan_execution_verifier=verifier,
    )
    registry.register(
        kind="tool",
        manifest=CapabilityManifest(
            capability_id="engine-capability",
            version="1.0.0",
            category="basic",
            status="available",
            intents=["environment"],
            tool_names=["check_engine_availability"],
            risk_level="read_only",
            source="aerospace_agent.mcp.tools.environment_tools",
        ),
        executor_name="check_engine_availability",
        handler=check_engine_availability,
        input_model=EngineInput,
        entrypoint="aerospace_agent.mcp.tools.environment_tools.check_engine_availability",
        adapter_path=Path("aerospace_agent/mcp/tools/environment_tools.py"),
        recovery_class="read_only",
    )
    raw = _plan().model_dump(mode="python")
    raw.pop("plan_sha256")
    raw["execution_snapshot"] = PlanExecutionSnapshot(
        capability_snapshots=[registry.snapshot("engine-capability")],
        registry_snapshot_sha256="c" * 64,
        captured_at="2026-07-13T08:00:00+00:00",
    )
    plan = build_task_plan(raw)
    dag = CheckpointedDAGExecutor(
        database_path=tmp_path / "dag.sqlite",
        workspace_root=Path.cwd(),
        registry=registry,
        execution_service=ExecutionService(
            registry,
            checkpoint_store=ExecutionCheckpointStore(tmp_path / "execution-checkpoints.sqlite"),
        ),
        plan_verifier=verifier,
        metadata_vocabulary=CanonicalMetadataVocabulary(),
    )
    outcome = dag.execute(
        plan,
        project_id=plan.project_id,
        thread_id=plan.thread_id,
        root_run_id=plan.root_run_id,
    )
    assert outcome.state is not None
    assert outcome.status == "completed", (
        [item.status for item in outcome.state.step_states],
        {key: value.status for key, value in outcome.step_results.items()},
    )

    assessment = _assessment(
        checks=[
            CheckResult(
                check_id="schema",
                passed=True,
                severity="error",
                message="validated",
                evidence_refs=[outcome.step_results["step-1"].audit_id],
            )
        ]
    )
    review = ReviewService(dag).review(
        plan=plan,
        state=outcome.state,
        step_results=outcome.step_results,
        assessment=assessment,
    )

    assert review.status == "passed", (
        review.unresolved_items,
        review.goal_satisfied,
        review.boundary_compliant,
        review.constraints_satisfied,
        review.checkpoint_valid,
        review.evidence_sufficient,
        review.tool_execution_safe,
    )
    assert review.recommended_action == "respond"
    assert (review.project_id, review.thread_id, review.root_run_id) == (
        plan.project_id,
        plan.thread_id,
        plan.root_run_id,
    )
    assert review.plan_sha256 == plan.plan_sha256


def test_mismatched_state_identity_fails_closed_instead_of_reviewing_other_run() -> None:
    plan = _plan()
    state = _state(plan).model_copy(update={"thread_id": "other-thread"})

    review = ReviewService().review(
        plan=plan,
        state=state,
        step_results={"step-1": _success_result()},
        assessment=_assessment(),
    )

    assert review.status == "failed"
    assert not review.checkpoint_valid
    assert review.recommended_action == "stop"
    assert "identity" in review.unresolved_items[0]


def test_unpersisted_but_plausible_state_cannot_receive_passing_review() -> None:
    plan = _plan()
    review = ReviewService().review(
        plan=plan,
        state=_state(plan),
        step_results={"step-1": _success_result()},
        assessment=_assessment(),
    )

    assert review.status == "partial"
    assert not review.checkpoint_valid
    assert review.recommended_action == "replan"


def test_partial_or_unsupported_work_cannot_be_declared_complete() -> None:
    plan = _plan()
    review = ReviewService().review(
        plan=plan,
        state=_state(plan, status="interrupted", last_checkpoint_id="checkpoint-interrupted"),
        step_results={},
        assessment=_assessment(
            evidence_sufficient=False,
            unresolved_items=["step interrupted"],
            unsupported_claims=["mission result is correct"],
        ),
    )

    assert review.status == "partial"
    assert not review.goal_satisfied
    assert review.recommended_action == "retry"
    assert review.unsupported_claims == ["mission result is correct"]


def test_failed_domain_or_boundary_review_cannot_pass() -> None:
    plan = _plan()
    domain_review = DomainReview(
        domain="orbit_design",
        status="failed",
        validator="orbit-validator",
        checks=[
            CheckResult(
                check_id="frame",
                passed=False,
                severity="critical",
                message="frame mismatch",
            )
        ],
    )
    review = ReviewService().review(
        plan=plan,
        state=_state(plan),
        step_results={"step-1": _success_result()},
        assessment=_assessment(
            boundary_compliant=False,
            domain_reviews=[domain_review],
        ),
    )

    assert review.status == "failed"
    assert review.recommended_action == "rollback"
    assert not review.boundary_compliant


def test_confirmation_block_produces_needs_confirmation_not_success() -> None:
    plan = _plan()
    blocked = ToolResult(
        status="blocked",
        error=ToolError(
            code="confirmation_required",
            message="confirmation required",
            recoverability="not_applicable",
        ),
        audit_id="audit-blocked",
        operation_id="operation-1",
        recovery_class="manual_recovery",
    )
    review = ReviewService().review(
        plan=plan,
        state=_state(plan, status="blocked"),
        step_results={"step-1": blocked},
        assessment=_assessment(goal_satisfied=False),
    )

    assert review.status == "needs_confirmation"
    assert review.recommended_action == "request_confirmation"
    assert not review.goal_satisfied

