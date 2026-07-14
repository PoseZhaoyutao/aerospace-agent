from __future__ import annotations

from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilitySnapshot,
    GoalBoundary,
    PlanExecutionSnapshot,
    PlanStep,
    VerificationCheck,
)
from aerospace_agent.langgraph_agent.agent_core.planning import (
    PlanExecutionVerifier,
    build_task_plan,
    compute_task_plan_sha256,
)


def _payload(**overrides):
    data = {
        "plan_id": "plan-1",
        "project_id": "project-1",
        "thread_id": "thread-1",
        "root_run_id": "run-1",
        "goal": GoalBoundary(objective="Read environment status"),
        "steps": [
            PlanStep(
                step_id="step-1",
                title="Check",
                description="Read engine availability",
                executor_type="basic_tool",
                capability="engine-capability",
                tool_name="check_engine_availability",
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
    data.update(overrides)
    return data


def test_build_plan_computes_canonical_hash_and_verifier_persists_exact_binding(tmp_path) -> None:
    plan = build_task_plan(_payload())
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")

    verifier.register_plan(plan)
    restarted = PlanExecutionVerifier(tmp_path / "plans.sqlite")

    assert plan.plan_sha256 == compute_task_plan_sha256(plan)
    assert restarted.verify(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=plan.plan_sha256,
        step_id="step-1",
        kind="tool",
        capability_id="engine-capability",
        executor_name="check_engine_availability",
        arguments={},
        capability_snapshot=plan.execution_snapshot.capability_snapshots[0],
        workflow_snapshot=None,
        registry_snapshot_sha256=plan.execution_snapshot.registry_snapshot_sha256,
    )


def test_plan_binding_rejects_cross_namespace_step_or_executor_reuse(tmp_path) -> None:
    plan = build_task_plan(_payload())
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    verifier.register_plan(plan)
    base = {
        "project_id": "project-1",
        "thread_id": "thread-1",
        "root_run_id": "run-1",
        "plan_id": "plan-1",
        "plan_sha256": plan.plan_sha256,
        "step_id": "step-1",
        "kind": "tool",
        "capability_id": "engine-capability",
        "executor_name": "check_engine_availability",
        "arguments": {},
        "capability_snapshot": plan.execution_snapshot.capability_snapshots[0],
        "workflow_snapshot": None,
        "registry_snapshot_sha256": plan.execution_snapshot.registry_snapshot_sha256,
    }

    for changed in (
        {"project_id": "other"},
        {"thread_id": "other"},
        {"root_run_id": "other"},
        {"step_id": "other"},
        {"capability_id": "other"},
        {"executor_name": "other"},
        {"kind": "workflow"},
    ):
        assert not verifier.verify(**{**base, **changed})


def test_plan_is_immutable_and_hash_drift_is_rejected(tmp_path) -> None:
    plan = build_task_plan(_payload())
    verifier = PlanExecutionVerifier(tmp_path / "plans.sqlite")
    verifier.register_plan(plan)

    changed = build_task_plan(_payload(goal=GoalBoundary(objective="Different")))
    try:
        verifier.register_plan(changed)
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("same plan_id with changed content was accepted")


def test_schema_version_is_explicit(tmp_path) -> None:
    assert PlanExecutionVerifier(tmp_path / "plans.sqlite").schema_version() == 1
