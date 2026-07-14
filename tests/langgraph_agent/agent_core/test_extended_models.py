from __future__ import annotations

import pytest
from pydantic import ValidationError

from aerospace_agent.langgraph_agent.agent_core.models import (
    ArtifactRef,
    ArtifactSchemaManifest,
    CapabilityGap,
    CapabilitySnapshot,
    CheckpointRef,
    CheckResult,
    DependencyCandidate,
    DomainArtifact,
    DomainDataMetadata,
    DomainExecutionOutput,
    DomainReview,
    ExecutionRun,
    HandoffRecord,
    ReviewResult,
    SessionMemory,
    WorkflowManifest,
    WorkflowStepPolicy,
)


SHA = "b" * 64


def _check(passed: bool = True) -> CheckResult:
    return CheckResult(
        check_id="check-1",
        passed=passed,
        severity="info" if passed else "error",
        message="checked",
        evidence_refs=["artifact:1"],
    )


def test_passed_review_cannot_contradict_completion_claim():
    data = {
        "review_id": "review-1",
        "project_id": "project-1",
        "thread_id": "thread-1",
        "plan_id": "plan-1",
        "root_run_id": "run-1",
        "plan_sha256": SHA,
        "status": "passed",
        "goal_satisfied": True,
        "boundary_compliant": True,
        "constraints_satisfied": True,
        "checkpoint_valid": True,
        "evidence_sufficient": True,
        "tool_execution_safe": True,
        "checks": [_check()],
        "domain_reviews": [
            DomainReview(
                domain="orbit_design",
                status="passed",
                validator="orbit-review-v1",
                checks=[_check()],
            )
        ],
        "recommended_action": "respond",
        "confidence": 0.9,
        "reviewed_at": "2026-07-13T12:00:00+08:00",
    }
    assert ReviewResult.model_validate(data).status == "passed"

    for field, bad_value in (
        ("goal_satisfied", False),
        ("boundary_compliant", False),
        ("constraints_satisfied", False),
        ("checkpoint_valid", False),
        ("evidence_sufficient", False),
        ("tool_execution_safe", False),
    ):
        invalid = dict(data)
        invalid[field] = bad_value
        with pytest.raises(ValidationError):
            ReviewResult.model_validate(invalid)

    noncritical = dict(data)
    noncritical["unresolved_items"] = ["non-critical follow-up"]
    assert ReviewResult.model_validate(noncritical).status == "passed"

    critical = dict(data)
    critical["checks"] = [
        CheckResult(
            check_id="critical-1",
            passed=False,
            severity="critical",
            message="critical failure",
        )
    ]
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(critical)

    failed_domain = dict(data)
    failed_domain["domain_reviews"] = [
        DomainReview(domain="orbit_design", status="failed", validator="v1")
    ]
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(failed_domain)

    critical_domain_check = dict(data)
    critical_domain_check["domain_reviews"] = [
        DomainReview(
            domain="orbit_design",
            status="passed",
            validator="v1",
            checks=[
                CheckResult(
                    check_id="critical-domain",
                    passed=False,
                    severity="critical",
                    message="critical domain failure",
                )
            ],
        )
    ]
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(critical_domain_check)

    stop = dict(data)
    stop["recommended_action"] = "stop"
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(stop)


def test_execution_run_enforces_scheduled_no_rag_contract():
    scheduled = ExecutionRun(
        root_run_id="job:job-1:1",
        project_id="project-1",
        thread_id=None,
        kind="scheduled",
        retrieval_budget=0,
        retrieval_state="unavailable",
        version=1,
    )
    assert scheduled.retrieval_budget == 0

    with pytest.raises(ValidationError):
        ExecutionRun(
            root_run_id="job:job-1:1",
            project_id="project-1",
            thread_id=None,
            kind="scheduled",
            retrieval_budget=1,
            retrieval_state="available",
            version=1,
        )


def test_user_execution_run_requires_trace_fields_for_rag_lifecycle_state():
    base = {
        "root_run_id": "run-1",
        "project_id": "project-1",
        "thread_id": "thread-1",
        "kind": "user",
        "retrieval_budget": 1,
        "retrieval_state": "claimed",
        "version": 1,
    }
    with pytest.raises(ValidationError):
        ExecutionRun.model_validate(base)

    claimed = dict(base)
    claimed.update(
        retrieval_query_hash=SHA,
        retrieval_claimed_at="2026-07-13T12:00:00+08:00",
        retrieval_lease_expires_at="2026-07-13T12:01:00+08:00",
        retrieval_claimer_id="worker-1",
    )
    assert ExecutionRun.model_validate(claimed).retrieval_state == "claimed"

    in_flight = dict(claimed)
    in_flight["retrieval_state"] = "in_flight"
    with pytest.raises(ValidationError):
        ExecutionRun.model_validate(in_flight)
    in_flight["retrieval_attempt_started_at"] = "2026-07-13T12:00:10+08:00"
    assert ExecutionRun.model_validate(in_flight).retrieval_state == "in_flight"

    for state in ("consumed", "consumed_unknown"):
        incomplete = dict(base)
        incomplete.update(
            retrieval_state=state,
            retrieval_query_hash=SHA,
            retrieval_attempt_started_at="2026-07-13T12:00:10+08:00",
        )
        with pytest.raises(ValidationError):
            ExecutionRun.model_validate(incomplete)

        complete = dict(in_flight)
        complete["retrieval_state"] = state
        assert ExecutionRun.model_validate(complete).retrieval_state == state

    with pytest.raises(ValidationError):
        ExecutionRun(
            root_run_id="job:job-1:1",
            project_id="project-1",
            thread_id=None,
            kind="scheduled",
            retrieval_budget=0,
            retrieval_state="unavailable",
            retrieval_query_hash=SHA,
            retrieval_claimed_at="2026-07-13T12:00:00+08:00",
            version=1,
        )


def test_session_memory_requires_same_namespace_provenance():
    checkpoint = CheckpointRef(
        project_id="project-1",
        thread_id="thread-1",
        checkpoint_id="cp-1",
    )
    memory = SessionMemory(
        memory_id="memory-1",
        project_id="project-1",
        thread_id="thread-1",
        kind="constraint",
        content="Use SI units",
        source_checkpoints=[checkpoint],
        source_content_hash=SHA,
        truth_status="user_stated",
        confidence=1.0,
        created_at="2026-07-13T12:00:00+08:00",
        updated_at="2026-07-13T12:00:00+08:00",
    )
    assert memory.source_checkpoints[0].checkpoint_id == "cp-1"

    invalid = memory.model_dump(mode="json")
    invalid["source_checkpoints"][0]["thread_id"] = "other-thread"
    with pytest.raises(ValidationError):
        SessionMemory.model_validate(invalid)


def test_domain_artifact_contracts_bind_hash_schema_and_checkpoint():
    checkpoint = CheckpointRef(project_id="project-1", thread_id="thread-1", checkpoint_id="cp-1")
    reference = ArtifactRef(
        uri="artifact://sha256/" + SHA,
        sha256=SHA,
        media_type="application/json",
        byte_length=128,
    )
    schema = ArtifactSchemaManifest(
        schema_id="orbit-state",
        schema_version="1.0",
        payload_json_schema={"type": "object"},
        required_quantity_fields=["position"],
        requires_frame=True,
        requires_time_system=True,
        requires_epoch_field=True,
    )
    artifact = DomainArtifact(
        artifact_id="artifact-1",
        payload_ref=reference,
        schema_id=schema.schema_id,
        schema_version=schema.schema_version,
        metadata=DomainDataMetadata(
            quantity_units={"position": "m"},
            frame_id="GCRF",
            time_system="UTC",
            epoch_field="epoch",
        ),
        source_capability=CapabilitySnapshot(
            capability_id="space.propagate_orbit",
            version="1.0.0",
            manifest_sha256=SHA,
            adapter_sha256=SHA,
        ),
        source_checkpoints=[checkpoint],
        provenance=["run:1"],
    )
    output = DomainExecutionOutput(artifacts=[artifact])
    handoff = HandoffRecord(
        handoff_id="handoff-1",
        plan_id="plan-1",
        source_step_id="s1",
        target_step_id="s2",
        source_artifact_ids=[artifact.artifact_id],
        target_artifact_ids=[],
        validation_check_ids=["check-1"],
        checkpoint=checkpoint,
    )
    assert output.artifacts[0].payload_ref.byte_length == 128
    assert handoff.source_artifact_ids == ["artifact-1"]

    with pytest.raises(ValidationError):
        ArtifactRef(uri="artifact://bad", sha256="bad", media_type="text/plain", byte_length=-1)


def test_frozen_model_copy_revalidates_and_refreezes_updates():
    metadata = DomainDataMetadata(quantity_units={"position": "m"}, frame_id="GCRF")
    copied = metadata.model_copy(update={"quantity_units": {"position": "km"}})
    assert copied.quantity_units["position"] == "km"
    with pytest.raises(TypeError):
        copied.quantity_units["position"] = "cm"
    with pytest.raises(ValidationError):
        metadata.model_copy(update={"unknown": "field"})


def test_automatable_workflow_is_only_approved_idempotent_read_only_work():
    safe_step = WorkflowStepPolicy(
        step_id="s1",
        executor_type="tool",
        capability_id="core.files",
        risk_level="read_only",
        recovery_class="read_only",
        idempotent=True,
    )
    manifest = WorkflowManifest(
        workflow_id="wf-1",
        version="1.0.0",
        workflow_schema_version="1.0",
        input_schema={"type": "object"},
        steps=[safe_step],
        workflow_sha256=SHA,
        manifest_sha256=SHA,
        approval_record_id="approval-1",
        approval_scope="scheduled_read_only",
        automatable=True,
    )
    assert manifest.automatable is True

    unsafe = manifest.model_dump(mode="json")
    unsafe["steps"][0]["risk_level"] = "project_write"
    with pytest.raises(ValidationError):
        WorkflowManifest.model_validate(unsafe)


def test_capability_gap_preserves_reviewed_dependency_candidates():
    candidate = DependencyCandidate(
        name="example-lib",
        source_type="package",
        source_uri="https://example.invalid/example-lib",
        version_or_commit="1.2.3",
        license="MIT",
        compatibility="uncertain",
        maintenance_status="unknown",
        risks=["not yet validated"],
    )
    gap = CapabilityGap(
        capability_id="domain.simulation",
        requested_by_step_id="s2",
        description="No verified simulation executor",
        required_contract={"type": "object"},
        candidates=[candidate],
        resolution="defer",
    )
    assert gap.resolution == "defer"
