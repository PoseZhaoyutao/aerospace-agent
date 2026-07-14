from __future__ import annotations

import importlib
import json
import sqlite3
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aerospace_agent.langgraph_agent.agent_core.approval import (
    CapabilityApprovalLedger,
    CapabilityApprovalVerifier,
    approval_signature_payload,
)
from aerospace_agent.langgraph_agent.agent_core.models import WorkflowManifest
from aerospace_agent.langgraph_agent.agent_core.rag_gate import ExecutionRunStore
from aerospace_agent.langgraph_agent.agent_core.workflows import (
    WorkflowRegistry,
    canonical_manifest_sha256,
    canonical_workflow_sha256,
    workflow_approval_digest,
)


def _scheduler_module():
    try:
        return importlib.import_module("aerospace_agent.langgraph_agent.agent_core.scheduler")
    except ModuleNotFoundError as exc:
        pytest.fail(f"internal scheduler module is missing: {exc}")


def _body(*, safe: bool = True) -> dict:
    return {
        "workflow_id": "workflow.read_environment",
        "version": "1.0.0",
        "workflow_schema_version": "1.0",
        "input_schema": {
            "type": "object",
            "properties": {
                "engine": {"type": "string"},
                "token": {"type": "string", "x-sensitive": True},
            },
            "required": ["engine"],
            "additionalProperties": False,
        },
        "steps": [
            {
                "step_id": "check",
                "executor_type": "tool",
                "capability_id": "environment",
                "risk_level": "read_only" if safe else "project_write",
                "recovery_class": "read_only" if safe else "manual_recovery",
                "idempotent": safe,
            }
        ],
    }


def _manifest(body: dict, *, safe: bool = True) -> WorkflowManifest:
    data = {
        **body,
        "workflow_sha256": canonical_workflow_sha256(body),
        "manifest_sha256": "0" * 64,
        "approval_record_id": "approval-1",
        "approval_scope": "scheduled_read_only" if safe else "interactive_only",
        "automatable": safe,
    }
    data["manifest_sha256"] = canonical_manifest_sha256(data)
    return WorkflowManifest.model_validate(data)


def _workflow_registry(tmp_path: Path, *, safe: bool = True):
    body = _body(safe=safe)
    manifest = _manifest(body, safe=safe)
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    ledger = CapabilityApprovalLedger(
        tmp_path / "approval.sqlite", trusted_public_keys={"operator": public}
    )
    digest = workflow_approval_digest(manifest)
    ledger.append(
        approval_record_id=manifest.approval_record_id,
        key_id="operator",
        digest=digest,
        signature_b64=b64encode(private.sign(approval_signature_payload(digest))).decode(),
        created_at="2026-07-13T08:00:00+00:00",
    )
    registry = WorkflowRegistry(
        tmp_path / "workflows.sqlite",
        approval_verifier=CapabilityApprovalVerifier(ledger),
    )
    registry.register(manifest=manifest, workflow_body=body)
    return registry, ledger, manifest


def _service(tmp_path: Path, current: list[datetime], *, safe: bool = True):
    module = _scheduler_module()
    registry, ledger, manifest = _workflow_registry(tmp_path, safe=safe)
    runs = ExecutionRunStore(tmp_path / "runs.sqlite", clock=lambda: current[0])
    service = module.SchedulerService(
        tmp_path / "scheduler.sqlite",
        workflow_registry=registry,
        execution_run_store=runs,
        clock=lambda: current[0],
    )
    return service, runs, registry, ledger, manifest


def test_payloads_are_content_addressed_exact_and_due_time_is_aware(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, *_ = _service(tmp_path, now)
    due = now[0] + timedelta(minutes=5)

    first = service.create_reminder(
        project_id="project-1", thread_id="thread-1", due_at=due, message="inspect telemetry"
    )
    second = service.create_reminder(
        project_id="project-1", thread_id="thread-1", due_at=due, message="inspect telemetry"
    )

    assert first.payload_id == second.payload_id
    assert service.payload_count() == 1
    payload = service.get_payload(first.payload_id)
    assert payload.body == {"kind": "reminder", "message": "inspect telemetry"}
    assert payload.payload_sha256 == first.payload_sha256
    assert first.timezone == "UTC"
    with pytest.raises(ValueError, match="timezone-aware"):
        service.create_reminder(
            project_id="project-1",
            thread_id="thread-1",
            due_at=datetime(2026, 7, 13, 9, 0),
            message="invalid",
        )


def test_workflow_job_locks_snapshot_masks_inputs_and_creates_no_rag_run_on_claim(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, runs, _, _, manifest = _service(tmp_path, now)
    job = service.create_workflow(
        project_id="project-1",
        thread_id="thread-1",
        due_at=now[0],
        workflow_id=manifest.workflow_id,
        workflow_version=manifest.version,
        inputs={"engine": "gmat", "token": "secret"},
        max_retries=1,
        retry_delay_seconds=10,
    )

    assert job.workflow_sha256 == manifest.workflow_sha256
    assert job.manifest_sha256 == manifest.manifest_sha256
    assert job.approval_record_id == manifest.approval_record_id
    payload = service.get_payload(job.payload_id)
    assert payload.body["inputs"]["token"] == "secret"
    assert payload.masked_body["inputs"]["token"] == "***"

    claimed = service.claim_due(worker_id="worker-1", lease_seconds=30)
    assert claimed is not None
    assert claimed.status == "claimed" and claimed.attempt == 1
    assert claimed.job_run_id == f"job:{job.job_id}:1"
    run = runs.get(claimed.job_run_id)
    assert run.kind == "scheduled"
    assert run.retrieval_budget == 0 and run.retrieval_state == "unavailable"
    assert service.claim_due(worker_id="worker-2", lease_seconds=30) is None


def test_claim_revalidates_payload_workflow_approval_and_automation_policy(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, _, _, ledger, manifest = _service(tmp_path / "revoked", now)
    revoked = service.create_workflow(
        project_id="project-1",
        thread_id=None,
        due_at=now[0],
        workflow_id=manifest.workflow_id,
        workflow_version=manifest.version,
        inputs={"engine": "gmat"},
    )
    ledger.revoke(
        approval_record_id=manifest.approval_record_id,
        revoked_at="2026-07-13T08:00:00+00:00",
        reason="test revocation",
    )

    assert service.claim_due(worker_id="worker-1") is None
    assert service.get_job(revoked.job_id).status == "blocked"

    tampered_service, *_ = _service(tmp_path / "tampered", now)
    tampered = tampered_service.create_reminder(
        project_id="project-1", thread_id=None, due_at=now[0], message="original"
    )
    with sqlite3.connect(tmp_path / "tampered" / "scheduler.sqlite") as connection:
        connection.execute(
            "UPDATE scheduled_job_payloads SET body_json = ? WHERE payload_id = ?",
            (json.dumps({"kind": "reminder", "message": "changed"}), tampered.payload_id),
        )
    assert tampered_service.claim_due(worker_id="worker-1") is None
    assert tampered_service.get_job(tampered.job_id).status == "blocked"

    unsafe_service, _, _, _, unsafe = _service(tmp_path / "unsafe", now, safe=False)
    with pytest.raises(ValueError, match="automatable|read-only"):
        unsafe_service.create_workflow(
            project_id="project-1",
            thread_id=None,
            due_at=now[0],
            workflow_id=unsafe.workflow_id,
            workflow_version=unsafe.version,
            inputs={"engine": "gmat"},
        )


def test_atomic_optimistic_cancel_and_safe_point_recovery_class(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, *_ = _service(tmp_path, now)
    future = service.create_reminder(
        project_id="project-1",
        thread_id=None,
        due_at=now[0] + timedelta(hours=1),
        message="future",
    )
    assert service.cancel(future.job_id, expected_version=future.version + 1) is None
    cancelled = service.cancel(future.job_id, expected_version=future.version)
    assert cancelled is not None and cancelled.status == "cancelled"

    claimed_job = service.create_reminder(
        project_id="project-1", thread_id=None, due_at=now[0], message="claimed"
    )
    claimed = service.claim_due(worker_id="worker-1")
    assert claimed is not None and claimed.job_id == claimed_job.job_id
    requested = service.cancel(claimed.job_id, expected_version=claimed.version)
    assert requested is not None and requested.status == "cancel_requested"
    manual = service.honor_cancel(
        requested.job_id,
        expected_version=requested.version,
        worker_id="worker-1",
        interruptible=False,
    )
    assert manual.status == "failed"
    assert manual.recovery_class == "manual_recovery"


def test_retry_is_only_for_locked_safe_workflows_and_attempt_ids_increment(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, runs, _, _, manifest = _service(tmp_path, now)
    created = service.create_workflow(
        project_id="project-1",
        thread_id=None,
        due_at=now[0],
        workflow_id=manifest.workflow_id,
        workflow_version=manifest.version,
        inputs={"engine": "gmat"},
        max_retries=1,
        retry_delay_seconds=10,
    )
    claimed = service.claim_due(worker_id="worker-1")
    assert claimed is not None and claimed.job_id == created.job_id
    running = service.mark_running(
        claimed.job_id, expected_version=claimed.version, worker_id="worker-1"
    )
    waiting = service.mark_failed(
        running.job_id,
        expected_version=running.version,
        worker_id="worker-1",
        retryable=True,
    )
    assert waiting.status == "retry_wait"
    assert service.claim_due(worker_id="worker-2") is None

    now[0] += timedelta(seconds=10)
    retried = service.claim_due(worker_id="worker-2")
    assert retried is not None and retried.attempt == 2
    assert retried.job_run_id == f"job:{created.job_id}:2"
    assert runs.get(retried.job_run_id).retrieval_budget == 0

    reminder = service.create_reminder(
        project_id="project-1", thread_id=None, due_at=now[0], message="no retry"
    )
    reminder_claim = service.claim_due(worker_id="worker-3")
    assert reminder_claim is not None and reminder_claim.job_id == reminder.job_id
    reminder_run = service.mark_running(
        reminder.job_id,
        expected_version=reminder_claim.version,
        worker_id="worker-3",
    )
    failed = service.mark_failed(
        reminder.job_id,
        expected_version=reminder_run.version,
        worker_id="worker-3",
        retryable=True,
    )
    assert failed.status == "failed"


def test_overdue_recovery_and_successful_state_machine(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, *_ = _service(tmp_path, now)
    created = service.create_reminder(
        project_id="project-1",
        thread_id=None,
        due_at=now[0] + timedelta(minutes=1),
        message="late reminder",
    )
    now[0] += timedelta(minutes=2)

    assert service.recover_overdue() == 1
    assert service.get_job(created.job_id).status == "overdue"
    claimed = service.claim_due(worker_id="worker-1")
    assert claimed is not None and claimed.status == "claimed"
    running = service.mark_running(
        claimed.job_id, expected_version=claimed.version, worker_id="worker-1"
    )
    succeeded = service.mark_succeeded(
        running.job_id,
        expected_version=running.version,
        worker_id="worker-1",
        last_checkpoint_id="checkpoint-1",
    )
    assert succeeded.status == "succeeded"
    assert succeeded.last_checkpoint_id == "checkpoint-1"


def test_expired_lease_rejects_mark_running(tmp_path: Path) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, *_ = _service(tmp_path, now)
    created = service.create_reminder(
        project_id="project-1", thread_id=None, due_at=now[0], message="expire"
    )
    claimed = service.claim_due(worker_id="worker-1", lease_seconds=1)
    assert claimed is not None and claimed.job_id == created.job_id
    now[0] += timedelta(seconds=2)

    with pytest.raises(RuntimeError, match="lease|transition"):
        service.mark_running(
            claimed.job_id,
            expected_version=claimed.version,
            worker_id="worker-1",
        )


@pytest.mark.parametrize("transition", ["succeeded", "failed"])
def test_expired_running_lease_rejects_terminal_worker_transition(
    tmp_path: Path, transition: str
) -> None:
    now = [datetime(2026, 7, 13, 8, 0, tzinfo=UTC)]
    service, *_ = _service(tmp_path, now)
    created = service.create_reminder(
        project_id="project-1", thread_id=None, due_at=now[0], message="expire"
    )
    claimed = service.claim_due(worker_id="worker-1", lease_seconds=1)
    assert claimed is not None and claimed.job_id == created.job_id
    running = service.mark_running(
        claimed.job_id,
        expected_version=claimed.version,
        worker_id="worker-1",
    )
    now[0] += timedelta(seconds=2)

    with pytest.raises(RuntimeError, match="lease|transition"):
        if transition == "succeeded":
            service.mark_succeeded(
                running.job_id,
                expected_version=running.version,
                worker_id="worker-1",
            )
        else:
            service.mark_failed(
                running.job_id,
                expected_version=running.version,
                worker_id="worker-1",
                retryable=False,
            )

