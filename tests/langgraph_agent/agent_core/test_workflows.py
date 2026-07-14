from __future__ import annotations

from base64 import b64encode
from pathlib import Path
import sqlite3

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aerospace_agent.langgraph_agent.agent_core.approval import (
    CapabilityApprovalLedger,
    CapabilityApprovalVerifier,
    approval_signature_payload,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    WorkflowManifest,
    WorkflowStepPolicy,
)
from aerospace_agent.langgraph_agent.agent_core.workflows import (
    WorkflowRegistry,
    canonical_manifest_sha256,
    canonical_workflow_sha256,
    workflow_approval_digest,
)


def _body() -> dict:
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
                "risk_level": "read_only",
                "recovery_class": "read_only",
                "idempotent": True,
            }
        ],
    }


def _manifest(body: dict | None = None, **overrides) -> WorkflowManifest:
    body = body or _body()
    data = {
        **body,
        "workflow_sha256": canonical_workflow_sha256(body),
        "manifest_sha256": "0" * 64,
        "approval_record_id": "approval-1",
        "approval_scope": "scheduled_read_only",
        "automatable": True,
    }
    data.update(overrides)
    data["manifest_sha256"] = canonical_manifest_sha256(data)
    return WorkflowManifest.model_validate(data)


def _approval(tmp_path: Path):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    ledger = CapabilityApprovalLedger(
        tmp_path / "approval.sqlite", trusted_public_keys={"operator": public}
    )
    return private, ledger, CapabilityApprovalVerifier(ledger)


def _approve(private, ledger, manifest: WorkflowManifest) -> None:
    digest = workflow_approval_digest(manifest)
    ledger.append(
        approval_record_id=manifest.approval_record_id,
        key_id="operator",
        digest=digest,
        signature_b64=b64encode(private.sign(approval_signature_payload(digest))).decode(),
        created_at="2026-07-13T08:00:00+00:00",
    )


def _registry(tmp_path: Path, manifest: WorkflowManifest | None = None) -> WorkflowRegistry:
    private, ledger, verifier = _approval(tmp_path)
    if manifest is not None:
        _approve(private, ledger, manifest)
    return WorkflowRegistry(tmp_path / "workflows.sqlite", approval_verifier=verifier)


def test_register_persists_exact_locked_workflow_and_survives_restart(tmp_path) -> None:
    body = _body()
    manifest = _manifest(body)
    registry = _registry(tmp_path, manifest)

    snapshot = registry.register(manifest=manifest, workflow_body=body)
    restarted = WorkflowRegistry(
        tmp_path / "workflows.sqlite",
        approval_verifier=registry._approval_verifier,
    )

    assert snapshot.workflow_sha256 == manifest.workflow_sha256
    assert restarted.get("workflow.read_environment", "1.0.0").manifest == manifest
    assert restarted.get("workflow.read_environment", "1.0.0").body == body


def test_registration_rejects_body_manifest_or_approval_drift(tmp_path) -> None:
    body = _body()
    manifest = _manifest(body)

    with pytest.raises(ValueError, match="workflow body hash"):
        _registry(tmp_path / "body").register(
            manifest=manifest,
            workflow_body={**body, "version": "1.0.1"},
        )
    with pytest.raises(ValueError, match="manifest hash"):
        _registry(tmp_path / "manifest").register(
            manifest=manifest.model_copy(update={"manifest_sha256": "f" * 64}),
            workflow_body=body,
        )
    with pytest.raises(PermissionError, match="approval"):
        _registry(tmp_path / "approval").register(
            manifest=manifest,
            workflow_body=body,
        )


def test_same_identity_cannot_be_replaced_by_different_content(tmp_path) -> None:
    body = _body()
    first = _manifest(body)
    registry = _registry(tmp_path, first)
    registry.register(manifest=first, workflow_body=body)
    changed = {**body, "input_schema": {"type": "object", "properties": {}}}
    second = _manifest(changed, approval_record_id="approval-2")
    private, ledger, verifier = _approval(tmp_path / "second-approval")
    _approve(private, ledger, second)
    registry._approval_verifier = verifier

    with pytest.raises(ValueError, match="immutable"):
        registry.register(manifest=second, workflow_body=changed)


def test_input_validation_and_sensitive_masking_use_locked_schema(tmp_path) -> None:
    body = _body()
    manifest = _manifest(body)
    registry = _registry(tmp_path, manifest)
    registry.register(manifest=manifest, workflow_body=body)

    assert registry.validate_inputs(
        "workflow.read_environment", "1.0.0", {"engine": "gmat", "token": "secret"}
    ) == {"engine": "gmat", "token": "secret"}
    assert registry.mask_inputs(
        "workflow.read_environment", "1.0.0", {"engine": "gmat", "token": "secret"}
    ) == {"engine": "gmat", "token": "***"}
    with pytest.raises(ValueError, match="workflow inputs"):
        registry.validate_inputs("workflow.read_environment", "1.0.0", {})


def test_automation_policy_is_not_inferred_from_workflow_name(tmp_path) -> None:
    body = _body()
    policy = WorkflowStepPolicy.model_validate(body["steps"][0]).model_copy(
        update={"risk_level": "project_write"}
    )
    body = {**body, "steps": [policy.model_dump(mode="json")]}
    unsafe = _manifest(
        body,
        automatable=False,
        approval_scope="interactive_only",
    )
    registry = _registry(tmp_path, unsafe)
    registry.register(manifest=unsafe, workflow_body=body)

    assert not registry.is_scheduled_read_only("workflow.read_environment", "1.0.0")


def test_schema_version_is_explicit(tmp_path) -> None:
    _, _, verifier = _approval(tmp_path)
    registry = WorkflowRegistry(tmp_path / "workflows.sqlite", approval_verifier=verifier)
    assert registry.schema_version() == 1


def test_workflow_migration_failure_is_locally_degraded(tmp_path) -> None:
    database = tmp_path / "future-workflows.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 999")
    _, _, verifier = _approval(tmp_path)

    registry = WorkflowRegistry(database, approval_verifier=verifier)

    assert registry.availability()["available"] is False
    with pytest.raises(RuntimeError, match="unavailable"):
        registry.get("missing", "1.0.0")
