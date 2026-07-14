from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from base64 import b64encode
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from aerospace_agent.langgraph_agent.agent_core.approval import (
    CapabilityApprovalLedger,
    CapabilityApprovalVerifier,
    approval_signature_payload,
)
from aerospace_agent.langgraph_agent.agent_core.confirmation import (
    ConfirmationError,
    ConfirmationService,
)
from aerospace_agent.langgraph_agent.agent_core.capabilities import CapabilityRegistry
from aerospace_agent.langgraph_agent.agent_core.integrations import (
    CapabilityAcquisitionService,
    IntegrationTrustService,
    content_sha256,
)
from aerospace_agent.langgraph_agent.agent_core.models import (
    CapabilityGap,
    CheckpointRef,
    DependencyCandidate,
)


NOW = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)


def _write(path: Path, content: str | bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_read_only(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        child.chmod(0o555 if child.is_dir() else 0o444)
    path.chmod(0o555)


def _integration_fixture(workspace: Path, *, capability_id: str, import_root: str):
    if import_root == "aerospace_agent.integrations":
        adapter_rel = f"aerospace_agent/integrations/{capability_id}/adapter.py"
        entrypoint = f"aerospace_agent.integrations.{capability_id}.adapter:run"
    elif import_root == "aerospace_agent.mcp.tools":
        adapter_rel = f"aerospace_agent/mcp/tools/{capability_id}_adapter.py"
        entrypoint = f"aerospace_agent.mcp.tools.{capability_id}_adapter:run"
    else:
        adapter_rel = f"aerospace_agent/domains/{capability_id}_adapter.py"
        entrypoint = f"aerospace_agent.domains.{capability_id}_adapter:run"
    adapter_sha = _write(workspace / adapter_rel, "def run(value):\n    return value\n")

    cache_staging = workspace / "data/langgraph/capability_sources/staging"
    _write(cache_staging / "source.py", "VALUE = 1\n")
    cache_sha = content_sha256(cache_staging)
    cache_rel = f"data/langgraph/capability_sources/{cache_sha}"
    cache = workspace / cache_rel
    cache_staging.rename(cache)
    _make_read_only(cache)

    commit = "a" * 40
    lock_rel = f"aerospace_agent/integrations/{capability_id}/dependency.lock"
    lock_sha = _write(
        workspace / lock_rel,
        f"source=git\ncommit={commit}\nsource_sha256={cache_sha}\n",
    )
    license_rel = f"aerospace_agent/integrations/{capability_id}/LICENSE.evidence"
    license_sha = _write(workspace / license_rel, "SPDX-License-Identifier: MIT\n")
    version_rel = f"aerospace_agent/integrations/{capability_id}/VERSION.evidence"
    version_sha = _write(workspace / version_rel, f"commit={commit}\n")
    validation_rel = f"aerospace_agent/integrations/{capability_id}/validation.json"
    validation_sha = _write(
        workspace / validation_rel,
        json.dumps({"passed": True, "commands": ["pytest tests/capabilities/test_contract.py"]}),
    )
    capability_rel = f"aerospace_agent/integrations/{capability_id}/capability.yaml"
    capability_payload = {
        "capability_id": capability_id,
        "version": "1.0.0",
        "category": "basic",
        "status": "available",
        "intents": ["fixture"],
        "tool_names": [f"{capability_id}.run"],
        "risk_level": "read_only",
        "required_dependencies": [f"fixture@{commit}"],
        "validators": ["tests/capabilities/test_contract.py"],
        "source": entrypoint.split(":", 1)[0],
    }
    capability_sha = _write(workspace / capability_rel, yaml.safe_dump(capability_payload, sort_keys=True))

    manifest_rel = f"aerospace_agent/integrations/{capability_id}/manifest.yaml"
    manifest_payload = {
        "schema_version": "1.0",
        "capability_id": capability_id,
        "adapter_entrypoint": entrypoint,
        "adapter_path": adapter_rel,
        "adapter_sha256": adapter_sha,
        "source_type": "git",
        "source_uri": "https://example.invalid/repository.git",
        "version_or_commit": commit,
        "license": "MIT",
        "compatibility": "compatible",
        "lock_path": lock_rel,
        "lock_sha256": lock_sha,
        "source_cache_path": cache_rel,
        "source_cache_sha256": cache_sha,
        "license_evidence_path": license_rel,
        "license_evidence_sha256": license_sha,
        "version_evidence_path": version_rel,
        "version_evidence_sha256": version_sha,
        "validation_result_path": validation_rel,
        "validation_result_sha256": validation_sha,
        "validation_commands": ["pytest tests/capabilities/test_contract.py"],
        "capability_manifest_path": capability_rel,
        "capability_manifest_sha256": capability_sha,
        "capability_manifest_version": "1.0.0",
    }
    _write(workspace / manifest_rel, yaml.safe_dump(manifest_payload, sort_keys=True))
    return manifest_payload, workspace / manifest_rel


def _runtime(workspace: Path):
    external = workspace.parent / f"operator-key-{workspace.name}"
    external.mkdir(exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    private_path = external / "operator.private"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    ledger = CapabilityApprovalLedger(
        workspace / "data/langgraph/approval.sqlite",
        trusted_public_keys={"operator-1": public},
    )
    verifier = CapabilityApprovalVerifier(ledger)
    trust = IntegrationTrustService(workspace, approval_verifier=verifier)
    return private_key, private_path, ledger, trust


def _approve(private_key, ledger, digest):
    ledger.append(
        approval_record_id=f"approval-{digest[:12]}",
        key_id="operator-1",
        digest=digest,
        signature_b64=b64encode(private_key.sign(approval_signature_payload(digest))).decode("ascii"),
        created_at=NOW.isoformat(),
    )


@pytest.mark.parametrize(
    "import_root",
    ["aerospace_agent.mcp.tools", "aerospace_agent.integrations", "aerospace_agent.domains"],
)
def test_combined_digest_requires_external_ed25519_approval_for_all_import_roots(
    tmp_path, import_root
):
    _integration_fixture(tmp_path, capability_id="fixture_cap", import_root=import_root)
    private_key, private_path, ledger, trust = _runtime(tmp_path)

    unapproved = trust.verify("fixture_cap")
    assert unapproved.status == "unavailable"
    assert unapproved.reason == "approval_missing_or_invalid"
    assert private_path.is_relative_to(tmp_path.parent) and not private_path.is_relative_to(tmp_path)
    assert not hasattr(trust, "private_key")

    _approve(private_key, ledger, unapproved.combined_digest)
    approved = trust.verify("fixture_cap")
    assert approved.status == "available"
    assert approved.approval_record_id
    assert trust.capability_manifest("fixture_cap").status == "available"

    manifest = yaml.safe_load((tmp_path / "aerospace_agent/integrations/fixture_cap/manifest.yaml").read_text(encoding="utf-8"))
    adapter = tmp_path / manifest["adapter_path"]
    adapter.chmod(0o644)
    adapter.write_text("def run(value):\n    return {'drift': value}\n", encoding="utf-8")
    drift = trust.verify("fixture_cap")
    assert drift.status == "unavailable"
    assert drift.reason == "adapter_hash_mismatch"
    assert trust.capability_manifest("fixture_cap").status == "unavailable"


def test_lock_cache_local_code_and_evidence_boundaries_fail_closed(tmp_path):
    manifest, manifest_path = _integration_fixture(
        tmp_path, capability_id="boundary_cap", import_root="aerospace_agent.integrations"
    )
    _, _, _, trust = _runtime(tmp_path)

    cache = tmp_path / manifest["source_cache_path"]
    source = cache / "source.py"
    source.chmod(0o644)
    assert trust.verify("boundary_cap").reason == "git_cache_not_read_only"
    source.chmod(0o444)

    lock = tmp_path / manifest["lock_path"]
    lock.write_text("drift\n", encoding="utf-8")
    assert trust.verify("boundary_cap").reason == "lock_hash_mismatch"
    lock.write_text(
        f"source=git\ncommit={manifest['version_or_commit']}\nsource_sha256={manifest['source_cache_sha256']}\n",
        encoding="utf-8",
    )

    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    loaded["adapter_entrypoint"] = "outside_project.adapter:run"
    manifest_path.write_text(yaml.safe_dump(loaded, sort_keys=True), encoding="utf-8")
    assert trust.verify("boundary_cap").reason == "entrypoint_outside_allowed_roots"

    loaded["adapter_entrypoint"] = "aerospace_agent.integrations.boundary_cap.adapter:run"
    loaded["source_type"] = "local_code"
    loaded["source_uri"] = None
    loaded["source_cache_path"] = None
    loaded["source_cache_sha256"] = None
    loaded["local_patch_journal_path"] = "../outside-journal.json"
    loaded["local_patch_journal_sha256"] = "0" * 64
    manifest_path.write_text(yaml.safe_dump(loaded, sort_keys=True), encoding="utf-8")
    assert trust.verify("boundary_cap").reason == "path_outside_workspace"


def test_manifest_or_evidence_drift_requires_a_new_approval(tmp_path):
    _, manifest_path = _integration_fixture(
        tmp_path, capability_id="drift_cap", import_root="aerospace_agent.integrations"
    )
    private_key, _, ledger, trust = _runtime(tmp_path)
    first = trust.verify("drift_cap")
    _approve(private_key, ledger, first.combined_digest)
    assert trust.verify("drift_cap").status == "available"

    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    evidence = tmp_path / loaded["license_evidence_path"]
    evidence.write_text("SPDX-License-Identifier: Apache-2.0\n", encoding="utf-8")
    assert trust.verify("drift_cap").reason == "license_evidence_hash_mismatch"

    loaded["license"] = "Apache-2.0"
    loaded["license_evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest_path.write_text(yaml.safe_dump(loaded, sort_keys=True), encoding="utf-8")
    changed = trust.verify("drift_cap")
    assert changed.status == "unavailable"
    assert changed.reason == "approval_missing_or_invalid"
    assert changed.combined_digest != first.combined_digest
    _approve(private_key, ledger, changed.combined_digest)
    assert trust.verify("drift_cap").status == "available"


def test_active_registry_route_is_invalidated_when_integration_drifts(tmp_path):
    manifest_data, _ = _integration_fixture(
        tmp_path, capability_id="runtime_cap", import_root="aerospace_agent.integrations"
    )
    private_key, _, ledger, trust = _runtime(tmp_path)
    pending = trust.verify("runtime_cap")
    _approve(private_key, ledger, pending.combined_digest)
    manifest = trust.capability_manifest("runtime_cap")
    registry = CapabilityRegistry(
        [manifest],
        approval_verifier=lambda item: trust.verify(item.capability_id).status == "available",
    )
    assert registry.get("runtime_cap").status == "available"

    adapter = tmp_path / manifest_data["adapter_path"]
    adapter.write_text("def run(value):\n    return {'drift': value}\n", encoding="utf-8")

    assert registry.get("runtime_cap").status == "unavailable"
    assert registry.candidates_for_intents(["fixture"]) == []


def test_package_distribution_and_local_patch_journal_are_verified_as_artifacts(tmp_path):
    manifest, manifest_path = _integration_fixture(
        tmp_path, capability_id="artifact_cap", import_root="aerospace_agent.integrations"
    )
    _, _, _, trust = _runtime(tmp_path)
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    distribution_rel = "data/langgraph/capability_sources/distributions/artifact_cap-1.2.3.whl"
    distribution_sha = _write(tmp_path / distribution_rel, b"wheel-bytes")
    loaded.update(
        source_type="package",
        source_uri="https://example.invalid/artifact_cap-1.2.3.whl",
        version_or_commit="1.2.3",
        source_cache_path=None,
        source_cache_sha256=None,
        distribution_path=distribution_rel,
        distribution_sha256=distribution_sha,
    )
    loaded["lock_sha256"] = _write(
        tmp_path / loaded["lock_path"], "artifact_cap==1.2.3\n"
    )
    loaded["version_evidence_sha256"] = _write(
        tmp_path / loaded["version_evidence_path"], "version=1.2.3\n"
    )
    manifest_path.write_text(yaml.safe_dump(loaded, sort_keys=True), encoding="utf-8")
    assert trust.verify("artifact_cap").reason == "approval_missing_or_invalid"
    (tmp_path / distribution_rel).write_bytes(b"drift")
    assert trust.verify("artifact_cap").reason == "distribution_hash_mismatch"

    loaded["source_type"] = "local_code"
    loaded["source_uri"] = None
    loaded["distribution_path"] = None
    loaded["distribution_sha256"] = None
    journal_rel = "data/langgraph/operations/local-artifact-cap.json"
    journal = {
        "operation_id": "local-artifact-cap",
        "status": "committed",
        "recovery_class": "reversible",
        "target_paths": [manifest["adapter_path"]],
        "postimage_sha256": manifest["adapter_sha256"],
    }
    loaded["local_patch_journal_path"] = journal_rel
    loaded["local_patch_journal_sha256"] = _write(
        tmp_path / journal_rel, json.dumps(journal, sort_keys=True)
    )
    manifest_path.write_text(yaml.safe_dump(loaded, sort_keys=True), encoding="utf-8")
    assert trust.verify("artifact_cap").reason == "approval_missing_or_invalid"
    journal["status"] = "planned"
    loaded["local_patch_journal_sha256"] = _write(
        tmp_path / journal_rel, json.dumps(journal, sort_keys=True)
    )
    manifest_path.write_text(yaml.safe_dump(loaded, sort_keys=True), encoding="utf-8")
    assert trust.verify("artifact_cap").reason == "local_patch_journal_not_committed"


def test_one_build_per_capability_run_and_resume_exact_original_checkpoint(tmp_path):
    _integration_fixture(tmp_path, capability_id="missing_cap", import_root="aerospace_agent.integrations")
    private_key, _, ledger, trust = _runtime(tmp_path)
    acquisition = CapabilityAcquisitionService(
        tmp_path,
        database_path=tmp_path / "data/langgraph/acquisition.sqlite",
        project_id="project-1",
        clock=lambda: NOW,
    )
    confirmation = ConfirmationService(
        tmp_path / "data/langgraph/confirmation.sqlite",
        clock=lambda: NOW,
    )
    checkpoint = CheckpointRef(
        project_id="project-1", thread_id="thread-1", checkpoint_id="checkpoint-original"
    )
    gap = CapabilityGap(
        capability_id="missing_cap",
        requested_by_step_id="step-build",
        description="fixture capability is absent",
        required_contract={"type": "object"},
        candidates=[
            DependencyCandidate(
                name="fixture-repository",
                source_type="git",
                source_uri="https://example.invalid/repository.git",
                version_or_commit="a" * 40,
                license="MIT",
                compatibility="compatible",
                maintenance_status="active",
            )
        ],
        resolution="integrate_git",
    )
    staged = acquisition.stage_gap(
        gap,
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        original_checkpoint=checkpoint,
    )
    assert staged.status == "staged"
    with pytest.raises(ConfirmationError):
        acquisition.authorize_build(
            staged.gap_id,
            candidate_name="fixture-repository",
            confirmation_service=confirmation,
            confirmation_id="missing",
        )

    action_hash = acquisition.build_action_hash(staged.gap_id, "fixture-repository")
    grant = confirmation.issue(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        operation_id=f"build:{staged.gap_id}",
        action_hash=action_hash,
    )
    attempt = acquisition.authorize_build(
        staged.gap_id,
        candidate_name="fixture-repository",
        confirmation_service=confirmation,
        confirmation_id=grant.confirmation_id,
    )
    assert attempt.attempt_number == 1
    assert attempt.status == "authorized"

    second_gap = acquisition.stage_gap(
        gap,
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        original_checkpoint=checkpoint,
    )
    second_hash = acquisition.build_action_hash(second_gap.gap_id, "fixture-repository")
    second_grant = confirmation.issue(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        operation_id=f"build:{second_gap.gap_id}",
        action_hash=second_hash,
    )
    with pytest.raises(ValueError, match="one build attempt"):
        acquisition.authorize_build(
            second_gap.gap_id,
            candidate_name="fixture-repository",
            confirmation_service=confirmation,
            confirmation_id=second_grant.confirmation_id,
        )

    with pytest.raises(RuntimeError, match="registered"):
        acquisition.resume_from_original_checkpoint(staged.gap_id, trust_service=trust)
    verification = trust.verify("missing_cap")
    acquired = acquisition.record_acquisition(attempt.attempt_id, trust_service=trust)
    assert acquired.status == "acquired"
    validated = acquisition.record_validation(attempt.attempt_id, trust_service=trust)
    assert validated.status == "validated"
    with pytest.raises(RuntimeError, match="validated"):
        acquisition.record_validation(attempt.attempt_id, trust_service=trust)
    _approve(private_key, ledger, verification.combined_digest)
    registered = acquisition.register(attempt.attempt_id, trust_service=trust)
    assert registered.status == "registered"
    resumed = acquisition.resume_from_original_checkpoint(staged.gap_id, trust_service=trust)
    assert resumed == checkpoint
    assert confirmation.continuation_checkpoint(grant.confirmation_id)["checkpoint_id"] == checkpoint.checkpoint_id


def test_acquisition_migration_failure_is_locally_degraded(tmp_path):
    database = tmp_path / "future-acquisition.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 999")

    service = CapabilityAcquisitionService(
        tmp_path,
        database_path=database,
        project_id="project-1",
    )

    assert service.availability()["available"] is False
    with pytest.raises(RuntimeError, match="unavailable"):
        service.get_gap("missing")


def test_acquisition_restart_recovers_consumed_build_confirmation(tmp_path, monkeypatch):
    confirmation = ConfirmationService(tmp_path / "confirmation.sqlite", clock=lambda: NOW)
    database = tmp_path / "acquisition.sqlite"
    acquisition = CapabilityAcquisitionService(
        tmp_path,
        database_path=database,
        project_id="project-1",
        clock=lambda: NOW,
        confirmation_service=confirmation,
    )
    checkpoint = CheckpointRef(
        project_id="project-1", thread_id="thread-1", checkpoint_id="checkpoint-crash"
    )
    gap = CapabilityGap(
        capability_id="crash_cap",
        requested_by_step_id="build",
        description="crash recovery",
        required_contract={"type": "object"},
        candidates=[DependencyCandidate(
            name="crash-repository",
            source_type="git",
            source_uri="https://example.invalid/crash.git",
            version_or_commit="c" * 40,
            license="MIT",
            compatibility="compatible",
            maintenance_status="active",
        )],
        resolution="integrate_git",
    )
    staged = acquisition.stage_gap(
        gap,
        thread_id="thread-1",
        root_run_id="run-crash",
        plan_id="plan-crash",
        original_checkpoint=checkpoint,
    )
    grant = confirmation.issue(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-crash",
        operation_id=f"build:{staged.gap_id}",
        action_hash=acquisition.build_action_hash(staged.gap_id, "crash-repository"),
    )

    def crash_before_authorized(_attempt):
        raise KeyboardInterrupt("simulated acquisition crash")

    monkeypatch.setattr(acquisition, "_mark_authorized", crash_before_authorized)
    with pytest.raises(KeyboardInterrupt, match="simulated acquisition crash"):
        acquisition.authorize_build(
            staged.gap_id,
            candidate_name="crash-repository",
            confirmation_service=confirmation,
            confirmation_id=grant.confirmation_id,
        )

    restarted = CapabilityAcquisitionService(
        tmp_path,
        database_path=database,
        project_id="project-1",
        clock=lambda: NOW,
        confirmation_service=confirmation,
    )
    with sqlite3.connect(database) as connection:
        attempt_id = connection.execute(
            "SELECT attempt_id FROM acquisition_build_attempts WHERE gap_id = ?",
            (staged.gap_id,),
        ).fetchone()[0]
    assert restarted.get_attempt(attempt_id).status == "authorized"

