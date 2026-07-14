from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from pathlib import Path

import pytest

from aerospace_agent.langgraph_agent.agent_core.models import (
    ArtifactRef,
    ArtifactSchemaManifest,
    CapabilitySnapshot,
    CheckpointRef,
    CrossDomainHandoff,
    DomainArtifact,
    DomainDataMetadata,
    HandoffConversion,
    HandoffRecord,
)


SHA = "a" * 64


def _artifact_module():
    try:
        return importlib.import_module("aerospace_agent.langgraph_agent.agent_core.artifacts")
    except ModuleNotFoundError as exc:
        pytest.fail(f"ArtifactStore module is missing: {exc}")


def _schema() -> ArtifactSchemaManifest:
    return ArtifactSchemaManifest(
        schema_id="orbit-state",
        schema_version="1.0",
        payload_json_schema={
            "type": "object",
            "required": ["position", "epoch"],
            "properties": {
                "position": {"type": "array", "minItems": 3, "maxItems": 3},
                "epoch": {"type": "string"},
            },
        },
        required_quantity_fields=["position"],
        requires_frame=True,
        requires_time_system=True,
        requires_epoch_field=True,
    )


def _write_payload(root: Path, payload: dict) -> ArtifactRef:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = root / "data" / "langgraph" / "artifacts" / "sha256" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return ArtifactRef(
        uri=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type="application/json",
        byte_length=len(encoded),
    )


def _domain_artifact(root: Path, *, payload: dict | None = None) -> DomainArtifact:
    payload = payload or {"position": [1.0, 2.0, 3.0], "epoch": "2026-07-13T00:00:00Z"}
    return DomainArtifact(
        artifact_id="source-artifact",
        payload_ref=_write_payload(root, payload),
        schema_id="orbit-state",
        schema_version="1.0",
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
        source_checkpoints=[
            CheckpointRef(project_id="project-1", thread_id="thread-1", checkpoint_id="cp-1")
        ],
    )


def _store(root: Path, **overrides):
    module = _artifact_module()
    values = {
        "workspace_root": root,
        "schema_manifests": [_schema()],
        "schema_approval_verifier": lambda _manifest: True,
        "checkpoint_verifier": lambda checkpoint: checkpoint.checkpoint_id == "cp-1",
        "capability_snapshot_verifier": lambda snapshot: snapshot.manifest_sha256 == SHA,
    }
    values.update(overrides)
    return module.ArtifactStore(**values)


def test_store_is_the_only_content_addressed_resolver_and_rechecks_content(tmp_path: Path) -> None:
    artifact = _domain_artifact(tmp_path)
    resolved = _store(tmp_path).resolve(
        artifact,
        project_id="project-1",
        thread_id="thread-1",
    )

    assert resolved.artifact.artifact_id == "source-artifact"
    assert resolved.payload["position"] == [1.0, 2.0, 3.0]
    assert resolved.schema_manifest.schema_id == "orbit-state"

    content_path = (
        tmp_path
        / "data"
        / "langgraph"
        / "artifacts"
        / "sha256"
        / artifact.payload_ref.sha256
    )
    content_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256|byte length"):
        _store(tmp_path).resolve(artifact, project_id="project-1", thread_id="thread-1")


def test_store_fails_closed_on_uri_schema_metadata_snapshot_and_checkpoint(tmp_path: Path) -> None:
    artifact = _domain_artifact(tmp_path)

    escaped = artifact.model_copy(
        update={
            "payload_ref": artifact.payload_ref.model_copy(
                update={"uri": "file:///outside/payload.json"}
            )
        }
    )
    with pytest.raises(ValueError, match="artifact URI"):
        _store(tmp_path).resolve(escaped, project_id="project-1", thread_id="thread-1")

    with pytest.raises(ValueError, match="approved"):
        _store(tmp_path, schema_approval_verifier=lambda _manifest: False)

    invalid_payload = _domain_artifact(tmp_path, payload={"epoch": "2026-07-13T00:00:00Z"})
    with pytest.raises(ValueError, match="payload schema"):
        _store(tmp_path).resolve(
            invalid_payload, project_id="project-1", thread_id="thread-1"
        )

    missing_metadata = artifact.model_copy(
        update={"metadata": artifact.metadata.model_copy(update={"frame_id": None})}
    )
    with pytest.raises(ValueError, match="frame"):
        _store(tmp_path).resolve(
            missing_metadata, project_id="project-1", thread_id="thread-1"
        )

    with pytest.raises(ValueError, match="source capability snapshot"):
        _store(tmp_path, capability_snapshot_verifier=lambda _snapshot: False).resolve(
            artifact, project_id="project-1", thread_id="thread-1"
        )

    with pytest.raises(ValueError, match="checkpoint namespace"):
        _store(tmp_path).resolve(artifact, project_id="project-1", thread_id="other-thread")

    with pytest.raises(ValueError, match="checkpoint"):
        _store(tmp_path, checkpoint_verifier=lambda _checkpoint: False).resolve(
            artifact, project_id="project-1", thread_id="thread-1"
        )


def test_handoff_resolves_real_source_and_conversion_output_and_checks_metadata(
    tmp_path: Path,
) -> None:
    source = _domain_artifact(tmp_path)
    target = _domain_artifact(tmp_path).model_copy(
        update={
            "artifact_id": "target-artifact",
            "metadata": DomainDataMetadata(
                quantity_units={"position": "km"},
                frame_id="ITRF",
                time_system="TAI",
                epoch_field="epoch",
            ),
        }
    )
    planned = CrossDomainHandoff(
        source_step_id="source-step",
        target_step_id="target-step",
        source_domain="orbit_design",
        target_domain="simulation",
        reason="convert propagated state",
        required_inputs=["position"],
        expected_outputs=["position"],
        source_output_mapping={"position": "state.position"},
        target_input_mapping={"state.position": "position"},
        source_metadata=source.metadata,
        target_metadata=target.metadata,
        conversion=HandoffConversion(
            converter_capability="space.transform_frame",
            input_mapping={"position": "position"},
            output_mapping={"position": "position"},
            validation_check_id="conversion-check",
        ),
    )
    record = HandoffRecord(
        handoff_id="handoff-1",
        plan_id="plan-1",
        source_step_id="source-step",
        target_step_id="target-step",
        source_artifact_ids=["source-artifact"],
        target_artifact_ids=["target-artifact"],
        conversion_capability="space.transform_frame",
        validation_check_ids=["conversion-check"],
        checkpoint=CheckpointRef(
            project_id="project-1", thread_id="thread-1", checkpoint_id="cp-1"
        ),
    )

    module = _artifact_module()
    binding = module.ArtifactCheckpointBinding(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        step_id="source-step",
        checkpoint_id="cp-1",
        phase="after",
    )
    resolved = _store(tmp_path).validate_handoff(
        planned,
        record,
        source_artifacts={source.artifact_id: source},
        target_artifacts={target.artifact_id: target},
        project_id="project-1",
        thread_id="thread-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        source_snapshot=source.source_capability,
        target_snapshot=source.source_capability,
        source_checkpoint_binding=binding,
        validation_results={"conversion-check": True},
    )
    assert [item.artifact.artifact_id for item in resolved.sources] == ["source-artifact"]
    assert [item.artifact.artifact_id for item in resolved.targets] == ["target-artifact"]

    wrong_target = target.model_copy(update={"metadata": source.metadata})
    with pytest.raises(ValueError, match="target metadata"):
        _store(tmp_path).validate_handoff(
            planned,
            record,
            source_artifacts={source.artifact_id: source},
            target_artifacts={target.artifact_id: wrong_target},
            project_id="project-1",
            thread_id="thread-1",
            plan_id="plan-1",
            plan_sha256=SHA,
            source_snapshot=source.source_capability,
            target_snapshot=source.source_capability,
            source_checkpoint_binding=binding,
            validation_results={"conversion-check": True},
        )


def test_handoff_fails_closed_on_plan_mapping_snapshot_and_real_validation_mismatch(
    tmp_path: Path,
) -> None:
    module = _artifact_module()
    source = _domain_artifact(tmp_path)
    planned = CrossDomainHandoff(
        source_step_id="source-step",
        target_step_id="target-step",
        source_domain="orbit_design",
        target_domain="simulation",
        reason="direct state handoff",
        required_inputs=["position"],
        expected_outputs=["position"],
        source_output_mapping={"position": "state.position"},
        target_input_mapping={"state.position": "position"},
        source_metadata=source.metadata,
        target_metadata=source.metadata,
    )
    record = HandoffRecord(
        handoff_id="handoff-1",
        plan_id="plan-1",
        source_step_id="source-step",
        target_step_id="target-step",
        source_artifact_ids=[source.artifact_id],
        validation_check_ids=["handoff-check"],
        checkpoint=CheckpointRef(
            project_id="project-1", thread_id="thread-1", checkpoint_id="cp-1"
        ),
    )
    binding = module.ArtifactCheckpointBinding(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        step_id="source-step",
        checkpoint_id="cp-1",
        phase="after",
    )
    arguments = {
        "source_artifacts": {source.artifact_id: source},
        "target_artifacts": {},
        "project_id": "project-1",
        "thread_id": "thread-1",
        "plan_id": "plan-1",
        "plan_sha256": SHA,
        "source_snapshot": source.source_capability,
        "target_snapshot": source.source_capability,
        "source_checkpoint_binding": binding,
        "validation_results": {"handoff-check": True},
    }

    accepted = _store(tmp_path).validate_handoff(planned, record, **arguments)
    assert accepted.target_inputs == {"position": [1.0, 2.0, 3.0]}

    with pytest.raises(ValueError, match="plan"):
        _store(tmp_path).validate_handoff(
            planned, record, **{**arguments, "plan_id": "other-plan"}
        )
    with pytest.raises(ValueError, match="validation"):
        _store(tmp_path).validate_handoff(
            planned,
            record,
            **{**arguments, "validation_results": {"handoff-check": False}},
        )
    with pytest.raises(ValueError, match="snapshot"):
        _store(tmp_path).validate_handoff(
            planned,
            record,
            **{
                **arguments,
                "source_snapshot": source.source_capability.model_copy(
                    update={"manifest_sha256": "b" * 64}
                ),
            },
        )
    broken_mapping = planned.model_copy(
        update={"target_input_mapping": {"missing.transfer": "position"}}
    )
    with pytest.raises(ValueError, match="mapping"):
        _store(tmp_path).validate_handoff(broken_mapping, record, **arguments)


def test_artifact_and_handoff_records_are_immutable_durable_and_revalidated_on_read(
    tmp_path: Path,
) -> None:
    module = _artifact_module()
    source = _domain_artifact(tmp_path)
    planned = CrossDomainHandoff(
        source_step_id="source-step",
        target_step_id="target-step",
        source_domain="orbit_design",
        target_domain="simulation",
        reason="durable handoff",
        required_inputs=["position"],
        expected_outputs=["position"],
        source_output_mapping={"position": "state.position"},
        target_input_mapping={"state.position": "position"},
        source_metadata=source.metadata,
        target_metadata=source.metadata,
    )
    record = HandoffRecord(
        handoff_id="handoff-durable",
        plan_id="plan-1",
        source_step_id="source-step",
        target_step_id="target-step",
        source_artifact_ids=[source.artifact_id],
        validation_check_ids=["handoff-check"],
        checkpoint=CheckpointRef(
            project_id="project-1", thread_id="thread-1", checkpoint_id="cp-1"
        ),
    )
    binding = module.ArtifactCheckpointBinding(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        step_id="source-step",
        checkpoint_id="cp-1",
        phase="after",
    )
    store = _store(tmp_path)
    resolved = store.validate_handoff(
        planned,
        record,
        source_artifacts={source.artifact_id: source},
        target_artifacts={},
        project_id="project-1",
        thread_id="thread-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        source_snapshot=source.source_capability,
        target_snapshot=source.source_capability,
        source_checkpoint_binding=binding,
        validation_results={"handoff-check": True},
    )
    artifact_records, handoff_records = store.persist_exchange(
        artifacts=[source],
        handoffs=[resolved],
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        step_id="source-step",
        checkpoint_binding=binding,
        source_snapshot=source.source_capability,
    )

    assert [item.artifact_id for item in artifact_records] == ["source-artifact"]
    assert [item.handoff_id for item in handoff_records] == ["handoff-durable"]
    restarted = _store(tmp_path)
    loaded = restarted.load_handoff(
        planned,
        "handoff-durable",
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        source_snapshot=source.source_capability,
        target_snapshot=source.source_capability,
    )
    assert loaded.target_inputs == {"position": [1.0, 2.0, 3.0]}

    changed = source.model_copy(update={"artifact_id": "source-artifact"})
    changed_payload = changed.model_copy(
        update={"assumptions": ["different immutable record"]}
    )
    with pytest.raises(ValueError, match="immutable"):
        restarted.persist_exchange(
            artifacts=[changed_payload],
            handoffs=[resolved],
            project_id="project-1",
            thread_id="thread-1",
            root_run_id="run-1",
            plan_id="plan-1",
            plan_sha256=SHA,
            step_id="source-step",
            checkpoint_binding=binding,
            source_snapshot=source.source_capability,
        )


def _persist_exchange_fixture(tmp_path: Path):
    module = _artifact_module()
    source = _domain_artifact(tmp_path)
    planned = CrossDomainHandoff(
        source_step_id="source-step",
        target_step_id="target-step",
        source_domain="orbit_design",
        target_domain="simulation",
        reason="durable handoff",
        required_inputs=["position"],
        expected_outputs=["position"],
        source_output_mapping={"position": "state.position"},
        target_input_mapping={"state.position": "position"},
        source_metadata=source.metadata,
        target_metadata=source.metadata,
    )
    record = HandoffRecord(
        handoff_id="handoff-envelope",
        plan_id="plan-1",
        source_step_id="source-step",
        target_step_id="target-step",
        source_artifact_ids=[source.artifact_id],
        validation_check_ids=["handoff-check"],
        checkpoint=CheckpointRef(
            project_id="project-1", thread_id="thread-1", checkpoint_id="cp-1"
        ),
    )
    binding = module.ArtifactCheckpointBinding(
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        step_id="source-step",
        checkpoint_id="cp-1",
        phase="after",
    )
    store = _store(tmp_path)
    resolved = store.validate_handoff(
        planned,
        record,
        source_artifacts={source.artifact_id: source},
        target_artifacts={},
        project_id="project-1",
        thread_id="thread-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        source_snapshot=source.source_capability,
        target_snapshot=source.source_capability,
        source_checkpoint_binding=binding,
        validation_results={"handoff-check": True},
    )
    store.persist_exchange(
        artifacts=[source],
        handoffs=[resolved],
        project_id="project-1",
        thread_id="thread-1",
        root_run_id="run-1",
        plan_id="plan-1",
        plan_sha256=SHA,
        step_id="source-step",
        checkpoint_binding=binding,
        source_snapshot=source.source_capability,
    )
    return store, source, planned


def test_artifact_envelope_hash_covers_denormalized_database_identity(
    tmp_path: Path,
) -> None:
    store, source, _planned = _persist_exchange_fixture(tmp_path)
    with sqlite3.connect(tmp_path / "data" / "langgraph" / "artifact_records.sqlite") as db:
        db.execute(
            "UPDATE artifact_records SET artifact_id='alias-artifact' WHERE artifact_id=?",
            (source.artifact_id,),
        )

    with pytest.raises(ValueError, match="hash mismatch"):
        store.load_artifact_record(
            "alias-artifact",
            project_id="project-1",
            thread_id="thread-1",
            root_run_id="run-1",
            plan_id="plan-1",
            plan_sha256=SHA,
        )


def test_handoff_envelope_hash_covers_database_fields_and_mappings(
    tmp_path: Path,
) -> None:
    store, source, planned = _persist_exchange_fixture(tmp_path)
    with sqlite3.connect(tmp_path / "data" / "langgraph" / "artifact_records.sqlite") as db:
        db.execute(
            "UPDATE handoff_records SET target_step_id='tampered-target' WHERE handoff_id=?",
            ("handoff-envelope",),
        )

    with pytest.raises(ValueError, match="hash mismatch"):
        store.load_handoff(
            planned,
            "handoff-envelope",
            project_id="project-1",
            thread_id="thread-1",
            root_run_id="run-1",
            plan_id="plan-1",
            plan_sha256=SHA,
            source_snapshot=source.source_capability,
            target_snapshot=source.source_capability,
        )

