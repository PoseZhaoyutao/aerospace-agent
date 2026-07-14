"""Content-addressed domain artifact resolution and handoff validation.

No caller receives a filesystem path.  Domain payloads enter the runtime only
through :class:`ArtifactStore`, which revalidates every persisted claim before
returning decoded JSON.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as JsonSchemaValidationError
from pydantic import JsonValue

from .models import (
    ArtifactSchemaManifest,
    CapabilitySnapshot,
    CheckpointRef,
    CrossDomainHandoff,
    DomainArtifact,
    FrozenContractModel,
    HandoffRecord,
)


class ResolvedArtifact(FrozenContractModel):
    """Verified artifact plus its decoded payload and approved schema."""

    artifact: DomainArtifact
    payload: JsonValue
    schema_manifest: ArtifactSchemaManifest


class ArtifactCheckpointBinding(FrozenContractModel):
    """Exact immutable DAG checkpoint identity asserted by an artifact."""

    project_id: str
    thread_id: str
    root_run_id: str
    plan_id: str
    plan_sha256: str
    step_id: str
    checkpoint_id: str
    phase: Literal["before", "after"]


class ResolvedHandoff(FrozenContractModel):
    """All concrete artifacts accepted for a cross-domain handoff."""

    record: HandoffRecord
    planned: CrossDomainHandoff
    sources: list[ResolvedArtifact]
    targets: list[ResolvedArtifact]
    target_inputs: dict[str, JsonValue]
    source_snapshot: CapabilitySnapshot
    target_snapshot: CapabilitySnapshot
    source_checkpoint_binding: ArtifactCheckpointBinding
    validation_results: dict[str, bool]


class ArtifactRecord(FrozenContractModel):
    record_id: str
    artifact_id: str
    project_id: str
    thread_id: str
    root_run_id: str
    plan_id: str
    plan_sha256: str
    step_id: str
    artifact_sha256: str
    source_snapshot: CapabilitySnapshot
    checkpoint_binding: ArtifactCheckpointBinding
    artifact: DomainArtifact
    created_at: str


class RecoveredExchange(FrozenContractModel):
    checkpoint_binding: ArtifactCheckpointBinding
    source_artifacts: list[DomainArtifact]
    artifact_ids: list[str]
    handoff_ids: list[str]


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _artifact_envelope_sha256(
    *,
    project_id: str,
    plan_id: str,
    artifact_id: str,
    record_json: str,
) -> str:
    return _canonical_sha256(
        {
            "project_id": project_id,
            "plan_id": plan_id,
            "artifact_id": artifact_id,
            "record": json.loads(record_json),
        }
    )


def _handoff_envelope_sha256(row: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "project_id": row["project_id"],
            "plan_id": row["plan_id"],
            "handoff_id": row["handoff_id"],
            "target_step_id": row["target_step_id"],
            "root_run_id": row["root_run_id"],
            "thread_id": row["thread_id"],
            "plan_sha256": row["plan_sha256"],
            "planned": json.loads(str(row["planned_json"])),
            "record": json.loads(str(row["record_json"])),
            "binding": json.loads(str(row["binding_json"])),
            "source_snapshot": json.loads(str(row["source_snapshot_json"])),
            "target_snapshot": json.loads(str(row["target_snapshot_json"])),
            "validation_results": json.loads(str(row["validation_results_json"])),
        }
    )


def _hash_matches(expected: str, actual: object) -> bool:
    return isinstance(actual, str) and hmac.compare_digest(expected, actual)


class ArtifactStore:
    """Sole resolver for project-local ``artifact://sha256/<digest>`` URIs."""

    def __init__(
        self,
        workspace_root: str | Path,
        schema_manifests: Iterable[ArtifactSchemaManifest],
        *,
        schema_approval_verifier: Callable[[ArtifactSchemaManifest], bool],
        checkpoint_verifier: Callable[[CheckpointRef], bool],
        capability_snapshot_verifier: Callable[[CapabilitySnapshot], bool],
        database_path: str | Path | None = None,
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve(strict=True)
        self._artifact_root = (
            self._workspace_root / "data" / "langgraph" / "artifacts" / "sha256"
        )
        self._checkpoint_verifier = checkpoint_verifier
        self._capability_snapshot_verifier = capability_snapshot_verifier
        self._database_path = Path(database_path) if database_path is not None else (
            self._workspace_root / "data" / "langgraph" / "artifact_records.sqlite"
        )
        if not self._database_path.is_absolute():
            self._database_path = self._workspace_root / self._database_path
        self._database_path = self._database_path.resolve()
        try:
            self._database_path.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("artifact record database must stay inside workspace") from exc
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._schemas: dict[tuple[str, str], ArtifactSchemaManifest] = {}

        for manifest in schema_manifests:
            verified = ArtifactSchemaManifest.model_validate(manifest.model_dump(mode="python"))
            if not schema_approval_verifier(verified):
                raise ValueError(
                    f"artifact schema is not approved: {verified.schema_id}@{verified.schema_version}"
                )
            try:
                Draft202012Validator.check_schema(verified.payload_json_schema)
            except SchemaError as exc:
                raise ValueError(
                    f"approved artifact schema is invalid: {verified.schema_id}@{verified.schema_version}"
                ) from exc
            key = (verified.schema_id, verified.schema_version)
            if key in self._schemas:
                raise ValueError(f"duplicate approved artifact schema: {key[0]}@{key[1]}")
            self._schemas[key] = verified
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 1:
                raise RuntimeError(f"unsupported artifact record schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.executescript(
                    """
                    CREATE TABLE artifact_records (
                        project_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        artifact_id TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        record_sha256 TEXT NOT NULL,
                        PRIMARY KEY(project_id, plan_id, artifact_id)
                    );
                    CREATE TABLE handoff_records (
                        project_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        handoff_id TEXT NOT NULL,
                        target_step_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        plan_sha256 TEXT NOT NULL,
                        planned_json TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        binding_json TEXT NOT NULL,
                        source_snapshot_json TEXT NOT NULL,
                        target_snapshot_json TEXT NOT NULL,
                        validation_results_json TEXT NOT NULL,
                        record_sha256 TEXT NOT NULL,
                        PRIMARY KEY(project_id, plan_id, handoff_id)
                    );
                    CREATE INDEX handoff_target_idx
                    ON handoff_records(project_id, plan_id, target_step_id);
                    """
                )
                connection.execute("PRAGMA user_version = 1")
                connection.commit()

    def resolve(
        self,
        artifact: DomainArtifact,
        *,
        project_id: str,
        thread_id: str,
    ) -> ResolvedArtifact:
        """Resolve and revalidate an artifact without exposing its local path."""

        schema = self._approved_schema(artifact)
        self._validate_source_identity(artifact, project_id=project_id, thread_id=thread_id)
        content = self._read_content(artifact)
        payload = self._decode_and_validate_payload(content, artifact, schema)
        self._validate_metadata(payload, artifact, schema)
        return ResolvedArtifact(artifact=artifact, payload=payload, schema_manifest=schema)

    def validate_handoff(
        self,
        planned: CrossDomainHandoff,
        record: HandoffRecord,
        *,
        source_artifacts: Mapping[str, DomainArtifact],
        target_artifacts: Mapping[str, DomainArtifact],
        project_id: str,
        thread_id: str,
        plan_id: str,
        plan_sha256: str,
        source_snapshot: CapabilitySnapshot,
        target_snapshot: CapabilitySnapshot,
        source_checkpoint_binding: ArtifactCheckpointBinding,
        validation_results: Mapping[str, bool],
    ) -> ResolvedHandoff:
        """Validate concrete source, conversion output, and target handoff data."""

        if record.plan_id != plan_id:
            raise ValueError("handoff record plan identity does not match the active plan")
        if (record.source_step_id, record.target_step_id) != (
            planned.source_step_id,
            planned.target_step_id,
        ):
            raise ValueError("handoff record step identity does not match the plan")
        if (
            source_checkpoint_binding.project_id != project_id
            or source_checkpoint_binding.thread_id != thread_id
            or source_checkpoint_binding.plan_id != plan_id
            or source_checkpoint_binding.plan_sha256 != plan_sha256
            or source_checkpoint_binding.step_id != planned.source_step_id
            or source_checkpoint_binding.checkpoint_id != record.checkpoint.checkpoint_id
            or source_checkpoint_binding.phase != "after"
        ):
            raise ValueError("handoff source checkpoint plan/step/phase binding is invalid")
        self._validate_checkpoint(record.checkpoint, project_id=project_id, thread_id=thread_id)

        sources = self._resolve_named(
            record.source_artifact_ids,
            source_artifacts,
            role="source",
            project_id=project_id,
            thread_id=thread_id,
        )
        targets = self._resolve_named(
            record.target_artifact_ids,
            target_artifacts,
            role="target",
            project_id=project_id,
            thread_id=thread_id,
        )

        for resolved in sources:
            if resolved.artifact.source_capability != source_snapshot:
                raise ValueError("actual source capability snapshot does not match the plan")
            if resolved.artifact.metadata != planned.source_metadata:
                raise ValueError("actual source metadata does not match planned source metadata")
            if record.checkpoint not in resolved.artifact.source_checkpoints:
                raise ValueError("source artifact does not cite the bound handoff checkpoint")
        for resolved in targets:
            if resolved.artifact.source_capability != target_snapshot:
                raise ValueError("actual target capability snapshot does not match the plan")
            if resolved.artifact.metadata != planned.target_metadata:
                raise ValueError("actual target metadata does not match planned target metadata")

        self._validate_payload_fields(sources, planned.expected_outputs, role="source")
        if targets:
            target_fields = (
                list(planned.conversion.output_mapping.values())
                if planned.conversion is not None
                else planned.required_inputs
            )
            self._validate_payload_fields(targets, target_fields, role="target")

        declared_checks = set(record.validation_check_ids)
        if set(validation_results) != declared_checks or not declared_checks:
            raise ValueError("handoff validation results do not exactly match declared checks")
        if any(validation_results[check_id] is not True for check_id in declared_checks):
            raise ValueError("handoff validation check did not pass")

        target_inputs = self._apply_mappings(planned, sources, targets)

        if planned.conversion is None:
            if record.conversion_capability is not None:
                raise ValueError("handoff record declares an unplanned conversion capability")
        else:
            if not targets:
                raise ValueError("converted handoff requires a concrete target artifact")
            if record.conversion_capability != planned.conversion.converter_capability:
                raise ValueError("handoff conversion capability does not match the plan")
            if planned.conversion.validation_check_id not in record.validation_check_ids:
                raise ValueError("handoff conversion validation check is missing")

        return ResolvedHandoff(
            record=record,
            planned=planned,
            sources=sources,
            targets=targets,
            target_inputs=target_inputs,
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            source_checkpoint_binding=source_checkpoint_binding,
            validation_results=dict(validation_results),
        )

    def persist_exchange(
        self,
        *,
        artifacts: Iterable[DomainArtifact],
        handoffs: Iterable[ResolvedHandoff],
        project_id: str,
        thread_id: str,
        root_run_id: str,
        plan_id: str,
        plan_sha256: str,
        step_id: str,
        checkpoint_binding: ArtifactCheckpointBinding,
        source_snapshot: CapabilitySnapshot,
    ) -> tuple[list[ArtifactRecord], list[HandoffRecord]]:
        """Atomically append immutable artifact and handoff records."""

        if (
            checkpoint_binding.project_id,
            checkpoint_binding.thread_id,
            checkpoint_binding.root_run_id,
            checkpoint_binding.plan_id,
            checkpoint_binding.plan_sha256,
            checkpoint_binding.step_id,
            checkpoint_binding.phase,
        ) != (
            project_id,
            thread_id,
            root_run_id,
            plan_id,
            plan_sha256,
            step_id,
            "after",
        ):
            raise ValueError("artifact persistence checkpoint binding is not exact")
        now = datetime.now(UTC).isoformat()
        concrete: dict[str, DomainArtifact] = {
            item.artifact_id: DomainArtifact.model_validate(item.model_dump(mode="python"))
            for item in artifacts
        }
        artifact_snapshots: dict[str, CapabilitySnapshot] = {
            artifact_id: source_snapshot for artifact_id in concrete
        }
        resolved_handoffs = list(handoffs)
        for handoff in resolved_handoffs:
            if (
                handoff.record.plan_id != plan_id
                or handoff.record.source_step_id != step_id
                or handoff.source_checkpoint_binding != checkpoint_binding
                or handoff.source_snapshot != source_snapshot
            ):
                raise ValueError("handoff persistence identity does not match source execution")
            for resolved in [*handoff.sources, *handoff.targets]:
                concrete.setdefault(resolved.artifact.artifact_id, resolved.artifact)
            for resolved in handoff.sources:
                artifact_snapshots[resolved.artifact.artifact_id] = handoff.source_snapshot
            for resolved in handoff.targets:
                artifact_snapshots[resolved.artifact.artifact_id] = handoff.target_snapshot

        artifact_records: list[ArtifactRecord] = []
        expected_checkpoint = CheckpointRef(
            project_id=project_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_binding.checkpoint_id,
        )
        for artifact in concrete.values():
            self.resolve(artifact, project_id=project_id, thread_id=thread_id)
            artifact_snapshot = artifact_snapshots.get(artifact.artifact_id, source_snapshot)
            if artifact.source_capability != artifact_snapshot:
                raise ValueError("artifact source snapshot does not match persisted execution")
            if expected_checkpoint not in artifact.source_checkpoints:
                raise ValueError("artifact does not cite the persisted after checkpoint")
            artifact_hash = _canonical_sha256(artifact)
            artifact_records.append(
                ArtifactRecord(
                    record_id=f"artifact-record:{artifact_hash}",
                    artifact_id=artifact.artifact_id,
                    project_id=project_id,
                    thread_id=thread_id,
                    root_run_id=root_run_id,
                    plan_id=plan_id,
                    plan_sha256=plan_sha256,
                    step_id=step_id,
                    artifact_sha256=artifact_hash,
                    source_snapshot=artifact_snapshot,
                    checkpoint_binding=checkpoint_binding,
                    artifact=artifact,
                    created_at=now,
                )
            )

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for index, record in enumerate(artifact_records):
                record_json = _canonical_json(record)
                record_hash = _artifact_envelope_sha256(
                    project_id=project_id,
                    plan_id=plan_id,
                    artifact_id=record.artifact_id,
                    record_json=record_json,
                )
                prior = connection.execute(
                    "SELECT * FROM artifact_records WHERE project_id=? AND plan_id=? AND artifact_id=?",
                    (project_id, plan_id, record.artifact_id),
                ).fetchone()
                if prior is not None:
                    prior_hash = _artifact_envelope_sha256(
                        project_id=str(prior["project_id"]),
                        plan_id=str(prior["plan_id"]),
                        artifact_id=str(prior["artifact_id"]),
                        record_json=str(prior["record_json"]),
                    )
                    if not _hash_matches(prior_hash, prior["record_sha256"]):
                        connection.rollback()
                        raise ValueError("persisted artifact record hash mismatch")
                    persisted = ArtifactRecord.model_validate(
                        json.loads(prior["record_json"])
                    )
                    current_identity = record.model_dump(mode="json")
                    persisted_identity = persisted.model_dump(mode="json")
                    current_identity.pop("created_at", None)
                    persisted_identity.pop("created_at", None)
                    if current_identity != persisted_identity:
                        connection.rollback()
                        raise ValueError(
                            "immutable artifact record already exists with different content"
                        )
                    artifact_records[index] = persisted
                else:
                    connection.execute(
                        "INSERT INTO artifact_records(project_id, plan_id, artifact_id, record_json, record_sha256) VALUES (?, ?, ?, ?, ?)",
                        (project_id, plan_id, record.artifact_id, record_json, record_hash),
                    )
            for resolved in resolved_handoffs:
                record_json = _canonical_json(resolved.record)
                planned_json = _canonical_json(resolved.planned)
                binding_json = _canonical_json(resolved.source_checkpoint_binding)
                source_snapshot_json = _canonical_json(resolved.source_snapshot)
                target_snapshot_json = _canonical_json(resolved.target_snapshot)
                validation_json = _canonical_json(resolved.validation_results)
                envelope_hash = _handoff_envelope_sha256(
                    {
                        "project_id": project_id,
                        "plan_id": plan_id,
                        "handoff_id": resolved.record.handoff_id,
                        "target_step_id": resolved.record.target_step_id,
                        "root_run_id": root_run_id,
                        "thread_id": thread_id,
                        "plan_sha256": plan_sha256,
                        "planned_json": planned_json,
                        "record_json": record_json,
                        "binding_json": binding_json,
                        "source_snapshot_json": source_snapshot_json,
                        "target_snapshot_json": target_snapshot_json,
                        "validation_results_json": validation_json,
                    }
                )
                prior = connection.execute(
                    "SELECT * FROM handoff_records WHERE project_id=? AND plan_id=? AND handoff_id=?",
                    (project_id, plan_id, resolved.record.handoff_id),
                ).fetchone()
                if prior is not None:
                    prior_hash = _handoff_envelope_sha256(prior)
                    if not _hash_matches(prior_hash, prior["record_sha256"]):
                        connection.rollback()
                        raise ValueError("persisted handoff record hash mismatch")
                    if not hmac.compare_digest(str(prior["record_sha256"]), envelope_hash):
                        connection.rollback()
                        raise ValueError("immutable handoff record already exists with different content")
                if prior is None:
                    connection.execute(
                        """
                        INSERT INTO handoff_records(
                            project_id, plan_id, handoff_id, target_step_id, root_run_id,
                            thread_id, plan_sha256, planned_json, record_json, binding_json,
                            source_snapshot_json, target_snapshot_json,
                            validation_results_json, record_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            plan_id,
                            resolved.record.handoff_id,
                            resolved.record.target_step_id,
                            root_run_id,
                            thread_id,
                            plan_sha256,
                            planned_json,
                            record_json,
                            binding_json,
                            source_snapshot_json,
                            target_snapshot_json,
                            validation_json,
                            envelope_hash,
                        ),
                    )
            connection.commit()
        return artifact_records, [item.record for item in resolved_handoffs]

    def load_handoff(
        self,
        planned: CrossDomainHandoff,
        handoff_id: str,
        *,
        project_id: str,
        thread_id: str,
        root_run_id: str,
        plan_id: str,
        plan_sha256: str,
        source_snapshot: CapabilitySnapshot,
        target_snapshot: CapabilitySnapshot,
    ) -> ResolvedHandoff:
        """Reload and fully revalidate an immutable handoff before target dispatch."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM handoff_records WHERE project_id=? AND plan_id=? AND handoff_id=?",
                (project_id, plan_id, handoff_id),
            ).fetchone()
            if row is None:
                raise ValueError("persisted handoff record is missing")
            try:
                envelope_hash = _handoff_envelope_sha256(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("persisted handoff record hash mismatch") from exc
            if not _hash_matches(envelope_hash, row["record_sha256"]):
                raise ValueError("persisted handoff record hash mismatch")
            if (
                row["thread_id"],
                row["root_run_id"],
                row["plan_sha256"],
                row["planned_json"],
                row["source_snapshot_json"],
                row["target_snapshot_json"],
            ) != (
                thread_id,
                root_run_id,
                plan_sha256,
                _canonical_json(planned),
                _canonical_json(source_snapshot),
                _canonical_json(target_snapshot),
            ):
                raise ValueError("persisted handoff plan or frozen snapshot binding mismatch")
            record = HandoffRecord.model_validate(json.loads(row["record_json"]))
            binding = ArtifactCheckpointBinding.model_validate(
                json.loads(row["binding_json"])
            )
            validation_results = json.loads(row["validation_results_json"])
            artifact_ids = [*record.source_artifact_ids, *record.target_artifact_ids]
            artifacts: dict[str, DomainArtifact] = {}
            for artifact_id in artifact_ids:
                artifact_row = connection.execute(
                    "SELECT * FROM artifact_records WHERE project_id=? AND plan_id=? AND artifact_id=?",
                    (project_id, plan_id, artifact_id),
                ).fetchone()
                if artifact_row is None:
                    raise ValueError(f"persisted artifact record is missing: {artifact_id}")
                try:
                    artifact_hash = _artifact_envelope_sha256(
                        project_id=str(artifact_row["project_id"]),
                        plan_id=str(artifact_row["plan_id"]),
                        artifact_id=str(artifact_row["artifact_id"]),
                        record_json=str(artifact_row["record_json"]),
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError("persisted artifact record hash mismatch") from exc
                if not _hash_matches(artifact_hash, artifact_row["record_sha256"]):
                    raise ValueError("persisted artifact record hash mismatch")
                artifact_record = ArtifactRecord.model_validate(
                    json.loads(artifact_row["record_json"])
                )
                if (
                    artifact_record.project_id,
                    artifact_record.thread_id,
                    artifact_record.root_run_id,
                    artifact_record.plan_id,
                    artifact_record.plan_sha256,
                ) != (project_id, thread_id, root_run_id, plan_id, plan_sha256):
                    raise ValueError("persisted artifact namespace or plan binding mismatch")
                artifacts[artifact_id] = artifact_record.artifact
        return self.validate_handoff(
            planned,
            record,
            source_artifacts={
                item: artifacts[item] for item in record.source_artifact_ids
            },
            target_artifacts={
                item: artifacts[item] for item in record.target_artifact_ids
            },
            project_id=project_id,
            thread_id=thread_id,
            plan_id=plan_id,
            plan_sha256=plan_sha256,
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            source_checkpoint_binding=binding,
            validation_results=validation_results,
        )

    def handoff_ids_for_target(
        self,
        *,
        project_id: str,
        plan_id: str,
        target_step_id: str,
    ) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT handoff_id FROM handoff_records
                WHERE project_id=? AND plan_id=? AND target_step_id=?
                ORDER BY handoff_id
                """,
                (project_id, plan_id, target_step_id),
            ).fetchall()
        return [str(row["handoff_id"]) for row in rows]

    def recover_exchange(
        self,
        *,
        project_id: str,
        thread_id: str,
        root_run_id: str,
        plan_id: str,
        plan_sha256: str,
        step_id: str,
        source_snapshot: CapabilitySnapshot,
        expected_target_step_ids: Iterable[str],
    ) -> RecoveredExchange | None:
        """Return one complete append-only exchange left before its DAG checkpoint."""

        with closing(self._connect()) as connection:
            artifact_rows = connection.execute(
                "SELECT * FROM artifact_records WHERE project_id=? AND plan_id=?",
                (project_id, plan_id),
            ).fetchall()
            handoff_rows = connection.execute(
                "SELECT * FROM handoff_records WHERE project_id=? AND plan_id=?",
                (project_id, plan_id),
            ).fetchall()
        records: list[ArtifactRecord] = []
        for row in artifact_rows:
            try:
                envelope_hash = _artifact_envelope_sha256(
                    project_id=str(row["project_id"]),
                    plan_id=str(row["plan_id"]),
                    artifact_id=str(row["artifact_id"]),
                    record_json=str(row["record_json"]),
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("persisted artifact record hash mismatch") from exc
            if not _hash_matches(envelope_hash, row["record_sha256"]):
                raise ValueError("persisted artifact record hash mismatch")
            record = ArtifactRecord.model_validate(json.loads(row["record_json"]))
            if record.step_id == step_id:
                records.append(record)
        if not records:
            return None
        bindings = {record.checkpoint_binding for record in records}
        if len(bindings) != 1:
            raise ValueError("orphaned artifact exchange has inconsistent checkpoint bindings")
        binding = next(iter(bindings))
        if (
            binding.project_id,
            binding.thread_id,
            binding.root_run_id,
            binding.plan_id,
            binding.plan_sha256,
            binding.step_id,
            binding.phase,
        ) != (
            project_id,
            thread_id,
            root_run_id,
            plan_id,
            plan_sha256,
            step_id,
            "after",
        ):
            raise ValueError("orphaned artifact checkpoint binding mismatch")
        for record in records:
            self.load_artifact_record(
                record.artifact_id,
                project_id=project_id,
                thread_id=thread_id,
                root_run_id=root_run_id,
                plan_id=plan_id,
                plan_sha256=plan_sha256,
            )
        source_artifacts = [
            record.artifact for record in records if record.source_snapshot == source_snapshot
        ]
        if not source_artifacts:
            raise ValueError("orphaned exchange has no source-domain artifact")

        outgoing: list[HandoffRecord] = []
        for row in handoff_rows:
            try:
                envelope_hash = _handoff_envelope_sha256(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("persisted handoff record hash mismatch") from exc
            if not _hash_matches(envelope_hash, row["record_sha256"]):
                raise ValueError("persisted handoff record hash mismatch")
            record = HandoffRecord.model_validate(json.loads(row["record_json"]))
            if record.source_step_id == step_id:
                if record.checkpoint.checkpoint_id != binding.checkpoint_id:
                    raise ValueError("orphaned handoff checkpoint binding mismatch")
                outgoing.append(record)
        if {record.target_step_id for record in outgoing} != set(expected_target_step_ids):
            raise ValueError("orphaned exchange does not contain the complete planned handoff set")
        return RecoveredExchange(
            checkpoint_binding=binding,
            source_artifacts=source_artifacts,
            artifact_ids=sorted(record.artifact_id for record in records),
            handoff_ids=sorted(record.handoff_id for record in outgoing),
        )

    def load_artifact_record(
        self,
        artifact_id: str,
        *,
        project_id: str,
        thread_id: str,
        root_run_id: str,
        plan_id: str,
        plan_sha256: str,
    ) -> ArtifactRecord:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM artifact_records WHERE project_id=? AND plan_id=? AND artifact_id=?",
                (project_id, plan_id, artifact_id),
            ).fetchone()
        if row is None:
            raise ValueError("persisted artifact record is missing")
        try:
            envelope_hash = _artifact_envelope_sha256(
                project_id=str(row["project_id"]),
                plan_id=str(row["plan_id"]),
                artifact_id=str(row["artifact_id"]),
                record_json=str(row["record_json"]),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("persisted artifact record hash mismatch") from exc
        if not _hash_matches(envelope_hash, row["record_sha256"]):
            raise ValueError("persisted artifact record hash mismatch")
        record = ArtifactRecord.model_validate(json.loads(row["record_json"]))
        if (
            record.project_id,
            record.thread_id,
            record.root_run_id,
            record.plan_id,
            record.plan_sha256,
        ) != (project_id, thread_id, root_run_id, plan_id, plan_sha256):
            raise ValueError("persisted artifact namespace or plan binding mismatch")
        self.resolve(record.artifact, project_id=project_id, thread_id=thread_id)
        return record

    @staticmethod
    def _apply_mappings(
        planned: CrossDomainHandoff,
        sources: list[ResolvedArtifact],
        targets: list[ResolvedArtifact],
    ) -> dict[str, JsonValue]:
        if planned.conversion is not None:
            if not targets:
                raise ValueError("converted handoff mapping requires target artifacts")
            converted_inputs: dict[str, JsonValue] = {}
            for target_input, artifact_field in planned.conversion.output_mapping.items():
                values = [
                    item.payload[artifact_field]
                    for item in targets
                    if isinstance(item.payload, dict) and artifact_field in item.payload
                ]
                if len(values) != 1 or target_input in converted_inputs:
                    raise ValueError(
                        f"converted handoff mapping field is ambiguous or missing: {artifact_field}"
                    )
                converted_inputs[target_input] = values[0]
            if set(converted_inputs) != set(planned.required_inputs):
                raise ValueError(
                    "converted handoff mapping does not exactly produce required inputs"
                )
            return converted_inputs
        if not sources:
            raise ValueError("handoff mapping requires at least one source artifact")
        transfer: dict[str, JsonValue] = {}
        for source_field, transfer_field in planned.source_output_mapping.items():
            found = False
            for source in sources:
                if isinstance(source.payload, dict) and source_field in source.payload:
                    if transfer_field in transfer:
                        raise ValueError("handoff source mapping produced a duplicate transfer field")
                    transfer[transfer_field] = source.payload[source_field]
                    found = True
                    break
            if not found:
                raise ValueError(f"handoff source mapping field is missing: {source_field}")
        target_inputs: dict[str, JsonValue] = {}
        for transfer_field, target_field in planned.target_input_mapping.items():
            if transfer_field not in transfer:
                raise ValueError(
                    f"handoff target mapping references missing transfer field: {transfer_field}"
                )
            if target_field in target_inputs:
                raise ValueError("handoff target mapping produced a duplicate target field")
            target_inputs[target_field] = transfer[transfer_field]
        if set(target_inputs) != set(planned.required_inputs):
            raise ValueError("handoff target mapping does not exactly produce required inputs")
        return target_inputs

    def _approved_schema(self, artifact: DomainArtifact) -> ArtifactSchemaManifest:
        key = (artifact.schema_id, artifact.schema_version)
        try:
            return self._schemas[key]
        except KeyError as exc:
            raise ValueError(f"artifact schema is not approved: {key[0]}@{key[1]}") from exc

    def _validate_source_identity(
        self,
        artifact: DomainArtifact,
        *,
        project_id: str,
        thread_id: str,
    ) -> None:
        if not self._capability_snapshot_verifier(artifact.source_capability):
            raise ValueError("source capability snapshot is not current and approved")
        for checkpoint in artifact.source_checkpoints:
            self._validate_checkpoint(
                checkpoint,
                project_id=project_id,
                thread_id=thread_id,
            )

    def _validate_checkpoint(
        self,
        checkpoint: CheckpointRef,
        *,
        project_id: str,
        thread_id: str,
    ) -> None:
        if checkpoint.project_id != project_id or checkpoint.thread_id != thread_id:
            raise ValueError("checkpoint namespace does not match artifact request")
        if not self._checkpoint_verifier(checkpoint):
            raise ValueError("checkpoint identity could not be verified")

    def _read_content(self, artifact: DomainArtifact) -> bytes:
        reference = artifact.payload_ref
        parsed = urlparse(reference.uri)
        digest = parsed.path.removeprefix("/")
        if (
            parsed.scheme != "artifact"
            or parsed.netloc != "sha256"
            or parsed.params
            or parsed.query
            or parsed.fragment
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest != reference.sha256.lower()
        ):
            raise ValueError("artifact URI must be artifact://sha256/<matching lowercase digest>")

        candidate = self._artifact_root / digest
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self._artifact_root.resolve())
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ValueError("artifact URI does not resolve to project-local content") from exc
        if not resolved.is_file():
            raise ValueError("artifact URI must resolve to a regular project-local file")

        content = resolved.read_bytes()
        actual_digest = hashlib.sha256(content).hexdigest()
        if actual_digest != reference.sha256.lower():
            raise ValueError("artifact SHA-256 does not match persisted content")
        if len(content) != reference.byte_length:
            raise ValueError("artifact byte length does not match persisted content")
        return content

    @staticmethod
    def _decode_and_validate_payload(
        content: bytes,
        artifact: DomainArtifact,
        schema: ArtifactSchemaManifest,
    ) -> JsonValue:
        if artifact.payload_ref.media_type != "application/json" and not artifact.payload_ref.media_type.endswith(
            "+json"
        ):
            raise ValueError("domain artifact payload must use a JSON media type")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("domain artifact payload is not valid UTF-8 JSON") from exc
        try:
            Draft202012Validator(schema.payload_json_schema).validate(payload)
        except JsonSchemaValidationError as exc:
            raise ValueError(f"payload schema validation failed: {exc.message}") from exc
        return payload

    @staticmethod
    def _validate_metadata(
        payload: JsonValue,
        artifact: DomainArtifact,
        schema: ArtifactSchemaManifest,
    ) -> None:
        if not isinstance(payload, dict):
            if schema.required_quantity_fields or schema.requires_epoch_field:
                raise ValueError("metadata requirements need an object payload")
            payload_fields: set[str] = set()
        else:
            payload_fields = set(payload)

        for field_name in schema.required_quantity_fields:
            if field_name not in payload_fields:
                raise ValueError(f"required quantity payload field is missing: {field_name}")
            unit = artifact.metadata.quantity_units.get(field_name)
            if not unit:
                raise ValueError(f"required quantity metadata is missing: {field_name}")
        if schema.requires_frame and not artifact.metadata.frame_id:
            raise ValueError("artifact schema requires frame metadata")
        if schema.requires_time_system and not artifact.metadata.time_system:
            raise ValueError("artifact schema requires time-system metadata")
        if schema.requires_epoch_field:
            epoch_field = artifact.metadata.epoch_field
            if not epoch_field or epoch_field not in payload_fields:
                raise ValueError("artifact schema requires valid epoch-field metadata")

    def _resolve_named(
        self,
        artifact_ids: list[str],
        artifacts: Mapping[str, DomainArtifact],
        *,
        role: str,
        project_id: str,
        thread_id: str,
    ) -> list[ResolvedArtifact]:
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError(f"duplicate concrete {role} artifact ID")
        resolved: list[ResolvedArtifact] = []
        for artifact_id in artifact_ids:
            try:
                artifact = artifacts[artifact_id]
            except KeyError as exc:
                raise ValueError(f"missing concrete {role} artifact: {artifact_id}") from exc
            if artifact.artifact_id != artifact_id:
                raise ValueError(f"{role} artifact key does not match artifact identity")
            resolved.append(
                self.resolve(artifact, project_id=project_id, thread_id=thread_id)
            )
        return resolved

    @staticmethod
    def _validate_payload_fields(
        artifacts: list[ResolvedArtifact],
        fields: list[str],
        *,
        role: str,
    ) -> None:
        if not fields:
            return
        for artifact in artifacts:
            if not isinstance(artifact.payload, dict):
                raise ValueError(f"{role} payload must be an object for declared handoff fields")
            missing = [field for field in fields if field not in artifact.payload]
            if missing:
                raise ValueError(
                    f"{role} payload is missing declared handoff fields: {', '.join(missing)}"
                )


__all__ = [
    "ArtifactCheckpointBinding",
    "ArtifactStore",
    "ResolvedArtifact",
    "ResolvedHandoff",
]

