"""Immutable, approval-bound workflow registry.

The registry stores the exact workflow body and manifest used at approval
time.  A workflow ID or version is never treated as sufficient authority.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .approval import CapabilityApprovalVerifier
from .models import FrozenContractModel, WorkflowManifest, WorkflowSnapshot


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"workflow content is not canonical JSON: {exc}") from exc


def canonical_workflow_sha256(workflow_body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(workflow_body)).hexdigest()


def canonical_manifest_sha256(manifest: WorkflowManifest | Mapping[str, Any]) -> str:
    data = dict(_json_value(manifest))
    data.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(data)).hexdigest()


def workflow_approval_digest(manifest: WorkflowManifest) -> str:
    material = {
        "workflow_id": manifest.workflow_id,
        "version": manifest.version,
        "workflow_sha256": manifest.workflow_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "approval_record_id": manifest.approval_record_id,
    }
    return hashlib.sha256(_canonical_bytes(material)).hexdigest()


class LockedWorkflow(FrozenContractModel):
    manifest: WorkflowManifest
    body: dict[str, Any]


class WorkflowRegistry:
    """Versioned SQLite registry for exact approved workflow snapshots."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        database_path: str | Path,
        *,
        approval_verifier: CapabilityApprovalVerifier,
    ) -> None:
        if not isinstance(approval_verifier, CapabilityApprovalVerifier):
            raise TypeError("approval_verifier must be CapabilityApprovalVerifier")
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._approval_verifier = approval_verifier
        self._available = True
        self._degraded_reason: str | None = None
        try:
            self._migrate()
        except Exception as exc:
            self._available = False
            self._degraded_reason = str(exc)

    def availability(self) -> dict[str, Any]:
        return {"available": self._available, "reason": self._degraded_reason}

    def _require_available(self) -> None:
        if not self._available:
            raise RuntimeError(
                f"workflow registry is unavailable: {self._degraded_reason or 'migration failed'}"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported workflow schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE workflows (
                        workflow_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        workflow_sha256 TEXT NOT NULL,
                        manifest_sha256 TEXT NOT NULL,
                        approval_record_id TEXT NOT NULL,
                        approval_digest TEXT NOT NULL,
                        body_json TEXT NOT NULL,
                        manifest_json TEXT NOT NULL,
                        PRIMARY KEY(workflow_id, version)
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def schema_version(self) -> int:
        self._require_available()
        with closing(self._connect()) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def register(
        self,
        *,
        manifest: WorkflowManifest,
        workflow_body: Mapping[str, Any],
    ) -> WorkflowSnapshot:
        self._require_available()
        checked = WorkflowManifest.model_validate(manifest.model_dump(mode="python"))
        body = dict(_json_value(workflow_body))
        self._validate_body_manifest(checked, body)
        digest = workflow_approval_digest(checked)
        approval = self._approval_verifier.verify_digest(digest)
        if approval is None or approval.approval_record_id != checked.approval_record_id:
            raise PermissionError("workflow approval is missing or invalid")
        body_json = _canonical_bytes(body).decode("utf-8")
        manifest_json = _canonical_bytes(checked).decode("utf-8")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT workflow_sha256, manifest_sha256 FROM workflows "
                "WHERE workflow_id = ? AND version = ?",
                (checked.workflow_id, checked.version),
            ).fetchone()
            if existing is not None:
                if (
                    existing["workflow_sha256"] != checked.workflow_sha256
                    or existing["manifest_sha256"] != checked.manifest_sha256
                ):
                    connection.rollback()
                    raise ValueError("workflow identity/version is immutable")
                connection.rollback()
                return self._snapshot(checked)
            connection.execute(
                """
                INSERT INTO workflows(
                    workflow_id, version, workflow_sha256, manifest_sha256,
                    approval_record_id, approval_digest, body_json, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checked.workflow_id,
                    checked.version,
                    checked.workflow_sha256,
                    checked.manifest_sha256,
                    checked.approval_record_id,
                    digest,
                    body_json,
                    manifest_json,
                ),
            )
            connection.commit()
        return self._snapshot(checked)

    def get(self, workflow_id: str, version: str) -> LockedWorkflow:
        self._require_available()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM workflows WHERE workflow_id = ? AND version = ?",
                (workflow_id, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"workflow not found: {workflow_id}@{version}")
        try:
            manifest = WorkflowManifest.model_validate(json.loads(row["manifest_json"]))
            body = json.loads(row["body_json"])
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError("stored workflow is invalid") from exc
        self._validate_body_manifest(manifest, body)
        digest = workflow_approval_digest(manifest)
        approval = self._approval_verifier.verify_digest(digest)
        if (
            digest != row["approval_digest"]
            or approval is None
            or approval.approval_record_id != manifest.approval_record_id
        ):
            raise PermissionError("workflow approval is missing, revoked, or invalid")
        return LockedWorkflow(manifest=manifest, body=body)

    def snapshot(self, workflow_id: str, version: str) -> WorkflowSnapshot:
        return self._snapshot(self.get(workflow_id, version).manifest)

    def validate_inputs(
        self,
        workflow_id: str,
        version: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        locked = self.get(workflow_id, version)
        data = dict(_json_value(inputs))
        try:
            Draft202012Validator(locked.manifest.input_schema).validate(data)
        except (ValidationError, SchemaError) as exc:
            raise ValueError(f"workflow inputs do not match locked schema: {exc.message}") from exc
        return data

    def mask_inputs(
        self,
        workflow_id: str,
        version: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        data = self.validate_inputs(workflow_id, version, inputs)
        schema = self.get(workflow_id, version).manifest.input_schema
        properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
        return {
            key: "***"
            if isinstance(properties.get(key), Mapping)
            and properties[key].get("x-sensitive") is True
            else value
            for key, value in data.items()
        }

    def is_scheduled_read_only(self, workflow_id: str, version: str) -> bool:
        manifest = self.get(workflow_id, version).manifest
        return bool(
            manifest.automatable
            and manifest.approval_scope == "scheduled_read_only"
            and manifest.steps
            and all(
                step.risk_level == "read_only"
                and step.recovery_class == "read_only"
                and step.idempotent
                and step.executor_type not in {"human", "capability_builder"}
                for step in manifest.steps
            )
        )

    @staticmethod
    def _snapshot(manifest: WorkflowManifest) -> WorkflowSnapshot:
        return WorkflowSnapshot(
            workflow_id=manifest.workflow_id,
            version=manifest.version,
            workflow_sha256=manifest.workflow_sha256,
            manifest_sha256=manifest.manifest_sha256,
            approval_record_id=manifest.approval_record_id,
        )

    @staticmethod
    def _validate_body_manifest(
        manifest: WorkflowManifest,
        body: Mapping[str, Any],
    ) -> None:
        if canonical_workflow_sha256(body) != manifest.workflow_sha256:
            raise ValueError("workflow body hash does not match manifest")
        if canonical_manifest_sha256(manifest) != manifest.manifest_sha256:
            raise ValueError("workflow manifest hash does not match manifest")
        expected = {
            "workflow_id": manifest.workflow_id,
            "version": manifest.version,
            "workflow_schema_version": manifest.workflow_schema_version,
            "input_schema": manifest.input_schema,
            "steps": [step.model_dump(mode="json") for step in manifest.steps],
        }
        for key, value in expected.items():
            if _json_value(body.get(key)) != _json_value(value):
                raise ValueError(f"workflow body {key} does not match manifest")
        try:
            Draft202012Validator.check_schema(manifest.input_schema)
        except SchemaError as exc:
            raise ValueError(f"workflow input schema is invalid: {exc.message}") from exc


__all__ = [
    "LockedWorkflow",
    "WorkflowRegistry",
    "canonical_manifest_sha256",
    "canonical_workflow_sha256",
    "workflow_approval_digest",
]
