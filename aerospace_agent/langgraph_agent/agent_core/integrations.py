"""Trust verification and controlled acquisition for repository integrations.

This module never imports an adapter and never holds signing authority.  It
hashes an integration's complete evidence set, verifies an operator signature
through the public-key-only approval ledger, and otherwise reports the
capability as unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import Field, field_validator, model_validator

from .approval import CapabilityApprovalVerifier
from .capabilities import ALLOWED_IMPORT_ROOTS
from .confirmation import ConfirmationService, compute_action_hash
from .models import (
    CapabilityGap,
    CapabilityManifest,
    CheckpointRef,
    ContractModel,
    DependencyCandidate,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(path: str | Path) -> str:
    """Hash a file or a directory tree without including mutable mode bits."""

    target = Path(path)
    if target.is_symlink():
        raise ValueError("content-addressed sources cannot contain symbolic links")
    if target.is_file():
        return _file_sha256(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    files = sorted(target.rglob("*"), key=lambda item: item.relative_to(target).as_posix())
    for item in files:
        relative = item.relative_to(target).as_posix()
        if item.is_symlink():
            raise ValueError("content-addressed sources cannot contain symbolic links")
        marker = "d" if item.is_dir() else "f"
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(marker.encode("ascii"))
        digest.update(b"\0")
        if item.is_file():
            digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class IntegrationManifest(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    capability_id: str
    adapter_entrypoint: str
    adapter_path: str
    adapter_sha256: str
    source_type: Literal["installed", "package", "git", "local_code"]
    source_uri: str | None = None
    version_or_commit: str
    license: str
    compatibility: Literal["compatible", "uncertain", "incompatible"]
    lock_path: str
    lock_sha256: str
    source_cache_path: str | None = None
    source_cache_sha256: str | None = None
    distribution_path: str | None = None
    distribution_sha256: str | None = None
    local_patch_journal_path: str | None = None
    local_patch_journal_sha256: str | None = None
    license_evidence_path: str
    license_evidence_sha256: str
    version_evidence_path: str
    version_evidence_sha256: str
    validation_result_path: str
    validation_result_sha256: str
    validation_commands: list[str] = Field(min_length=1)
    capability_manifest_path: str
    capability_manifest_sha256: str
    capability_manifest_version: str

    @field_validator(
        "adapter_sha256",
        "lock_sha256",
        "source_cache_sha256",
        "distribution_sha256",
        "local_patch_journal_sha256",
        "license_evidence_sha256",
        "version_evidence_sha256",
        "validation_result_sha256",
        "capability_manifest_sha256",
    )
    @classmethod
    def _hashes(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("evidence hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _source_contract(self) -> "IntegrationManifest":
        if not _IDENTIFIER.fullmatch(self.capability_id):
            raise ValueError("capability_id is not a safe identifier")
        if not self.version_or_commit.strip() or not self.license.strip():
            raise ValueError("exact version/commit and license are required")
        if self.source_type == "git":
            if not _COMMIT.fullmatch(self.version_or_commit):
                raise ValueError("git integrations require an exact commit SHA")
            if not self.source_uri or not self.source_cache_path or not self.source_cache_sha256:
                raise ValueError("git integrations require URI and content-addressed cache evidence")
        if self.source_type == "package":
            if not self.source_uri or not self.distribution_path or not self.distribution_sha256:
                raise ValueError("package integrations require URI and hashed distribution artifact")
        if self.source_type == "local_code":
            if self.source_uri is not None:
                raise ValueError("local_code cannot reference an external source URI")
            if self.source_cache_path is not None or self.source_cache_sha256 is not None:
                raise ValueError("local_code cannot reference a source cache")
            if not self.local_patch_journal_path or not self.local_patch_journal_sha256:
                raise ValueError("local_code requires operation-journal patch evidence")
        return self


class IntegrationVerification(ContractModel):
    capability_id: str
    status: Literal["available", "unavailable"]
    combined_digest: str
    reason: str
    approval_record_id: str | None = None
    key_id: str | None = None


class _Assessment:
    def __init__(
        self,
        *,
        manifest: IntegrationManifest | None,
        combined_digest: str,
        reason: str | None,
        capability_manifest: CapabilityManifest | None,
    ) -> None:
        self.manifest = manifest
        self.combined_digest = combined_digest
        self.reason = reason
        self.capability_manifest = capability_manifest


class IntegrationTrustService:
    """Verify exact integration evidence using only trusted public keys."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        approval_verifier: CapabilityApprovalVerifier,
    ) -> None:
        if not isinstance(approval_verifier, CapabilityApprovalVerifier):
            raise TypeError("approval_verifier must be CapabilityApprovalVerifier")
        self._workspace = Path(workspace_root).resolve()
        self._approval_verifier = approval_verifier

    def verify(self, capability_id: str) -> IntegrationVerification:
        assessment = self._assess(capability_id)
        if assessment.reason is not None:
            return IntegrationVerification(
                capability_id=capability_id,
                status="unavailable",
                combined_digest=assessment.combined_digest,
                reason=assessment.reason,
            )
        record = self._approval_verifier.verify_digest(assessment.combined_digest)
        if record is None:
            return IntegrationVerification(
                capability_id=capability_id,
                status="unavailable",
                combined_digest=assessment.combined_digest,
                reason="approval_missing_or_invalid",
            )
        return IntegrationVerification(
            capability_id=capability_id,
            status="available",
            combined_digest=assessment.combined_digest,
            reason="approved",
            approval_record_id=record.approval_record_id,
            key_id=record.key_id,
        )

    def capability_manifest(self, capability_id: str) -> CapabilityManifest:
        assessment = self._assess(capability_id)
        capability = assessment.capability_manifest
        if capability is None and assessment.manifest is not None:
            try:
                path = self._resolve(assessment.manifest.capability_manifest_path)
                capability = CapabilityManifest.model_validate(
                    yaml.safe_load(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise ValueError(
                    assessment.reason or "capability manifest is unavailable"
                ) from exc
        if capability is None:
            raise ValueError(assessment.reason or "capability manifest is unavailable")
        manifest = CapabilityManifest.model_validate(capability.model_dump(mode="python"))
        verification = self.verify(capability_id)
        manifest.status = verification.status
        return manifest

    def _assess(self, capability_id: str) -> _Assessment:
        fallback_digest = hashlib.sha256(f"missing:{capability_id}".encode("utf-8")).hexdigest()
        if not _IDENTIFIER.fullmatch(capability_id):
            return _Assessment(
                manifest=None,
                combined_digest=fallback_digest,
                reason="invalid_capability_id",
                capability_manifest=None,
            )
        anchor_relative = f"aerospace_agent/integrations/{capability_id}/manifest.yaml"
        try:
            anchor = self._resolve(anchor_relative)
        except ValueError:
            return _Assessment(
                manifest=None,
                combined_digest=fallback_digest,
                reason="path_outside_workspace",
                capability_manifest=None,
            )
        if not anchor.is_file():
            return _Assessment(
                manifest=None,
                combined_digest=fallback_digest,
                reason="integration_manifest_missing",
                capability_manifest=None,
            )
        raw_manifest_hash = _file_sha256(anchor)
        try:
            loaded = yaml.safe_load(anchor.read_text(encoding="utf-8"))
            manifest = IntegrationManifest.model_validate(loaded)
        except Exception:
            return _Assessment(
                manifest=None,
                combined_digest=raw_manifest_hash,
                reason="integration_manifest_invalid",
                capability_manifest=None,
            )
        if manifest.capability_id != capability_id:
            return _Assessment(
                manifest=manifest,
                combined_digest=raw_manifest_hash,
                reason="capability_id_mismatch",
                capability_manifest=None,
            )

        module, separator, attribute = manifest.adapter_entrypoint.partition(":")
        if not separator or not module or not attribute or not any(
            module == root or module.startswith(f"{root}.") for root in ALLOWED_IMPORT_ROOTS
        ):
            return _Assessment(
                manifest=manifest,
                combined_digest=raw_manifest_hash,
                reason="entrypoint_outside_allowed_roots",
                capability_manifest=None,
            )
        try:
            paths = self._evidence_paths(manifest)
        except ValueError:
            return _Assessment(
                manifest=manifest,
                combined_digest=raw_manifest_hash,
                reason="path_outside_workspace",
                capability_manifest=None,
            )

        adapter = paths["adapter"]
        if not adapter.is_file():
            return self._failed(manifest, raw_manifest_hash, "adapter_missing")
        if not self._module_matches_path(module, manifest.adapter_path):
            return self._failed(manifest, raw_manifest_hash, "entrypoint_adapter_path_mismatch")
        if _file_sha256(adapter) != manifest.adapter_sha256:
            return self._failed(manifest, raw_manifest_hash, "adapter_hash_mismatch")

        lock = paths["lock"]
        if not lock.is_file():
            return self._failed(manifest, raw_manifest_hash, "lock_missing")
        if _file_sha256(lock) != manifest.lock_sha256:
            return self._failed(manifest, raw_manifest_hash, "lock_hash_mismatch")
        try:
            lock_text = lock.read_text(encoding="utf-8")
        except UnicodeError:
            return self._failed(manifest, raw_manifest_hash, "lock_invalid")
        if manifest.version_or_commit not in lock_text:
            return self._failed(manifest, raw_manifest_hash, "lock_version_evidence_missing")
        if "-e " in lock_text or "editable" in lock_text.casefold():
            return self._failed(manifest, raw_manifest_hash, "editable_dependency_forbidden")

        if manifest.source_cache_path is not None:
            cache = paths["source_cache"]
            if not cache.exists():
                return self._failed(manifest, raw_manifest_hash, "source_cache_missing")
            try:
                actual_cache_hash = content_sha256(cache)
            except ValueError:
                return self._failed(manifest, raw_manifest_hash, "source_cache_invalid")
            if actual_cache_hash != manifest.source_cache_sha256:
                return self._failed(manifest, raw_manifest_hash, "source_cache_hash_mismatch")
            expected_parent = self._workspace / "data" / "langgraph" / "capability_sources"
            try:
                cache.relative_to(expected_parent.resolve())
            except ValueError:
                return self._failed(manifest, raw_manifest_hash, "source_cache_outside_cache_root")
            if cache.name != manifest.source_cache_sha256:
                return self._failed(manifest, raw_manifest_hash, "source_cache_not_content_addressed")
            if manifest.source_type == "git" and not self._read_only_tree(cache):
                return self._failed(manifest, raw_manifest_hash, "git_cache_not_read_only")
            if manifest.source_cache_sha256 not in lock_text:
                return self._failed(manifest, raw_manifest_hash, "lock_source_hash_missing")

        if manifest.distribution_path is not None:
            distribution = paths["distribution"]
            if not distribution.is_file():
                return self._failed(manifest, raw_manifest_hash, "distribution_missing")
            if _file_sha256(distribution) != manifest.distribution_sha256:
                return self._failed(manifest, raw_manifest_hash, "distribution_hash_mismatch")
            cache_root = (self._workspace / "data/langgraph/capability_sources").resolve()
            try:
                distribution.relative_to(cache_root)
            except ValueError:
                return self._failed(
                    manifest, raw_manifest_hash, "distribution_outside_cache_root"
                )

        if manifest.local_patch_journal_path is not None:
            journal = paths["local_patch_journal"]
            if not journal.is_file():
                return self._failed(manifest, raw_manifest_hash, "local_patch_journal_missing")
            if _file_sha256(journal) != manifest.local_patch_journal_sha256:
                return self._failed(manifest, raw_manifest_hash, "local_patch_journal_hash_mismatch")
            try:
                journal_data = json.loads(journal.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return self._failed(manifest, raw_manifest_hash, "local_patch_journal_invalid")
            if not isinstance(journal_data, dict) or journal_data.get("status") != "committed":
                return self._failed(
                    manifest, raw_manifest_hash, "local_patch_journal_not_committed"
                )
            if journal_data.get("recovery_class") != "reversible":
                return self._failed(
                    manifest, raw_manifest_hash, "local_patch_journal_not_reversible"
                )
            if manifest.adapter_path not in journal_data.get("target_paths", []):
                return self._failed(
                    manifest, raw_manifest_hash, "local_patch_journal_target_mismatch"
                )
            if journal_data.get("postimage_sha256") != manifest.adapter_sha256:
                return self._failed(
                    manifest, raw_manifest_hash, "local_patch_journal_postimage_mismatch"
                )

        evidence_checks = (
            ("license_evidence", manifest.license_evidence_sha256, "license_evidence_hash_mismatch"),
            ("version_evidence", manifest.version_evidence_sha256, "version_evidence_hash_mismatch"),
            ("validation_result", manifest.validation_result_sha256, "validation_result_hash_mismatch"),
            (
                "capability_manifest",
                manifest.capability_manifest_sha256,
                "capability_manifest_hash_mismatch",
            ),
        )
        for key, expected, reason in evidence_checks:
            path = paths[key]
            if not path.is_file():
                return self._failed(manifest, raw_manifest_hash, f"{key}_missing")
            if _file_sha256(path) != expected:
                return self._failed(manifest, raw_manifest_hash, reason)

        try:
            license_text = paths["license_evidence"].read_text(encoding="utf-8")
            version_text = paths["version_evidence"].read_text(encoding="utf-8")
        except UnicodeError:
            return self._failed(manifest, raw_manifest_hash, "license_or_version_evidence_invalid")
        if manifest.license not in license_text:
            return self._failed(manifest, raw_manifest_hash, "license_evidence_missing")
        if manifest.version_or_commit not in version_text:
            return self._failed(manifest, raw_manifest_hash, "version_evidence_missing")
        if manifest.compatibility != "compatible":
            return self._failed(manifest, raw_manifest_hash, "compatibility_not_verified")

        try:
            validation = json.loads(paths["validation_result"].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self._failed(manifest, raw_manifest_hash, "validation_result_invalid")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            return self._failed(manifest, raw_manifest_hash, "validation_failed")
        if validation.get("commands") != manifest.validation_commands:
            return self._failed(manifest, raw_manifest_hash, "validation_command_mismatch")

        try:
            capability_loaded = yaml.safe_load(
                paths["capability_manifest"].read_text(encoding="utf-8")
            )
            capability = CapabilityManifest.model_validate(capability_loaded)
        except Exception:
            return self._failed(manifest, raw_manifest_hash, "capability_manifest_invalid")
        if capability.capability_id != capability_id:
            return self._failed(manifest, raw_manifest_hash, "capability_manifest_id_mismatch")
        if capability.version != manifest.capability_manifest_version:
            return self._failed(manifest, raw_manifest_hash, "capability_manifest_version_mismatch")
        if capability.status != "available":
            return self._failed(manifest, raw_manifest_hash, "capability_manifest_not_available")
        if capability.source != module:
            return self._failed(manifest, raw_manifest_hash, "capability_source_mismatch")
        if not capability.validators:
            return self._failed(manifest, raw_manifest_hash, "capability_validation_evidence_missing")

        combined = self._combined_digest(manifest, raw_manifest_hash)
        return _Assessment(
            manifest=manifest,
            combined_digest=combined,
            reason=None,
            capability_manifest=capability,
        )

    @staticmethod
    def _failed(manifest: IntegrationManifest, digest: str, reason: str) -> _Assessment:
        return _Assessment(
            manifest=manifest, combined_digest=digest, reason=reason, capability_manifest=None
        )

    def _evidence_paths(self, manifest: IntegrationManifest) -> dict[str, Path]:
        paths = {
            "adapter": self._resolve(manifest.adapter_path),
            "lock": self._resolve(manifest.lock_path),
            "license_evidence": self._resolve(manifest.license_evidence_path),
            "version_evidence": self._resolve(manifest.version_evidence_path),
            "validation_result": self._resolve(manifest.validation_result_path),
            "capability_manifest": self._resolve(manifest.capability_manifest_path),
        }
        if manifest.source_cache_path is not None:
            paths["source_cache"] = self._resolve(manifest.source_cache_path)
        if manifest.distribution_path is not None:
            paths["distribution"] = self._resolve(manifest.distribution_path)
        if manifest.local_patch_journal_path is not None:
            paths["local_patch_journal"] = self._resolve(manifest.local_patch_journal_path)
        return paths

    def _resolve(self, relative: str) -> Path:
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError("path is outside workspace")
        resolved = (self._workspace / raw).resolve()
        try:
            resolved.relative_to(self._workspace)
        except ValueError as exc:
            raise ValueError("path is outside workspace") from exc
        return resolved

    @staticmethod
    def _module_matches_path(module: str, relative: str) -> bool:
        expected = module.replace(".", "/")
        normalized = Path(relative).as_posix()
        return normalized in {f"{expected}.py", f"{expected}/__init__.py"}

    @staticmethod
    def _read_only_tree(root: Path) -> bool:
        for item in [root, *root.rglob("*")]:
            if item.is_symlink():
                return False
            if item.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                return False
        return True

    @staticmethod
    def _combined_digest(manifest: IntegrationManifest, manifest_sha256: str) -> str:
        evidence = {
            "domain": "zyt-agent-integration-baseline-v1",
            "manifest_sha256": manifest_sha256,
            "capability_id": manifest.capability_id,
            "adapter_entrypoint": manifest.adapter_entrypoint,
            "adapter_sha256": manifest.adapter_sha256,
            "lock_sha256": manifest.lock_sha256,
            "source_cache_sha256": manifest.source_cache_sha256,
            "distribution_sha256": manifest.distribution_sha256,
            "local_patch_journal_sha256": manifest.local_patch_journal_sha256,
            "license_evidence_sha256": manifest.license_evidence_sha256,
            "version_evidence_sha256": manifest.version_evidence_sha256,
            "validation_result_sha256": manifest.validation_result_sha256,
            "capability_manifest_sha256": manifest.capability_manifest_sha256,
            "source_type": manifest.source_type,
            "source_uri": manifest.source_uri,
            "version_or_commit": manifest.version_or_commit,
            "license": manifest.license,
        }
        encoded = json.dumps(
            evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class AcquisitionGapRecord(ContractModel):
    gap_id: str
    capability_id: str
    requested_by_step_id: str
    project_id: str
    thread_id: str
    root_run_id: str
    plan_id: str
    original_checkpoint: CheckpointRef
    gap: CapabilityGap
    status: Literal["staged", "authorized", "acquired", "validated", "registered", "resumed"]
    created_at: str


class BuildAttempt(ContractModel):
    attempt_id: str
    gap_id: str
    capability_id: str
    root_run_id: str
    candidate: DependencyCandidate
    attempt_number: int = Field(ge=1, le=1)
    status: Literal["prepared", "authorized", "acquired", "validated", "registered", "failed"]
    confirmation_id: str
    candidate_digest: str
    evidence: dict[str, Any] | None = None
    integration_digest: str | None = None
    created_at: str


class CapabilityAcquisitionService:
    """Persist capability gaps and authorize at most one build per run/capability."""

    _SCHEMA_VERSION = 2

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        database_path: str | Path,
        project_id: str,
        clock: Callable[[], datetime] = _utc_now,
        confirmation_service: ConfirmationService | None = None,
    ) -> None:
        self._workspace = Path(workspace_root).resolve()
        self._database_path = Path(database_path).resolve()
        self._project_id = project_id
        self._clock = clock
        self._confirmation_service = confirmation_service
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._available = True
        self._degraded_reason: str | None = None
        try:
            self._migrate()
            if confirmation_service is not None:
                self._reconcile_authorizations()
        except Exception as exc:
            self._available = False
            self._degraded_reason = str(exc)

    def availability(self) -> dict[str, Any]:
        return {"available": self._available, "reason": self._degraded_reason}

    def _require_available(self) -> None:
        if not self._available:
            raise RuntimeError(
                f"capability acquisition is unavailable: {self._degraded_reason or 'migration failed'}"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported acquisition schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE acquisition_gaps (
                        gap_id TEXT PRIMARY KEY,
                        capability_id TEXT NOT NULL,
                        requested_by_step_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        plan_id TEXT NOT NULL,
                        original_checkpoint_json TEXT NOT NULL,
                        gap_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE acquisition_build_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        gap_id TEXT NOT NULL,
                        capability_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        candidate_json TEXT NOT NULL,
                        attempt_number INTEGER NOT NULL CHECK(attempt_number = 1),
                        status TEXT NOT NULL,
                        confirmation_id TEXT NOT NULL,
                        candidate_digest TEXT NOT NULL,
                        evidence_json TEXT,
                        integration_digest TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(root_run_id, capability_id),
                        FOREIGN KEY(gap_id) REFERENCES acquisition_gaps(gap_id)
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()
                version = self._SCHEMA_VERSION
            if version == 1:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "ALTER TABLE acquisition_build_attempts ADD COLUMN candidate_digest TEXT"
                )
                connection.execute(
                    "ALTER TABLE acquisition_build_attempts ADD COLUMN evidence_json TEXT"
                )
                connection.execute(
                    "ALTER TABLE acquisition_build_attempts ADD COLUMN integration_digest TEXT"
                )
                rows = connection.execute(
                    "SELECT attempt_id, candidate_json FROM acquisition_build_attempts"
                ).fetchall()
                for row in rows:
                    candidate = json.loads(row["candidate_json"])
                    connection.execute(
                        "UPDATE acquisition_build_attempts SET candidate_digest = ? WHERE attempt_id = ?",
                        (self._digest(candidate), row["attempt_id"]),
                    )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def stage_gap(
        self,
        gap: CapabilityGap,
        *,
        thread_id: str,
        root_run_id: str,
        plan_id: str,
        original_checkpoint: CheckpointRef,
    ) -> AcquisitionGapRecord:
        self._require_available()
        gap = CapabilityGap.model_validate(gap.model_dump(mode="python"))
        checkpoint = CheckpointRef.model_validate(original_checkpoint.model_dump(mode="python"))
        if checkpoint.project_id != self._project_id or checkpoint.thread_id != thread_id:
            raise ValueError("original checkpoint namespace mismatch")
        if not gap.candidates:
            raise ValueError("capability acquisition requires at least one reviewed candidate")
        created_at = self._as_utc(self._clock()).isoformat()
        record = AcquisitionGapRecord(
            gap_id=uuid4().hex,
            capability_id=gap.capability_id,
            requested_by_step_id=gap.requested_by_step_id,
            project_id=self._project_id,
            thread_id=thread_id,
            root_run_id=root_run_id,
            plan_id=plan_id,
            original_checkpoint=checkpoint,
            gap=gap,
            status="staged",
            created_at=created_at,
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO acquisition_gaps(
                    gap_id, capability_id, requested_by_step_id, project_id, thread_id,
                    root_run_id, plan_id, original_checkpoint_json, gap_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.gap_id,
                    record.capability_id,
                    record.requested_by_step_id,
                    record.project_id,
                    record.thread_id,
                    record.root_run_id,
                    record.plan_id,
                    self._json(record.original_checkpoint.model_dump(mode="json")),
                    self._json(record.gap.model_dump(mode="json")),
                    record.status,
                    record.created_at,
                ),
            )
            connection.commit()
        return record

    def build_action_hash(self, gap_id: str, candidate_name: str) -> str:
        self._require_available()
        record = self.get_gap(gap_id)
        candidate = self._candidate(record, candidate_name)
        return compute_action_hash(
            tool_name="capability_builder.build",
            arguments={
                "gap_id": gap_id,
                "capability_id": record.capability_id,
                "candidate": candidate.model_dump(mode="json"),
                "plan_id": record.plan_id,
                "project_id": record.project_id,
                "thread_id": record.thread_id,
                "root_run_id": record.root_run_id,
                "requested_by_step_id": record.requested_by_step_id,
                "original_checkpoint": record.original_checkpoint.model_dump(mode="json"),
                "required_contract": record.gap.required_contract,
            },
            target_paths=[
                f"aerospace_agent/integrations/{record.capability_id}/manifest.yaml"
            ],
            run_id=record.root_run_id,
            risk_level="high_risk",
        )

    def authorize_build(
        self,
        gap_id: str,
        *,
        candidate_name: str,
        confirmation_service: ConfirmationService,
        confirmation_id: str,
    ) -> BuildAttempt:
        self._require_available()
        if not isinstance(confirmation_service, ConfirmationService):
            raise TypeError("confirmation_service must be ConfirmationService")
        record = self.get_gap(gap_id)
        candidate = self._candidate(record, candidate_name)
        self._validate_candidate(candidate)
        candidate_json = candidate.model_dump(mode="json")
        attempt = BuildAttempt(
            attempt_id=uuid4().hex,
            gap_id=record.gap_id,
            capability_id=record.capability_id,
            root_run_id=record.root_run_id,
            candidate=candidate,
            attempt_number=1,
            status="prepared",
            confirmation_id=confirmation_id,
            candidate_digest=self._digest(candidate_json),
            created_at=self._as_utc(self._clock()).isoformat(),
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO acquisition_build_attempts(
                        attempt_id, gap_id, capability_id, root_run_id, candidate_json,
                        attempt_number, status, confirmation_id, candidate_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, 1, 'prepared', ?, ?, ?)
                    """,
                    (
                        attempt.attempt_id,
                        attempt.gap_id,
                        attempt.capability_id,
                        attempt.root_run_id,
                        self._json(candidate_json),
                        attempt.confirmation_id,
                        attempt.candidate_digest,
                        attempt.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise ValueError("one build attempt per capability/run is allowed") from exc
            connection.commit()
        try:
            confirmation_service.consume(
                confirmation_id=confirmation_id,
                project_id=record.project_id,
                thread_id=record.thread_id,
                root_run_id=record.root_run_id,
                operation_id=f"build:{record.gap_id}",
                action_hash=self.build_action_hash(record.gap_id, candidate_name),
                continuation_checkpoint={
                    **record.original_checkpoint.model_dump(mode="json"),
                    "plan_id": record.plan_id,
                    "gap_id": record.gap_id,
                    "capability_id": record.capability_id,
                    "candidate_digest": attempt.candidate_digest,
                },
            )
        except Exception:
            if confirmation_service.continuation_checkpoint(confirmation_id) is None:
                with closing(self._connect()) as connection:
                    connection.execute(
                        "DELETE FROM acquisition_build_attempts WHERE attempt_id = ? AND status = 'prepared'",
                        (attempt.attempt_id,),
                    )
                    connection.commit()
            raise
        return self._mark_authorized(attempt)

    def _mark_authorized(self, attempt: BuildAttempt) -> BuildAttempt:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE acquisition_build_attempts SET status = 'authorized' "
                "WHERE attempt_id = ? AND status = 'prepared'",
                (attempt.attempt_id,),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("build authorization state changed concurrently")
            connection.execute(
                "UPDATE acquisition_gaps SET status = 'authorized' WHERE gap_id = ?",
                (attempt.gap_id,),
            )
            connection.commit()
        return self.get_attempt(attempt.attempt_id)

    def record_acquisition(
        self,
        attempt_id: str,
        *,
        trust_service: IntegrationTrustService,
    ) -> BuildAttempt:
        """Persist the exact acquired source and evidence baseline before validation."""

        self._require_available()
        if not isinstance(trust_service, IntegrationTrustService):
            raise TypeError("trust_service must be IntegrationTrustService")
        attempt = self.get_attempt(attempt_id)
        if attempt.status != "authorized":
            raise RuntimeError("build attempt must be authorized before acquisition evidence")
        assessment = trust_service._assess(attempt.capability_id)
        if assessment.reason is not None or assessment.manifest is None:
            raise RuntimeError(f"integration acquisition evidence is invalid: {assessment.reason}")
        manifest = assessment.manifest
        candidate = attempt.candidate
        if (
            manifest.source_type != candidate.source_type
            or manifest.source_uri != candidate.source_uri
            or manifest.version_or_commit != candidate.version_or_commit
            or manifest.license != candidate.license
        ):
            raise RuntimeError("integration evidence does not match the approved candidate")
        evidence = {
            "candidate_digest": attempt.candidate_digest,
            "source_type": manifest.source_type,
            "source_uri": manifest.source_uri,
            "version_or_commit": manifest.version_or_commit,
            "license": manifest.license,
            "lock_sha256": manifest.lock_sha256,
            "source_cache_sha256": manifest.source_cache_sha256,
            "distribution_sha256": manifest.distribution_sha256,
            "local_patch_journal_sha256": manifest.local_patch_journal_sha256,
            "validation_result_sha256": manifest.validation_result_sha256,
            "combined_digest": assessment.combined_digest,
        }
        return self._transition_attempt(
            attempt_id,
            expected="authorized",
            target="acquired",
            evidence=evidence,
            integration_digest=assessment.combined_digest,
        )

    def record_validation(
        self,
        attempt_id: str,
        *,
        trust_service: IntegrationTrustService,
    ) -> BuildAttempt:
        self._require_available()
        if not isinstance(trust_service, IntegrationTrustService):
            raise TypeError("trust_service must be IntegrationTrustService")
        attempt = self.get_attempt(attempt_id)
        if attempt.status != "acquired":
            raise RuntimeError("build attempt must be acquired before it can be validated")
        assessment = trust_service._assess(attempt.capability_id)
        if assessment.reason is not None:
            raise RuntimeError(f"isolated validation evidence is invalid: {assessment.reason}")
        if assessment.combined_digest != attempt.integration_digest:
            raise RuntimeError("integration payload drifted after acquisition")
        return self._transition_attempt(
            attempt_id,
            expected="acquired",
            target="validated",
            evidence=attempt.evidence,
            integration_digest=attempt.integration_digest,
        )

    def register(
        self,
        attempt_id: str,
        *,
        trust_service: IntegrationTrustService,
    ) -> BuildAttempt:
        self._require_available()
        if not isinstance(trust_service, IntegrationTrustService):
            raise TypeError("trust_service must be IntegrationTrustService")
        attempt = self.get_attempt(attempt_id)
        if attempt.status != "validated":
            raise RuntimeError("build attempt must be validated before registration")
        verification = trust_service.verify(attempt.capability_id)
        if verification.status != "available":
            raise RuntimeError("signed integration approval is required before registration")
        if verification.combined_digest != attempt.integration_digest:
            raise RuntimeError("integration payload drifted before registration")
        return self._transition_attempt(
            attempt_id,
            expected="validated",
            target="registered",
            evidence=attempt.evidence,
            integration_digest=attempt.integration_digest,
        )

    def resume_from_original_checkpoint(
        self,
        gap_id: str,
        *,
        trust_service: IntegrationTrustService,
    ) -> CheckpointRef:
        self._require_available()
        if not isinstance(trust_service, IntegrationTrustService):
            raise TypeError("trust_service must be IntegrationTrustService")
        record = self.get_gap(gap_id)
        with closing(self._connect()) as connection:
            attempt = connection.execute(
                "SELECT integration_digest FROM acquisition_build_attempts "
                "WHERE gap_id = ? AND status = 'registered'",
                (gap_id,),
            ).fetchone()
        if attempt is None:
            raise RuntimeError("capability build has not been registered")
        verification = trust_service.verify(record.capability_id)
        if (
            verification.status != "available"
            or verification.combined_digest != attempt["integration_digest"]
        ):
            raise RuntimeError("an approved integration baseline is required before resume")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE acquisition_gaps SET status = 'resumed' WHERE gap_id = ?",
                (gap_id,),
            )
            connection.commit()
        return CheckpointRef.model_validate(record.original_checkpoint.model_dump(mode="python"))

    def get_gap(self, gap_id: str) -> AcquisitionGapRecord:
        self._require_available()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM acquisition_gaps WHERE gap_id = ?", (gap_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"capability gap not found: {gap_id}")
        return AcquisitionGapRecord(
            gap_id=row["gap_id"],
            capability_id=row["capability_id"],
            requested_by_step_id=row["requested_by_step_id"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            root_run_id=row["root_run_id"],
            plan_id=row["plan_id"],
            original_checkpoint=CheckpointRef.model_validate(
                json.loads(row["original_checkpoint_json"])
            ),
            gap=CapabilityGap.model_validate(json.loads(row["gap_json"])),
            status=row["status"],
            created_at=row["created_at"],
        )

    def get_attempt(self, attempt_id: str) -> BuildAttempt:
        self._require_available()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM acquisition_build_attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"build attempt not found: {attempt_id}")
        candidate_data = json.loads(row["candidate_json"])
        if self._digest(candidate_data) != row["candidate_digest"]:
            raise RuntimeError("persisted acquisition candidate payload drifted")
        return BuildAttempt(
            attempt_id=row["attempt_id"],
            gap_id=row["gap_id"],
            capability_id=row["capability_id"],
            root_run_id=row["root_run_id"],
            candidate=DependencyCandidate.model_validate(candidate_data),
            attempt_number=row["attempt_number"],
            status=row["status"],
            confirmation_id=row["confirmation_id"],
            candidate_digest=row["candidate_digest"],
            evidence=json.loads(row["evidence_json"]) if row["evidence_json"] else None,
            integration_digest=row["integration_digest"],
            created_at=row["created_at"],
        )

    def _transition_attempt(
        self,
        attempt_id: str,
        *,
        expected: str,
        target: str,
        evidence: Mapping[str, Any] | None,
        integration_digest: str | None,
    ) -> BuildAttempt:
        attempt = self.get_attempt(attempt_id)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE acquisition_build_attempts SET status = ?, evidence_json = ?, "
                "integration_digest = ? WHERE attempt_id = ? AND status = ?",
                (
                    target,
                    self._json(evidence) if evidence is not None else None,
                    integration_digest,
                    attempt_id,
                    expected,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError(f"build attempt must be {expected} before {target}")
            connection.execute(
                "UPDATE acquisition_gaps SET status = ? WHERE gap_id = ?",
                (target, attempt.gap_id),
            )
            connection.commit()
        return self.get_attempt(attempt_id)

    def _reconcile_authorizations(self) -> None:
        assert self._confirmation_service is not None
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT attempt_id, gap_id, confirmation_id, candidate_digest "
                "FROM acquisition_build_attempts WHERE status = 'prepared'"
            ).fetchall()
        for row in rows:
            try:
                checkpoint = self._confirmation_service.continuation_checkpoint(
                    row["confirmation_id"]
                )
            except Exception:
                checkpoint = None
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                if checkpoint and checkpoint.get("candidate_digest") == row["candidate_digest"]:
                    connection.execute(
                        "UPDATE acquisition_build_attempts SET status = 'authorized' "
                        "WHERE attempt_id = ? AND status = 'prepared'",
                        (row["attempt_id"],),
                    )
                    connection.execute(
                        "UPDATE acquisition_gaps SET status = 'authorized' WHERE gap_id = ?",
                        (row["gap_id"],),
                    )
                else:
                    connection.execute(
                        "DELETE FROM acquisition_build_attempts "
                        "WHERE attempt_id = ? AND status = 'prepared'",
                        (row["attempt_id"],),
                    )
                connection.commit()

    @staticmethod
    def _candidate(record: AcquisitionGapRecord, name: str) -> DependencyCandidate:
        for candidate in record.gap.candidates:
            if candidate.name == name:
                return DependencyCandidate.model_validate(candidate.model_dump(mode="python"))
        raise KeyError(f"dependency candidate not found: {name}")

    @staticmethod
    def _validate_candidate(candidate: DependencyCandidate) -> None:
        if candidate.compatibility != "compatible":
            raise ValueError("candidate compatibility must be verified")
        if not candidate.version_or_commit or not candidate.license:
            raise ValueError("candidate requires version and license evidence")
        if candidate.source_type in {"package", "git"} and not candidate.source_uri:
            raise ValueError("external candidate requires an exact source URI")
        if candidate.source_type == "git" and not _COMMIT.fullmatch(candidate.version_or_commit):
            raise ValueError("git candidate requires an exact commit SHA")

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def _digest(cls, value: Mapping[str, Any]) -> str:
        return hashlib.sha256(cls._json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("acquisition clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


__all__ = [
    "AcquisitionGapRecord",
    "BuildAttempt",
    "CapabilityAcquisitionService",
    "IntegrationManifest",
    "IntegrationTrustService",
    "IntegrationVerification",
    "content_sha256",
]
