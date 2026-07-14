"""Single, fail-closed authorization and execution boundary for all executor kinds."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .approval import ApprovalRecord, CapabilityApprovalVerifier
from .capabilities import ALLOWED_IMPORT_ROOTS
from .confirmation import ConfirmationError, ConfirmationService, compute_action_hash
from .execution_checkpoints import ExecutionCheckpointStore
from .journal import OperationJournal
from .models import (
    CapabilityManifest,
    CapabilitySnapshot,
    ContractModel,
    DomainExecutionOutput,
    ToolError,
    ToolResult,
    WorkflowSnapshot,
)
from .planning import PlanExecutionVerifier


ExecutionKind = Literal["tool", "workflow", "domain", "capability_builder", "human"]
RecoveryClass = Literal["read_only", "reversible", "compensatable", "manual_recovery"]
ExecutorHandler = Callable[..., dict[str, Any] | ToolResult]
_SHA256 = set("0123456789abcdefABCDEF")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _validate_optional_sha(value: str | None) -> str | None:
    if value is not None and (len(value) != 64 or any(character not in _SHA256 for character in value)):
        raise ValueError("must be a 64-character SHA-256 hex digest")
    return value


class ExecutionRequest(ContractModel):
    kind: ExecutionKind
    capability_id: str
    executor_name: str
    operation_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmation_id: str | None = None
    origin: Literal["direct", "planned"] = "direct"
    step_id: str | None = None
    domain_state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_origin(self) -> "ExecutionRequest":
        if self.origin == "planned" and not self.step_id:
            raise ValueError("planned execution requires step_id")
        if self.origin == "direct" and self.step_id is not None:
            raise ValueError("direct execution cannot carry step_id")
        if self.kind != "domain" and self.domain_state:
            raise ValueError("domain_state is only valid for domain execution")
        return self


class ExecutionContext(ContractModel):
    project_id: str
    thread_id: str | None
    root_run_id: str
    workspace_root: str
    capability_snapshot: CapabilitySnapshot
    workflow_snapshot: WorkflowSnapshot | None = None
    plan_id: str | None = None
    plan_sha256: str | None = None
    registry_snapshot_sha256: str | None = None

    _plan_hash = field_validator("plan_sha256")(_validate_optional_sha)
    _registry_hash = field_validator("registry_snapshot_sha256")(_validate_optional_sha)

    @model_validator(mode="after")
    def validate_plan_pair(self) -> "ExecutionContext":
        if (self.plan_id is None) != (self.plan_sha256 is None):
            raise ValueError("plan_id and plan_sha256 must be provided together")
        return self


_AUTHORIZATION_FACTORY = object()


class AuthorizedExecutor:
    """Opaque, expiring, one-shot authorization that contains no handler."""

    __slots__ = (
        "kind",
        "capability_id",
        "executor_name",
        "operation_id",
        "recovery_class",
        "__token",
    )

    def __init__(
        self,
        *,
        kind: ExecutionKind,
        capability_id: str,
        executor_name: str,
        operation_id: str,
        recovery_class: RecoveryClass,
        token: str,
        factory: object,
    ) -> None:
        if factory is not _AUTHORIZATION_FACTORY:
            raise TypeError("AuthorizedExecutor can only be created by ExecutionRegistry")
        self.kind = kind
        self.capability_id = capability_id
        self.executor_name = executor_name
        self.operation_id = operation_id
        self.recovery_class = recovery_class
        self.__token = token

    def _authorization_token(self) -> str:
        return self.__token


@dataclass(frozen=True, slots=True)
class _Registration:
    kind: ExecutionKind
    manifest: CapabilityManifest
    executor_name: str
    handler: ExecutorHandler | None
    input_model: type[BaseModel]
    input_model_identity: str
    input_model_path: Path
    input_model_file_sha256: str
    input_model_source_sha256: str
    entrypoint: str
    adapter_path: Path
    adapter_sha256: str
    manifest_sha256: str
    recovery_class: RecoveryClass
    path_fields: tuple[str, ...]
    validation_evidence: tuple[tuple[str, str], ...]
    dependency_hashes: tuple[tuple[str, Path, str], ...]
    cache_hashes: tuple[tuple[str, Path, str], ...]
    requires_confirmation: bool
    approval_required: bool
    approval_record_id: str | None
    registration_digest: str
    runtime_trust_digest: str | None
    runtime_trust_verifier: Callable[[], Any] | None
    workflow_snapshot: WorkflowSnapshot | None
    operation_journal: OperationJournal | None


@dataclass(frozen=True, slots=True)
class _AuthorizationRecord:
    authorized: AuthorizedExecutor
    registration: _Registration
    request: ExecutionRequest
    context: ExecutionContext
    validated_arguments: dict[str, Any]
    confirmation_consumed: bool
    authorized_at: str
    expires_at: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_hash(manifest: CapabilityManifest) -> str:
    return _canonical_sha256(manifest.model_dump(mode="json"))


class ExecutionRegistry:
    """Resolve requests only after kind-specific trust, snapshot and safety checks."""

    _AUDIT_SCHEMA_VERSION = 1

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        audit_database_path: str | Path,
        confirmation_service: ConfirmationService | None = None,
        approval_verifier: CapabilityApprovalVerifier | None = None,
        plan_execution_verifier: PlanExecutionVerifier | None = None,
        clock: Callable[[], datetime] = _utc_now,
        authorization_ttl_seconds: int = 600,
    ) -> None:
        if not 1 <= authorization_ttl_seconds <= 600:
            raise ValueError("authorization TTL must be between 1 and 600 seconds")
        self._workspace_root = Path(workspace_root).resolve()
        self._audit_database_path = Path(audit_database_path)
        if not self._audit_database_path.is_absolute():
            self._audit_database_path = self._workspace_root / self._audit_database_path
        self._audit_database_path = self._audit_database_path.resolve()
        try:
            self._audit_database_path.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("audit database must stay inside workspace") from exc
        self._audit_database_path.parent.mkdir(parents=True, exist_ok=True)
        self._confirmation_service = confirmation_service
        if confirmation_service is not None and type(confirmation_service) is not ConfirmationService:
            raise TypeError("confirmation_service must be the concrete ConfirmationService")
        if approval_verifier is not None and not isinstance(
            approval_verifier, CapabilityApprovalVerifier
        ):
            raise TypeError("approval_verifier must be CapabilityApprovalVerifier")
        if plan_execution_verifier is not None and not isinstance(
            plan_execution_verifier, PlanExecutionVerifier
        ):
            raise TypeError("plan_execution_verifier must be PlanExecutionVerifier")
        self._approval_verifier = approval_verifier
        self._plan_execution_verifier = plan_execution_verifier
        self._clock = clock
        self._authorization_ttl_seconds = authorization_ttl_seconds
        self._registrations: dict[tuple[str, str, str], _Registration] = {}
        self._authorizations: dict[str, _AuthorizationRecord] = {}
        self._used_authorizations: dict[str, _AuthorizationRecord] = {}
        self._migrate_audit()

    def _now(self) -> datetime:
        return _as_utc(self._clock())

    def _audit_connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._audit_database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate_audit(self) -> None:
        with closing(self._audit_connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._AUDIT_SCHEMA_VERSION:
                raise RuntimeError(f"unsupported execution audit schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE execution_audit (
                        audit_id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        thread_id TEXT,
                        capability_id TEXT NOT NULL,
                        executor_name TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result_sha256 TEXT NOT NULL,
                        arguments_sha256 TEXT NOT NULL,
                        recovery_class TEXT NOT NULL,
                        error_code TEXT
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX execution_audit_run_idx "
                    "ON execution_audit(project_id, root_run_id, operation_id)"
                )
                connection.execute(f"PRAGMA user_version = {self._AUDIT_SCHEMA_VERSION}")
                connection.commit()

    def register(
        self,
        *,
        kind: ExecutionKind,
        manifest: CapabilityManifest,
        executor_name: str,
        handler: ExecutorHandler | None,
        input_model: type[BaseModel],
        entrypoint: str,
        adapter_path: str | Path,
        recovery_class: RecoveryClass,
        path_fields: Sequence[str] = (),
        dependency_files: Mapping[str, str | Path] | None = None,
        cache_files: Mapping[str, str | Path] | None = None,
        validation_evidence: Mapping[str, str] | None = None,
        requires_confirmation: bool = False,
        workflow_snapshot: WorkflowSnapshot | None = None,
        runtime_trust_digest: str | None = None,
        runtime_trust_verifier: Callable[[], Any] | None = None,
    ) -> None:
        validated_manifest = CapabilityManifest.model_validate(manifest.model_dump(mode="python"))
        self._validate_kind_policy(kind, validated_manifest, executor_name, handler, workflow_snapshot)
        if not isinstance(input_model, type) or not issubclass(input_model, BaseModel):
            raise TypeError("input_model must be a Pydantic BaseModel type")
        if {"confirmed", "operation_id"} & set(input_model.model_fields):
            raise ValueError("execution-managed fields cannot appear in the public input schema")
        if len(set(path_fields)) != len(tuple(path_fields)) or not set(path_fields).issubset(
            input_model.model_fields
        ):
            raise ValueError("path_fields must be unique fields declared by input_model")
        adapter = self._resolve_registration_path(adapter_path)
        self._validate_entrypoint_and_handler(
            kind=kind,
            manifest=validated_manifest,
            entrypoint=entrypoint,
            handler=handler,
            adapter_path=adapter,
            input_model=input_model,
        )
        dependency_files = dict(dependency_files or {})
        cache_files = dict(cache_files or {})
        validation_evidence = dict(validation_evidence or {})
        if set(dependency_files) != set(validated_manifest.required_dependencies):
            raise ValueError("dependency_files keys must exactly match manifest.required_dependencies")
        if set(validation_evidence) != set(validated_manifest.validators):
            raise ValueError("validation_evidence keys must exactly match manifest.validators")
        if any(not value for value in validation_evidence.values()):
            raise ValueError("validation evidence digests cannot be empty")
        dependency_hashes = self._capture_named_hashes(dependency_files)
        cache_hashes = self._capture_named_hashes(cache_files)
        adapter_hash = _sha256_file(adapter)
        manifest_hash = _manifest_hash(validated_manifest)
        input_model_evidence = self._capture_input_model_evidence(input_model)
        operation_journal = self._validate_recovery_binding(
            handler=handler,
            recovery_class=recovery_class,
            path_fields=path_fields,
        )
        matched_root = self._matched_import_root(entrypoint)
        if (runtime_trust_digest is None) != (runtime_trust_verifier is None):
            raise ValueError(
                "runtime_trust_digest and runtime_trust_verifier must be provided together"
            )
        if runtime_trust_digest is not None and (
            len(runtime_trust_digest) != 64
            or runtime_trust_digest != runtime_trust_digest.casefold()
            or any(character not in "0123456789abcdef" for character in runtime_trust_digest)
        ):
            raise ValueError("runtime_trust_digest must be a lowercase SHA-256 digest")
        if matched_root == "aerospace_agent.integrations" and runtime_trust_digest is None:
            raise ValueError("integration executor requires a runtime trust verifier")
        approval_required = kind in {"workflow", "capability_builder"} or (
            matched_root == "aerospace_agent.integrations"
        )
        if matched_root == "aerospace_agent.integrations" and not validated_manifest.validators:
            raise ValueError("integration executor requires validation evidence")
        registration_digest = self._registration_digest(
            kind=kind,
            manifest=validated_manifest,
            executor_name=executor_name,
            input_model=input_model,
            input_model_evidence=input_model_evidence,
            entrypoint=entrypoint,
            adapter_hash=adapter_hash,
            manifest_hash=manifest_hash,
            recovery_class=recovery_class,
            path_fields=path_fields,
            dependency_hashes=dependency_hashes,
            cache_hashes=cache_hashes,
            validation_evidence=validation_evidence,
            requires_confirmation=requires_confirmation,
            workflow_snapshot=workflow_snapshot,
            operation_journal=operation_journal,
            runtime_trust_digest=runtime_trust_digest,
        )
        approval_record = (
            self._approval_record(registration_digest) if approval_required else None
        )
        if approval_required and approval_record is None:
            raise ValueError("trusted approval verification is required for this executor")
        if (
            kind == "workflow"
            and approval_record is not None
            and workflow_snapshot is not None
            and approval_record.approval_record_id != workflow_snapshot.approval_record_id
        ):
            raise ValueError("workflow snapshot approval record does not match signed baseline")
        registration = _Registration(
            kind=kind,
            manifest=validated_manifest,
            executor_name=executor_name,
            handler=handler,
            input_model=input_model,
            input_model_identity=input_model_evidence[0],
            input_model_path=input_model_evidence[1],
            input_model_file_sha256=input_model_evidence[2],
            input_model_source_sha256=input_model_evidence[3],
            entrypoint=entrypoint,
            adapter_path=adapter,
            adapter_sha256=adapter_hash,
            manifest_sha256=manifest_hash,
            recovery_class=recovery_class,
            path_fields=tuple(path_fields),
            validation_evidence=tuple(sorted(validation_evidence.items())),
            dependency_hashes=dependency_hashes,
            cache_hashes=cache_hashes,
            requires_confirmation=requires_confirmation,
            approval_required=approval_required,
            approval_record_id=(
                approval_record.approval_record_id if approval_record is not None else None
            ),
            registration_digest=registration_digest,
            runtime_trust_digest=runtime_trust_digest,
            runtime_trust_verifier=runtime_trust_verifier,
            workflow_snapshot=workflow_snapshot,
            operation_journal=operation_journal,
        )
        key = (kind, validated_manifest.capability_id, executor_name)
        if key in self._registrations:
            raise ValueError(f"duplicate execution registration: {key}")
        self._registrations[key] = registration

    def _validate_recovery_binding(
        self,
        *,
        handler: ExecutorHandler | None,
        recovery_class: RecoveryClass,
        path_fields: Sequence[str],
    ) -> OperationJournal | None:
        if recovery_class == "compensatable":
            raise ValueError(
                "compensatable executor requires a verified compensation controller"
            )
        if recovery_class != "reversible":
            return None
        from .tools.files import FileService

        if handler is None or not inspect.ismethod(handler):
            raise ValueError("reversible executor requires a journal-managed bound handler")
        owner = handler.__self__
        file_service = getattr(owner, "file_service", None)
        if not isinstance(file_service, FileService):
            raise ValueError("reversible executor requires a trusted FileService adapter")
        if file_service.root != self._workspace_root:
            raise ValueError("FileService root must equal the execution workspace")
        journal = getattr(file_service, "journal", None)
        if not isinstance(journal, OperationJournal):
            raise ValueError("reversible executor requires OperationJournal proof")
        if not handler.__module__.startswith(
            "aerospace_agent.mcp.tools.core_tool_adapters"
        ):
            raise ValueError("only the current-repository FileService adapter is reversible")
        expected_path_fields = {
            "file_write": ("path",),
            "file_append": ("path",),
            "file_mkdir": ("path",),
            "file_copy": ("source", "destination"),
            "file_move": ("source", "destination"),
            "file_delete": ("path",),
        }
        expected = expected_path_fields.get(handler.__name__)
        if expected is None or tuple(path_fields) != expected:
            raise ValueError("reversible FileService adapter has incomplete path_fields")
        if "operation_id" not in inspect.signature(handler).parameters:
            raise ValueError("reversible executor must bind the execution operation_id")
        expected_database = self._workspace_root / ".agent_core" / "operation_journal.sqlite3"
        expected_backups = self._workspace_root / ".agent_core" / "preimages"
        if journal.database_path != expected_database or journal.backup_dir != expected_backups:
            raise ValueError("FileService recovery journal must use the fixed workspace paths")
        try:
            journal.database_path.relative_to(self._workspace_root)
            journal.backup_dir.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("recovery journal and backups must stay inside workspace") from exc
        return journal

    def preview_registration_digest(
        self,
        *,
        kind: ExecutionKind,
        manifest: CapabilityManifest,
        executor_name: str,
        handler: ExecutorHandler | None,
        input_model: type[BaseModel],
        entrypoint: str,
        adapter_path: str | Path,
        recovery_class: RecoveryClass,
        path_fields: Sequence[str] = (),
        dependency_files: Mapping[str, str | Path] | None = None,
        cache_files: Mapping[str, str | Path] | None = None,
        validation_evidence: Mapping[str, str] | None = None,
        requires_confirmation: bool = False,
        workflow_snapshot: WorkflowSnapshot | None = None,
        runtime_trust_digest: str | None = None,
    ) -> str:
        """Return the exact digest an external operator must approve."""

        checked = CapabilityManifest.model_validate(manifest.model_dump(mode="python"))
        self._validate_kind_policy(kind, checked, executor_name, handler, workflow_snapshot)
        if not isinstance(input_model, type) or not issubclass(input_model, BaseModel):
            raise TypeError("input_model must be a Pydantic BaseModel type")
        if {"confirmed", "operation_id"} & set(input_model.model_fields):
            raise ValueError("execution-managed fields cannot appear in the public input schema")
        if len(set(path_fields)) != len(tuple(path_fields)) or not set(path_fields).issubset(
            input_model.model_fields
        ):
            raise ValueError("path_fields must be unique fields declared by input_model")
        adapter = self._resolve_registration_path(adapter_path)
        self._validate_entrypoint_and_handler(
            kind=kind,
            manifest=checked,
            entrypoint=entrypoint,
            handler=handler,
            adapter_path=adapter,
            input_model=input_model,
        )
        dependencies = dict(dependency_files or {})
        caches = dict(cache_files or {})
        evidence = dict(validation_evidence or {})
        if set(dependencies) != set(checked.required_dependencies):
            raise ValueError("dependency_files keys must exactly match manifest.required_dependencies")
        if set(evidence) != set(checked.validators):
            raise ValueError("validation_evidence keys must exactly match manifest.validators")
        operation_journal = self._validate_recovery_binding(
            handler=handler,
            recovery_class=recovery_class,
            path_fields=path_fields,
        )
        return self._registration_digest(
            kind=kind,
            manifest=checked,
            executor_name=executor_name,
            input_model=input_model,
            input_model_evidence=self._capture_input_model_evidence(input_model),
            entrypoint=entrypoint,
            adapter_hash=_sha256_file(adapter),
            manifest_hash=_manifest_hash(checked),
            recovery_class=recovery_class,
            path_fields=path_fields,
            dependency_hashes=self._capture_named_hashes(dependencies),
            cache_hashes=self._capture_named_hashes(caches),
            validation_evidence=evidence,
            requires_confirmation=requires_confirmation,
            workflow_snapshot=workflow_snapshot,
            operation_journal=operation_journal,
            runtime_trust_digest=runtime_trust_digest,
        )

    @staticmethod
    def _registration_digest(
        *,
        kind: ExecutionKind,
        manifest: CapabilityManifest,
        executor_name: str,
        input_model: type[BaseModel],
        input_model_evidence: tuple[str, Path, str, str],
        entrypoint: str,
        adapter_hash: str,
        manifest_hash: str,
        recovery_class: RecoveryClass,
        path_fields: Sequence[str],
        dependency_hashes: tuple[tuple[str, Path, str], ...],
        cache_hashes: tuple[tuple[str, Path, str], ...],
        validation_evidence: Mapping[str, str],
        requires_confirmation: bool,
        workflow_snapshot: WorkflowSnapshot | None,
        operation_journal: OperationJournal | None,
        runtime_trust_digest: str | None,
    ) -> str:
        return _canonical_sha256(
            {
                "kind": kind,
                "capability_id": manifest.capability_id,
                "executor_name": executor_name,
                "entrypoint": entrypoint,
                "manifest_sha256": manifest_hash,
                "adapter_sha256": adapter_hash,
                "dependencies": [(name, digest) for name, _, digest in dependency_hashes],
                "caches": [(name, digest) for name, _, digest in cache_hashes],
                "input_schema": input_model.model_json_schema(),
                "input_model": {
                    "qualified_name": input_model_evidence[0],
                    "file_sha256": input_model_evidence[2],
                    "implementation_sha256": input_model_evidence[3],
                },
                "path_fields": sorted(set(path_fields)),
                "recovery_class": recovery_class,
                "requires_confirmation": bool(requires_confirmation),
                "validation_evidence": sorted(validation_evidence.items()),
                "runtime_trust_digest": runtime_trust_digest,
                "recovery_binding": (
                    {
                        "journal_database": str(operation_journal.database_path),
                        "backup_directory": str(operation_journal.backup_dir),
                    }
                    if operation_journal is not None
                    else None
                ),
                "workflow_snapshot": (
                    workflow_snapshot.model_dump(mode="json")
                    if workflow_snapshot is not None
                    else None
                ),
            }
        )

    def _validate_kind_policy(
        self,
        kind: ExecutionKind,
        manifest: CapabilityManifest,
        executor_name: str,
        handler: ExecutorHandler | None,
        workflow_snapshot: WorkflowSnapshot | None,
    ) -> None:
        if kind == "human":
            if handler is not None:
                raise ValueError("human executor must not have a handler")
            if manifest.category not in {"workflow", "project"}:
                raise ValueError("human executor requires workflow/project category")
            return
        if handler is None:
            raise ValueError(f"{kind} executor requires a handler")
        if kind == "tool":
            if manifest.category not in {"basic", "space_basic", "memory", "project"}:
                raise ValueError("tool executor has incompatible manifest category")
            if executor_name not in manifest.tool_names:
                raise ValueError("tool executor must be declared in manifest.tool_names")
        elif kind == "workflow":
            if manifest.category != "workflow":
                raise ValueError("workflow executor requires workflow manifest category")
            if workflow_snapshot is None:
                raise ValueError("workflow executor requires a locked workflow snapshot")
            if (
                workflow_snapshot.workflow_id != executor_name
                or workflow_snapshot.version != manifest.version
            ):
                raise ValueError("workflow snapshot identity/version mismatch")
        elif kind == "domain":
            if manifest.category != "domain" or not (
                manifest.source == "aerospace_agent.domains"
                or manifest.source.startswith("aerospace_agent.domains.")
            ):
                raise ValueError("domain executor requires aerospace_agent.domains manifest")
        elif kind == "capability_builder":
            if manifest.category != "project" or not (
                manifest.source == "aerospace_agent.integrations"
                or manifest.source.startswith("aerospace_agent.integrations.")
            ):
                raise ValueError("capability_builder requires approved integrations project manifest")
            if manifest.risk_level != "high_risk":
                raise ValueError("capability_builder must be high_risk")

    def _validate_entrypoint_and_handler(
        self,
        *,
        kind: ExecutionKind,
        manifest: CapabilityManifest,
        entrypoint: str,
        handler: ExecutorHandler | None,
        adapter_path: Path,
        input_model: type[BaseModel],
    ) -> None:
        matched_root = self._matched_import_root(entrypoint)
        relative_root = Path(*matched_root.split("."))
        allowed_directory = (self._workspace_root / relative_root).resolve()
        try:
            adapter_path.relative_to(allowed_directory)
        except ValueError as exc:
            raise ValueError("adapter path does not belong to entrypoint import root") from exc
        if not adapter_path.is_file():
            raise ValueError(f"adapter file does not exist: {adapter_path}")
        if not (
            manifest.source == matched_root
            or manifest.source.startswith(f"{matched_root}.")
        ):
            raise ValueError("manifest source is outside the matched import root")
        if not (
            entrypoint.startswith(f"{manifest.source}.") or entrypoint == manifest.source
        ):
            raise ValueError("entrypoint does not belong to manifest source")
        if kind == "human":
            return
        assert handler is not None
        target = inspect.unwrap(handler.__func__ if inspect.ismethod(handler) else handler)
        actual_entrypoint = f"{target.__module__}.{target.__qualname__}"
        if entrypoint != actual_entrypoint:
            raise ValueError(
                f"entrypoint does not match actual handler identity: {actual_entrypoint}"
            )
        source_file = inspect.getsourcefile(target)
        if source_file is None or Path(source_file).resolve() != adapter_path:
            raise ValueError("handler source file does not match adapter path")
        signature = inspect.signature(handler)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            accepted = {
                name
                for name, parameter in signature.parameters.items()
                if parameter.kind
                in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            }
            unexpected = set(input_model.model_fields) - accepted
            if unexpected:
                raise ValueError(
                    f"input model fields are not accepted by handler: {sorted(unexpected)}"
                )

    @staticmethod
    def _matched_import_root(entrypoint: str) -> str:
        matched = [
            root
            for root in ALLOWED_IMPORT_ROOTS
            if entrypoint == root or entrypoint.startswith(f"{root}.")
        ]
        if not matched:
            raise ValueError("entrypoint is outside allowed current-repository import roots")
        return max(matched, key=len)

    def _resolve_registration_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self._workspace_root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError(f"registration path is outside workspace: {resolved}") from exc
        return resolved

    def _capture_named_hashes(
        self,
        files: Mapping[str, str | Path],
    ) -> tuple[tuple[str, Path, str], ...]:
        captured: list[tuple[str, Path, str]] = []
        for name in sorted(files):
            path = self._resolve_registration_path(files[name])
            if not path.is_file():
                raise ValueError(f"trusted dependency/cache file does not exist: {path}")
            captured.append((name, path, _sha256_file(path)))
        return tuple(captured)

    def _capture_input_model_evidence(
        self,
        input_model: type[BaseModel],
    ) -> tuple[str, Path, str, str]:
        source_file = inspect.getsourcefile(input_model)
        if source_file is None:
            raise ValueError("input model must have inspectable source code")
        path = Path(source_file).resolve()
        try:
            path.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("input model source must stay inside the current workspace") from exc
        identity = f"{input_model.__module__}.{input_model.__qualname__}"
        return (
            identity,
            path,
            _sha256_file(path),
            self._input_model_implementation_sha256(input_model),
        )

    @staticmethod
    def _input_model_implementation_sha256(input_model: type[BaseModel]) -> str:
        try:
            return hashlib.sha256(inspect.getsource(input_model).encode("utf-8")).hexdigest()
        except (OSError, TypeError):
            # Pydantic ``create_model`` classes have no class source range.  Bind
            # their qualified identity, strict schema, and every attached
            # validator callable instead of silently trusting schema alone.
            validators: list[dict[str, str]] = []
            decorators = getattr(input_model, "__pydantic_decorators__", None)
            for group_name in (
                "validators",
                "field_validators",
                "root_validators",
                "field_serializers",
                "model_serializers",
                "model_validators",
                "computed_fields",
            ):
                group = getattr(decorators, group_name, {}) if decorators is not None else {}
                for name, info in sorted(group.items()):
                    function = getattr(info, "func", None)
                    target = inspect.unwrap(
                        function.__func__ if inspect.ismethod(function) else function
                    ) if callable(function) else None
                    if target is None:
                        implementation = "missing"
                    else:
                        try:
                            implementation = inspect.getsource(target)
                        except (OSError, TypeError):
                            code = getattr(target, "__code__", None)
                            implementation = (
                                repr((code.co_code, code.co_consts, code.co_names))
                                if code is not None
                                else f"{target.__module__}.{target.__qualname__}"
                            )
                    validators.append(
                        {
                            "group": group_name,
                            "name": name,
                            "sha256": hashlib.sha256(
                                implementation.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
            return _canonical_sha256(
                {
                    "identity": f"{input_model.__module__}.{input_model.__qualname__}",
                    "schema": input_model.model_json_schema(),
                    "validators": validators,
                }
            )

    def _approval_record(self, digest: str) -> ApprovalRecord | None:
        if self._approval_verifier is None:
            return None
        return self._approval_verifier.verify_digest(digest)

    def snapshot(self, capability_id: str) -> CapabilitySnapshot:
        matches = [
            item for item in self._registrations.values() if item.manifest.capability_id == capability_id
        ]
        if not matches:
            raise KeyError(f"unknown capability_id: {capability_id}")
        baseline = matches[0]
        if any(
            item.manifest_sha256 != baseline.manifest_sha256
            or item.adapter_sha256 != baseline.adapter_sha256
            for item in matches[1:]
        ):
            raise RuntimeError(f"capability registrations do not share one snapshot: {capability_id}")
        return CapabilitySnapshot(
            capability_id=capability_id,
            version=baseline.manifest.version,
            manifest_sha256=baseline.manifest_sha256,
            adapter_sha256=baseline.adapter_sha256,
        )

    def preview_action_hash(
        self,
        execution_request: ExecutionRequest,
        execution_context: ExecutionContext,
    ) -> str:
        request, context, registration = self._validated_request_context(
            execution_request, execution_context
        )
        arguments = self._validate_input_and_paths(registration, request.arguments)
        target_paths = [
            str(arguments[field])
            for field in registration.path_fields
            if field in arguments and arguments[field] is not None
        ]
        return compute_action_hash(
            tool_name=request.executor_name,
            arguments=arguments,
            target_paths=target_paths,
            run_id=context.root_run_id,
            risk_level=registration.manifest.risk_level,
        )

    def resolve(
        self,
        execution_request: ExecutionRequest,
        execution_context: ExecutionContext,
    ) -> AuthorizedExecutor | ToolResult:
        request = ExecutionRequest.model_validate(execution_request.model_dump(mode="python"))
        context = ExecutionContext.model_validate(execution_context.model_dump(mode="python"))
        registration = self._registrations.get(
            (request.kind, request.capability_id, request.executor_name)
        )
        if registration is None:
            return self._failure(
                request,
                context,
                status="unavailable",
                code="unavailable",
                message="executor is not registered for requested kind/capability/name",
            )
        policy_failure = self._validate_resolution_policy(request, context, registration)
        if policy_failure is not None:
            return policy_failure
        try:
            validated_arguments = self._validate_input_and_paths(registration, request.arguments)
        except (ValueError, TypeError) as exc:
            code = (
                "path_outside_workspace"
                if "path" in str(exc).casefold()
                else "invalid_arguments"
            )
            status = "blocked" if code == "path_outside_workspace" else "invalid_arguments"
            return self._failure(
                request,
                context,
                registration=registration,
                status=status,
                code=code,
                message=str(exc),
            )
        confirmation_consumed = False
        if self._requires_confirmation(registration, validated_arguments):
            confirmation_error = self._consume_confirmation(
                request, context, registration, validated_arguments
            )
            if confirmation_error is not None:
                return self._failure(
                    request,
                    context,
                    registration=registration,
                    status="blocked",
                    code=confirmation_error.code,
                    message=str(confirmation_error),
                )
            confirmation_consumed = True
        now = self._now()
        token = uuid4().hex
        authorized = AuthorizedExecutor(
            kind=request.kind,
            capability_id=request.capability_id,
            executor_name=request.executor_name,
            operation_id=request.operation_id,
            recovery_class=registration.recovery_class,
            token=token,
            factory=_AUTHORIZATION_FACTORY,
        )
        self._authorizations[token] = _AuthorizationRecord(
            authorized=authorized,
            registration=registration,
            request=request,
            context=context,
            validated_arguments=validated_arguments,
            confirmation_consumed=confirmation_consumed,
            authorized_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=self._authorization_ttl_seconds)).isoformat(),
        )
        return authorized

    def _validated_request_context(
        self,
        execution_request: ExecutionRequest,
        execution_context: ExecutionContext,
    ) -> tuple[ExecutionRequest, ExecutionContext, _Registration]:
        request = ExecutionRequest.model_validate(execution_request.model_dump(mode="python"))
        context = ExecutionContext.model_validate(execution_context.model_dump(mode="python"))
        registration = self._registrations.get(
            (request.kind, request.capability_id, request.executor_name)
        )
        if registration is None:
            raise ValueError("executor is not registered")
        failure = self._validate_resolution_policy(request, context, registration, audit=False)
        if failure is not None:
            raise ValueError(failure.error.message if failure.error else "execution policy blocked")
        return request, context, registration

    def _validate_resolution_policy(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
        registration: _Registration,
        *,
        audit: bool = True,
    ) -> ToolResult | None:
        def fail(status: str, code: str, message: str) -> ToolResult:
            if audit:
                return self._failure(
                    request,
                    context,
                    registration=registration,
                    status=status,
                    code=code,
                    message=message,
                )
            return ToolResult(
                status=status,
                error=ToolError(code=code, message=message, recoverability="not_applicable"),
                audit_id=uuid4().hex,
                operation_id=request.operation_id,
                recovery_class=registration.recovery_class,
            )

        if registration.manifest.status != "available":
            return fail("unavailable", "unavailable", f"capability status is {registration.manifest.status}")
        if Path(context.workspace_root).resolve() != self._workspace_root:
            return fail("blocked", "path_outside_workspace", "execution workspace mismatch")
        if context.capability_snapshot != self.snapshot(request.capability_id):
            return fail("blocked", "conflict", "capability execution snapshot mismatch")
        if registration.kind == "workflow":
            if context.workflow_snapshot != registration.workflow_snapshot:
                return fail("blocked", "conflict", "workflow execution snapshot mismatch")
        requires_plan = registration.kind in {"domain", "capability_builder", "human"}
        if requires_plan and request.origin != "planned":
            return fail("blocked", "conflict", f"{registration.kind} execution requires planned origin")
        if (
            request.origin == "direct"
            and self._plan_execution_verifier is not None
            and self._plan_execution_verifier.has_plan_for_run(
                project_id=context.project_id,
                thread_id=context.thread_id,
                root_run_id=context.root_run_id,
            )
        ):
            return fail("blocked", "conflict", "root run is governed by an immutable TaskPlan")
        if request.origin == "planned":
            if (
                context.plan_id is None
                or context.plan_sha256 is None
                or context.registry_snapshot_sha256 is None
                or request.step_id is None
            ):
                return fail("blocked", "conflict", "planned execution requires exact plan and step binding")
            try:
                planned_arguments = self._validate_input_and_paths(registration, request.arguments)
            except (ValueError, TypeError):
                return fail("blocked", "conflict", "planned execution inputs are invalid")
            if self._plan_execution_verifier is None or not self._plan_execution_verifier.verify(
                project_id=context.project_id,
                thread_id=context.thread_id,
                root_run_id=context.root_run_id,
                plan_id=context.plan_id,
                plan_sha256=context.plan_sha256,
                step_id=request.step_id,
                kind=request.kind,
                capability_id=request.capability_id,
                executor_name=request.executor_name,
                arguments=planned_arguments,
                capability_snapshot=context.capability_snapshot,
                workflow_snapshot=context.workflow_snapshot,
                registry_snapshot_sha256=context.registry_snapshot_sha256,
                domain_state=request.domain_state,
            ):
                return fail("blocked", "conflict", "plan step binding verification failed")
        elif context.plan_id is not None:
            return fail("blocked", "conflict", "direct execution cannot reuse a plan snapshot")
        if registration.approval_required:
            record = self._approval_record(registration.registration_digest)
            if (
                record is None
                or record.approval_record_id != registration.approval_record_id
            ):
                return fail("unavailable", "unavailable", "approval baseline is missing or invalid")
        if registration.runtime_trust_verifier is not None:
            try:
                verification = registration.runtime_trust_verifier()
                if isinstance(verification, Mapping):
                    trust_status = verification.get("status")
                    trust_digest = verification.get("combined_digest")
                else:
                    trust_status = getattr(verification, "status", None)
                    trust_digest = getattr(verification, "combined_digest", None)
            except Exception as exc:
                return fail(
                    "unavailable",
                    "unavailable",
                    f"runtime trust verification failed: {exc}",
                )
            if (
                trust_status != "available"
                or trust_digest != registration.runtime_trust_digest
            ):
                return fail(
                    "unavailable",
                    "unavailable",
                    "runtime trust baseline is unavailable or drifted",
                )
        try:
            if not self._trusted_hashes_match(registration):
                return fail("unavailable", "unavailable", "adapter/dependency/cache hash mismatch")
        except OSError as exc:
            return fail("unavailable", "unavailable", f"trusted execution file unavailable: {exc}")
        return None

    def _validate_input_and_paths(
        self,
        registration: _Registration,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            validated = registration.input_model.model_validate(dict(arguments))
        except Exception as exc:
            raise ValueError(f"input schema validation failed: {exc}") from exc
        normalized = validated.model_dump(mode="python", exclude_none=False)
        for field in registration.path_fields:
            raw_path = normalized.get(field)
            if raw_path is None:
                continue
            if not isinstance(raw_path, str):
                raise ValueError(f"path field must be a string: {field}")
            path = Path(raw_path)
            if not path.is_absolute():
                path = self._workspace_root / path
            resolved = path.resolve()
            try:
                resolved.relative_to(self._workspace_root)
            except ValueError as exc:
                raise ValueError(f"path resolves outside workspace: {field}") from exc
            normalized[field] = str(resolved)
        _canonical_sha256(normalized)
        return normalized

    @staticmethod
    def _requires_confirmation(
        registration: _Registration,
        arguments: Mapping[str, Any] | None = None,
    ) -> bool:
        # TerminalService performs a second, command-level classification. A
        # version/help/status invocation is read-only and must not be blocked
        # by the catalog's worst-case manual-recovery registration. Mutating
        # or long-running argv still consumes the normal confirmation token.
        if (
            registration.executor_name == "terminal.run"
            and arguments is not None
            and ExecutionRegistry._terminal_arguments_are_read_only(arguments)
        ):
            return False
        return (
            registration.requires_confirmation
            or registration.manifest.risk_level == "high_risk"
            or registration.recovery_class in {"compensatable", "manual_recovery"}
            or registration.kind == "capability_builder"
        )

    @staticmethod
    def _terminal_arguments_are_read_only(arguments: Mapping[str, Any]) -> bool:
        argv = arguments.get("argv")
        if not isinstance(argv, (list, tuple)) or not argv:
            return False
        values = [str(item).casefold() for item in argv]
        flags = {"--version", "-v", "-vv", "--help", "-h"}
        if len(values) > 1 and all(item in flags for item in values[1:]):
            return True
        executable = Path(values[0]).name
        return executable in {"git", "git.exe"} and tuple(values[1:2]) in {
            ("status",),
            ("diff",),
            ("log",),
        }

    def _consume_confirmation(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
        registration: _Registration,
        validated_arguments: Mapping[str, Any],
    ) -> ConfirmationError | None:
        if self._confirmation_service is None or request.confirmation_id is None:
            return ConfirmationError("confirmation_required", "protected operation requires confirmation")
        target_paths = [
            str(validated_arguments[field])
            for field in registration.path_fields
            if field in validated_arguments and validated_arguments[field] is not None
        ]
        action_hash = compute_action_hash(
            tool_name=request.executor_name,
            arguments=validated_arguments,
            target_paths=target_paths,
            run_id=context.root_run_id,
            risk_level=registration.manifest.risk_level,
        )
        try:
            self._confirmation_service.consume(
                confirmation_id=request.confirmation_id,
                project_id=context.project_id,
                thread_id=context.thread_id,
                root_run_id=context.root_run_id,
                operation_id=request.operation_id,
                action_hash=action_hash,
                continuation_checkpoint={
                    "checkpoint_id": f"confirmation:{request.operation_id}",
                    "plan_id": context.plan_id,
                    "plan_sha256": context.plan_sha256,
                },
            )
        except ConfirmationError as exc:
            return exc
        return None

    @staticmethod
    def _trusted_hashes_match(registration: _Registration) -> bool:
        if not registration.adapter_path.is_file():
            return False
        if _sha256_file(registration.adapter_path) != registration.adapter_sha256:
            return False
        if (
            not registration.input_model_path.is_file()
            or _sha256_file(registration.input_model_path)
            != registration.input_model_file_sha256
        ):
            return False
        if (
            ExecutionRegistry._input_model_implementation_sha256(registration.input_model)
            != registration.input_model_source_sha256
        ):
            return False
        for _, path, expected in (*registration.dependency_hashes, *registration.cache_hashes):
            if not path.is_file() or _sha256_file(path) != expected:
                return False
        return True

    def _consume_authorization(self, authorized: AuthorizedExecutor) -> _AuthorizationRecord | None:
        token = authorized._authorization_token()
        record = self._authorizations.get(token)
        if record is None or record.authorized is not authorized:
            return None
        del self._authorizations[token]
        self._used_authorizations[token] = record
        return record

    def _authorization_record(self, authorized: AuthorizedExecutor) -> _AuthorizationRecord | None:
        token = authorized._authorization_token()
        return self._authorizations.get(token) or self._used_authorizations.get(token)

    def _revalidate_authorization(self, record: _AuthorizationRecord) -> ToolResult | None:
        if self._now() >= datetime.fromisoformat(record.expires_at).astimezone(UTC):
            return self._failure(
                record.request,
                record.context,
                registration=record.registration,
                status="blocked",
                code="conflict",
                message="authorized executor expired before execution",
                record=False,
            )
        failure = self._validate_resolution_policy(
            record.request, record.context, record.registration, audit=False
        )
        if failure is not None:
            return failure
        try:
            current_arguments = self._validate_input_and_paths(
                record.registration, record.request.arguments
            )
        except (ValueError, TypeError) as exc:
            return self._failure(
                record.request,
                record.context,
                registration=record.registration,
                status="blocked",
                code="conflict",
                message=f"execution inputs changed or became invalid: {exc}",
                record=False,
            )
        if current_arguments != record.validated_arguments:
            return self._failure(
                record.request,
                record.context,
                registration=record.registration,
                status="blocked",
                code="conflict",
                message="normalized execution arguments changed after authorization",
                record=False,
            )
        return None

    def _failure(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
        *,
        status: Literal["blocked", "invalid_arguments", "unavailable"],
        code: Literal[
            "invalid_arguments",
            "path_outside_workspace",
            "confirmation_required",
            "confirmation_expired",
            "confirmation_replayed",
            "unavailable",
            "conflict",
        ],
        message: str,
        registration: _Registration | None = None,
        record: bool = True,
    ) -> ToolResult:
        result = ToolResult(
            status=status,
            error=ToolError(
                code=code,
                message=message,
                recoverability="retryable" if code in {"conflict", "unavailable"} else "not_applicable",
            ),
            audit_id=uuid4().hex,
            operation_id=request.operation_id,
            recovery_class=registration.recovery_class if registration else "manual_recovery",
        )
        if record:
            self._record_result(request, context, result, registration=registration)
        return result

    def _record_result(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
        result: ToolResult,
        *,
        registration: _Registration | None,
        arguments: Mapping[str, Any] | None = None,
    ) -> bool:
        result_hash = _canonical_sha256(result.model_dump(mode="json"))
        arguments_hash = _canonical_sha256(
            dict(arguments) if arguments is not None else request.arguments
        )
        risk_level = registration.manifest.risk_level if registration else "unknown"
        values = (
            result.audit_id,
            request.operation_id,
            context.root_run_id,
            context.project_id,
            context.thread_id,
            request.capability_id,
            request.executor_name,
            request.kind,
            risk_level,
            self._now().isoformat(),
            result.status,
            result_hash,
            arguments_hash,
            result.recovery_class,
            result.error.code if result.error else None,
        )
        try:
            with closing(self._audit_connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO execution_audit(
                        audit_id, operation_id, root_run_id, project_id, thread_id,
                        capability_id, executor_name, kind, risk_level, created_at,
                        status, result_sha256, arguments_sha256, recovery_class, error_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                connection.commit()
            return True
        except (OSError, sqlite3.Error) as exc:
            fallback = self._audit_database_path.with_name("execution_audit_fallback.jsonl")
            payload = {
                "audit_id": result.audit_id,
                "operation_id": request.operation_id,
                "root_run_id": context.root_run_id,
                "project_id": context.project_id,
                "thread_id": context.thread_id,
                "capability_id": request.capability_id,
                "executor_name": request.executor_name,
                "kind": request.kind,
                "risk_level": risk_level,
                "created_at": self._now().isoformat(),
                "status": result.status,
                "result_sha256": result_hash,
                "arguments_sha256": arguments_hash,
                "recovery_class": result.recovery_class,
                "error_code": result.error.code if result.error else None,
                "audit_storage_error": type(exc).__name__,
            }
            try:
                encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
                descriptor = os.open(
                    fallback,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                try:
                    os.write(descriptor, encoded)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
            return False

    def audit_records(self) -> list[dict[str, Any]]:
        with closing(self._audit_connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM execution_audit ORDER BY rowid"
            ).fetchall()
        return [dict(row) for row in rows]


class ExecutionService:
    """The only component allowed to invoke a registered handler."""

    def __init__(
        self,
        registry: ExecutionRegistry,
        *,
        checkpoint_store: ExecutionCheckpointStore,
    ) -> None:
        if not isinstance(checkpoint_store, ExecutionCheckpointStore):
            raise TypeError("checkpoint_store must be ExecutionCheckpointStore")
        self._registry = registry
        self._checkpoint_store = checkpoint_store

    def execute(self, authorized: AuthorizedExecutor) -> ToolResult:
        record = self._registry._consume_authorization(authorized)
        if record is None:
            known = self._registry._authorization_record(authorized)
            if known is None:
                raise TypeError("AuthorizedExecutor does not belong to this registry")
            result = ToolResult(
                status="blocked",
                error=ToolError(
                    code="conflict",
                    message="authorization is invalid or already consumed",
                    recoverability="not_applicable",
                ),
                audit_id=uuid4().hex,
                operation_id=authorized.operation_id,
                recovery_class=authorized.recovery_class,
            )
            self._registry._record_result(
                known.request,
                known.context,
                result,
                registration=known.registration,
                arguments=known.validated_arguments,
            )
            return result

        revalidation_failure = self._registry._revalidate_authorization(record)
        if revalidation_failure is not None:
            return self._checkpoint_or_failure(record, revalidation_failure, side_effect_possible=False)

        try:
            claim_status, prior_result = self._checkpoint_store.claim_step(
                record.request,
                record.context,
                record.validated_arguments,
            )
        except Exception as exc:
            result = ToolResult(
                status="blocked",
                error=ToolError(
                    code="conflict",
                    message=f"planned step idempotency claim failed: {exc}",
                    recoverability="manual_recovery",
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class="manual_recovery",
            )
            self._registry._record_result(
                record.request,
                record.context,
                result,
                registration=record.registration,
                arguments=record.validated_arguments,
            )
            return result
        if claim_status == "completed" and prior_result is not None:
            return prior_result
        if claim_status != "claimed":
            result = ToolResult(
                status="blocked",
                error=ToolError(
                    code="conflict",
                    message="planned step input is already in flight or has unknown side effects",
                    recoverability="manual_recovery",
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class="manual_recovery",
            )
            self._registry._record_result(
                record.request,
                record.context,
                result,
                registration=record.registration,
                arguments=record.validated_arguments,
            )
            return result

        registration = record.registration
        if registration.kind == "human":
            result = ToolResult(
                status="interrupted",
                result={"instruction": record.validated_arguments},
                error=ToolError(
                    code="interrupted",
                    message="human step requires an external response",
                    recoverability="manual_recovery",
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class="manual_recovery",
            )
            return self._checkpoint_or_failure(record, result, side_effect_possible=False)

        try:
            assert registration.handler is not None
            call_arguments = dict(record.validated_arguments)
            if registration.operation_journal is not None:
                call_arguments["operation_id"] = record.request.operation_id
            if record.confirmation_consumed and "confirmed" in inspect.signature(
                registration.handler
            ).parameters:
                call_arguments["confirmed"] = True
            if record.confirmation_consumed and "confirmation_consumed" in inspect.signature(
                registration.handler
            ).parameters:
                call_arguments["confirmation_consumed"] = True
            if record.confirmation_consumed and "confirmation_consumed" in inspect.signature(
                registration.handler
            ).parameters:
                call_arguments["confirmation_consumed"] = True
            if registration.kind == "domain" and record.request.domain_state:
                parameters = inspect.signature(registration.handler).parameters
                if "state" not in parameters and not any(
                    item.kind == inspect.Parameter.VAR_KEYWORD
                    for item in parameters.values()
                ):
                    raise TypeError("domain executor does not accept required bound state")
                call_arguments["state"] = dict(record.request.domain_state)
            raw_result = registration.handler(**call_arguments)
            if registration.kind == "domain":
                try:
                    domain_output = DomainExecutionOutput.model_validate(raw_result)
                except Exception as exc:
                    raise TypeError(
                        f"domain executor must return a valid DomainExecutionOutput: {exc}"
                    ) from exc
                if not domain_output.artifacts:
                    raise TypeError(
                        "domain executor DomainExecutionOutput must contain at least one artifact"
                    )
                result = ToolResult(
                    status="success",
                    result={
                        "domain_output": domain_output.model_dump(mode="json"),
                    },
                    audit_id=uuid4().hex,
                    operation_id=record.request.operation_id,
                    recovery_class=registration.recovery_class,
                )
            elif isinstance(raw_result, ToolResult):
                if registration.operation_journal is not None:
                    result = self._verified_journal_result(record, raw_result)
                else:
                    data = raw_result.model_dump(mode="python")
                    data.update(
                        {
                            "audit_id": uuid4().hex,
                            "operation_id": record.request.operation_id,
                            "recovery_class": registration.recovery_class,
                        }
                    )
                    result = ToolResult.model_validate(data)
            elif isinstance(raw_result, dict):
                if registration.operation_journal is not None:
                    raise TypeError("reversible executor must return a journal-bound ToolResult")
                _canonical_sha256(raw_result)
                result = ToolResult(
                    status="success",
                    result=raw_result,
                    audit_id=uuid4().hex,
                    operation_id=record.request.operation_id,
                    recovery_class=registration.recovery_class,
                )
            else:
                raise TypeError("executor handler must return dict or ToolResult")
        except Exception as exc:
            rollback_message = ""
            recovery_class = registration.recovery_class
            recoverability = (
                "reversible"
                if registration.recovery_class == "reversible"
                else "manual_recovery"
            )
            if registration.operation_journal is not None:
                operation = registration.operation_journal.get(record.request.operation_id)
                if operation is not None:
                    try:
                        registration.operation_journal.rollback(record.request.operation_id)
                        rollback_message = "; journal preimages restored and verified"
                    except Exception as rollback_exc:
                        rollback_message = f"; rollback failed: {rollback_exc}"
                        recovery_class = "manual_recovery"
                        recoverability = "manual_recovery"
            result = ToolResult(
                status="failed",
                error=ToolError(
                    code="failed",
                    message=f"executor failed: {exc}{rollback_message}",
                    recoverability=recoverability,
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class=recovery_class,
            )
        return self._checkpoint_or_failure(record, result, side_effect_possible=True)

    @staticmethod
    def _verified_journal_result(
        record: _AuthorizationRecord,
        raw_result: ToolResult,
    ) -> ToolResult:
        journal = record.registration.operation_journal
        assert journal is not None
        if raw_result.operation_id != record.request.operation_id:
            return ToolResult(
                status="failed",
                error=ToolError(
                    code="failed",
                    message="reversible executor returned a mismatched operation ID",
                    recoverability="manual_recovery",
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class="manual_recovery",
            )
        operation = journal.get(record.request.operation_id)
        if raw_result.status == "success":
            if operation is None and raw_result.recovery_class == "read_only":
                # A reversible adapter may truthfully report an idempotent
                # no-op (for example mkdir on an existing directory).  No
                # rollback proof is required because no mutation occurred.
                return ToolResult.model_validate(raw_result.model_dump(mode="python"))
            if (
                operation is None
                or operation.get("status") != "completed"
                or operation.get("audit_id") != raw_result.audit_id
                or raw_result.recovery_class != "reversible"
            ):
                return ToolResult(
                    status="failed",
                    error=ToolError(
                        code="failed",
                        message="reversible execution lacks a completed journal proof",
                        recoverability="manual_recovery",
                    ),
                    audit_id=uuid4().hex,
                    operation_id=record.request.operation_id,
                    recovery_class="manual_recovery",
                )
            return ToolResult.model_validate(raw_result.model_dump(mode="python"))
        if operation is None:
            return ToolResult.model_validate(raw_result.model_dump(mode="python"))
        try:
            journal.rollback(record.request.operation_id)
        except Exception as exc:
            return ToolResult(
                status="failed",
                error=ToolError(
                    code="failed",
                    message=f"executor failed and verified rollback did not complete: {exc}",
                    recoverability="manual_recovery",
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class="manual_recovery",
            )
        return ToolResult(
            status="failed",
            error=ToolError(
                code="failed",
                message="executor failed; journal preimages were restored and verified",
                recoverability="reversible",
            ),
            audit_id=uuid4().hex,
            operation_id=record.request.operation_id,
            recovery_class="reversible",
        )

    def _checkpoint_or_failure(
        self,
        record: _AuthorizationRecord,
        result: ToolResult,
        *,
        side_effect_possible: bool,
    ) -> ToolResult:
        try:
            receipt = self._checkpoint_store.write(
                record.request,
                record.context,
                result,
                arguments=record.validated_arguments,
            )
            if (
                receipt.project_id != record.context.project_id
                or receipt.thread_id != record.context.thread_id
                or receipt.root_run_id != record.context.root_run_id
                or receipt.operation_id != record.request.operation_id
            ):
                raise RuntimeError("checkpoint receipt identity mismatch")
            persisted_receipt, persisted_result = self._checkpoint_store.get(
                receipt.checkpoint_id
            )
            if persisted_receipt != receipt or persisted_result != result:
                raise RuntimeError("checkpoint receipt/result readback mismatch")
        except Exception as exc:
            failure = ToolResult(
                status="failed",
                error=ToolError(
                    code="failed",
                    message=f"checkpoint write failed after execution boundary: {exc}",
                    recoverability=(
                        "manual_recovery" if side_effect_possible else "retryable"
                    ),
                ),
                audit_id=uuid4().hex,
                operation_id=record.request.operation_id,
                recovery_class="manual_recovery",
            )
            self._registry._record_result(
                record.request,
                record.context,
                failure,
                registration=record.registration,
                arguments=record.validated_arguments,
            )
            return failure
        if self._registry._record_result(
            record.request,
            record.context,
            result,
            registration=record.registration,
            arguments=record.validated_arguments,
        ):
            return result
        return ToolResult(
            status="failed",
            error=ToolError(
                code="failed",
                message="execution completed but primary audit persistence failed; fallback attempted",
                recoverability="manual_recovery" if side_effect_possible else "retryable",
            ),
            audit_id=uuid4().hex,
            operation_id=record.request.operation_id,
            recovery_class="manual_recovery",
        )


__all__ = [
    "AuthorizedExecutor",
    "ExecutionContext",
    "ExecutionKind",
    "ExecutionRegistry",
    "ExecutionRequest",
    "ExecutionService",
]
