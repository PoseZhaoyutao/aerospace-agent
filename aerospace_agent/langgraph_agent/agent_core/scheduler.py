"""Internal, SQLite-backed scheduler for reminders and approved workflows.

This module never registers an operating-system job.  Payloads are immutable,
content-addressed JSON, while jobs contain only references and locked workflow
approval snapshots.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue

from .models import ContractModel, WorkflowManifest
from .rag_gate import ExecutionRunStore
from .workflows import WorkflowRegistry


JobStatus = Literal[
    "scheduled",
    "claimed",
    "running",
    "succeeded",
    "overdue",
    "retry_wait",
    "failed",
    "blocked",
    "cancel_requested",
    "cancelled",
]


class ScheduledPayload(ContractModel):
    payload_id: str
    schema_version: Literal["1.0"] = "1.0"
    body: dict[str, JsonValue]
    payload_sha256: str
    created_at: str
    masked_body: dict[str, JsonValue]


class ScheduledJob(ContractModel):
    job_id: str
    payload_id: str
    payload_sha256: str
    project_id: str
    thread_id: str | None
    created_at: str
    due_at: str
    timezone: str
    available_at: str
    workflow_id: str | None = None
    workflow_version: str | None = None
    workflow_sha256: str | None = None
    manifest_sha256: str | None = None
    approval_record_id: str | None = None
    risk_snapshot: dict[str, JsonValue] = Field(default_factory=dict)
    status: JobStatus
    attempt: int = Field(ge=0)
    max_retries: int = Field(ge=0)
    retry_delay_seconds: int = Field(ge=1)
    lease_holder: str | None = None
    lease_expires_at: str | None = None
    version: int = Field(ge=1)
    last_checkpoint_id: str | None = None
    job_run_id: str | None = None
    recovery_class: Literal["read_only", "manual_recovery"] = "read_only"


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
        raise ValueError(f"scheduled payload is not canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware_datetime(value: datetime | str, *, field_name: str) -> datetime:
    if isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid timezone-aware timestamp") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError(f"{field_name} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _timezone_name(value: datetime) -> str:
    return str(getattr(value.tzinfo, "key", None) or value.tzinfo)


class SchedulerService:
    """Versioned internal queue with atomic leases and optimistic cancellation."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        database_path: str | Path,
        *,
        workflow_registry: WorkflowRegistry,
        execution_run_store: ExecutionRunStore,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(workflow_registry, WorkflowRegistry):
            raise TypeError("workflow_registry must be WorkflowRegistry")
        if not isinstance(execution_run_store, ExecutionRunStore):
            raise TypeError("execution_run_store must be ExecutionRunStore")
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._workflow_registry = workflow_registry
        self._execution_run_store = execution_run_store
        self._clock = clock
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _now(self) -> datetime:
        value = self._clock()
        return _aware_datetime(value, field_name="scheduler clock").astimezone(UTC)

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported scheduler schema version: {version}")
            if version != 0:
                return
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE scheduled_job_payloads (
                    payload_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    body_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    masked_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE scheduled_jobs (
                    job_id TEXT PRIMARY KEY,
                    payload_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    thread_id TEXT,
                    created_at TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    workflow_id TEXT,
                    workflow_version TEXT,
                    workflow_sha256 TEXT,
                    manifest_sha256 TEXT,
                    approval_record_id TEXT,
                    risk_snapshot_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    max_retries INTEGER NOT NULL,
                    retry_delay_seconds INTEGER NOT NULL,
                    lease_holder TEXT,
                    lease_expires_at TEXT,
                    version INTEGER NOT NULL,
                    last_checkpoint_id TEXT,
                    job_run_id TEXT,
                    recovery_class TEXT NOT NULL,
                    FOREIGN KEY(payload_id) REFERENCES scheduled_job_payloads(payload_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX scheduled_jobs_due_idx "
                "ON scheduled_jobs(status, available_at, due_at)"
            )
            connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
            connection.commit()

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def create_reminder(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        due_at: datetime | str,
        message: str,
    ) -> ScheduledJob:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("reminder message must be a non-empty string")
        body = {"kind": "reminder", "message": message}
        return self._create_job(
            project_id=project_id,
            thread_id=thread_id,
            due_at=due_at,
            body=body,
            masked_body=body,
            workflow_manifest=None,
            risk_snapshot={"kind": "reminder", "read_only": True, "idempotent": True},
            max_retries=0,
            retry_delay_seconds=1,
        )

    def create_workflow(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        due_at: datetime | str,
        workflow_id: str,
        workflow_version: str,
        inputs: Mapping[str, Any],
        max_retries: int = 0,
        retry_delay_seconds: int = 30,
    ) -> ScheduledJob:
        if not isinstance(max_retries, int) or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if not isinstance(retry_delay_seconds, int) or not 1 <= retry_delay_seconds <= 3600:
            raise ValueError("retry_delay_seconds must be between 1 and 3600")
        locked = self._workflow_registry.get(workflow_id, workflow_version)
        if not self._workflow_registry.is_scheduled_read_only(workflow_id, workflow_version):
            raise ValueError("scheduled workflow must be automatable and fully read-only/idempotent")
        checked_inputs = self._workflow_registry.validate_inputs(
            workflow_id, workflow_version, inputs
        )
        masked_inputs = self._workflow_registry.mask_inputs(
            workflow_id, workflow_version, checked_inputs
        )
        body = {
            "kind": "workflow",
            "workflow_id": workflow_id,
            "workflow_version": workflow_version,
            "inputs": checked_inputs,
            "input_hash": _sha256(checked_inputs),
        }
        masked_body = {**body, "inputs": masked_inputs}
        return self._create_job(
            project_id=project_id,
            thread_id=thread_id,
            due_at=due_at,
            body=body,
            masked_body=masked_body,
            workflow_manifest=locked.manifest,
            risk_snapshot=self._workflow_risk_snapshot(locked.manifest),
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

    def _create_job(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        due_at: datetime | str,
        body: dict[str, JsonValue],
        masked_body: dict[str, JsonValue],
        workflow_manifest: WorkflowManifest | None,
        risk_snapshot: dict[str, JsonValue],
        max_retries: int,
        retry_delay_seconds: int,
    ) -> ScheduledJob:
        parsed_due = _aware_datetime(due_at, field_name="due_at")
        due_utc = parsed_due.astimezone(UTC).isoformat()
        created_at = self._now().isoformat()
        digest = _sha256(body)
        payload_id = digest
        body_json = _canonical_bytes(body).decode("utf-8")
        masked_json = _canonical_bytes(masked_body).decode("utf-8")
        job_id = str(uuid.uuid4())
        manifest = workflow_manifest
        values = (
            job_id,
            payload_id,
            digest,
            project_id,
            thread_id,
            created_at,
            due_utc,
            _timezone_name(parsed_due),
            due_utc,
            manifest.workflow_id if manifest else None,
            manifest.version if manifest else None,
            manifest.workflow_sha256 if manifest else None,
            manifest.manifest_sha256 if manifest else None,
            manifest.approval_record_id if manifest else None,
            _canonical_bytes(risk_snapshot).decode("utf-8"),
            "scheduled",
            0,
            max_retries,
            retry_delay_seconds,
            1,
            "read_only",
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT body_json, payload_sha256 FROM scheduled_job_payloads WHERE payload_id = ?",
                (payload_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO scheduled_job_payloads(
                        payload_id, schema_version, body_json, payload_sha256,
                        created_at, masked_json
                    ) VALUES (?, '1.0', ?, ?, ?, ?)
                    """,
                    (payload_id, body_json, digest, created_at, masked_json),
                )
            elif existing["body_json"] != body_json or existing["payload_sha256"] != digest:
                connection.rollback()
                raise RuntimeError("content-addressed scheduled payload collision")
            connection.execute(
                """
                INSERT INTO scheduled_jobs(
                    job_id, payload_id, payload_sha256, project_id, thread_id,
                    created_at, due_at, timezone, available_at,
                    workflow_id, workflow_version, workflow_sha256,
                    manifest_sha256, approval_record_id, risk_snapshot_json,
                    status, attempt, max_retries, retry_delay_seconds,
                    version, recovery_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            connection.commit()
        return self.get_job(job_id)

    def get_payload(self, payload_id: str) -> ScheduledPayload:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_job_payloads WHERE payload_id = ?", (payload_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"scheduled payload not found: {payload_id}")
        return self._payload_from_row(row)

    def payload_count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM scheduled_job_payloads").fetchone()[0])

    def get_job(self, job_id: str) -> ScheduledJob:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM scheduled_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"scheduled job not found: {job_id}")
        return self._job_from_row(row)

    def list_jobs(self, *, project_id: str, thread_id: str | None = None) -> list[ScheduledJob]:
        with closing(self._connect()) as connection:
            if thread_id is None:
                rows = connection.execute(
                    "SELECT * FROM scheduled_jobs WHERE project_id = ? ORDER BY created_at, job_id",
                    (project_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM scheduled_jobs WHERE project_id = ? AND thread_id = ? "
                    "ORDER BY created_at, job_id",
                    (project_id, thread_id),
                ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def recover_overdue(self) -> int:
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                """
                UPDATE scheduled_jobs
                SET status = 'overdue', version = version + 1
                WHERE status = 'scheduled' AND due_at < ?
                """,
                (now,),
            ).rowcount
            connection.commit()
        return int(count)

    def claim_due(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 30,
    ) -> ScheduledJob | None:
        if not worker_id:
            raise ValueError("worker_id cannot be empty")
        if not 1 <= lease_seconds <= 300:
            raise ValueError("scheduler lease must be between 1 and 300 seconds")
        self.recover_overdue()
        self._recover_expired_leases()
        now = self._now()
        now_text = now.isoformat()
        with closing(self._connect()) as connection:
            candidates = connection.execute(
                """
                SELECT job_id, version FROM scheduled_jobs
                WHERE status IN ('scheduled', 'overdue', 'retry_wait')
                  AND available_at <= ?
                ORDER BY available_at, created_at, job_id
                """,
                (now_text,),
            ).fetchall()
        for candidate in candidates:
            job = self.get_job(candidate["job_id"])
            try:
                self._revalidate_job(job)
            except (KeyError, PermissionError, RuntimeError, ValueError, TypeError):
                self._block(job.job_id, expected_version=int(candidate["version"]))
                continue
            next_attempt = job.attempt + 1
            job_run_id = f"job:{job.job_id}:{next_attempt}"
            lease = (now + timedelta(seconds=lease_seconds)).isoformat()
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    """
                    UPDATE scheduled_jobs SET
                        status = 'claimed', attempt = ?, lease_holder = ?,
                        lease_expires_at = ?, job_run_id = ?, version = version + 1
                    WHERE job_id = ? AND version = ?
                      AND status IN ('scheduled', 'overdue', 'retry_wait')
                      AND available_at <= ?
                    """,
                    (
                        next_attempt,
                        worker_id,
                        lease,
                        job_run_id,
                        job.job_id,
                        int(candidate["version"]),
                        now_text,
                    ),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    continue
                connection.commit()
            try:
                self._ensure_scheduled_run(job_run_id, project_id=job.project_id)
            except (KeyError, ValueError, sqlite3.Error):
                claimed = self.get_job(job.job_id)
                self._block(claimed.job_id, expected_version=claimed.version)
                continue
            return self.get_job(job.job_id)
        return None

    def _ensure_scheduled_run(self, root_run_id: str, *, project_id: str) -> None:
        try:
            self._execution_run_store.create_scheduled_run(
                root_run_id=root_run_id, project_id=project_id
            )
        except sqlite3.IntegrityError:
            existing = self._execution_run_store.get(root_run_id)
            if existing.kind != "scheduled" or existing.project_id != project_id:
                raise ValueError("job run identity collision")

    def mark_running(
        self,
        job_id: str,
        *,
        expected_version: int,
        worker_id: str,
    ) -> ScheduledJob:
        return self._worker_transition(
            job_id,
            expected_version=expected_version,
            worker_id=worker_id,
            from_status="claimed",
            to_status="running",
        )

    def mark_succeeded(
        self,
        job_id: str,
        *,
        expected_version: int,
        worker_id: str,
        last_checkpoint_id: str | None = None,
    ) -> ScheduledJob:
        return self._worker_transition(
            job_id,
            expected_version=expected_version,
            worker_id=worker_id,
            from_status="running",
            to_status="succeeded",
            last_checkpoint_id=last_checkpoint_id,
            clear_lease=True,
        )

    def mark_failed(
        self,
        job_id: str,
        *,
        expected_version: int,
        worker_id: str,
        retryable: bool,
    ) -> ScheduledJob:
        job = self.get_job(job_id)
        if job.version != expected_version or job.status != "running" or job.lease_holder != worker_id:
            raise RuntimeError("job cannot be failed by this worker/version")
        payload = self.get_payload(job.payload_id)
        can_retry = (
            retryable
            and payload.body.get("kind") == "workflow"
            and job.attempt <= job.max_retries
        )
        if can_retry:
            try:
                self._revalidate_job(job)
            except (KeyError, PermissionError, RuntimeError, ValueError, TypeError):
                can_retry = False
        next_status = "retry_wait" if can_retry else "failed"
        available = (
            self._now() + timedelta(seconds=job.retry_delay_seconds)
            if can_retry
            else _aware_datetime(job.available_at, field_name="available_at")
        ).astimezone(UTC).isoformat()
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE scheduled_jobs SET status = ?, available_at = ?,
                    lease_holder = NULL, lease_expires_at = NULL, version = version + 1
                WHERE job_id = ? AND version = ? AND status = 'running'
                  AND lease_holder = ? AND lease_expires_at > ?
                """,
                (next_status, available, job_id, expected_version, worker_id, now),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("job failure transition lost optimistic race")
            connection.commit()
        return self.get_job(job_id)

    def cancel(self, job_id: str, *, expected_version: int) -> ScheduledJob | None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM scheduled_jobs WHERE job_id = ? AND version = ?",
                (job_id, expected_version),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            if row["status"] in {"scheduled", "overdue", "retry_wait"}:
                target = "cancelled"
            elif row["status"] in {"claimed", "running"}:
                target = "cancel_requested"
            else:
                connection.rollback()
                return None
            connection.execute(
                "UPDATE scheduled_jobs SET status = ?, version = version + 1 "
                "WHERE job_id = ? AND version = ?",
                (target, job_id, expected_version),
            )
            connection.commit()
        return self.get_job(job_id)

    def honor_cancel(
        self,
        job_id: str,
        *,
        expected_version: int,
        worker_id: str,
        interruptible: bool,
    ) -> ScheduledJob:
        target = "cancelled" if interruptible else "failed"
        recovery = "read_only" if interruptible else "manual_recovery"
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE scheduled_jobs SET status = ?, recovery_class = ?,
                    lease_holder = NULL, lease_expires_at = NULL, version = version + 1
                WHERE job_id = ? AND version = ? AND status = 'cancel_requested'
                  AND lease_holder = ? AND lease_expires_at > ?
                """,
                (target, recovery, job_id, expected_version, worker_id, now),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("cancel safe point does not match worker/version")
            connection.commit()
        return self.get_job(job_id)

    def _worker_transition(
        self,
        job_id: str,
        *,
        expected_version: int,
        worker_id: str,
        from_status: str,
        to_status: str,
        last_checkpoint_id: str | None = None,
        clear_lease: bool = False,
    ) -> ScheduledJob:
        lease_clause = ", lease_holder = NULL, lease_expires_at = NULL" if clear_lease else ""
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                f"""
                UPDATE scheduled_jobs SET status = ?, last_checkpoint_id = ?,
                    version = version + 1 {lease_clause}
                WHERE job_id = ? AND version = ? AND status = ? AND lease_holder = ?
                  AND lease_expires_at > ?
                """,
                (
                    to_status,
                    last_checkpoint_id,
                    job_id,
                    expected_version,
                    from_status,
                    worker_id,
                    now,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("job transition does not match worker/version/state")
            connection.commit()
        return self.get_job(job_id)

    def _block(self, job_id: str, *, expected_version: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE scheduled_jobs SET status = 'blocked', lease_holder = NULL,
                    lease_expires_at = NULL, version = version + 1
                WHERE job_id = ? AND version = ?
                  AND status IN ('scheduled', 'overdue', 'retry_wait', 'claimed')
                """,
                (job_id, expected_version),
            )
            connection.commit()

    def _recover_expired_leases(self) -> None:
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE scheduled_jobs SET status = 'overdue', lease_holder = NULL,
                    lease_expires_at = NULL, available_at = ?, version = version + 1
                WHERE status = 'claimed' AND lease_expires_at < ?
                """,
                (now, now),
            )
            connection.execute(
                """
                UPDATE scheduled_jobs SET status = 'failed', recovery_class = 'manual_recovery',
                    lease_holder = NULL, lease_expires_at = NULL, version = version + 1
                WHERE status IN ('running', 'cancel_requested') AND lease_expires_at < ?
                """,
                (now,),
            )
            connection.commit()

    def _revalidate_job(self, job: ScheduledJob) -> None:
        payload = self.get_payload(job.payload_id)
        body = dict(payload.body)
        digest = _sha256(body)
        if (
            payload.schema_version != "1.0"
            or digest != payload.payload_sha256
            or digest != payload.payload_id
            or digest != job.payload_sha256
        ):
            raise ValueError("scheduled payload hash or schema version mismatch")
        kind = body.get("kind")
        if kind == "reminder":
            if set(body) != {"kind", "message"} or not isinstance(body.get("message"), str) or not body[
                "message"
            ].strip():
                raise ValueError("invalid reminder payload schema")
            expected_risk = {"kind": "reminder", "read_only": True, "idempotent": True}
            if job.workflow_id is not None or job.risk_snapshot != expected_risk:
                raise ValueError("reminder workflow/risk snapshot mismatch")
            return
        if kind != "workflow" or set(body) != {
            "kind",
            "workflow_id",
            "workflow_version",
            "inputs",
            "input_hash",
        }:
            raise ValueError("invalid workflow payload schema")
        if not isinstance(body.get("inputs"), dict) or body["input_hash"] != _sha256(body["inputs"]):
            raise ValueError("scheduled workflow input hash mismatch")
        if body["workflow_id"] != job.workflow_id or body["workflow_version"] != job.workflow_version:
            raise ValueError("scheduled workflow payload identity mismatch")
        locked = self._workflow_registry.get(job.workflow_id or "", job.workflow_version or "")
        manifest = locked.manifest
        if (
            manifest.workflow_sha256 != job.workflow_sha256
            or manifest.manifest_sha256 != job.manifest_sha256
            or manifest.approval_record_id != job.approval_record_id
        ):
            raise ValueError("scheduled workflow immutable snapshot mismatch")
        if not self._workflow_registry.is_scheduled_read_only(manifest.workflow_id, manifest.version):
            raise ValueError("scheduled workflow is no longer automatable/read-only")
        self._workflow_registry.validate_inputs(manifest.workflow_id, manifest.version, body["inputs"])
        if job.risk_snapshot != self._workflow_risk_snapshot(manifest):
            raise ValueError("scheduled workflow step policy snapshot mismatch")

    @staticmethod
    def _workflow_risk_snapshot(manifest: WorkflowManifest) -> dict[str, JsonValue]:
        return {
            "kind": "workflow",
            "approval_scope": manifest.approval_scope,
            "automatable": manifest.automatable,
            "steps": [
                {
                    "step_id": step.step_id,
                    "executor_type": step.executor_type,
                    "risk_level": step.risk_level,
                    "recovery_class": step.recovery_class,
                    "idempotent": step.idempotent,
                }
                for step in manifest.steps
            ],
        }

    @staticmethod
    def _payload_from_row(row: sqlite3.Row) -> ScheduledPayload:
        try:
            body = json.loads(row["body_json"])
            masked = json.loads(row["masked_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("stored scheduled payload is invalid JSON") from exc
        return ScheduledPayload(
            payload_id=row["payload_id"],
            schema_version=row["schema_version"],
            body=body,
            payload_sha256=row["payload_sha256"],
            created_at=row["created_at"],
            masked_body=masked,
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> ScheduledJob:
        try:
            risk_snapshot = json.loads(row["risk_snapshot_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("stored scheduled job risk snapshot is invalid JSON") from exc
        return ScheduledJob(
            job_id=row["job_id"],
            payload_id=row["payload_id"],
            payload_sha256=row["payload_sha256"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            created_at=row["created_at"],
            due_at=row["due_at"],
            timezone=row["timezone"],
            available_at=row["available_at"],
            workflow_id=row["workflow_id"],
            workflow_version=row["workflow_version"],
            workflow_sha256=row["workflow_sha256"],
            manifest_sha256=row["manifest_sha256"],
            approval_record_id=row["approval_record_id"],
            risk_snapshot=risk_snapshot,
            status=row["status"],
            attempt=int(row["attempt"]),
            max_retries=int(row["max_retries"]),
            retry_delay_seconds=int(row["retry_delay_seconds"]),
            lease_holder=row["lease_holder"],
            lease_expires_at=row["lease_expires_at"],
            version=int(row["version"]),
            last_checkpoint_id=row["last_checkpoint_id"],
            job_run_id=row["job_run_id"],
            recovery_class=row["recovery_class"],
        )


__all__ = ["ScheduledJob", "ScheduledPayload", "SchedulerService"]

