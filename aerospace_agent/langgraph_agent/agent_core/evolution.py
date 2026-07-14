"""Evidence-bound staging and human approval for durable agent evolution.

Discovery may create candidates, but this module deliberately exposes no path
from a staged candidate to project memory or the workflow directory without a
single-use :class:`ConfirmationService` grant bound to the exact candidate
content and original checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from pydantic import Field

from .confirmation import ConfirmationService, compute_action_hash
from .models import CheckpointRef, ContractModel, SessionMemory


CandidateKind = Literal["project_memory", "workflow"]
CandidateStatus = Literal["staged", "active"]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EvolutionCandidate(ContractModel):
    candidate_id: str
    kind: CandidateKind
    status: CandidateStatus
    project_id: str
    thread_id: str
    root_run_id: str
    source_checkpoint: CheckpointRef
    source_memory_id: str | None = None
    workflow_id: str | None = None
    workflow_version: str | None = None
    payload: dict[str, Any]
    content_sha256: str
    target_path: str
    created_at: str
    activated_at: str | None = None
    confirmation_id: str | None = None


class EvolutionService:
    """Stage candidates in SQLite and activate them only after confirmation."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        database_path: str | Path,
        project_id: str,
        clock: Callable[[], datetime] = _utc_now,
        **_: Any,
    ) -> None:
        if not project_id:
            raise ValueError("project_id is required")
        self._workspace = Path(workspace_root).resolve()
        self._database_path = Path(database_path).resolve()
        self._project_id = project_id
        self._clock = clock
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def availability(self) -> dict[str, Any]:
        """Expose a stable capability-health contract to the runtime catalog."""

        return {"available": True, "service": "evolution_candidates"}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported evolution schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE evolution_candidates (
                        candidate_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        source_checkpoint_json TEXT NOT NULL,
                        source_memory_id TEXT,
                        workflow_id TEXT,
                        workflow_version TEXT,
                        payload_json TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        target_path TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        activated_at TEXT,
                        confirmation_id TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE evolution_candidate_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        candidate_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        FOREIGN KEY(candidate_id) REFERENCES evolution_candidates(candidate_id)
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def stage_session_promotion(
        self,
        memory: SessionMemory,
        *,
        root_run_id: str,
    ) -> EvolutionCandidate:
        memory = SessionMemory.model_validate(memory.model_dump(mode="python"))
        if memory.project_id != self._project_id:
            raise ValueError("session memory project namespace does not match evolution service")
        if memory.truth_status in {"superseded", "retracted"}:
            raise ValueError("inactive session memory cannot be promoted")
        checkpoint = memory.source_checkpoints[-1]
        payload = {
            "memory_id": memory.memory_id,
            "kind": memory.kind,
            "content": memory.content,
            "truth_status": memory.truth_status,
            "confidence": memory.confidence,
            "source_content_hash": memory.source_content_hash,
            "source_checkpoints": [
                item.model_dump(mode="json") for item in memory.source_checkpoints
            ],
        }
        candidate_id = uuid4().hex
        target_path = f"memory/project/decisions/promoted-{candidate_id}.md"
        return self._stage(
            candidate_id=candidate_id,
            kind="project_memory",
            thread_id=memory.thread_id,
            root_run_id=root_run_id,
            source_checkpoint=checkpoint,
            source_memory_id=memory.memory_id,
            workflow_id=None,
            workflow_version=None,
            payload=payload,
            target_path=target_path,
        )

    def stage_workflow_candidate(
        self,
        *,
        thread_id: str,
        root_run_id: str,
        workflow_id: str,
        version: str,
        workflow_body: Mapping[str, Any],
        manifest: Mapping[str, Any],
        source_checkpoint: CheckpointRef,
    ) -> EvolutionCandidate:
        if not _IDENTIFIER.fullmatch(workflow_id) or not _IDENTIFIER.fullmatch(version):
            raise ValueError("workflow ID and version must be safe identifiers")
        checkpoint = CheckpointRef.model_validate(source_checkpoint.model_dump(mode="python"))
        if checkpoint.project_id != self._project_id or checkpoint.thread_id != thread_id:
            raise ValueError("workflow candidate checkpoint namespace mismatch")
        payload = {
            "workflow_id": workflow_id,
            "version": version,
            "workflow_body": dict(workflow_body),
            "manifest": dict(manifest),
        }
        candidate_id = uuid4().hex
        target_path = f"workflows/evolved/{workflow_id}-{version}-{candidate_id}.yaml"
        return self._stage(
            candidate_id=candidate_id,
            kind="workflow",
            thread_id=thread_id,
            root_run_id=root_run_id,
            source_checkpoint=checkpoint,
            source_memory_id=None,
            workflow_id=workflow_id,
            workflow_version=version,
            payload=payload,
            target_path=target_path,
        )

    def _stage(
        self,
        *,
        candidate_id: str,
        kind: CandidateKind,
        thread_id: str,
        root_run_id: str,
        source_checkpoint: CheckpointRef,
        source_memory_id: str | None,
        workflow_id: str | None,
        workflow_version: str | None,
        payload: dict[str, Any],
        target_path: str,
    ) -> EvolutionCandidate:
        if not thread_id or not root_run_id:
            raise ValueError("thread_id and root_run_id are required")
        content_hash = _canonical_sha256(payload)
        created_at = self._as_utc(self._clock()).isoformat()
        candidate = EvolutionCandidate(
            candidate_id=candidate_id,
            kind=kind,
            status="staged",
            project_id=self._project_id,
            thread_id=thread_id,
            root_run_id=root_run_id,
            source_checkpoint=source_checkpoint,
            source_memory_id=source_memory_id,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            payload=payload,
            content_sha256=content_hash,
            target_path=target_path,
            created_at=created_at,
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO evolution_candidates(
                    candidate_id, kind, status, project_id, thread_id, root_run_id,
                    source_checkpoint_json, source_memory_id, workflow_id, workflow_version,
                    payload_json, content_sha256, target_path, created_at,
                    activated_at, confirmation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    candidate.candidate_id,
                    candidate.kind,
                    candidate.status,
                    candidate.project_id,
                    candidate.thread_id,
                    candidate.root_run_id,
                    self._json(candidate.source_checkpoint.model_dump(mode="json")),
                    candidate.source_memory_id,
                    candidate.workflow_id,
                    candidate.workflow_version,
                    self._json(candidate.payload),
                    candidate.content_sha256,
                    candidate.target_path,
                    candidate.created_at,
                ),
            )
            self._append_event(connection, candidate.candidate_id, "staged", created_at, {})
            connection.commit()
        return candidate

    def activation_action_hash(self, candidate_id: str) -> str:
        candidate = self.get(candidate_id)
        return compute_action_hash(
            tool_name="evolution.activate",
            arguments={
                "candidate_id": candidate.candidate_id,
                "kind": candidate.kind,
                "content_sha256": candidate.content_sha256,
            },
            target_paths=[candidate.target_path],
            run_id=candidate.root_run_id,
            risk_level="project_write",
        )

    def activate(
        self,
        candidate_id: str,
        *,
        confirmation_service: ConfirmationService,
        confirmation_id: str,
    ) -> EvolutionCandidate:
        if not isinstance(confirmation_service, ConfirmationService):
            raise TypeError("confirmation_service must be ConfirmationService")
        candidate = self.get(candidate_id)
        confirmation_service.consume(
            confirmation_id=confirmation_id,
            project_id=candidate.project_id,
            thread_id=candidate.thread_id,
            root_run_id=candidate.root_run_id,
            operation_id=f"activate:{candidate.candidate_id}",
            action_hash=self.activation_action_hash(candidate.candidate_id),
            continuation_checkpoint={
                **candidate.source_checkpoint.model_dump(mode="json"),
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
            },
        )
        if candidate.status != "staged":
            raise ValueError("candidate is already active")

        target = self._workspace_path(candidate.target_path)
        if target.exists():
            raise FileExistsError(f"candidate target already exists: {candidate.target_path}")
        rendered = self._render(candidate)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        activated_at = self._as_utc(self._clock()).isoformat()
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(rendered)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard-link publication is atomic and fails if another writer
            # created the target after our existence check.  Unlike replace,
            # it can never overwrite an approved project-memory/workflow file.
            os.link(temporary, target)
            temporary.unlink()
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    """
                    UPDATE evolution_candidates
                    SET status = 'active', activated_at = ?, confirmation_id = ?
                    WHERE candidate_id = ? AND status = 'staged'
                    """,
                    (activated_at, confirmation_id, candidate.candidate_id),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    raise ValueError("candidate is no longer staged")
                self._append_event(
                    connection,
                    candidate.candidate_id,
                    "approved",
                    activated_at,
                    {"confirmation_id": confirmation_id, "target_path": candidate.target_path},
                )
                connection.commit()
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return self.get(candidate.candidate_id)

    def get(self, candidate_id: str) -> EvolutionCandidate:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM evolution_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"evolution candidate not found: {candidate_id}")
        return self._from_row(row)

    def list_active(self) -> list[EvolutionCandidate]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM evolution_candidates
                WHERE project_id = ? AND status = 'active'
                ORDER BY activated_at, candidate_id
                """,
                (self._project_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def events(self, candidate_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT event_type, occurred_at, details_json
                FROM evolution_candidate_events
                WHERE candidate_id = ? ORDER BY event_id
                """,
                (candidate_id,),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "occurred_at": row["occurred_at"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        candidate_id: str,
        event_type: str,
        occurred_at: str,
        details: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO evolution_candidate_events(
                candidate_id, event_type, occurred_at, details_json
            ) VALUES (?, ?, ?, ?)
            """,
            (candidate_id, event_type, occurred_at, EvolutionService._json(details)),
        )

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("evolution clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _workspace_path(self, relative_path: str) -> Path:
        raw = Path(relative_path)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError("candidate target path is outside workspace")
        resolved = (self._workspace / raw).resolve()
        try:
            resolved.relative_to(self._workspace)
        except ValueError as exc:
            raise ValueError("candidate target path is outside workspace") from exc
        return resolved

    @staticmethod
    def _render(candidate: EvolutionCandidate) -> str:
        if candidate.kind == "workflow":
            return yaml.safe_dump(candidate.payload, allow_unicode=True, sort_keys=True)
        metadata = {
            "schema_version": "1.0",
            "candidate_id": candidate.candidate_id,
            "source_memory_id": candidate.source_memory_id,
            "project_id": candidate.project_id,
            "thread_id": candidate.thread_id,
            "root_run_id": candidate.root_run_id,
            "source_checkpoint": candidate.source_checkpoint.model_dump(mode="json"),
            "truth_status": candidate.payload["truth_status"],
            "confidence": candidate.payload["confidence"],
            "source_content_hash": candidate.payload["source_content_hash"],
        }
        frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=True).rstrip()
        return f"---\n{frontmatter}\n---\n\n{candidate.payload['content']}\n"

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EvolutionCandidate:
        return EvolutionCandidate(
            candidate_id=row["candidate_id"],
            kind=row["kind"],
            status=row["status"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            root_run_id=row["root_run_id"],
            source_checkpoint=CheckpointRef.model_validate(
                json.loads(row["source_checkpoint_json"])
            ),
            source_memory_id=row["source_memory_id"],
            workflow_id=row["workflow_id"],
            workflow_version=row["workflow_version"],
            payload=json.loads(row["payload_json"]),
            content_sha256=row["content_sha256"],
            target_path=row["target_path"],
            created_at=row["created_at"],
            activated_at=row["activated_at"],
            confirmation_id=row["confirmation_id"],
        )


__all__ = ["EvolutionCandidate", "EvolutionService"]
