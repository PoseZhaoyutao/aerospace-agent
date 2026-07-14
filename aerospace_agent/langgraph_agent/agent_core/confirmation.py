"""Persistent, single-use confirmation grants for protected operations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .models import ConfirmationGrant


ConfirmationErrorCode = Literal[
    "confirmation_required",
    "confirmation_expired",
    "confirmation_replayed",
]


class ConfirmationError(RuntimeError):
    """Typed policy error that callers convert to a structured ToolResult."""

    def __init__(self, code: ConfirmationErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("confirmation clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def compute_action_hash(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    target_paths: Sequence[str],
    run_id: str,
    risk_level: str,
) -> str:
    """Hash the canonical operation identity protected by a confirmation."""

    canonical = {
        "arguments": dict(arguments),
        "risk_level": risk_level,
        "run_id": run_id,
        "target_paths": sorted(set(target_paths)),
        "tool_name": tool_name,
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ConfirmationService:
    """Issue and atomically consume grants from a versioned SQLite store."""

    _SCHEMA_VERSION = 2

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported confirmation schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE confirmation_grants (
                        confirmation_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT,
                        root_run_id TEXT NOT NULL,
                        operation_id TEXT NOT NULL,
                        action_hash TEXT NOT NULL,
                        issued_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        used_at TEXT,
                        continuation_checkpoint_json TEXT
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX confirmation_operation_idx "
                    "ON confirmation_grants(project_id, operation_id)"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()
            elif version == 1:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "ALTER TABLE confirmation_grants ADD COLUMN continuation_checkpoint_json TEXT"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def issue(
        self,
        *,
        project_id: str,
        thread_id: str | None,
        root_run_id: str,
        operation_id: str,
        action_hash: str,
        ttl_seconds: int = 600,
    ) -> ConfirmationGrant:
        if not 1 <= ttl_seconds <= 600:
            raise ValueError("confirmation ttl_seconds must be between 1 and 600")
        issued_at = _as_utc(self._clock())
        expires_at = issued_at + timedelta(seconds=ttl_seconds)
        grant = ConfirmationGrant(
            confirmation_id=uuid4().hex,
            project_id=project_id,
            thread_id=thread_id,
            root_run_id=root_run_id,
            operation_id=operation_id,
            action_hash=action_hash,
            issued_at=issued_at.isoformat(),
            expires_at=expires_at.isoformat(),
        )
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO confirmation_grants(
                    confirmation_id, project_id, thread_id, root_run_id,
                    operation_id, action_hash, issued_at, expires_at, used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    grant.confirmation_id,
                    grant.project_id,
                    grant.thread_id,
                    grant.root_run_id,
                    grant.operation_id,
                    grant.action_hash,
                    grant.issued_at,
                    grant.expires_at,
                ),
            )
            connection.commit()
        return grant

    def consume(
        self,
        *,
        confirmation_id: str,
        project_id: str,
        thread_id: str | None,
        root_run_id: str,
        operation_id: str,
        action_hash: str,
        continuation_checkpoint: Mapping[str, Any],
    ) -> ConfirmationGrant:
        now = _as_utc(self._clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM confirmation_grants WHERE confirmation_id = ?",
                (confirmation_id,),
            ).fetchone()
            if row is None:
                raise ConfirmationError("confirmation_required", "confirmation grant not found")
            grant = self._from_row(row)
            if grant.used_at is not None:
                raise ConfirmationError("confirmation_replayed", "confirmation grant was already used")
            if now >= datetime.fromisoformat(grant.expires_at).astimezone(UTC):
                raise ConfirmationError("confirmation_expired", "confirmation grant has expired")
            expected = (
                grant.project_id,
                grant.thread_id,
                grant.root_run_id,
                grant.operation_id,
                grant.action_hash,
            )
            supplied = (project_id, thread_id, root_run_id, operation_id, action_hash)
            if supplied != expected:
                raise ConfirmationError(
                    "confirmation_required",
                    "confirmation grant does not match operation context or action hash",
                )
            checkpoint = dict(continuation_checkpoint)
            identity = {
                "project_id": project_id,
                "thread_id": thread_id,
                "root_run_id": root_run_id,
                "operation_id": operation_id,
            }
            if any(key in checkpoint and checkpoint[key] != value for key, value in identity.items()):
                raise ConfirmationError(
                    "confirmation_required",
                    "continuation checkpoint identity does not match confirmation context",
                )
            checkpoint.update(identity)
            used_at = now.isoformat()
            updated = connection.execute(
                """
                UPDATE confirmation_grants
                SET used_at = ?, continuation_checkpoint_json = ?
                WHERE confirmation_id = ? AND used_at IS NULL
                """,
                (
                    used_at,
                    json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    confirmation_id,
                ),
            )
            if updated.rowcount != 1:
                raise ConfirmationError("confirmation_replayed", "confirmation grant was already used")
            connection.commit()
        return grant.model_copy(update={"used_at": used_at})

    def continuation_checkpoint(self, confirmation_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT continuation_checkpoint_json FROM confirmation_grants
                WHERE confirmation_id = ?
                """,
                (confirmation_id,),
            ).fetchone()
        if row is None or row["continuation_checkpoint_json"] is None:
            return None
        loaded = json.loads(row["continuation_checkpoint_json"])
        if not isinstance(loaded, dict):
            raise RuntimeError("invalid stored continuation checkpoint")
        return loaded

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ConfirmationGrant:
        return ConfirmationGrant(
            confirmation_id=row["confirmation_id"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            root_run_id=row["root_run_id"],
            operation_id=row["operation_id"],
            action_hash=row["action_hash"],
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            used_at=row["used_at"],
        )


__all__ = [
    "ConfirmationError",
    "ConfirmationErrorCode",
    "ConfirmationService",
    "compute_action_hash",
]
