"""Durable session memory with strict project/thread namespace isolation."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from .models import CheckpointRef, ContractModel, SessionMemory


MemoryKind = Literal["fact", "preference", "decision", "constraint", "task_state", "artifact", "open_item"]
TruthStatus = Literal["user_stated", "verified", "assumption", "superseded", "retracted"]
CheckpointValidator = Callable[[CheckpointRef, str], bool]
_ACTIVE_STATUSES = ("user_stated", "verified", "assumption")
_SEARCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "about",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "how",
        "is",
        "it",
        "me",
        "of",
        "on",
        "please",
        "recall",
        "remember",
        "tell",
        "that",
        "the",
        "to",
        "was",
        "what",
        "were",
        "which",
        "with",
        "you",
    }
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SessionSummary(ContractModel):
    project_id: str
    thread_id: str
    revision: int = Field(ge=1)
    current_goal: str
    preferences: list[str] = Field(default_factory=list)
    confirmed_constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    completed_items: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    source_checkpoints: list[CheckpointRef] = Field(min_length=1)
    created_at: str

    @model_validator(mode="after")
    def validate_namespace(self) -> Self:
        if any(
            item.project_id != self.project_id or item.thread_id != self.thread_id
            for item in self.source_checkpoints
        ):
            raise ValueError("summary checkpoint namespace mismatch")
        return self


class SessionMemoryService:
    """The sole SQL boundary for one fixed ``(project_id, thread_id)``."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        project_id: str,
        thread_id: str,
        checkpoint_validator: CheckpointValidator,
    ) -> None:
        if not project_id or not thread_id:
            raise ValueError("project_id and thread_id are required")
        self._database_path = Path(database_path)
        self._project_id = project_id
        self._thread_id = thread_id
        self._checkpoint_validator = checkpoint_validator
        if not self._database_path.is_file():
            raise RuntimeError("project_not_initialized: session memory database is missing")
        with closing(self._connect(read_only=True)) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != 1:
            raise RuntimeError("project_memory_migration_failed: session memory schema mismatch")

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            connection = sqlite3.connect(
                f"file:{self._database_path.as_posix()}?mode=ro",
                uri=True,
                timeout=30.0,
            )
        else:
            connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def remember(
        self,
        *,
        kind: MemoryKind,
        content: str,
        source_checkpoints: Sequence[CheckpointRef],
        source_content_hash: str,
        truth_status: Literal["user_stated", "verified", "assumption"],
        confidence: float,
    ) -> SessionMemory:
        self._validate_checkpoints(source_checkpoints, source_content_hash)
        now = _now_iso()
        memory = SessionMemory(
            memory_id=uuid4().hex,
            project_id=self._project_id,
            thread_id=self._thread_id,
            kind=kind,
            content=content,
            source_checkpoints=list(source_checkpoints),
            source_content_hash=source_content_hash,
            truth_status=truth_status,
            confidence=confidence,
            created_at=now,
            updated_at=now,
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_memory(connection, memory)
            self._touch_thread(connection, now)
            connection.commit()
        return memory

    def search(self, query: str, *, include_history: bool = False, limit: int = 20) -> list[SessionMemory]:
        if not 1 <= limit <= 100:
            raise ValueError("memory search limit must be between 1 and 100")
        tokens = [
            token
            for token in re.findall(r"\w+", query, flags=re.UNICODE)
            if token.casefold() not in _SEARCH_STOPWORDS
        ]
        if not tokens:
            return []
        expression = " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        status_clause, status_parameters = self._status_clause(include_history)
        sql = f"""
            SELECT m.*
            FROM session_memories_fts AS f
            JOIN session_memories AS m ON m.memory_id = f.memory_id
            WHERE session_memories_fts MATCH ?
              AND m.project_id = ?
              AND m.thread_id = ?
              AND {status_clause}
            ORDER BY m.updated_at DESC, m.memory_id DESC
            LIMIT ?
        """
        parameters = [expression, self._project_id, self._thread_id, *status_parameters, limit]
        with closing(self._connect(read_only=True)) as connection:
            rows = connection.execute(sql, parameters).fetchall()
            # SQLite's default FTS tokenizer does not split or index Chinese
            # substrings reliably.  Keep FTS as the fast path, then fall back
            # to a namespace-scoped substring scan whenever it returns no
            # result or the query contains CJK text.  This preserves the
            # project/thread isolation contract while making short Chinese
            # queries usable across restarts.
            if not rows or any("\u4e00" <= char <= "\u9fff" for char in query):
                fallback_sql = f"""
                    SELECT * FROM session_memories
                    WHERE project_id = ? AND thread_id = ? AND {status_clause}
                    ORDER BY updated_at DESC, memory_id DESC
                """
                fallback_parameters = [self._project_id, self._thread_id, *status_parameters]
                candidates = connection.execute(fallback_sql, fallback_parameters).fetchall()
                folded_tokens = [token.casefold() for token in tokens]
                cjk_terms: list[str] = []
                for run in re.findall(r"[\u4e00-\u9fff]{2,}", query):
                    # A natural Chinese request often contains an instruction
                    # prefix (for example, "请回忆") around the actual memory
                    # key.  Match overlapping terms so that "请回忆干质量"
                    # can retrieve a memory containing only "干质量".
                    for size in range(min(len(run), 6), 1, -1):
                        cjk_terms.extend(
                            run[index : index + size]
                            for index in range(0, len(run) - size + 1)
                        )
                non_cjk_tokens = [
                    token for token in folded_tokens if not any("\u4e00" <= char <= "\u9fff" for char in token)
                ]
                seen = {str(row["memory_id"]) for row in rows}
                for candidate in candidates:
                    content = str(candidate["content"]).casefold()
                    cjk_match = not cjk_terms or any(term in content for term in cjk_terms)
                    if (
                        cjk_match
                        and all(token in content for token in non_cjk_tokens)
                        and str(candidate["memory_id"]) not in seen
                    ):
                        rows.append(candidate)
                rows = rows[:limit]
        return [self._memory_from_row(row) for row in rows]

    def list(
        self,
        *,
        include_history: bool = False,
        limit: int = 100,
        kind: MemoryKind | None = None,
    ) -> list[SessionMemory]:
        if not 1 <= limit <= 500:
            raise ValueError("memory list limit must be between 1 and 500")
        status_clause, status_parameters = self._status_clause(include_history)
        kind_clause = ""
        kind_parameters: list[str] = []
        if kind is not None:
            kind_clause = "AND kind = ?"
            kind_parameters.append(kind)
        sql = f"""
            SELECT * FROM session_memories
            WHERE project_id = ? AND thread_id = ? AND {status_clause} {kind_clause}
            ORDER BY updated_at DESC, memory_id DESC
            LIMIT ?
        """
        parameters = [
            self._project_id,
            self._thread_id,
            *status_parameters,
            *kind_parameters,
            limit,
        ]
        with closing(self._connect(read_only=True)) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def update(
        self,
        memory_id: str,
        *,
        content: str,
        source_checkpoints: Sequence[CheckpointRef],
        source_content_hash: str,
        truth_status: Literal["user_stated", "verified", "assumption"],
        confidence: float,
    ) -> SessionMemory:
        self._validate_checkpoints(source_checkpoints, source_content_hash)
        now = _now_iso()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM session_memories
                WHERE memory_id = ? AND project_id = ? AND thread_id = ?
                """,
                (memory_id, self._project_id, self._thread_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"session memory not found: {memory_id}")
            old = self._memory_from_row(row)
            if old.truth_status not in _ACTIVE_STATUSES:
                raise ValueError("only an active session memory can be updated")
            new = SessionMemory(
                memory_id=uuid4().hex,
                project_id=self._project_id,
                thread_id=self._thread_id,
                kind=old.kind,
                content=content,
                source_checkpoints=list(source_checkpoints),
                source_content_hash=source_content_hash,
                truth_status=truth_status,
                confidence=confidence,
                supersedes=old.memory_id,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                UPDATE session_memories SET truth_status = 'superseded', updated_at = ?
                WHERE memory_id = ? AND project_id = ? AND thread_id = ?
                """,
                (now, old.memory_id, self._project_id, self._thread_id),
            )
            self._insert_memory(connection, new)
            self._touch_thread(connection, now)
            connection.commit()
        return new

    def forget(self, memory_id: str) -> SessionMemory:
        now = _now_iso()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE session_memories SET truth_status = 'retracted', updated_at = ?
                WHERE memory_id = ? AND project_id = ? AND thread_id = ?
                  AND truth_status IN ('user_stated', 'verified', 'assumption')
                """,
                (now, memory_id, self._project_id, self._thread_id),
            )
            if updated.rowcount != 1:
                raise KeyError(f"active session memory not found: {memory_id}")
            row = connection.execute(
                "SELECT * FROM session_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            self._touch_thread(connection, now)
            connection.commit()
        assert row is not None
        return self._memory_from_row(row)

    def clear(self, *, confirmation_consumed: bool) -> int:
        if not confirmation_consumed:
            raise PermissionError("bulk session-memory clear requires consumed confirmation")
        now = _now_iso()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE session_memories SET truth_status = 'retracted', updated_at = ?
                WHERE project_id = ? AND thread_id = ?
                  AND truth_status IN ('user_stated', 'verified', 'assumption')
                """,
                (now, self._project_id, self._thread_id),
            )
            self._touch_thread(connection, now)
            connection.commit()
            return int(updated.rowcount)

    def save_summary(
        self,
        *,
        current_goal: str,
        confirmed_constraints: Sequence[str],
        decisions: Sequence[str],
        completed_items: Sequence[str],
        open_items: Sequence[str],
        artifacts: Sequence[str],
        assumptions: Sequence[str],
        source_checkpoints: Sequence[CheckpointRef],
        preferences: Sequence[str] = (),
    ) -> SessionSummary:
        self._validate_checkpoints(source_checkpoints, "0" * 64)
        now = _now_iso()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(revision), 0) + 1 FROM session_summaries
                    WHERE project_id = ? AND thread_id = ?
                    """,
                    (self._project_id, self._thread_id),
                ).fetchone()[0]
            )
            summary = SessionSummary(
                project_id=self._project_id,
                thread_id=self._thread_id,
                revision=revision,
                current_goal=current_goal,
                preferences=list(preferences),
                confirmed_constraints=list(confirmed_constraints),
                decisions=list(decisions),
                completed_items=list(completed_items),
                open_items=list(open_items),
                artifacts=list(artifacts),
                assumptions=list(assumptions),
                source_checkpoints=list(source_checkpoints),
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO session_summaries(
                    project_id, thread_id, revision, summary_json,
                    source_checkpoints_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self._project_id,
                    self._thread_id,
                    revision,
                    json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                    json.dumps(
                        [item.model_dump(mode="json") for item in source_checkpoints],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    now,
                ),
            )
            self._touch_thread(connection, now)
            connection.commit()
        return summary

    def latest_summary(self) -> SessionSummary | None:
        with closing(self._connect(read_only=True)) as connection:
            row = connection.execute(
                """
                SELECT summary_json FROM session_summaries
                WHERE project_id = ? AND thread_id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (self._project_id, self._thread_id),
            ).fetchone()
        if row is None:
            return None
        return SessionSummary.model_validate(json.loads(row["summary_json"]))

    @staticmethod
    def summary_due(
        *,
        turn_count: int,
        context_ratio: float,
        task_state_changed: bool = False,
        user_corrected: bool = False,
    ) -> bool:
        if turn_count < 0 or not 0.0 <= context_ratio <= 1.0:
            raise ValueError("invalid summary policy inputs")
        return (
            (turn_count > 0 and turn_count % 6 == 0)
            or context_ratio >= 0.70
            or task_state_changed
            or user_corrected
        )

    def _validate_checkpoints(
        self,
        checkpoints: Sequence[CheckpointRef],
        source_content_hash: str,
    ) -> None:
        if not checkpoints:
            raise ValueError("at least one source checkpoint is required")
        for checkpoint in checkpoints:
            if checkpoint.project_id != self._project_id or checkpoint.thread_id != self._thread_id:
                raise ValueError("source checkpoint namespace mismatch")
            if not self._checkpoint_validator(checkpoint, source_content_hash):
                raise ValueError("source checkpoint validation failed")

    @staticmethod
    def _insert_memory(connection: sqlite3.Connection, memory: SessionMemory) -> None:
        connection.execute(
            """
            INSERT INTO session_memories(
                memory_id, project_id, thread_id, kind, content,
                source_checkpoints_json, source_content_hash, truth_status,
                confidence, supersedes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.memory_id,
                memory.project_id,
                memory.thread_id,
                memory.kind,
                memory.content,
                json.dumps(
                    [item.model_dump(mode="json") for item in memory.source_checkpoints],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                memory.source_content_hash,
                memory.truth_status,
                memory.confidence,
                memory.supersedes,
                memory.created_at,
                memory.updated_at,
            ),
        )
        connection.execute(
            "INSERT INTO session_memories_fts(memory_id, content) VALUES (?, ?)",
            (memory.memory_id, memory.content),
        )

    def _touch_thread(self, connection: sqlite3.Connection, now: str) -> None:
        connection.execute(
            """
            INSERT INTO session_threads(project_id, thread_id, created_at, updated_at, turn_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(project_id, thread_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                turn_count = session_threads.turn_count + 1
            """,
            (self._project_id, self._thread_id, now, now),
        )

    @staticmethod
    def _status_clause(include_history: bool) -> tuple[str, list[str]]:
        if include_history:
            return "truth_status IN (?, ?, ?, ?, ?)", [
                "user_stated", "verified", "assumption", "superseded", "retracted"
            ]
        return "truth_status IN (?, ?, ?)", list(_ACTIVE_STATUSES)

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> SessionMemory:
        return SessionMemory(
            memory_id=row["memory_id"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            kind=row["kind"],
            content=row["content"],
            source_checkpoints=[
                CheckpointRef.model_validate(item)
                for item in json.loads(row["source_checkpoints_json"])
            ],
            source_content_hash=row["source_content_hash"],
            truth_status=row["truth_status"],
            confidence=float(row["confidence"]),
            supersedes=row["supersedes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


__all__ = ["CheckpointValidator", "SessionMemoryService", "SessionSummary"]
