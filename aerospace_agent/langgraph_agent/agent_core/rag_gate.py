"""Conditional private-RAG policy and one-budget-per-root-run state machine."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .models import ContractModel, ExecutionRun


class RagGateDecision(ContractModel):
    retrieve: bool
    reason: Literal[
        "low_confidence",
        "explicit_evidence_request",
        "planner_requested",
        "explicit_no_retrieval",
        "high_confidence",
        "route_not_eligible",
    ]


_NEGATIVE_PHRASES = (
    "不需要来源",
    "不要来源",
    "不要核实",
    "无需核实",
    "不要引用知识库",
    "不引用知识库",
    "no sources",
    "do not verify",
    "don't verify",
    "no knowledge base",
)
_EVIDENCE_PHRASES = (
    "依据",
    "来源",
    "核实",
    "引用知识库",
    "evidence",
    "source",
    "verify",
)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    for phrase in phrases:
        candidate = phrase.casefold()
        if re.fullmatch(r"[a-z][a-z ]*", candidate):
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])",
                normalized,
            ):
                return True
        elif candidate in normalized:
            return True
    return False


def decide_private_rag(
    *,
    route: str,
    confidence: float,
    user_text: str,
    planner_request: str | None = None,
    confidence_threshold: float = 0.60,
) -> RagGateDecision:
    """Apply the three positive triggers with explicit denial taking precedence."""

    if not 0.0 <= confidence <= 1.0 or not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("RAG confidence values must be between 0 and 1")
    if _contains_phrase(user_text, _NEGATIVE_PHRASES):
        return RagGateDecision(retrieve=False, reason="explicit_no_retrieval")
    if _contains_phrase(user_text, _EVIDENCE_PHRASES):
        if route in {"knowledge_qa", "complex_task"}:
            return RagGateDecision(retrieve=True, reason="explicit_evidence_request")
    if route == "complex_task" and planner_request == "retrieve":
        return RagGateDecision(retrieve=True, reason="planner_requested")
    if route == "knowledge_qa" and confidence < confidence_threshold:
        return RagGateDecision(retrieve=True, reason="low_confidence")
    if route != "knowledge_qa":
        return RagGateDecision(retrieve=False, reason="route_not_eligible")
    return RagGateDecision(retrieve=False, reason="high_confidence")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ExecutionRunStore:
    """Versioned SQLite store for atomic retrieval-budget transitions."""

    _SCHEMA_VERSION = 1

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

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("execution run clock must be timezone-aware")
        return value.astimezone(UTC)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported execution run schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE execution_runs (
                        root_run_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT,
                        kind TEXT NOT NULL,
                        continuation_of TEXT,
                        retrieval_budget INTEGER NOT NULL,
                        retrieval_state TEXT NOT NULL,
                        retrieval_reason TEXT NOT NULL,
                        retrieval_query_hash TEXT,
                        retrieval_claimed_at TEXT,
                        retrieval_lease_expires_at TEXT,
                        retrieval_attempt_started_at TEXT,
                        retrieval_claimer_id TEXT,
                        version INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX execution_run_project_idx "
                    "ON execution_runs(project_id, thread_id, root_run_id)"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def create_user_run(
        self,
        *,
        root_run_id: str,
        project_id: str,
        thread_id: str,
        continuation_of: str | None = None,
    ) -> ExecutionRun:
        run = ExecutionRun(
            root_run_id=root_run_id,
            project_id=project_id,
            thread_id=thread_id,
            kind="user",
            continuation_of=continuation_of,
            retrieval_budget=1,
            retrieval_state="available",
            version=1,
        )
        self._insert(run)
        return run

    def create_scheduled_run(self, *, root_run_id: str, project_id: str) -> ExecutionRun:
        run = ExecutionRun(
            root_run_id=root_run_id,
            project_id=project_id,
            thread_id=None,
            kind="scheduled",
            retrieval_budget=0,
            retrieval_state="unavailable",
            version=1,
        )
        self._insert(run)
        return run

    def _insert(self, run: ExecutionRun) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO execution_runs(
                    root_run_id, project_id, thread_id, kind, continuation_of,
                    retrieval_budget, retrieval_state, retrieval_reason,
                    retrieval_query_hash, retrieval_claimed_at,
                    retrieval_lease_expires_at, retrieval_attempt_started_at,
                    retrieval_claimer_id, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(run),
            )
            connection.commit()

    def get(self, root_run_id: str) -> ExecutionRun:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE root_run_id = ?",
                (root_run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"execution run not found: {root_run_id}")
        return self._from_row(row)

    def claim(
        self,
        *,
        root_run_id: str,
        expected_version: int,
        claimer_id: str,
        query: str,
        reason: str,
        lease_seconds: int = 30,
    ) -> ExecutionRun | None:
        if not 1 <= lease_seconds <= 300:
            raise ValueError("RAG claim lease must be between 1 and 300 seconds")
        now = self._now()
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        lease = now + timedelta(seconds=lease_seconds)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE execution_runs SET
                    retrieval_state = 'claimed', retrieval_reason = ?,
                    retrieval_query_hash = ?, retrieval_claimed_at = ?,
                    retrieval_lease_expires_at = ?, retrieval_claimer_id = ?,
                    version = version + 1
                WHERE root_run_id = ? AND kind = 'user'
                  AND retrieval_budget = 1 AND retrieval_state = 'available'
                  AND version = ?
                """,
                (
                    reason,
                    query_hash,
                    now.isoformat(),
                    lease.isoformat(),
                    claimer_id,
                    root_run_id,
                    expected_version,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                return None
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE root_run_id = ?",
                (root_run_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._from_row(row)

    def mark_in_flight(
        self,
        *,
        root_run_id: str,
        expected_version: int,
        claimer_id: str,
    ) -> ExecutionRun:
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE execution_runs SET
                    retrieval_state = 'in_flight',
                    retrieval_attempt_started_at = ?, version = version + 1
                WHERE root_run_id = ? AND retrieval_state = 'claimed'
                  AND retrieval_claimer_id = ? AND version = ?
                  AND retrieval_attempt_started_at IS NULL
                """,
                (now, root_run_id, claimer_id, expected_version),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("RAG claim cannot transition to in_flight")
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE root_run_id = ?",
                (root_run_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._from_row(row)

    def consume(
        self,
        *,
        root_run_id: str,
        expected_version: int,
        claimer_id: str,
    ) -> ExecutionRun:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE execution_runs SET retrieval_state = 'consumed', version = version + 1
                WHERE root_run_id = ? AND retrieval_state = 'in_flight'
                  AND retrieval_claimer_id = ? AND version = ?
                """,
                (root_run_id, claimer_id, expected_version),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("RAG attempt cannot transition to consumed")
            row = connection.execute(
                "SELECT * FROM execution_runs WHERE root_run_id = ?",
                (root_run_id,),
            ).fetchone()
            connection.commit()
        assert row is not None
        return self._from_row(row)

    def recover_expired(self) -> dict[str, int]:
        now = self._now().isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            released = connection.execute(
                """
                UPDATE execution_runs SET
                    retrieval_state = 'available', retrieval_reason = '',
                    retrieval_query_hash = NULL, retrieval_claimed_at = NULL,
                    retrieval_lease_expires_at = NULL,
                    retrieval_attempt_started_at = NULL,
                    retrieval_claimer_id = NULL, version = version + 1
                WHERE retrieval_state = 'claimed'
                  AND retrieval_attempt_started_at IS NULL
                  AND retrieval_lease_expires_at < ?
                """,
                (now,),
            ).rowcount
            unknown = connection.execute(
                """
                UPDATE execution_runs SET
                    retrieval_state = 'consumed_unknown', version = version + 1
                WHERE retrieval_state = 'in_flight'
                  AND retrieval_attempt_started_at IS NOT NULL
                  AND retrieval_lease_expires_at < ?
                """,
                (now,),
            ).rowcount
            connection.commit()
        return {"released": int(released), "consumed_unknown": int(unknown)}

    @staticmethod
    def _values(run: ExecutionRun) -> tuple[Any, ...]:
        return (
            run.root_run_id,
            run.project_id,
            run.thread_id,
            run.kind,
            run.continuation_of,
            run.retrieval_budget,
            run.retrieval_state,
            run.retrieval_reason,
            run.retrieval_query_hash,
            run.retrieval_claimed_at,
            run.retrieval_lease_expires_at,
            run.retrieval_attempt_started_at,
            run.retrieval_claimer_id,
            run.version,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ExecutionRun:
        return ExecutionRun(
            root_run_id=row["root_run_id"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            kind=row["kind"],
            continuation_of=row["continuation_of"],
            retrieval_budget=int(row["retrieval_budget"]),
            retrieval_state=row["retrieval_state"],
            retrieval_reason=row["retrieval_reason"],
            retrieval_query_hash=row["retrieval_query_hash"],
            retrieval_claimed_at=row["retrieval_claimed_at"],
            retrieval_lease_expires_at=row["retrieval_lease_expires_at"],
            retrieval_attempt_started_at=row["retrieval_attempt_started_at"],
            retrieval_claimer_id=row["retrieval_claimer_id"],
            version=int(row["version"]),
        )


class RagGateService:
    """Run a private retriever only after claiming and marking one budget in-flight."""

    def __init__(self, store: ExecutionRunStore) -> None:
        self._store = store

    def retrieve_once(
        self,
        *,
        run: ExecutionRun,
        decision: RagGateDecision,
        query: str,
        claimer_id: str,
        retriever: Callable[[str], Any],
    ) -> Any:
        if not decision.retrieve:
            return None
        claimed = self._store.claim(
            root_run_id=run.root_run_id,
            expected_version=run.version,
            claimer_id=claimer_id,
            query=query,
            reason=decision.reason,
        )
        if claimed is None:
            return None
        in_flight = self._store.mark_in_flight(
            root_run_id=run.root_run_id,
            expected_version=claimed.version,
            claimer_id=claimer_id,
        )
        try:
            return retriever(query)
        finally:
            self._store.consume(
                root_run_id=run.root_run_id,
                expected_version=in_flight.version,
                claimer_id=claimer_id,
            )


__all__ = [
    "ExecutionRunStore",
    "RagGateDecision",
    "RagGateService",
    "decide_private_rag",
]
