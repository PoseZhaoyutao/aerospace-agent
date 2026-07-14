"""Durable checkpoint receipts for the execution boundary."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ContractModel, ToolResult


class ExecutionCheckpointReceipt(ContractModel):
    checkpoint_id: str
    project_id: str
    thread_id: str | None
    root_run_id: str
    operation_id: str
    persisted_at: str


class ExecutionCheckpointStore:
    """Versioned SQLite checkpoint writer used by ``ExecutionService``."""

    _SCHEMA_VERSION = 2

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported execution checkpoint schema: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE execution_checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT,
                        root_run_id TEXT NOT NULL,
                        operation_id TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        persisted_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX execution_checkpoint_run_idx ON execution_checkpoints("
                    "project_id, thread_id, root_run_id, operation_id)"
                )
                self._create_step_claims(connection)
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()
            elif version == 1:
                connection.execute("BEGIN IMMEDIATE")
                self._create_step_claims(connection)
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    @staticmethod
    def _create_step_claims(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE execution_step_claims (
                project_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                root_run_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                checkpoint_id TEXT,
                result_json TEXT,
                claimed_at TEXT NOT NULL,
                PRIMARY KEY(project_id, thread_id, root_run_id, plan_id, step_id, input_hash)
            )
            """
        )

    @staticmethod
    def input_hash(arguments: Any) -> str:
        encoded = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def claim_step(self, request: Any, context: Any, arguments: Any) -> tuple[str, ToolResult | None]:
        """Atomically claim one immutable planned-step input."""

        if request.origin != "planned":
            return "claimed", None
        if context.thread_id is None or context.plan_id is None or request.step_id is None:
            raise ValueError("planned execution claim requires thread, plan and step identity")
        digest = self.input_hash(arguments)
        claimed_at = datetime.now(UTC).isoformat()
        key = (
            context.project_id,
            context.thread_id,
            context.root_run_id,
            context.plan_id,
            request.step_id,
            digest,
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status, result_json FROM execution_step_claims
                WHERE project_id = ? AND thread_id = ? AND root_run_id = ?
                  AND plan_id = ? AND step_id = ? AND input_hash = ?
                """,
                key,
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO execution_step_claims(
                        project_id, thread_id, root_run_id, plan_id, step_id,
                        input_hash, status, claimed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'in_flight', ?)
                    """,
                    (*key, claimed_at),
                )
                connection.commit()
                return "claimed", None
            if row["status"] == "retryable":
                updated = connection.execute(
                    """
                    UPDATE execution_step_claims
                    SET status = 'in_flight', checkpoint_id = NULL, result_json = NULL,
                        claimed_at = ?
                    WHERE project_id = ? AND thread_id = ? AND root_run_id = ?
                      AND plan_id = ? AND step_id = ? AND input_hash = ?
                      AND status = 'retryable'
                    """,
                    (claimed_at, *key),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    return "blocked", None
                connection.commit()
                return "claimed", None
            connection.rollback()
        if row["status"] == "completed" and row["result_json"]:
            return "completed", ToolResult.model_validate(json.loads(row["result_json"]))
        return "blocked", None

    def write(
        self,
        request: Any,
        context: Any,
        result: ToolResult,
        *,
        arguments: Any | None = None,
    ) -> ExecutionCheckpointReceipt:
        persisted_at = datetime.now(UTC).isoformat()
        checkpoint_id = f"execution:{context.root_run_id}:{request.operation_id}:{result.audit_id}"
        receipt = ExecutionCheckpointReceipt(
            checkpoint_id=checkpoint_id,
            project_id=context.project_id,
            thread_id=context.thread_id,
            root_run_id=context.root_run_id,
            operation_id=request.operation_id,
            persisted_at=persisted_at,
        )
        result_json = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO execution_checkpoints(
                    checkpoint_id, project_id, thread_id, root_run_id,
                    operation_id, result_json, persisted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.checkpoint_id,
                    receipt.project_id,
                    receipt.thread_id,
                    receipt.root_run_id,
                    receipt.operation_id,
                    result_json,
                    receipt.persisted_at,
                ),
            )
            if request.origin == "planned":
                digest = self.input_hash(request.arguments if arguments is None else arguments)
                if result.status == "success":
                    claim_status = "completed"
                    claim_result = result_json
                elif result.recovery_class == "read_only":
                    claim_status = "retryable"
                    claim_result = result_json
                else:
                    claim_status = "unknown"
                    claim_result = result_json
                updated = connection.execute(
                    """
                    UPDATE execution_step_claims
                    SET status = ?, checkpoint_id = ?, result_json = ?
                    WHERE project_id = ? AND thread_id = ? AND root_run_id = ?
                      AND plan_id = ? AND step_id = ? AND input_hash = ?
                      AND status = 'in_flight'
                    """,
                    (
                        claim_status,
                        receipt.checkpoint_id,
                        claim_result,
                        context.project_id,
                        context.thread_id,
                        context.root_run_id,
                        context.plan_id,
                        request.step_id,
                        digest,
                    ),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    raise RuntimeError("planned step claim is missing or not in flight")
            connection.commit()
        return receipt

    def get(self, checkpoint_id: str) -> tuple[ExecutionCheckpointReceipt, ToolResult]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM execution_checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"execution checkpoint not found: {checkpoint_id}")
        receipt = ExecutionCheckpointReceipt(
            checkpoint_id=row["checkpoint_id"],
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            root_run_id=row["root_run_id"],
            operation_id=row["operation_id"],
            persisted_at=row["persisted_at"],
        )
        return receipt, ToolResult.model_validate(json.loads(row["result_json"]))


__all__ = ["ExecutionCheckpointReceipt", "ExecutionCheckpointStore"]
