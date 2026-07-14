"""Explicit project identity initialization and rebuildable project-memory index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import yaml

from .models import ContractModel


class ProjectStatus(ContractModel):
    state: Literal["uninitialized", "ready", "migration_failed"]
    project_id: str | None = None
    indexed_documents: int = 0
    message: str = ""


class ProjectInitializationConflict(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ProjectIdentityService:
    """Own the stable random project ID and its project-scoped memory stores."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        lock_timeout_seconds: float = 10.0,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.project_memory_root = self.workspace_root / "memory" / "project"
        self.manifest_path = self.project_memory_root / "manifest.yaml"
        self.lock_path = self.project_memory_root / ".init.lock"
        self.failure_path = self.workspace_root / "data" / "langgraph" / "project_memory_migration_failed.json"
        self.session_db_path = self.workspace_root / "data" / "langgraph" / "session_memory.sqlite"
        self.index_db_path = self.workspace_root / "data" / "langgraph" / "project_memory_index.sqlite"
        self.lock_timeout_seconds = float(lock_timeout_seconds)

    def status(self) -> ProjectStatus:
        if not self.manifest_path.is_file():
            return ProjectStatus(state="uninitialized")
        try:
            project_id = self._read_project_id()
            if self.failure_path.exists():
                return ProjectStatus(
                    state="migration_failed",
                    message="a project-memory migration failure is recorded",
                )
            self._assert_database_version(self.session_db_path)
            self._assert_database_version(self.index_db_path)
            return ProjectStatus(
                state="ready",
                project_id=project_id,
                indexed_documents=self._index_count(project_id),
            )
        except (OSError, ValueError, sqlite3.Error, yaml.YAMLError) as exc:
            return ProjectStatus(state="migration_failed", message=str(exc))

    def initialize(self) -> ProjectStatus:
        self.project_memory_root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.lock_timeout_seconds
        lock_fd: int | None = None
        while lock_fd is None:
            try:
                lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ProjectInitializationConflict("project initialization lock timed out")
                time.sleep(0.05)
        try:
            os.write(lock_fd, f"pid={os.getpid()} created_at={_now_iso()}\n".encode("utf-8"))
            os.fsync(lock_fd)
            os.close(lock_fd)
            lock_fd = None
            try:
                project_id = self._ensure_manifest()
                self._ensure_project_memory_layout()
                self._migrate_session_database()
                self._migrate_project_index()
                self.reindex()
                self.failure_path.unlink(missing_ok=True)
            except Exception as exc:
                self._record_migration_failure(exc)
                raise
            return ProjectStatus(
                state="ready",
                project_id=project_id,
                indexed_documents=self._index_count(project_id),
            )
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            self.lock_path.unlink(missing_ok=True)

    def reindex(self) -> ProjectStatus:
        project_id = self._read_project_id()
        self._assert_database_version(self.index_db_path)
        documents = self._scan_documents()
        now = _now_iso()
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT document_id FROM project_documents WHERE project_id = ?",
                    (project_id,),
                )
            ]
            for document_id in existing_ids:
                connection.execute(
                    "DELETE FROM project_documents_fts WHERE document_id = ?",
                    (document_id,),
                )
            connection.execute(
                "DELETE FROM project_documents WHERE project_id = ?",
                (project_id,),
            )
            for relative_path, content in documents:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                document_id = hashlib.sha256(
                    f"{project_id}\0{relative_path}\0{digest}".encode("utf-8")
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO project_documents(
                        document_id, project_id, relative_path, content_sha256, content, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (document_id, project_id, relative_path, digest, content, now),
                )
                connection.execute(
                    "INSERT INTO project_documents_fts(document_id, content) VALUES (?, ?)",
                    (document_id, content),
                )
            connection.commit()
        return ProjectStatus(
            state="ready",
            project_id=project_id,
            indexed_documents=len(documents),
        )

    def indexed_paths(self) -> list[str]:
        project_id = self._read_project_id()
        self._assert_database_version(self.index_db_path)
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            rows = connection.execute(
                """
                SELECT relative_path FROM project_documents
                WHERE project_id = ? ORDER BY relative_path
                """,
                (project_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, str]]:
        if not 1 <= limit <= 100:
            raise ValueError("project memory search limit must be between 1 and 100")
        tokens = [token for token in query.replace('"', " ").split() if token]
        if not tokens:
            return []
        status = self.status()
        if status.state != "ready" or status.project_id is None:
            raise RuntimeError(
                "project_not_initialized"
                if status.state == "uninitialized"
                else "project_memory_migration_failed"
            )
        expression = " AND ".join(f'"{token}"' for token in tokens)
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT d.relative_path, d.content, d.content_sha256
                FROM project_documents_fts AS f
                JOIN project_documents AS d ON d.document_id = f.document_id
                WHERE project_documents_fts MATCH ? AND d.project_id = ?
                ORDER BY rank LIMIT ?
                """,
                (expression, status.project_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_indexed(self, relative_path: str) -> str:
        status = self.status()
        if status.state != "ready" or status.project_id is None:
            raise RuntimeError(
                "project_not_initialized"
                if status.state == "uninitialized"
                else "project_memory_migration_failed"
            )
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            row = connection.execute(
                """
                SELECT content FROM project_documents
                WHERE project_id = ? AND relative_path = ?
                """,
                (status.project_id, relative_path),
            ).fetchone()
        if row is None:
            raise KeyError(f"indexed project document not found: {relative_path}")
        return str(row[0])

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, str]]:
        if not 1 <= limit <= 100:
            raise ValueError("project memory search limit must be between 1 and 100")
        tokens = [token for token in query.replace('"', " ").split() if token]
        if not tokens:
            return []
        status = self.status()
        if status.state != "ready" or status.project_id is None:
            raise RuntimeError(
                "project_not_initialized"
                if status.state == "uninitialized"
                else "project_memory_migration_failed"
            )
        expression = " AND ".join(f'"{token}"' for token in tokens)
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT d.relative_path, d.content, d.content_sha256
                FROM project_documents_fts AS f
                JOIN project_documents AS d ON d.document_id = f.document_id
                WHERE project_documents_fts MATCH ? AND d.project_id = ?
                ORDER BY rank LIMIT ?
                """,
                (expression, status.project_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def read_indexed(self, relative_path: str) -> str:
        status = self.status()
        if status.state != "ready" or status.project_id is None:
            raise RuntimeError(
                "project_not_initialized"
                if status.state == "uninitialized"
                else "project_memory_migration_failed"
            )
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            row = connection.execute(
                """
                SELECT content FROM project_documents
                WHERE project_id = ? AND relative_path = ?
                """,
                (status.project_id, relative_path),
            ).fetchone()
        if row is None:
            raise KeyError(f"indexed project document not found: {relative_path}")
        return str(row[0])

    def _ensure_manifest(self) -> str:
        if self.manifest_path.exists():
            return self._read_project_id()
        project_id = str(uuid4())
        payload = {
            "schema_version": "1.0",
            "project_id": project_id,
            "created_at": _now_iso(),
        }
        temporary = self.project_memory_root / f".manifest.{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
        return project_id

    def _read_project_id(self) -> str:
        loaded = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("schema_version") != "1.0":
            raise ValueError("invalid project identity manifest schema")
        raw_project_id = loaded.get("project_id")
        try:
            project_id = str(UUID(str(raw_project_id)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid project_id in project identity manifest") from exc
        return project_id

    def _ensure_project_memory_layout(self) -> None:
        self._write_if_missing(
            self.project_memory_root / "PROJECT.md",
            "# Project Memory\n\nOnly human-approved project facts belong here.\n",
        )
        self._write_if_missing(
            self.project_memory_root / "constraints.yaml",
            'schema_version: "1.0"\nconstraints: []\n',
        )
        (self.project_memory_root / "decisions").mkdir(exist_ok=True)
        (self.project_memory_root / "workflows").mkdir(exist_ok=True)

    @staticmethod
    def _write_if_missing(path: Path, content: str) -> None:
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            return

    def _migrate_session_database(self) -> None:
        self.session_db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.session_db_path)) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported session memory schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE session_threads (
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        turn_count INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(project_id, thread_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE session_memories (
                        memory_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        content TEXT NOT NULL,
                        source_checkpoints_json TEXT NOT NULL,
                        source_content_hash TEXT NOT NULL,
                        truth_status TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        supersedes TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX session_memory_namespace_idx "
                    "ON session_memories(project_id, thread_id, updated_at)"
                )
                connection.execute(
                    """
                    CREATE TABLE session_summaries (
                        project_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        summary_json TEXT NOT NULL,
                        source_checkpoints_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(project_id, thread_id, revision)
                    )
                    """
                )
                connection.execute(
                    "CREATE VIRTUAL TABLE session_memories_fts "
                    "USING fts5(memory_id UNINDEXED, content)"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def _migrate_project_index(self) -> None:
        self.index_db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported project index schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE project_documents (
                        document_id TEXT PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        content TEXT NOT NULL,
                        indexed_at TEXT NOT NULL,
                        UNIQUE(project_id, relative_path)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX project_document_namespace_idx "
                    "ON project_documents(project_id, relative_path)"
                )
                connection.execute(
                    "CREATE VIRTUAL TABLE project_documents_fts "
                    "USING fts5(document_id UNINDEXED, content)"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def _scan_documents(self) -> list[tuple[str, str]]:
        candidates: set[Path] = set()
        agents = self.workspace_root / "AGENTS.md"
        if agents.is_file():
            candidates.add(agents)
        candidates.update(path for path in self.workspace_root.glob("README*") if path.is_file())
        config_root = self.workspace_root / "config"
        if config_root.is_dir():
            for suffix in ("*.yaml", "*.yml", "*.json", "*.toml", "*.md"):
                candidates.update(path for path in config_root.rglob(suffix) if path.is_file())
        if self.project_memory_root.is_dir():
            candidates.update(
                path
                for path in self.project_memory_root.rglob("*")
                if path.is_file() and path.suffix.casefold() in {".md", ".yaml", ".yml"}
            )
        documents: list[tuple[str, str]] = []
        for path in sorted(candidates):
            relative = path.relative_to(self.workspace_root).as_posix()
            lowered = relative.casefold()
            if any(
                marker in lowered
                for marker in ("secret", "token", "password", "private-key", ".env")
            ):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            documents.append((relative, content))
        return documents

    @staticmethod
    def _assert_database_version(path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"project database is missing: {path.name}")
        with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version != 1:
            raise ValueError(f"project database schema mismatch: {path.name}")

    def _index_count(self, project_id: str) -> int:
        with closing(sqlite3.connect(self.index_db_path)) as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM project_documents WHERE project_id = ?",
                    (project_id,),
                ).fetchone()[0]
            )

    def _record_migration_failure(self, error: Exception) -> None:
        self.failure_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "recorded_at": _now_iso(),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        temporary = self.failure_path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.failure_path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["ProjectIdentityService", "ProjectInitializationConflict", "ProjectStatus"]
