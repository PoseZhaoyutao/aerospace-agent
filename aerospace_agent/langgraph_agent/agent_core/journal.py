"""Durable operation journal and content-addressed preimage backups."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class PreparedOperation:
    operation_id: str
    audit_id: str


class PreimageConflict(RuntimeError):
    """Raised when a target changes after its durable preimage was captured."""


class OperationJournal:
    """SQLite journal whose preimages are durable before mutation begins."""

    _SCHEMA_VERSION = 2

    def __init__(self, database_path: str | Path, *, backup_dir: str | Path | None = None):
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir = (
            Path(backup_dir).resolve()
            if backup_dir is not None
            else self.database_path.parent / "preimages"
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported operation journal schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE operations (
                        operation_id TEXT PRIMARY KEY,
                        audit_id TEXT NOT NULL UNIQUE,
                        action TEXT NOT NULL,
                        status TEXT NOT NULL,
                        recovery_class TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        failure_message TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE postimages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        path TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        sha256 TEXT,
                        byte_length INTEGER,
                        FOREIGN KEY(operation_id) REFERENCES operations(operation_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE preimages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        path TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        sha256 TEXT,
                        byte_length INTEGER,
                        backup_path TEXT,
                        FOREIGN KEY(operation_id) REFERENCES operations(operation_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX preimages_operation_idx ON preimages(operation_id, id)"
                )
                connection.execute(
                    "CREATE INDEX postimages_operation_idx ON postimages(operation_id, id)"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
            elif version == 1:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE postimages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        path TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        sha256 TEXT,
                        byte_length INTEGER,
                        FOREIGN KEY(operation_id) REFERENCES operations(operation_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX postimages_operation_idx ON postimages(operation_id, id)"
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")

    def _backup_file(self, path: Path) -> tuple[str, int, str]:
        data = path.read_bytes()
        digest = _sha256_bytes(data)
        target = self.backup_dir / "files" / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}.", dir=target.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if _sha256_bytes(temporary.read_bytes()) != digest:
                    raise OSError("preimage backup verification failed")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        elif _sha256_bytes(target.read_bytes()) != digest:
            raise OSError("content-addressed preimage backup is corrupt")
        return digest, len(data), str(target)

    def _directory_digest(self, path: Path) -> str:
        digest = hashlib.sha256()
        for entry in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
            relative = entry.relative_to(path).as_posix().encode("utf-8")
            if entry.is_symlink():
                digest.update(b"L\0" + relative + b"\0" + os.readlink(entry).encode("utf-8"))
            elif entry.is_dir():
                digest.update(b"D\0" + relative + b"\0")
            else:
                digest.update(b"F\0" + relative + b"\0" + entry.read_bytes())
        return digest.hexdigest()

    def _backup_directory(self, path: Path) -> tuple[str, int | None, str]:
        digest = self._directory_digest(path)
        parent = self.backup_dir / "directories"
        target = parent / digest
        parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=parent))
            try:
                shutil.rmtree(temporary)
                shutil.copytree(path, temporary, symlinks=True)
                if self._directory_digest(temporary) != digest:
                    raise OSError("directory preimage backup verification failed")
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        elif self._directory_digest(target) != digest:
            raise OSError("content-addressed directory backup is corrupt")
        return digest, None, str(target)

    def _capture(self, role: str, path: Path) -> dict[str, object]:
        if not path.exists() and not path.is_symlink():
            return {
                "role": role,
                "path": str(path),
                "kind": "missing",
                "sha256": None,
                "byte_length": None,
                "backup_path": None,
            }
        if path.is_dir():
            digest, length, backup_path = self._backup_directory(path)
            kind = "directory"
        else:
            digest, length, backup_path = self._backup_file(path)
            kind = "file"
        return {
            "role": role,
            "path": str(path),
            "kind": kind,
            "sha256": digest,
            "byte_length": length,
            "backup_path": backup_path,
        }

    def _describe(self, role: str, path: Path) -> dict[str, object]:
        if not path.exists() and not path.is_symlink():
            return {
                "role": role,
                "path": str(path),
                "kind": "missing",
                "sha256": None,
                "byte_length": None,
            }
        if path.is_dir():
            digest = self._directory_digest(path)
            length = None
            kind = "directory"
        else:
            data = path.read_bytes()
            digest = _sha256_bytes(data)
            length = len(data)
            kind = "file"
        return {
            "role": role,
            "path": str(path),
            "kind": kind,
            "sha256": digest,
            "byte_length": length,
        }

    def prepare(
        self,
        *,
        operation_id: str,
        action: str,
        paths: Mapping[str, str | Path],
        metadata: Mapping[str, object] | None = None,
    ) -> PreparedOperation:
        preimages = [self._capture(role, Path(path)) for role, path in paths.items()]
        audit_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO operations(
                    operation_id, audit_id, action, status, recovery_class,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, 'prepared', 'reversible', ?, ?)
                """,
                (
                    operation_id,
                    audit_id,
                    action,
                    json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":")),
                    _utc_now(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO preimages(
                    operation_id, role, path, kind, sha256, byte_length, backup_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        operation_id,
                        item["role"],
                        item["path"],
                        item["kind"],
                        item["sha256"],
                        item["byte_length"],
                        item["backup_path"],
                    )
                    for item in preimages
                ],
            )
        return PreparedOperation(operation_id=operation_id, audit_id=audit_id)

    def complete(self, operation_id: str) -> None:
        with self._connect() as connection:
            paths = connection.execute(
                "SELECT role, path FROM preimages WHERE operation_id = ? ORDER BY id",
                (operation_id,),
            ).fetchall()
        postimages = [self._describe(row["role"], Path(row["path"])) for row in paths]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """
                INSERT INTO postimages(
                    operation_id, role, path, kind, sha256, byte_length
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        operation_id,
                        item["role"],
                        item["path"],
                        item["kind"],
                        item["sha256"],
                        item["byte_length"],
                    )
                    for item in postimages
                ],
            )
            cursor = connection.execute(
                """
                UPDATE operations
                SET status = 'completed', completed_at = ?
                WHERE operation_id = ? AND status = 'prepared'
                """,
                (_utc_now(), operation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("operation is not in prepared state")

    def fail(self, operation_id: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET status = 'failed', completed_at = ?, failure_message = ?
                WHERE operation_id = ? AND status = 'prepared'
                """,
                (_utc_now(), message, operation_id),
            )

    def get(self, operation_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(str(result.pop("metadata_json")))
        return result

    def list_preimages(self, operation_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, path, kind, sha256, byte_length, backup_path "
                "FROM preimages WHERE operation_id = ? ORDER BY id",
                (operation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_postimages(self, operation_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, path, kind, sha256, byte_length "
                "FROM postimages WHERE operation_id = ? ORDER BY id",
                (operation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def assert_preimages_unchanged(self, operation_id: str) -> None:
        """Verify every current target still equals the durable preimage."""

        preimages = self.list_preimages(operation_id)
        if not preimages:
            raise PreimageConflict("operation has no durable preimage proof")
        for expected in preimages:
            actual = self._describe(str(expected["role"]), Path(str(expected["path"])))
            if any(
                actual[key] != expected[key]
                for key in ("role", "path", "kind", "sha256", "byte_length")
            ):
                raise PreimageConflict(
                    f"filesystem target drifted after preimage capture: {expected['role']}"
                )

    def rollback(self, operation_id: str) -> None:
        operation = self.get(operation_id)
        if operation is None:
            raise KeyError(f"operation not found: {operation_id}")
        if operation["recovery_class"] != "reversible":
            raise ValueError("operation is not reversible")
        if operation["status"] not in {"completed", "failed"}:
            raise RuntimeError("operation is not eligible for rollback")
        preimages = self.list_preimages(operation_id)
        postimages = self.list_postimages(operation_id)
        if operation["status"] == "completed":
            if len(postimages) != len(preimages):
                raise RuntimeError("rollback proof is incomplete")
            for expected in postimages:
                actual = self._describe(str(expected["role"]), Path(str(expected["path"])))
                if any(
                    actual[key] != expected[key]
                    for key in ("role", "path", "kind", "sha256", "byte_length")
                ):
                    raise RuntimeError("current filesystem differs from recorded postimage")
        with self._connect() as connection:
            connection.execute(
                "UPDATE operations SET status = 'rollback_requested' WHERE operation_id = ?",
                (operation_id,),
            )
        try:
            for item in reversed(preimages):
                path = Path(str(item["path"]))
                kind = str(item["kind"])
                if path.exists() or path.is_symlink():
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                if kind == "missing":
                    continue
                backup = Path(str(item["backup_path"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                if kind == "directory":
                    shutil.copytree(backup, path, symlinks=True)
                else:
                    descriptor, temporary_name = tempfile.mkstemp(
                        prefix=f".{path.name}.", dir=path.parent
                    )
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        shutil.copyfile(backup, temporary)
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
            for expected in preimages:
                actual = self._describe(str(expected["role"]), Path(str(expected["path"])))
                if any(
                    actual[key] != expected[key]
                    for key in ("role", "path", "kind", "sha256", "byte_length")
                ):
                    raise RuntimeError("rollback verification failed")
        except Exception as exc:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE operations SET status = 'manual_recovery', failure_message = ? "
                    "WHERE operation_id = ?",
                    (str(exc), operation_id),
                )
            raise
        with self._connect() as connection:
            connection.execute(
                "UPDATE operations SET status = 'rolled_back', completed_at = ? "
                "WHERE operation_id = ?",
                (_utc_now(), operation_id),
            )


__all__ = ["OperationJournal", "PreimageConflict", "PreparedOperation"]
