"""Append-only Ed25519 approval ledger.

Only public keys are accepted by this runtime.  Signatures must be produced by
an operator-held private key outside the workspace and supplied as records.
"""

from __future__ import annotations

import re
import sqlite3
from base64 import b64decode
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import field_validator

from .models import ContractModel


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_DOMAIN = b"zyt-agent-core-approval-v1\0"


def approval_signature_payload(digest: str) -> bytes:
    if not _SHA256.fullmatch(digest):
        raise ValueError("approval digest must be lowercase SHA-256")
    return _SIGNATURE_DOMAIN + digest.encode("ascii")


class ApprovalRecord(ContractModel):
    approval_record_id: str
    key_id: str
    digest: str
    signature_b64: str
    created_at: str

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("approval digest must be lowercase SHA-256")
        return value


class CapabilityApprovalLedger:
    """Persist signed baselines and revocation events without signing authority."""

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        database_path: str | Path,
        *,
        trusted_public_keys: Mapping[str, bytes | Ed25519PublicKey],
    ) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: dict[str, Ed25519PublicKey] = {}
        for key_id, value in trusted_public_keys.items():
            if not key_id:
                raise ValueError("approval key ID cannot be empty")
            self._keys[key_id] = (
                value
                if isinstance(value, Ed25519PublicKey)
                else Ed25519PublicKey.from_public_bytes(bytes(value))
            )
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate(self) -> None:
        with closing(self._connect()) as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > self._SCHEMA_VERSION:
                raise RuntimeError(f"unsupported approval schema version: {version}")
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE approval_records (
                        approval_record_id TEXT PRIMARY KEY,
                        key_id TEXT NOT NULL,
                        digest TEXT NOT NULL,
                        signature_b64 TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX approval_digest_idx ON approval_records(digest)"
                )
                connection.execute(
                    """
                    CREATE TABLE approval_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        approval_record_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        occurred_at TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        FOREIGN KEY(approval_record_id)
                            REFERENCES approval_records(approval_record_id)
                    )
                    """
                )
                connection.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
                connection.commit()

    def schema_version(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def append(
        self,
        *,
        approval_record_id: str,
        key_id: str,
        digest: str,
        signature_b64: str,
        created_at: str,
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_record_id=approval_record_id,
            key_id=key_id,
            digest=digest,
            signature_b64=signature_b64,
            created_at=created_at,
        )
        key = self._keys.get(key_id)
        if key is None:
            raise ValueError(f"approval references unknown trusted key: {key_id}")
        try:
            signature = b64decode(signature_b64, validate=True)
            key.verify(signature, approval_signature_payload(digest))
        except (ValueError, InvalidSignature) as exc:
            raise ValueError("approval signature is invalid") from exc
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM approval_records WHERE approval_record_id = ?",
                (approval_record_id,),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                raise ValueError("approval ledger is append-only")
            connection.execute(
                """
                INSERT INTO approval_records(
                    approval_record_id, key_id, digest, signature_b64, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.approval_record_id,
                    record.key_id,
                    record.digest,
                    record.signature_b64,
                    record.created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO approval_events(
                    approval_record_id, event_type, occurred_at, reason
                ) VALUES (?, 'approved', ?, '')
                """,
                (record.approval_record_id, record.created_at),
            )
            connection.commit()
        return record

    def revoke(self, *, approval_record_id: str, revoked_at: str, reason: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM approval_records WHERE approval_record_id = ?",
                (approval_record_id,),
            ).fetchone()
            already = connection.execute(
                "SELECT 1 FROM approval_events WHERE approval_record_id = ? "
                "AND event_type = 'revoked'",
                (approval_record_id,),
            ).fetchone()
            if exists is None:
                connection.rollback()
                raise KeyError(f"approval record not found: {approval_record_id}")
            if already is not None:
                connection.rollback()
                raise ValueError("approval is already revoked")
            connection.execute(
                """
                INSERT INTO approval_events(
                    approval_record_id, event_type, occurred_at, reason
                ) VALUES (?, 'revoked', ?, ?)
                """,
                (approval_record_id, revoked_at, reason),
            )
            connection.commit()

    def verify(self, digest: str) -> ApprovalRecord | None:
        approval_signature_payload(digest)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT r.* FROM approval_records AS r
                WHERE r.digest = ? AND NOT EXISTS (
                    SELECT 1 FROM approval_events AS e
                    WHERE e.approval_record_id = r.approval_record_id
                      AND e.event_type = 'revoked'
                ) ORDER BY r.rowid DESC
                """,
                (digest,),
            ).fetchall()
        for row in rows:
            record = ApprovalRecord(**dict(row))
            key = self._keys.get(record.key_id)
            if key is None:
                continue
            try:
                key.verify(
                    b64decode(record.signature_b64, validate=True),
                    approval_signature_payload(record.digest),
                )
            except (ValueError, InvalidSignature):
                continue
            return record
        return None

    def events(self, approval_record_id: str) -> list[dict[str, str]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT event_type, occurred_at, reason FROM approval_events
                WHERE approval_record_id = ? ORDER BY event_id
                """,
                (approval_record_id,),
            ).fetchall()
        return [dict(row) for row in rows]


class CapabilityApprovalVerifier:
    """Concrete verifier accepted by the execution boundary."""

    def __init__(self, ledger: CapabilityApprovalLedger) -> None:
        if not isinstance(ledger, CapabilityApprovalLedger):
            raise TypeError("ledger must be CapabilityApprovalLedger")
        self._ledger = ledger

    def verify_digest(self, digest: str) -> ApprovalRecord | None:
        return self._ledger.verify(digest)


__all__ = [
    "ApprovalRecord",
    "CapabilityApprovalLedger",
    "CapabilityApprovalVerifier",
    "approval_signature_payload",
]
