"""Durable journal and artifacts for evolution transactions."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def encode_bytes(value: bytes | None) -> str | None:
    return None if value is None else base64.b64encode(value).decode("ascii")


def decode_bytes(value: str | None) -> bytes | None:
    return None if value is None else base64.b64decode(value.encode("ascii"))


class EvolutionStore:
    """Persist transaction metadata with atomic JSON writes and append-only journal."""

    def __init__(self, data_dir: str | os.PathLike[str]):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def transaction_dir(self, evolution_id: str) -> Path:
        path = self.data_dir / str(evolution_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "staging").mkdir(exist_ok=True)
        (path / "backup").mkdir(exist_ok=True)
        return path

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def write_json(self, evolution_id: str, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.transaction_dir(evolution_id) / name
        self._atomic_json(path, payload)
        return path

    def read_json(self, evolution_id: str, name: str) -> dict[str, Any]:
        path = self.transaction_dir(evolution_id) / name
        return json.loads(path.read_text(encoding="utf-8"))

    def append_transition(self, evolution_id: str, state: str, **details: Any) -> dict[str, Any]:
        event = {"state": state, "timestamp": datetime.now(timezone.utc).isoformat(), **details}
        path = self.transaction_dir(evolution_id) / "journal.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def read_journal(self, evolution_id: str) -> list[dict[str, Any]]:
        path = self.transaction_dir(evolution_id) / "journal.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["EvolutionStore", "sha256_bytes", "sha256_file", "encode_bytes", "decode_bytes"]
