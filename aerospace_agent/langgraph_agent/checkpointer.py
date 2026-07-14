"""Checkpoint backends and graph-native conversation helpers.

The SQLite context manager is deliberately kept open for the lifetime of an
agent.  Closing it after every operation can invalidate the connection held by
``CompiledStateGraph`` and makes resume/replay subtly racy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional

from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_CHECKPOINT_DB = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "checkpoints.db")
)


def _ensure_parent(path: str | os.PathLike[str]) -> None:
    if str(path) == ":memory:":
        return
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def create_sqlite_checkpointer(db_path: Optional[str | os.PathLike[str]] = None):
    """Return the ``SqliteSaver`` context manager for *db_path*."""
    path = str(db_path or DEFAULT_CHECKPOINT_DB)
    _ensure_parent(path)
    return SqliteSaver.from_conn_string(path)


def create_memory_checkpointer() -> MemorySaver:
    """Return an in-memory saver (useful for tests and ephemeral clients)."""
    return InMemorySaver()


def get_checkpointer(backend: str = "sqlite", db_path: Optional[str | os.PathLike[str]] = None) -> Any:
    """Create a saver or saver context manager for the requested backend."""
    backend = str(backend).lower()
    if backend == "sqlite":
        return create_sqlite_checkpointer(db_path)
    if backend in {"memory", "in_memory", "inmemory"}:
        return create_memory_checkpointer()
    raise ValueError(f"unsupported checkpoint backend: {backend}; expected sqlite or memory")


def list_saved_threads(db_path: Optional[str | os.PathLike[str]] = None) -> list[str]:
    """List distinct persisted thread IDs through LangGraph's saver API.

    The SQLite schema is an implementation detail of ``SqliteSaver`` and has
    changed between LangGraph releases.  Calling ``list`` keeps this helper
    compatible with the installed saver instead of depending on table/column
    names.  Schema errors are intentionally allowed to surface to operators.
    """
    path = str(db_path or DEFAULT_CHECKPOINT_DB)
    if path != ":memory:" and not Path(path).exists():
        return []
    with create_sqlite_checkpointer(path) as saver:
        thread_ids = {
            str((item.config.get("configurable", {}) or {}).get("thread_id"))
            for item in saver.list(None, limit=None)
            if (item.config.get("configurable", {}) or {}).get("thread_id") is not None
        }
    return sorted(thread_ids)


def get_thread_checkpoints(thread_id: str, db_path: Optional[str | os.PathLike[str]] = None) -> list[dict[str, Any]]:
    """Return graph-native checkpoint descriptors, newest first.

    ``CheckpointTuple`` is the public compatibility boundary.  In particular
    we derive the parent ID and timestamp from its config/checkpoint instead
    of querying the private SQLite tables.
    """
    path = str(db_path or DEFAULT_CHECKPOINT_DB)
    if path != ":memory:" and not Path(path).exists():
        return []
    with create_sqlite_checkpointer(path) as saver:
        tuples = list(saver.list({"configurable": {"thread_id": str(thread_id)}}, limit=None))
    result: list[dict[str, Any]] = []
    for item in tuples:
        configurable = dict(item.config.get("configurable", {}) or {})
        parent_config = getattr(item, "parent_config", None) or {}
        parent_id = (parent_config.get("configurable", {}) or {}).get("checkpoint_id")
        checkpoint = dict(getattr(item, "checkpoint", {}) or {})
        metadata_value = dict(getattr(item, "metadata", {}) or {})
        checkpoint_id = configurable.get("checkpoint_id") or checkpoint.get("id")
        result.append({
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            "parent_checkpoint_id": parent_id,
            "type": metadata_value.get("source", ""),
            "metadata": metadata_value,
            "created_at": checkpoint.get("ts", checkpoint_id),
        })
    return result


def delete_thread_checkpoints(thread_id: str, db_path: Optional[str | os.PathLike[str]] = None) -> int:
    """Delete a thread through ``SqliteSaver.delete_thread`` and return count."""
    path = str(db_path or DEFAULT_CHECKPOINT_DB)
    if path != ":memory:" and not Path(path).exists():
        return 0
    with create_sqlite_checkpointer(path) as saver:
        before = len(list(saver.list({"configurable": {"thread_id": str(thread_id)}}, limit=None)))
        saver.delete_thread(str(thread_id))
    return before


__all__ = [
    "DEFAULT_CHECKPOINT_DB",
    "create_sqlite_checkpointer",
    "create_memory_checkpointer",
    "get_checkpointer",
    "list_saved_threads",
    "get_thread_checkpoints",
    "delete_thread_checkpoints",
]
