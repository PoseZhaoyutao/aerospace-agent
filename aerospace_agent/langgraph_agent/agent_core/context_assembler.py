"""Bounded project/session memory context with strict thread isolation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .models import CheckpointRef, ContractModel, SessionMemory
from .project_memory import ProjectIdentityService
from .session_memory import SessionMemoryService, SessionSummary


CheckpointValidator = Callable[[CheckpointRef, str], bool]


class MemoryContextSections(ContractModel):
    project_memory: list[str]
    session_summary: SessionSummary | None = None
    session_memories: list[SessionMemory]

    def prompt_sections(self) -> list[str]:
        sections: list[str] = []
        for item in self.project_memory:
            sections.append(f"[PROJECT MEMORY]\n{item}")
        if self.session_summary is not None:
            summary = self.session_summary
            sections.append(
                "[SESSION SUMMARY]\n"
                f"goal: {summary.current_goal}\n"
                f"preferences: {list(summary.preferences)}\n"
                f"constraints: {list(summary.confirmed_constraints)}\n"
                f"decisions: {list(summary.decisions)}\n"
                f"open_items: {list(summary.open_items)}\n"
                f"assumptions: {list(summary.assumptions)}"
            )
        for memory in self.session_memories:
            sections.append(
                "[SESSION MEMORY]\n"
                f"kind: {memory.kind}\n"
                f"truth_status: {memory.truth_status}\n"
                f"content: {memory.content}"
            )
        return sections


class MemoryContextAssembler:
    """Read approved project memory and one runtime-injected thread only."""

    def __init__(
        self,
        *,
        project_memory: ProjectIdentityService,
        session_database_path: str | Path,
        project_id: str,
        checkpoint_validator: CheckpointValidator,
        project_limit: int = 3,
        session_limit: int = 8,
        max_section_chars: int = 4_000,
    ) -> None:
        if not isinstance(project_memory, ProjectIdentityService):
            raise TypeError("project_memory must be ProjectIdentityService")
        status = project_memory.status()
        if status.state != "ready" or status.project_id != project_id:
            raise RuntimeError("project memory is not ready for the supplied project_id")
        self._project_memory = project_memory
        self._session_database_path = Path(session_database_path)
        self._project_id = project_id
        self._checkpoint_validator = checkpoint_validator
        self._project_limit = project_limit
        self._session_limit = session_limit
        self._max_section_chars = max_section_chars

    def assemble(self, *, thread_id: str, query: str) -> MemoryContextSections:
        if not thread_id:
            raise ValueError("thread_id is required")
        session = SessionMemoryService(
            self._session_database_path,
            project_id=self._project_id,
            thread_id=thread_id,
            checkpoint_validator=self._checkpoint_validator,
        )
        project_documents = []
        memories: list[SessionMemory] = []
        if query.strip():
            try:
                matches = self._project_memory.search(query, limit=self._project_limit)
            except Exception:
                matches = []
            project_documents = [
                f"source: {item['relative_path']}\n{item['content'][: self._max_section_chars]}"
                for item in matches
            ]
            for memory in session.search(query, limit=self._session_limit):
                if all(
                    self._checkpoint_validator(checkpoint, memory.source_content_hash)
                    for checkpoint in memory.source_checkpoints
                ):
                    memories.append(memory)
            # Stable user preferences and explicit constraints/decisions are
            # not query facts.  Keep them available for unrelated follow-ups
            # (for example, a greeting after "call me ...") while preserving
            # the fixed project/thread namespace and session limit.
            seen_memory_ids = {item.memory_id for item in memories}
            for kind in ("preference", "constraint", "decision"):
                for memory in session.list(kind=kind, limit=self._session_limit):
                    if len(memories) >= self._session_limit:
                        break
                    if memory.memory_id in seen_memory_ids:
                        continue
                    if all(
                        self._checkpoint_validator(checkpoint, memory.source_content_hash)
                        for checkpoint in memory.source_checkpoints
                    ):
                        memories.append(memory)
                        seen_memory_ids.add(memory.memory_id)
                if len(memories) >= self._session_limit:
                    break
        summary = session.latest_summary()
        if summary is not None and not all(
            self._checkpoint_validator(checkpoint, "0" * 64)
            for checkpoint in summary.source_checkpoints
        ):
            summary = None
        return MemoryContextSections(
            project_memory=project_documents,
            session_summary=summary,
            session_memories=memories,
        )


__all__ = ["MemoryContextAssembler", "MemoryContextSections"]
