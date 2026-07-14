"""Bounded, checkpoint-safe context assembly and artifact offloading.

The context assembler deliberately keeps the policy-bearing messages and the
latest user instruction in memory.  Large tool responses are content
addressed on disk and represented in the prompt by a short reference rather
than by their payload.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def _jsonable(value: Any) -> Any:
    """Convert common runtime values into deterministic JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=lambda k: str(k))}
    if isinstance(value, (list, tuple, set)):
        values = [_jsonable(item) for item in value]
        if isinstance(value, set):
            return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return values
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "content"):
        return _jsonable(getattr(value, "content"))
    return str(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(_jsonable(content), ensure_ascii=False, sort_keys=True)


def _estimate_tokens(text: str) -> int:
    # A deterministic, conservative estimate that does not require a model
    # tokenizer.  Four UTF-8 characters per token is intentionally generous.
    return max(1, math.ceil(len(text) / 4)) if text else 0


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a content-addressed JSON artifact on disk."""

    path: Path
    sha256: str
    media_type: str = "application/json"
    summary: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "media_type": self.media_type,
            "summary": self.summary,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass
class ContextAssembly:
    """Result returned by :meth:`ContextService.assemble`."""

    messages: list[BaseMessage]
    estimated_tokens: int
    artifact_refs: list[ArtifactRef] = field(default_factory=list)
    summary: str = ""
    _recent_messages: list[BaseMessage] = field(default_factory=list, repr=False)

    @property
    def artifacts(self) -> list[ArtifactRef]:
        return self.artifact_refs

    @property
    def recent_messages(self) -> list[BaseMessage]:
        return self._recent_messages

    @property
    def prompt(self) -> str:
        return "\n".join(_message_text(message) for message in self.messages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "estimated_tokens": self.estimated_tokens,
            "artifact_refs": [ref.as_dict() for ref in self.artifact_refs],
            "summary": self.summary,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class ContextService:
    """Assemble bounded context while preserving essential instructions."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        max_tokens: int = 4096,
        recent_turns: int = 4,
        artifact_chars: int = 12000,
        memory_context: Any = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if recent_turns < 1:
            raise ValueError("recent_turns must be positive")
        if artifact_chars < 1:
            raise ValueError("artifact_chars must be positive")
        self.workspace = Path(workspace)
        self.max_tokens = int(max_tokens)
        self.recent_turns = int(recent_turns)
        self.artifact_chars = int(artifact_chars)
        self.artifact_dir = self.workspace / "data" / "langgraph" / "artifacts"
        # Runtime-only dependency.  It owns the fixed project/thread namespace;
        # callers cannot pass arbitrary namespace identifiers to a memory store.
        self._memory_context = memory_context

    def set_memory_context(self, memory_context: Any) -> None:
        """Attach a namespace-bound memory reader after checkpointer startup."""

        self._memory_context = memory_context

    def _write_artifact(self, payload: Any) -> ArtifactRef:
        data = _canonical_json(payload)
        digest = hashlib.sha256(data).hexdigest()
        path = self.artifact_dir / f"{digest}.json"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Reuse only a file whose bytes verify against its content address.
        valid = False
        if path.exists():
            try:
                existing = path.read_bytes()
                valid = hashlib.sha256(existing).hexdigest() == digest and existing == data
            except OSError:
                valid = False
        if not valid:
            fd, temporary = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=self.artifact_dir)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary).replace(path)
            finally:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

        value = _jsonable(payload)
        if isinstance(value, Mapping):
            keys = ", ".join(str(key) for key in sorted(value)) or "none"
            summary = f"JSON artifact ({len(data)} bytes; keys: {keys})"
        else:
            summary = f"JSON artifact ({len(data)} bytes)"
        return ArtifactRef(path=path, sha256=digest, media_type="application/json", summary=summary)

    def _offload_tools(self, tool_results: Iterable[Any]) -> tuple[list[ArtifactRef], list[BaseMessage]]:
        refs: list[ArtifactRef] = []
        messages: list[BaseMessage] = []
        for result in tool_results:
            canonical = _canonical_json(result)
            if len(canonical) > self.artifact_chars:
                ref = self._write_artifact(result)
                refs.append(ref)
                messages.append(
                    AIMessage(
                        content=(
                            f"[TOOL ARTIFACT] {ref.summary}; path={ref.path}; "
                            f"sha256={ref.sha256}"
                        )
                    )
                )
            else:
                # Small results are safe to keep inline, but retain stable
                # serialization so assembly is deterministic.
                messages.append(AIMessage(content=f"[TOOL RESULT] {canonical.decode('utf-8')}"))
        return refs, messages

    @staticmethod
    def _summary(dropped: Sequence[BaseMessage]) -> str:
        if not dropped:
            return ""
        # Do not echo payloads into the summary.  Counts and role names are
        # enough to make the reduction auditable and deterministic.
        roles = ",".join(getattr(message, "type", "message") for message in dropped)
        return f"[CONTEXT SUMMARY] omitted {len(dropped)} earlier message(s): {roles}"

    def _fit_messages(
        self,
        messages: list[BaseMessage],
        *,
        required: set[int],
        current_index: int | None,
    ) -> list[BaseMessage]:
        """Trim non-essential messages until the token budget is satisfied."""

        def total(items: Sequence[BaseMessage]) -> int:
            return sum(_estimate_tokens(_message_text(item)) for item in items)

        items = list(messages)
        while total(items) > self.max_tokens:
            # Prefer dropping/truncating oldest optional messages.  Required
            # system messages and the latest user instruction are never lost.
            candidate = next(
                (
                    i
                    for i, item in enumerate(items)
                    if i not in required and i != current_index
                ),
                None,
            )
            if candidate is None:
                break
            text = _message_text(items[candidate])
            excess_tokens = total(items) - self.max_tokens
            keep_chars = max(1, len(text) - excess_tokens * 4 - 3)
            if keep_chars < len(text):
                # Include the ellipsis in the requested bound.  The previous
                # implementation could turn a three-character "..." into
                # four characters and loop forever under a tight budget.
                if keep_chars <= 3:
                    shortened = text[:keep_chars]
                else:
                    shortened = text[: keep_chars - 3] + "..."
                if len(shortened) < len(text):
                    items[candidate] = AIMessage(content=shortened)
                    continue
            # If truncation cannot reduce the message any further, remove the
            # optional item so the bounded loop always makes progress.
            if candidate is not None:
                items.pop(candidate)
                required = {i - 1 if i > candidate else i for i in required if i != candidate}
                if current_index is not None and current_index > candidate:
                    current_index -= 1
        return items

    def assemble(
        self,
        *,
        messages: Sequence[BaseMessage] | None = None,
        tool_results: Sequence[Any] | None = None,
        thread_id: str | None = None,
        current_request: str | None = None,
    ) -> ContextAssembly:
        # Memory/project context is prompt-only evidence.  It is stored in the
        # checkpoint because the graph message channel is durable, but it must
        # not become an input to the next assembly or it would be duplicated on
        # every turn.  The marker is controlled by this service, not by model
        # text, so user content that happens to contain the same label is kept.
        raw_messages = list(messages or [])
        latest_human_index = next(
            (
                index
                for index in range(len(raw_messages) - 1, -1, -1)
                if isinstance(raw_messages[index], HumanMessage)
            ),
            None,
        )
        source = [
            message
            for index, message in enumerate(raw_messages)
            if not bool(
                (getattr(message, "additional_kwargs", None) or {}).get(
                    "agent_context_ephemeral",
                    False,
                )
            )
            or (latest_human_index is not None and index > latest_human_index)
        ]
        system_indices = [i for i, message in enumerate(source) if isinstance(message, SystemMessage)]
        human_indices = [i for i, message in enumerate(source) if isinstance(message, HumanMessage)]
        current_index = human_indices[-1] if human_indices else (len(source) - 1 if source else None)
        recent_limit = self.recent_turns * 2

        protected_indices = set(system_indices)
        if current_index is not None:
            protected_indices.add(current_index)
        pool = [message for i, message in enumerate(source) if i not in protected_indices]
        # The latest user instruction is part of the recent-turn budget.
        pool_limit = max(0, recent_limit - (1 if current_index is not None else 0))
        recent = pool[-pool_limit:] if pool_limit else []
        dropped = pool[:-pool_limit] if pool_limit and len(pool) > pool_limit else pool
        summary_text = self._summary(dropped)
        refs, artifact_messages = self._offload_tools(tool_results or [])

        memory_messages: list[BaseMessage] = []
        if self._memory_context is not None and thread_id:
            query = current_request
            if query is None and current_index is not None:
                query = _message_text(source[current_index])
            sections = self._memory_context.assemble(
                thread_id=str(thread_id),
                query=str(query or ""),
            )
            for section in sections.prompt_sections():
                # Retrieved memory is evidence/context, not a policy-bearing
                # system instruction.  It may be trimmed before system and the
                # current user request when the context budget is tight.
                memory_messages.append(
                    AIMessage(
                        content=str(section),
                        additional_kwargs={"agent_context_ephemeral": True},
                    )
                )

        assembled: list[BaseMessage] = [source[i] for i in system_indices]
        assembled.extend(memory_messages)
        if summary_text:
            assembled.append(AIMessage(content=summary_text))
        assembled.extend(recent)
        if current_index is not None:
            current = source[current_index]
            if current not in assembled:
                assembled.append(current)
        assembled.extend(artifact_messages)

        # Required indices are relative to this newly assembled list.
        required_positions = {
            i for i, message in enumerate(assembled) if isinstance(message, SystemMessage)
        }
        current_position = next(
            (i for i in range(len(assembled) - 1, -1, -1) if isinstance(assembled[i], HumanMessage)),
            None,
        )
        assembled = self._fit_messages(
            assembled,
            required=required_positions,
            current_index=current_position,
        )
        estimated = sum(_estimate_tokens(_message_text(message)) for message in assembled)
        # The protected messages can be larger than the budget by themselves;
        # this is the only case where the hard bound cannot be met without
        # violating the preservation contract.
        recent_messages = [
            message
            for message in assembled
            if not isinstance(message, SystemMessage)
            and not _message_text(message).startswith("[CONTEXT SUMMARY]")
            and not _message_text(message).startswith("[TOOL ")
        ][-recent_limit:]
        return ContextAssembly(
            messages=assembled,
            estimated_tokens=estimated,
            artifact_refs=refs,
            summary=summary_text,
            _recent_messages=recent_messages,
        )


# Explicit alias used by a few callers that refer to an assembly as a result.
ContextResult = ContextAssembly

__all__ = ["ArtifactRef", "ContextAssembly", "ContextResult", "ContextService"]
