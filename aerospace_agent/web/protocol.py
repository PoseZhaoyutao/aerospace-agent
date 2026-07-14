"""Strict, versioned contracts shared by the WebUI gateway and frontend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aerospace_agent.langgraph_agent.schema import RunStatus

SCHEMA_VERSION = "1.0.0"


class WebModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Envelope(WebModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=128)


class RunStartRequest(Envelope):
    type: Literal["run.start"] = "run.start"
    message: str = Field(min_length=1, max_length=32_000)


class HealthResponse(WebModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    status: Literal["ready", "degraded"]
    service: Literal["aerospace-agent-webui"] = "aerospace-agent-webui"


class ThreadCreateRequest(WebModel):
    title: str | None = Field(default=None, max_length=256)


class ThreadSummary(WebModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    project_id: str = Field(min_length=1, max_length=256)
    thread_id: str = Field(min_length=1, max_length=128)
    title: str = Field(default="New chat", max_length=256)
    created_at: str | None = None
    updated_at: str | None = None
    checkpoint_id: str | None = None


class ThreadListResponse(WebModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    threads: list[ThreadSummary] = Field(default_factory=list)


class HistoryMessage(WebModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=32_000)
    checkpoint_id: str | None = None


class HistoryResponse(WebModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    thread: ThreadSummary
    messages: list[HistoryMessage] = Field(default_factory=list)


class RuntimeStatusResponse(WebModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    model: str = Field(default="unknown", max_length=256)
    connection: Literal["connecting", "connected", "disconnected", "error"] = "disconnected"
    active_runs: int = Field(default=0, ge=0)


class RunTerminalEvent(Envelope):
    type: Literal["run.completed", "run.interrupted", "run.failed"]
    status: RunStatus
    reason_code: Literal["human_approval_required"] | None = None
    answer: str = Field(default="", max_length=32_000)
    checkpoint_id: str | None = None
    citations: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, object]] = Field(default_factory=list)


def terminal_event_type(
    status: RunStatus | str,
    *,
    approval_required: bool = False,
) -> str:
    normalized = RunStatus(status)
    if approval_required or normalized is RunStatus.INTERRUPTED:
        return "run.interrupted"
    if normalized in {RunStatus.SUCCESS, RunStatus.PARTIAL}:
        return "run.completed"
    return "run.failed"


__all__ = [
    "SCHEMA_VERSION",
    "Envelope",
    "RunStartRequest",
    "HealthResponse",
    "ThreadCreateRequest",
    "ThreadSummary",
    "ThreadListResponse",
    "HistoryMessage",
    "HistoryResponse",
    "RuntimeStatusResponse",
    "RunTerminalEvent",
    "terminal_event_type",
]
