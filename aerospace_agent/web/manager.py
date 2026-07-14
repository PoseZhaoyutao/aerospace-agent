"""Scoped runtime manager for WebUI requests."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from aerospace_agent.langgraph_agent.schema import AgentOutput

from .projection import project_agent_output
from .protocol import (
    HistoryMessage,
    HistoryResponse,
    RunTerminalEvent,
    ThreadSummary,
)


class ThreadScopeError(ValueError):
    """The requested thread is not owned by this manager scope."""


@dataclass
class _ThreadRecord:
    summary: ThreadSummary
    lock: threading.Lock


class AgentRuntimeManager:
    """Own one Agent instance and enforce server-side thread isolation."""

    def __init__(
        self,
        *,
        agent_factory: Callable[[], Any],
        project_id: str,
        workspace_id: str,
    ) -> None:
        self.agent_factory = agent_factory
        self.project_id = project_id
        self.workspace_id = workspace_id
        self._agent: Any | None = None
        self._threads: dict[str, _ThreadRecord] = {}
        self._active: set[str] = set()
        self._state_lock = threading.RLock()
        self._started = False
        self._shutdown = False

    @property
    def agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError("runtime manager is not started")
        return self._agent

    @property
    def active_runs(self) -> int:
        with self._state_lock:
            return len(self._active)

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._agent = self.agent_factory()
            self._started = True
            self._shutdown = False

    def shutdown(self) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
            agent, self._agent = self._agent, None
        if agent is not None:
            closer = getattr(agent, "close", None)
            if callable(closer):
                closer()

    def create_thread(self, *, title: str | None = None) -> ThreadSummary:
        if not self._started:
            self.start()
        thread_id = f"web-{uuid.uuid4().hex}"
        summary = ThreadSummary(
            project_id=self.project_id,
            thread_id=thread_id,
            title=(title or "New chat")[:256],
        )
        with self._state_lock:
            self._threads[thread_id] = _ThreadRecord(summary=summary, lock=threading.Lock())
        return summary

    def list_threads(self) -> list[ThreadSummary]:
        with self._state_lock:
            return [item.summary for item in self._threads.values()]

    def get_thread(self, thread_id: str) -> ThreadSummary:
        with self._state_lock:
            record = self._threads.get(thread_id)
        if record is None:
            raise ThreadScopeError("thread is not owned by this project")
        return record.summary

    def _record(self, thread_id: str) -> _ThreadRecord:
        with self._state_lock:
            record = self._threads.get(thread_id)
        if record is None:
            raise ThreadScopeError("thread is not owned by this project")
        return record

    def run(
        self,
        thread_id: str,
        *,
        request_id: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> RunTerminalEvent:
        record = self._record(thread_id)
        if not record.lock.acquire(blocking=False):
            raise RuntimeError("thread already running")
        with self._state_lock:
            self._active.add(thread_id)
        try:
            result = self.agent.run(message, thread_id=thread_id, context=context or {})
            if not isinstance(result, AgentOutput):
                result = AgentOutput.model_validate(result)
            projected = project_agent_output(result)
            return RunTerminalEvent(
                type=projected["event_type"],
                request_id=request_id,
                thread_id=thread_id,
                status=projected["status"],
                reason_code=projected["reason_code"],
                answer=projected["answer"],
                checkpoint_id=projected["checkpoint_id"],
                citations=projected["citations"],
                warnings=projected["warnings"],
                errors=projected["errors"],
            )
        except Exception as exc:
            return RunTerminalEvent(
                type="run.failed",
                request_id=request_id,
                thread_id=thread_id,
                status="error",
                answer="",
                reason_code=None,
            ).model_copy(update={"answer": str(exc)[:1024]})
        finally:
            with self._state_lock:
                self._active.discard(thread_id)
            record.lock.release()

    def history(self, thread_id: str) -> HistoryResponse:
        summary = self.get_thread(thread_id)
        messages: list[HistoryMessage] = []
        getter = getattr(self.agent, "get_checkpoint_history", None)
        if callable(getter):
            for checkpoint in list(getter(thread_id) or []):
                values = checkpoint.get("values", {}) if isinstance(checkpoint, dict) else {}
                raw_messages = values.get("messages", []) if isinstance(values, dict) else []
                for message in raw_messages:
                    role = getattr(message, "type", None) or getattr(message, "role", None)
                    content = getattr(message, "content", None)
                    if isinstance(message, dict):
                        role = message.get("role") or message.get("type")
                        content = message.get("content")
                    if role in {"human", "user"}:
                        role = "user"
                    elif role in {"ai", "assistant"}:
                        role = "assistant"
                    else:
                        continue
                    messages.append(
                        HistoryMessage(
                            role=role,
                            content=str(content or "")[:32_000],
                            checkpoint_id=str(checkpoint.get("checkpoint_id") or "") or None,
                        )
                    )
        return HistoryResponse(thread=summary, messages=messages)


__all__ = ["AgentRuntimeManager", "ThreadScopeError"]
