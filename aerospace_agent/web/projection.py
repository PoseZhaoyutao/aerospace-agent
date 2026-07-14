"""Allowlisted, bounded projection of AgentOutput for untrusted browser clients."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from aerospace_agent.langgraph_agent.schema import AgentOutput, RunStatus

from .protocol import terminal_event_type

_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|secret|password|private[_-]?key)", re.I)


def _safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return "[TRUNCATED]"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:64]:
            name = str(key)
            result[name] = "[REDACTED]" if _SECRET_KEY.search(name) else _safe(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe(item, depth=depth + 1) for item in list(value)[:64]]
    return str(value)[:1024]


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "")[:limit]


def _error_item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        projected = _safe(value)
        return {
            str(key): item
            for key, item in projected.items()
            if str(key) not in {"traceback", "stack", "exception"}
        }
    return {"message": _bounded_text(value, 1024)}


def _has_approval_marker(output: AgentOutput) -> bool:
    for item in list(output.errors or []):
        if isinstance(item, dict) and str(item.get("code") or item.get("error_code")) == "human_approval_required":
            return True
    for item in list(output.tool_results or []):
        if getattr(item, "error_code", None) == "human_approval_required":
            return True
        if isinstance(item, dict) and str(item.get("error_code")) == "human_approval_required":
            return True
    return False


def project_agent_output(output: AgentOutput) -> dict[str, Any]:
    """Return only bounded primitive data intended for REST/WebSocket clients."""

    status = RunStatus(output.status)
    approval = status is RunStatus.INTERRUPTED and _has_approval_marker(output)
    event_type = terminal_event_type(status, approval_required=approval)
    errors = [_error_item(item) for item in list(output.errors or [])[:16]]
    for item in errors:
        if "message" in item:
            item["message"] = _bounded_text(item["message"], 1024)
    warnings = [_bounded_text(item, 512) for item in list(output.warnings or [])[:32]]
    citations = []
    for citation in list(output.citations or [])[:32]:
        data = citation.model_dump(mode="json") if hasattr(citation, "model_dump") else citation
        citations.append(_safe(data))
    projected: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": status.value,
        "event_type": event_type,
        "reason_code": "human_approval_required" if approval else None,
        "answer": _bounded_text(output.answer, 32_000),
        "checkpoint_id": _bounded_text(output.checkpoint_id, 256) or None,
        "citations": citations,
        "warnings": warnings,
        "errors": errors,
        "metrics": _safe(output.metrics or {}),
        "steps": int(output.steps or 0),
        "cycle_triggers": int(output.cycle_triggers or 0),
    }
    return projected


__all__ = ["project_agent_output"]
