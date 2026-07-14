"""Pure, checkpoint-backed loop detection for LangGraph state.

Detection history belongs to the serialized graph state (``state_fingerprints``
and ``intervention_count``), not to a process-local detector instance.  This
keeps independent threads isolated and makes replay/checkpoint behaviour
deterministic.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Tuple


def _normalise(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalise(value[k]) for k in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple, set)):
        items = [_normalise(item) for item in value]
        if isinstance(value, set):
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return items
    if hasattr(value, "model_dump"):
        return _normalise(value.model_dump(mode="json"))
    if hasattr(value, "content"):
        return _normalise(getattr(value, "content"))
    return str(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _normalise(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def fingerprint(
    action: str = "",
    tool_name: Optional[str] = None,
    target: Any = None,
    params: Optional[Mapping[str, Any]] = None,
    observation: Any = None,
    intent: Optional[str] = None,
    payload: Any = None,
    **extra: Any,
) -> str:
    """Return a stable SHA-256 identity for one logical action.

    The identity includes normalized action/tool/target, sorted parameters,
    intent and an observation digest.  Runtime depth/step counters are
    intentionally excluded: monotonic counters must not prevent recognizing
    an otherwise repeated action.
    """

    if payload is not None and params is None and observation is None:
        # ``payload`` is the compact compatibility spelling used by retrieve
        # and respond actions.
        params = payload if isinstance(payload, Mapping) else {"payload": payload}
    data: dict[str, Any] = {
        "action": str(action or ""),
        "tool_name": str(tool_name) if tool_name is not None else "",
        "target": _normalise(target),
        "params": _normalise(params or {}),
        "intent": str(intent) if intent is not None else "",
    }
    if observation is not None:
        data["observation_digest"] = hashlib.sha256(_canonical(observation)).hexdigest()
    for key, value in extra.items():
        if key not in {"step_count", "recursion_depth", "depth"}:
            data[str(key)] = _normalise(value)
    return hashlib.sha256(_canonical(data)).hexdigest()


def _state_fingerprints(state: Mapping[str, Any]) -> list[str]:
    values = state.get("state_fingerprints", state.get("cycle_hash_history", []))
    if values is None:
        return []
    return [str(value) for value in values]


def evaluate_cycle(
    state: Mapping[str, Any],
    *,
    action: str = "",
    tool_name: Optional[str] = None,
    target: Any = None,
    params: Optional[Mapping[str, Any]] = None,
    observation: Any = None,
    intent: Optional[str] = None,
    payload: Any = None,
    max_repeats: int = 3,
    max_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate one action and return only a serialized state delta.

    ``state`` is never modified.  The returned history and intervention count
    are derived solely from values already in that state, so two thread state
    dictionaries cannot leak counters into each other.
    """

    if max_repeats < 1:
        raise ValueError("max_repeats must be positive")
    current = fingerprint(
        action=action,
        tool_name=tool_name,
        target=target,
        params=params,
        observation=observation,
        intent=intent,
        payload=payload,
    )
    history = _state_fingerprints(state)
    repeats = history.count(current) + 1
    next_history = history + [current]
    prior_interventions = int(state.get("intervention_count", 0) or 0)
    intervention = repeats >= max_repeats
    delta: dict[str, Any] = {
        "state_fingerprints": next_history,
        # Keep the legacy channel synchronized for old checkpoints.
        "cycle_hash_history": next_history,
    }
    if max_steps is not None and int(state.get("step_count", 0) or 0) >= max_steps:
        intervention = True
        delta["termination_reason"] = "max_steps"
    if intervention:
        delta["intervention_count"] = prior_interventions + 1
        delta["cycle_count"] = int(state.get("cycle_count", 0) or 0) + 1
    else:
        delta["intervention_count"] = prior_interventions
    return delta


class CycleDetector:
    """Compatibility facade around :func:`evaluate_cycle`.

    This class intentionally stores thresholds only.  All counters are read
    from and returned to the caller's serializable state.
    """

    def __init__(self, max_repeats: int = 3, max_steps: int = 15) -> None:
        if max_repeats < 1 or max_steps < 1:
            raise ValueError("cycle thresholds must be positive")
        self.max_repeats = int(max_repeats)
        self.max_steps = int(max_steps)

    def compute_state_hash(self, state: Mapping[str, Any]) -> str:
        messages = state.get("messages", [])
        recent = []
        for message in list(messages)[-3:]:
            content = getattr(message, "content", message)
            recent.append(_normalise(content))
        calls = state.get("tool_calls", state.get("tool_requests", [])) or []
        return fingerprint(
            action=str(state.get("action", "state")),
            tool_name=(calls[-1].get("tool_name", calls[-1].get("name")) if calls and isinstance(calls[-1], Mapping) else None),
            target=state.get("target"),
            params={
                "recent_messages": recent,
                "decision": state.get("decision"),
                "tool_results": state.get("tool_results", []),
            },
            intent=state.get("intent", ""),
            observation=state.get("observation", state.get("final_answer")),
        )

    def check(self, state: Mapping[str, Any]) -> Tuple[bool, str, dict[str, Any]]:
        """Return ``(is_cycle, action, state_delta)`` for compatibility."""

        messages = state.get("messages", [])
        action = str(state.get("action", "state"))
        payload = state.get("payload")
        tool_name = state.get("tool_name")
        target = state.get("target")
        params = state.get("params")
        if not action or action == "state":
            # Hashing the complete state view keeps legacy callers working,
            # while still omitting monotonic counters from the fingerprint.
            action = "state"
            params = {
                "messages": [getattr(message, "content", message) for message in list(messages)[-3:]],
                "intent": state.get("intent", ""),
                "tool_calls": state.get("tool_calls", []),
            }
            payload = None
        delta = evaluate_cycle(
            state,
            action=action,
            tool_name=tool_name,
            target=target,
            params=params,
            observation=state.get("observation"),
            intent=state.get("intent"),
            payload=payload,
            max_repeats=self.max_repeats,
            max_steps=self.max_steps,
        )
        reason = "continue"
        is_cycle = False
        if delta.get("termination_reason") == "max_steps":
            reason, is_cycle = "force_stop", True
        elif int(delta.get("intervention_count", 0)) > int(state.get("intervention_count", 0) or 0):
            reason, is_cycle = "intervene", True
        return is_cycle, reason, delta

    def build_intervention_message(self) -> str:
        return (
            "[SYSTEM INTERVENTION] A repeated state/action was detected. "
            "Change the tool, target, or decomposition before continuing."
        )

    def build_force_stop_message(self) -> str:
        return f"[FORCE STOP] Maximum graph steps reached ({self.max_steps})."

    def reset(self) -> None:
        """Retained for callers of the old mutable detector; there is no state to reset."""

    def get_stats(self, state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        state = state or {}
        history = _state_fingerprints(state)
        return {
            "total_steps": int(state.get("step_count", len(history)) or 0),
            "unique_states": len(set(history)),
            "interventions": int(state.get("intervention_count", 0) or 0),
            "max_repeats_threshold": self.max_repeats,
            "max_steps_threshold": self.max_steps,
        }


__all__ = ["CycleDetector", "evaluate_cycle", "fingerprint"]
