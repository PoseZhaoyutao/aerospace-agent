"""Policy and eligibility helpers for reversible agent evolution.

The policy is deliberately conservative: proposals are accepted only for
known writable roots and only when the configured evolution gate says the
agent is idle and has enough interaction context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from ..schema import EvolutionProposal

ALLOWED_ROOT_NAMES = ("knowledge", "memory", "evolved_skills", "workflows/evolved")


def normalize_relative(value: str | Path) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."}:
        raise ValueError("evolution target must be a non-empty relative path")
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute() or PureWindowsPath(text).drive:
        raise ValueError("evolution target must be relative")
    if text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", text):
        raise ValueError("evolution target must be relative")
    parts = re.split(r"[\\/]", text)
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("evolution target contains an unsafe component")
    return "/".join(parts)


def validate_target_path(path: str | Path, workspace: str | Path, allowed_roots: list[str | Path] | None = None) -> Path:
    """Resolve *path* under one of the allowed roots, rejecting symlinks."""
    relative = normalize_relative(path)
    root = Path(workspace).resolve()
    roots = [Path(item) for item in (allowed_roots or ALLOWED_ROOT_NAMES)]
    for configured in roots:
        allowed = (root / configured).resolve()
        # A symlinked root is not a trusted writable root.
        current = root
        try:
            for component in Path(configured).parts:
                current = current / component
                if current.is_symlink():
                    raise ValueError("evolution root may not contain symlinks")
        except OSError as exc:
            raise ValueError("unable to inspect evolution root") from exc
        candidate = root / relative
        try:
            candidate.relative_to(allowed)
        except ValueError:
            continue
        # Check the lexical path and all existing components, then resolve.
        current = root
        for component in Path(relative).parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("evolution target may not traverse a symlink")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise ValueError("evolution target escapes allowed root") from exc
        return candidate
    raise ValueError(f"evolution target is outside allowed roots: {relative}")


@dataclass(frozen=True)
class Eligibility:
    due: bool
    reason: str


class EvolutionPolicy:
    def __init__(self, *, enabled: bool = True, idle_minutes: int = 10, min_turns: int = 6,
                 context_ratio: float = 0.8, allowed_roots: list[str | Path] | None = None):
        self.enabled = bool(enabled)
        self.idle_minutes = int(idle_minutes)
        self.min_turns = int(min_turns)
        self.context_ratio = float(context_ratio)
        self.allowed_roots = list(allowed_roots or ALLOWED_ROOT_NAMES)

    def is_due(self, *, idle_minutes: float = 0.0, turn_count: int = 0,
               context_tokens: int = 0, max_context_tokens: int = 0,
               already_applied: bool = False) -> Eligibility:
        if not self.enabled:
            return Eligibility(False, "disabled")
        if already_applied:
            return Eligibility(False, "already_applied")
        if float(idle_minutes) < self.idle_minutes:
            return Eligibility(False, "not_idle")
        enough_turns = int(turn_count) >= self.min_turns
        ratio = (float(context_tokens) / float(max_context_tokens)) if max_context_tokens else 0.0
        enough_context = ratio >= self.context_ratio
        if not (enough_turns or enough_context):
            return Eligibility(False, "insufficient_context")
        return Eligibility(True, "eligible")


def parse_llm_proposal(payload: Any) -> EvolutionProposal | None:
    """Strictly parse an LLM payload; malformed or extra data is a no-op."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    if not isinstance(payload, Mapping):
        return None
    allowed = {"thread_id", "run_id", "checkpoint_id", "rationale", "changes", "source", "unfinished_items", "required_validations"}
    if set(payload) - allowed:
        return None
    changes = payload.get("changes", [])
    if not isinstance(changes, list):
        return None
    for change in changes:
        if not isinstance(change, Mapping) or set(change) - {"operation", "path", "content"}:
            return None
    try:
        return EvolutionProposal.model_validate(dict(payload))
    except Exception:
        return None


__all__ = ["ALLOWED_ROOT_NAMES", "Eligibility", "EvolutionPolicy", "normalize_relative", "validate_target_path", "parse_llm_proposal"]
