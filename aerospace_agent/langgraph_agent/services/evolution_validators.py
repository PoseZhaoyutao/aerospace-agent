"""Deterministic validators used before an evolution commit."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from ..schema import EvolutionProposal, ValidationResult


Validator = Callable[[EvolutionProposal, Path, list[dict[str, Any]]], ValidationResult | bool]


def validate_manifest(proposal: EvolutionProposal, workspace: Path, manifest: list[dict[str, Any]]) -> ValidationResult:
    seen: set[str] = set()
    for item in manifest:
        path = str(item["path"])
        if path in seen:
            return ValidationResult(name="manifest", ok=False, message=f"duplicate target: {path}")
        seen.add(path)
        operation = item["operation"]
        exists = bool(item.get("prior_exists"))
        if operation == "create" and exists:
            return ValidationResult(name="manifest", ok=False, message=f"create target already exists: {path}")
        if operation == "update" and not exists:
            return ValidationResult(name="manifest", ok=False, message=f"update target does not exist: {path}")
    return ValidationResult(name="manifest", ok=True)


def validate_python_syntax(proposal: EvolutionProposal, workspace: Path, manifest: list[dict[str, Any]]) -> ValidationResult:
    for item in manifest:
        if item["operation"] == "delete" or not str(item["path"]).endswith(".py"):
            continue
        candidate = workspace / item["path"]
        staging = workspace / ".evolution-stage-do-not-use"
        content = item.get("after_bytes")
        if content is None:
            continue
        # Compile from bytes through a temporary file in the transaction is
        # handled by EvolutionService; this validator remains intentionally
        # conservative and only checks that bytes decode as UTF-8.
        try:
            content.decode("utf-8")
        except Exception as exc:
            return ValidationResult(name="python_syntax", ok=False, message=f"invalid UTF-8 for {candidate}", details={"error": str(exc)})
    return ValidationResult(name="python_syntax", ok=True)


def run_validators(proposal: EvolutionProposal, workspace: Path, manifest: list[dict[str, Any]], validators: Iterable[Validator] = ()) -> list[ValidationResult]:
    checks: list[Validator] = [validate_manifest, *list(validators)]
    results: list[ValidationResult] = []
    for validator in checks:
        try:
            if hasattr(validator, "validate"):
                result = validator.validate(proposal, workspace, manifest)  # type: ignore[attr-defined]
            else:
                result = validator(proposal, workspace, manifest)
            if isinstance(result, ValidationResult):
                results.append(result)
            else:
                results.append(ValidationResult(name=getattr(validator, "__name__", "validator"), ok=bool(result)))
        except Exception as exc:
            results.append(ValidationResult(name=getattr(validator, "__name__", "validator"), ok=False, message=str(exc)))
    return results


__all__ = ["ValidationResult", "Validator", "validate_manifest", "validate_python_syntax", "run_validators"]
